"""pt_search —— 超参搜索 + 邻域稳定区池化。

为什么**不报「最高分参数」**（这是本模块存在的理由，不是保守）：

目标函数只能取 val 窗（`cand_val_ic`）。因为 S03 的门禁判据本身就是拿 **test** 窗裁决
（`ct >= lt and ct > 0`，train_from_xy 里逐字抽出的那段）。拿 test 当搜索目标
= 直接拟合判据 —— 搜出来的「最好」就是门禁本身，再把它当证据汇报，等于把答案抄一遍。

可一旦用 val 当目标，val 就被选择过程污染了。所以本工具报的不是峰值，是两件事：

  1. **邻域稳定区池化**：取「离最高分不超过容差」的所有格，在网格格点上找最大连通块，
     报这个**区域**（以及区域内每组参数的取值范围与中心值），不报那一格峰值。
     区域只有一格 = 那是针尖不是稳定区 ⇒ 不报推荐值，判「判不了」。
     （借鉴 quant-research-skill 的 `scripts/region_pool.py` 与
     `references/03-surface.md`、`04-selection-bias.md`：别挑峰值，划稳定区再池化。）

  2. **强制留一段独立 holdout**：从 train 窗**末尾**切出 H 天，搜索全程既不当训练数据、
     也不当目标函数；搜索结束后用选出的参数在 holdout 上跑一次，与**同窗的现役模型**比。
     实现只用包内既有机制：把 `TRAIN_ARGS` 改成
     `train_days=252-H, val_days=21, test_days=21+H`。
     窗口本来就自面板末端切分（`train_dates = dates[-(t+v+te):-(v+te)]`），
     所以这样一改，val 窗正好落在那 H 天上，IC 仍由包内 `_ens_metrics` 算，
     **不另写一套口径**。

扫描只改**试次目录里的入口副本**（每个试次自己的一份 `train_from_xy.py`），
母本 / Z 包 / `_sync` 生成器一律只读；改写前后都留 sha1，这次跑的到底是哪一份入口可追。
默认 dry-run，`--commit` 才真训。

    python pt.py search --bundle . --jobs 2            # 只看计划（网格从包里找）
    python pt.py search --bundle . --jobs 2 --commit   # 真跑

网格**随包走**：本机 `pt_pack.py pack --grid` 把它装进包里，这里正常不用给。
包里没有网格时本工具会停下并说明去向本机要 —— 不要自己编一份。
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pt_common as C  # noqa: E402
import pt_train as T  # noqa: E402

# 搜索产物：这三个名字同时被 pt_pack 的 EXPECTED_SEARCH 认（collect 靠它们判断「跑过搜索」）
SEARCH_FILES = ("surface.csv", "region.json", "trials.csv")
ENTRY_NAME = "train_from_xy.py"
CFG_LITERAL = "SEG_CFG"
ARGS_LITERAL = "TRAIN_ARGS"

OBJECTIVE = "cand_val_ic"
# 搜索网格在包根的固定名字（本机 pack --grid 装的就是它）。
# 异地不带 --grid 时按这个名字找：路径可预测，不需要操作者记任何东西。
GRID_IN_ZIP = "SEARCH_GRID.json"
OBJECTIVE_WHY = ("目标只取 val 窗。test 窗是门禁（ct >= lt and ct > 0）的判据本身，"
                 "拿它当搜索目标等于直接拟合判据 —— 搜出来的最好就是门禁的答案。")

DEFAULT_HOLDOUT_DAYS = 21
DEFAULT_IC_TOL = 0.01          # IC 单位。判「两格是否算同一个区」的兜底容差
DEFAULT_SEC_PER_RUN = 40.0     # 只用于 dry-run 的耗时估算
MIN_REGION_CELLS = 2           # 连通块小于这个数 = 针尖，不是稳定区
THIN_REGION_FRACTION = 0.25    # 区域占全部格子的比例低于此 = 薄

# 能扫的键 —— **白名单**，每条都要写明它进了哪一步计算，并给出一条**能在源码里验的**证据。
#
# 为什么不能用黑名单：黑名单挡不住以后新加的键；而「改了其实没影响」的键会产出一张
# 每格数字都一模一样的表面，看起来像「这个参数不重要」，实际是「这个参数根本没接上」
# —— 这是本工具最不能出的错，因为它给出的是一个**看起来很干净**的结论。
#
# `in_src` 是正则，必须在入口源码的**非注释**部分真的命中。为什么要它：
# 光看"名字出现过"不够 —— 键可以只出现在配置字面量和打印文案里（`tilt` 就是），
# 也可以被整体展开而从不被点名（`**cfg["lgb"]` 不会写出 num_leaves）。
# 把"它怎么被用上的"写成一条可核验的证据，是唯一能自动查又不说谎的做法。
KNOB_EVIDENCE = {
    "ensemble_weight": {
        "why": "决定了候选侧集成的混合权，直接改 cand 的 IC",
        "in_src": r"""cfg(?:\.get\(|\[)\s*["']ensemble_weight["']""",
    },
    "lgb.num_leaves": {
        "why": "入口里 LGB 参数是整体展开进 lightgbm 的，改它就改了分箱与树的形状",
        "in_src": r"""\*\*\s*cfg\[["']lgb["']\]""",
    },
    "lgb.max_depth": {
        "why": "同 lgb.num_leaves（整体展开）",
        "in_src": r"""\*\*\s*cfg\[["']lgb["']\]""",
    },
    "lgb.learning_rate": {
        "why": "同 lgb.num_leaves（整体展开）",
        "in_src": r"""\*\*\s*cfg\[["']lgb["']\]""",
    },
    "lgb.min_child_samples": {
        "why": "同 lgb.num_leaves（整体展开）",
        "in_src": r"""\*\*\s*cfg\[["']lgb["']\]""",
    },
}
KNOB_FORBIDDEN = {
    "universe": "只是名字/标签，不参与任何计算；改了等于记录与实物不符",
    "horizon": "标签列由入口常量 FWD_COL=\"fwd21\" 与 Y/Y_target.npy 决定，"
               "改它只改记录、不改标签 —— 扫出来的差异全是噪声",
    "tilt": "**本包该键没接上计算**（全文只出现在打印文案里）。扫它只会得到一排相同的数，"
            "会被人读成「tilt 不影响」，那是假结论",
}

# 报告里老实列出的局限。不列出来，读的人会以为这些事已经做过了。
LIMITATIONS_BASE = [
    "目标函数是 val 窗，val 已被「选参数」这件事污染。holdout 只复核「降维之后还站不站得住」，"
    "不等于做了完整的多重比较校正。",
    "本工具**没做**多重比较校正：N 格 × 每格 M 个种子的朴素最优本身是有偏的。"
    "报区域而不是报峰值是对这个偏差的部分对冲，不是消除。",
    "网格格点之间只按声明顺序算相邻（邻域 = 某一维上下相邻一格），"
    "不代表参数空间里的真实距离（尤其 learning_rate 这类对数尺度参数）。",
    "holdout 那一次的训练窗比搜索短 H 天，它的 IC 水平**不能**与搜索期的 val IC 直接比；"
    "只能与**同一窗**的现役侧（live_val_ic）比。",
]


def _num(x):
    """version_row 里的数是字符串。空串/NA 一律当「没有」，绝不当 0。"""
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.upper() in ("NA", "NAN", "NONE"):
            return None
        v = float(s)
        if v != v:
            return None
        return v
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════ 入口改写
#
# 只改**试次目录里的副本**。做法：AST 定位 SEG_CFG / TRAIN_ARGS 两个字面量的字节区间，
# 只替换那一段源码；替换后用「把该字面量换成占位符再比 AST」证明**其它节点一个都没动**。

def _line_starts(data: bytes):
    starts = [0]
    for i, ch in enumerate(data):
        if ch == 0x0A:
            starts.append(i + 1)
    return starts


def _abs_pos(starts, lineno, col_offset):
    # 注意：Python 的 col_offset 是 **UTF-8 字节**偏移，不是字符偏移。
    # 这个文件里有中文，按字符算会错位 —— 所以全程在 bytes 上做。
    return starts[lineno - 1] + col_offset


def _find_literal_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return node.value
    return None


def _render_scalar(v):
    return json.dumps(v, ensure_ascii=False)


def _render_dict(cfg: dict, stmt_col: int, one_line: bool = False, nl: str = "\n") -> str:
    """按入口文件的原样式渲染。stmt_col = 赋值语句名那一列（模块级为 0）。

    SEG_CFG 是缩进 4 的多行（子字典 8），收尾花括号回到 stmt_col；
    TRAIN_ARGS 原文就是一行，故 one_line。

    `nl` 必须跟着**原文件**走：入口是 CRLF，用 "\\n" 渲染会静默少掉每行一个 \\r
    （实测：SEG_CFG 那 12 行正好少 11 字节），于是"原样回写"不再逐字节相同，
    而这个差异在真机上看不出来 —— 只有逐字节比才现形。
    """
    if one_line:
        return "{" + ", ".join("%s: %s" % (_render_scalar(k), _render_scalar(v))
                               for k, v in cfg.items()) + "}"
    pad = " " * stmt_col
    lines = ["{"]
    keys = list(cfg.items())
    for i, (k, v) in enumerate(keys):
        comma = "," if i < len(keys) - 1 else ""
        if isinstance(v, dict):
            lines.append("%s    %s: {" % (pad, _render_scalar(k)))
            sub = list(v.items())
            for j, (sk, sv) in enumerate(sub):
                lines.append("%s        %s: %s%s"
                             % (pad, _render_scalar(sk), _render_scalar(sv),
                                "," if j < len(sub) - 1 else ""))
            lines.append("%s    }%s" % (pad, comma))
        else:
            lines.append("%s    %s: %s%s" % (pad, _render_scalar(k), _render_scalar(v), comma))
    lines.append(pad + "}")
    return nl.join(lines)


def _replace_literal(data: bytes, tree, name, new_cfg, one_line=False):
    """把 <name> 的字面量整段替换掉，返回 (新字节, 旧值)。只动这一段。"""
    val = _find_literal_value(tree, name)
    if val is None:
        raise ValueError("入口里找不到模块级 %s 字面量" % name)
    if not isinstance(val, ast.Dict):
        raise ValueError("入口里的 %s 不是字典字面量（是 %s）" % (name, type(val).__name__))
    try:
        old = ast.literal_eval(val)
    except Exception as exc:
        raise ValueError("%s 不是纯字面量，本工具不敢改：%s" % (name, exc))
    if not isinstance(old, dict):
        raise ValueError("%s 求值出来不是字典" % name)

    starts = _line_starts(data)
    s = _abs_pos(starts, val.lineno, val.col_offset)
    e = _abs_pos(starts, val.end_lineno, val.end_col_offset)

    # 语句名所在列（模块级 = 0）。用它对齐缩进，让「没改的键」渲染出来与原文逐字相同。
    stmt_col = 0
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    stmt_col = tgt.col_offset
    nl = "\r\n" if b"\r\n" in data else "\n"
    rendered = _render_dict(new_cfg, stmt_col, one_line=one_line, nl=nl).encode("utf-8")
    return data[:s] + rendered + data[e:], old


def _ast_fingerprint_without(text: str, names) -> str:
    """把 `names` 这些字面量全换成占位符后 dump 整棵树。

    改前改后各算一次：相同 ⇒ **除这几个字面量外**什么都没碰。
    这是「改了哪儿」从口头声明变成可核对事实的那一步。

    `names` 必须是**这次允许被改的全部**：只列一个的话，另一个合法的改动会被
    判成"还有别的节点被动了"，把正常路径也拦下来（实测踩过：只占位 SEG_CFG 时，
    同时改 TRAIN_ARGS 的那条路 100% 失败，而只改 SEG_CFG 的 8 格全绿 ——
    所以每条分支都要真跑一遍，不能拿绿的那条推断没跑的那条）。
    """
    if isinstance(names, str):
        names = (names,)
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names:
                    node.value = ast.Constant(value="__PT_PLACEHOLDER__")
    return ast.dump(tree)


class EntryPatchError(Exception):
    pass


def make_entry_patcher(overrides: dict, train_args_override: dict | None, base_text: str):
    """造一个 run_one 用的 entry_patch 回调：改试次副本里的 SEG_CFG（和 TRAIN_ARGS）。"""
    fp_before = _ast_fingerprint_without(base_text, (CFG_LITERAL, ARGS_LITERAL))

    def _patch(trial_dir: Path) -> dict:
        target = Path(trial_dir) / ENTRY_NAME
        data = target.read_bytes()
        text_before = data.decode("utf-8")
        if text_before != base_text:
            # 试次里的副本理应等于母本那份。不等就是中间被谁动过 ⇒ 停，别在不明来路的入口上跑。
            raise EntryPatchError(
                "试次副本与预期入口不一致（可能母本被改过）。请重新打包，不要在来路不明的入口上跑搜索。")

        sha_before = C.sha1_file(target)
        out = data

        # 1) SEG_CFG：超参覆盖
        tree = ast.parse(out.decode("utf-8"))
        cur_cfg = ast.literal_eval(_find_literal_value(tree, CFG_LITERAL))
        new_cfg = copy.deepcopy(cur_cfg)
        for dotted, v in overrides.items():
            _set_dotted(new_cfg, dotted, v)
        out, old_cfg = _replace_literal(out, tree, CFG_LITERAL, new_cfg)

        got_args = None
        if train_args_override:
            tree = ast.parse(out.decode("utf-8"))
            cur_args = ast.literal_eval(_find_literal_value(tree, ARGS_LITERAL))
            new_args = dict(cur_args)
            new_args.update(train_args_override)
            out, old_args = _replace_literal(out, tree, ARGS_LITERAL, new_args, one_line=True)
            got_args = {"before": old_args, "after": new_args}

        # 改前改后，除这两个字面量外的 AST 必须逐字相同。
        # 两个一定要一起占位：只占一个的话，另一个的**合法**改动会被误判成"还有别的节点被动了"。
        text_after = out.decode("utf-8")
        if _ast_fingerprint_without(text_after, (CFG_LITERAL, ARGS_LITERAL)) != fp_before:
            raise EntryPatchError(
                "改写后发现 %s / %s 之外还有节点被改动，已中止（未落盘）"
                % (CFG_LITERAL, ARGS_LITERAL))

        # 落盘前最后一道：新字面量求值必须**恰好**等于预期的字典
        chk = ast.parse(text_after)
        if ast.literal_eval(_find_literal_value(chk, CFG_LITERAL)) != new_cfg:
            raise EntryPatchError("改写后的 SEG_CFG 求值不等于预期，已中止（未落盘）")
        if got_args is not None:
            got_now = ast.literal_eval(_find_literal_value(chk, ARGS_LITERAL))
            if got_now != got_args["after"]:
                raise EntryPatchError("改写后的 TRAIN_ARGS 求值不等于预期，已中止（未落盘）")

        target.write_bytes(out)
        return {"entry_sha1_before": sha_before,
                "entry_sha1_after": C.sha1_file(target),
                "seg_cfg_before": old_cfg, "seg_cfg_after": new_cfg,
                "train_args_before": (got_args or {}).get("before"),
                "train_args_after": (got_args or {}).get("after")}

    return _patch


def _set_dotted(d: dict, dotted: str, value):
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            raise ValueError("键 %s 在入口的配置里不存在（%s 不是子字典）" % (dotted, p))
        cur = cur[p]
    if parts[-1] not in cur:
        raise ValueError("键 %s 在入口的配置里不存在" % dotted)
    cur[parts[-1]] = value


def _get_dotted(d: dict, dotted: str):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


# ══════════════════════════════════════════════════════════════════════ 网格

def load_grid(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("网格文件的顶层必须是对象")
    grid = obj.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("网格文件里没有 `grid`，或它是空的。样例见 grid.example.json")
    out = {}
    for k, vs in grid.items():
        if not isinstance(vs, list) or not vs:
            raise ValueError("`%s` 的取值必须是**非空列表**" % k)
        seen, uniq = set(), []
        for v in vs:
            if isinstance(v, (dict, list)):
                raise ValueError("`%s` 里只允许标量（数/字符串/真假）" % k)
            key = repr(v)
            if key in seen:
                print("[pt] 注意：`%s` 的取值里有重复项 %r，已去掉一个" % (k, v))
                continue
            seen.add(key)
            uniq.append(v)
        out[k] = uniq
    return obj, out


def check_knobs(grid: dict, seg_cfg: dict, entry_text: str):
    """返回 (问题, 提示)。问题非空 ⇒ 拒绝跑。

    两道关：
      A. 键必须在**包内声明**（MANIFEST.seg_config）里真的存在 —— 不另造参数空间；
      B. 键必须在白名单里（白名单每条都写了「它进了哪一步计算」）。
         在黑名单里 / 不在白名单里，都拒绝，并说清楚为什么。
    """
    problems, notes = [], []
    for k in grid:
        if k in KNOB_FORBIDDEN:
            problems.append("`%s` 不能扫：%s" % (k, KNOB_FORBIDDEN[k]))
            continue
        if k not in KNOB_EVIDENCE:
            problems.append(
                "`%s` 不在本工具的可扫清单里。清单是**白名单**：只有逐个确认过"
                "「这个键真的进了计算」的键才允许扫。"
                "要加，请在 pt_search.py 的 KNOB_EVIDENCE 里写明它进了哪一步计算。" % k)
            continue
        cur = _get_dotted(seg_cfg, k)
        if cur is None:
            problems.append("`%s` 在包内声明的 seg_config 里不存在 —— 参数空间以包内声明为准，"
                            "不另造一套。" % k)
            continue
        # C. 实测证据：白名单里写明的那条"它怎么被用上的"必须真的在源码里命中。
        ev = KNOB_EVIDENCE[k]
        if not _evidence_in_src(entry_text, ev["in_src"]):
            problems.append(
                "`%s` 白名单里写的接入方式 `%s` **在这份入口源码里找不到**。"
                "要么这个包这一版根本没接上它，要么接入方式变了 —— 两种都不该扫："
                "扫一个没接上的键只会得到一排相同的数，会被读成「这个参数不重要」。"
                % (k, ev["in_src"]))
            continue
        notes.append("`%s` = %r  ->  %s" % (k, cur, ev["why"]))
    return problems, notes


def _strip_comments(text: str) -> str:
    """把注释换掉（保留行号/偏移）。证据必须命中**真代码**，命中一句注释不算数。"""
    import io
    import tokenize
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except Exception:
        return text                     # 解析不了就不剥（宁可严一点，也不能假装剥过了）
    lines = text.splitlines(keepends=True)
    out = list(lines)
    for tok in toks:
        if tok.type != tokenize.COMMENT:
            continue
        (r, c), (r2, c2) = tok.start, tok.end
        if r != r2 or not (1 <= r <= len(out)):
            continue
        ln = out[r - 1]
        nl = "\n" if ln.endswith("\n") else ""
        body = ln[:-len(nl)] if nl else ln
        out[r - 1] = body[:c] + " " * (len(body) - c) + nl
    return "".join(out)


def _evidence_in_src(text: str, pattern: str) -> bool:
    import re as _re
    return _re.search(pattern, _strip_comments(text)) is not None


def build_cells(grid: dict):
    """笛卡尔积。cell_id 由**声明顺序**决定，与 --jobs 无关（可复现）。"""
    dims = list(grid.keys())
    cells = []

    def rec(i, acc):
        if i == len(dims):
            cells.append({"cell_id": "c%02d" % len(cells),
                          "idx": tuple(acc),
                          "knobs": {d: grid[d][j] for d, j in zip(dims, acc)}})
            return
        for j in range(len(grid[dims[i]])):
            rec(i + 1, acc + [j])

    rec(0, [])
    return dims, cells


# ══════════════════════════════════════════════════════════════════════ 表面 / 区域

def _mean_sd(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    if len(v) == 1:
        return v[0], None
    return statistics.mean(v), statistics.stdev(v)


def build_surface(cells, dims, grid, trials):
    """每格一行。失败格单列 status，不当 0 计入。"""
    by_cell = {}
    for t in trials:
        by_cell.setdefault(t["cell_id"], []).append(t)
    rows, failed = [], 0
    for c in cells:
        rs = by_cell.get(c["cell_id"], [])
        objs = [_num((r.get("version_row") or {}).get(OBJECTIVE)) for r in rs]
        tests = [_num((r.get("version_row") or {}).get("cand_test_ic")) for r in rs]
        oks = [x for x in objs if x is not None]
        bad = [r for r in rs if r.get("rc") != 0]
        failed += len(bad)
        mean, sd = _mean_sd(oks)
        tmean, _ = _mean_sd(tests)
        gate = all((r.get("decision") == "replaced") for r in rs) if rs and not bad else False
        row = {"cell_id": c["cell_id"], "idx": list(c["idx"])}
        for d in dims:
            row["knob:" + d] = c["knobs"][d]
        row.update({
            "n_runs": len(rs), "n_ok": len(oks),
            "obj_mean": None if mean is None else round(mean, 6),
            "obj_sd": None if sd is None else round(sd, 6),
            "obj_min": None if not oks else round(min(oks), 6),
            "obj_max": None if not oks else round(max(oks), 6),
            "test_ic_mean": None if tmean is None else round(tmean, 6),
            "gate_pass": bool(gate),
            "status": ("ok" if oks else ("failed" if bad else "no_result")),
            "in_region": False, "region_id": None, "is_peak": False,
        })
        rows.append(row)
    return rows, failed


def region_pool(rows, dims, grid, tol):
    """在网格格点上找「离最高分不超过 tol」的最大连通块。连通 = 某一维上下相邻一格。

    回报区域而不是峰值。区域只有一格 ⇒ 判不了（针尖不是稳定区）。
    """
    key = {tuple(r["idx"]): r for r in rows if r["obj_mean"] is not None}
    if not key:
        return {"ok": False, "why": "没有任何一格算出目标值，无从谈稳定区。"}
    best = max(r["obj_mean"] for r in key.values())
    kept = {p for p, r in key.items() if r["obj_mean"] >= best - tol}
    peak = max(key.items(), key=lambda kv: (kv[1]["obj_mean"], kv[0]))[0]

    seen, comps = set(), []
    for start in sorted(kept):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for j in range(len(dims)):
                for step in (-1, 1):
                    nb = list(cur)
                    nb[j] += step
                    nb = tuple(nb)
                    if nb in kept and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c))
    biggest = comps[0] if comps else []
    ids = {}
    for i, c in enumerate(comps):
        for p in c:
            ids[p] = i

    params = {}
    for d in dims:
        # 区域里这一维出现过哪些取值；中心值 = 区域各格取值的**网格序号均值**四舍五入
        # 回最近的一个网格取值 —— 因为不在网格上的值（例如 0.01 与 0.02 的中间）复现不出来。
        vals = [key[p]["knob:" + d] for p in biggest]
        idxs = [grid[d].index(v) for v in vals]
        params[d] = {"values": sorted(set(vals)),
                     "central": grid[d][int(round(statistics.mean(idxs)))],
                     "value_indices": sorted(set(idxs))}
    return {
        "ok": True, "best_obj": best, "tol": tol,
        "peak": {"cell_id": key[peak]["cell_id"], "idx": list(peak),
                 "obj": key[peak]["obj_mean"],
                 "note": "仅记录，**不作为推荐**（峰值最容易被选择偏差挑中）"},
        "n_kept": len(kept), "n_cells": len(key),
        "components": [{"id": i, "size": len(c),
                        "cells": [key[p]["cell_id"] for p in c]}
                       for i, c in enumerate(comps)],
        "region_id": 0 if comps else None,
        "cells": [key[p]["cell_id"] for p in biggest],
        "coords": [list(p) for p in biggest],
        "size": len(biggest),
        "fraction": (len(biggest) / len(key)) if key else 0.0,
        "obj_mean": (statistics.mean([key[p]["obj_mean"] for p in biggest])
                     if biggest else None),
        "obj_min": min((key[p]["obj_mean"] for p in biggest), default=None),
        "obj_max": max((key[p]["obj_mean"] for p in biggest), default=None),
        "params": params,
        "cell_region_id": {key[p]["cell_id"]: ids.get(p) for p in key},
    }


def tolerance(rows, noise_floor):
    """容差：优先用「同格多种子的离散度」，再并上指纹里的噪声底，最后垫一个默认值。

    哪一种都不许静默：sources 里逐条写明这次用了什么。
    """
    within = []
    for r in rows:
        if r.get("obj_sd") is not None:
            within.append(r["obj_sd"])
    sources = []
    est = None
    if within:
        est = math.sqrt(sum(x * x for x in within) / len(within))
        sources.append("同格内多种子的合并标准差 %.6f（%d 格有 ≥2 个种子）" % (est, len(within)))
    else:
        sources.append("每格只有 1 个种子 ⇒ **估不出**同格离散度（判不了，退回兜底值）")
    nf = None
    if isinstance(noise_floor, dict):
        col = (noise_floor.get("columns") or {}).get(OBJECTIVE) or {}
        if col.get("std") is not None:
            nf = float(col["std"])
            sources.append("fingerprint.json 的噪声底 %s.std = %.6f" % (OBJECTIVE, nf))
    cands = [x for x in (est, nf) if x is not None and x > 0]
    val = max(cands) if cands else None
    if val is None or val < DEFAULT_IC_TOL:
        sources.append("兜底值 %.4f（IC 单位；这是本工具写死的，不是这个包测出来的）"
                       % DEFAULT_IC_TOL)
        val = max(val or 0.0, DEFAULT_IC_TOL)
    return val, sources


# ══════════════════════════════════════════════════════════════════════ 落盘

def _write_csv(path: Path, rows, cols):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════ 入口

def cmd_search(argv=None) -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description="attest 超参搜索（默认只看计划，不真跑）")
    ap.add_argument("--bundle", required=True, help="训练包目录（只读）")
    ap.add_argument("--grid", default="",
                    help="网格文件。**通常不用给**：本机打包时会把它装进包里，"
                         "这里留空就自动找包内的 %s" % GRID_IN_ZIP)
    ap.add_argument("--out", default=None, help="工作目录（试次/日志/产物落这里）")
    ap.add_argument("--workdir", dest="workdir_alias", default=None,
                    help="--out 的同义写法（与 eval/collect 一致）。两个都给了又不一致时会拒绝。")
    ap.add_argument("--config", default=None, help="pt_config.json 路径")
    ap.add_argument("--dispatch", default=None, help="DISPATCH.json 路径")
    ap.add_argument("--jobs", type=int, default=1,
                    help="并行跑几格。默认 1：并行不改随机种子，但会抢同一块 GPU、"
                         "让每次训练的墙钟失去可比性")
    ap.add_argument("--seeds", default=None, help="覆盖网格里的种子，逗号分隔")
    ap.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS,
                    help="从 train 窗末尾切出几天做独立 holdout（默认 %d）" % DEFAULT_HOLDOUT_DAYS)
    ap.add_argument("--region-tol", type=float, default=None,
                    help="覆盖「算同一个区」的容差（IC 单位）。默认自动估，见 tolerance()")
    ap.add_argument("--sec-per-run", type=float, default=DEFAULT_SEC_PER_RUN,
                    help="只用于 dry-run 的耗时估算（秒/次）")
    ap.add_argument("--timeout", type=int, default=None, help="单次训练超时（秒）")
    ap.add_argument("--allow-threads", type=int, default=None,
                    help="覆盖线程数（实验条件，改了会记进指纹）")
    ap.add_argument("--allow-master-write", action="store_true",
                    help="允许把试次写进母本根之下。默认禁止，防手滑污染母本")
    ap.add_argument("--commit", action="store_true", help="确认真跑。不加这个就只看计划")
    a = ap.parse_args(argv)

    cfg = C.load_config(a.config)
    # 归一要放在 cfg 载好**之后**：这段话术用 cfg 取联系人，
    # 插在前面会 UnboundLocalError —— 该给人话的时候抛栈，正是本工具最不该做的。
    a.out, _alias_err = C.resolve_out_alias(a.out, a.workdir_alias)
    if _alias_err:
        print(C.human_block(
            "两个开关指的不是同一个位置",
            _alias_err,
            "本工具不替你挑一个用 —— 挑错了会把结果写到你以为不在的地方。",
            "把其中一个删掉再跑一次。",
            cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
        return C.RC_USAGE
    if a.out:
        cfg["workdir"] = a.out
    if not cfg.get("workdir"):
        cfg["workdir"] = str(Path.cwd() / "_pt_work")
    bundle = C.normalize_path(a.bundle)
    workdir = C.normalize_path(cfg["workdir"])
    contact = cfg.get("contact", C.DEFAULT_CONFIG["contact"])

    meta = C.load_bundle_meta(bundle)
    manifest = meta.get("manifest") or {}
    pkgname = manifest.get("pkg") or bundle.name
    seg_cfg_decl = manifest.get("seg_config")

    def stop(headline, what, why, nxt, rc=C.RC_USAGE):
        print(C.human_block(headline, what, why, nxt, contact))
        return rc

    # 网格从哪来（三级，先到先用）。
    # 异地正常走中间那条：本机打包时就把网格装进包里了，操作者不用写一行 JSON ——
    # 参数空间以**包内声明**为准，让他自己编等于逼他猜，而猜错的后果不是报错，
    # 是一张"每格数字都一样"的表，看起来像结论。
    grid_path = None
    if a.grid:
        grid_path = Path(a.grid)
    else:
        _dobj, _dpath = C.load_dispatch(bundle, a.dispatch)
        _rec = (_dobj or {}).get("search_grid") or {}
        if _dpath and _rec:
            grid_path = bundle / (_rec.get("archive") or GRID_IN_ZIP)
        if grid_path is None or not Path(grid_path).is_file():
            grid_path = bundle / GRID_IN_ZIP
    if not Path(grid_path).is_file():
        return stop(
            "这个包里没有搜索网格",
            "既没给 --grid，包里也找不到 %s，DISPATCH.json 里也没记。" % GRID_IN_ZIP,
            "搜索扫哪些参数、扫哪些取值，是以**这个包自己的声明**为准定的，不能凭空编："
            "编出来的键多半在这个包上根本没接上计算，跑完只会得到一张每格数字都一样的表，"
            "那看起来像结论，其实什么都没扫到。",
            "向本机索要这个包的网格（本机用 pt_pack.py pack --grid 装进包里再发一次）。"
            "拿到后不用解开 zip 手工放：本机重发的包里就有。")
    print("[pt] 网格：%s" % grid_path)
    try:
        gobj, grid = load_grid(grid_path)
    except Exception as exc:
        return stop("网格文件读不了", "%s: %s" % (type(exc).__name__, exc),
                    "网格是这次搜索的全部输入，读不了就没法开始。",
                    "网格是随包发来的，你自己不要改：把这条发回本机核对。")

    # 目标函数：只认 val。test 是门禁判据，不能当目标。
    obj = gobj.get("objective", OBJECTIVE)
    if obj != OBJECTIVE:
        return stop("不支持这个搜索目标：%s" % obj,
                    "本工具只允许以 `%s` 为目标，你写的是 `%s`。" % (OBJECTIVE, obj),
                    OBJECTIVE_WHY,
                    "网格是随包发来的，你不要自己改：把这一条发回本机，由本机改完重发。")

    # 入口必须在，且必须是本工具认识的那种
    entry = bundle / ENTRY_NAME
    if not entry.is_file():
        return stop("包里没有训练入口", "找不到 %s。" % entry,
                    "搜索是在这个入口的副本上改参数再跑的，没有它无从改起。",
                    "确认拿到的是完整的训练包（应含 %s、X/、Y/、models/）。" % ENTRY_NAME)
    # 用 bytes 解码而不是 read_text：read_text 会做通用换行转换（CRLF -> LF），
    # 于是"母本那份"与"试次里那份"看起来不一样，改写前的比对会平白报错。
    entry_text = entry.read_bytes().decode("utf-8")
    try:
        tree = ast.parse(entry_text)
        has_cfg = _find_literal_value(tree, CFG_LITERAL) is not None
        has_args = _find_literal_value(tree, ARGS_LITERAL) is not None
    except SyntaxError as exc:
        return stop("训练入口解析不了", str(exc), "入口本身有问题。", "把包发回母本侧。")
    if not (has_cfg and has_args):
        return stop(
            "这个入口不是本工具认识的形状",
            "入口里找不到模块级的 %s / %s 字面量（找到 %s / %s）。"
            % (CFG_LITERAL, ARGS_LITERAL, has_cfg, has_args),
            "本工具靠改写这两个字面量的**试次副本**来扫参数；形状不同就不能盲改。",
            "把包发回母本侧确认导出格式。")

    seg_cfg = ast.literal_eval(_find_literal_value(tree, CFG_LITERAL))
    if isinstance(seg_cfg_decl, dict) and seg_cfg_decl != seg_cfg:
        print("[pt] 注意：包内 MANIFEST.seg_config 与入口里的 SEG_CFG **不一致**。")
        print("     以**入口**为准扫（真正跑的是它），但这条不一致本身就值得回报母本侧。")

    problems, notes = check_knobs(grid, seg_cfg, entry_text)
    if problems:
        print("这一份网格里有不能扫的键：\n")
        for p in problems:
            print("  - %s" % p)
        return stop("拒绝搜索：网格里有不该扫的键",
                    "共 %d 个键不通过（上面逐条列了原因）。" % len(problems),
                    "扫一个「改了不影响结果」的键，会产出一张每格数字都一样、"
                    "却看起来很有说服力的表格 —— 那比不扫更糟。",
                    "按上面的提示改网格：只扫清单里的键。")

    seeds = ([int(x) for x in a.seeds.split(",") if x.strip()] if a.seeds
             else list(gobj.get("seeds") or [1]))
    holdout_days = int(gobj.get("holdout_days", a.holdout_days))
    dims, cells = build_cells(grid)
    runs_planned = len(cells) * len(seeds) + 1        # +1 = holdout 那一次

    print("[pt] 包 %s   锚点 %s   工作目录 %s"
          % (pkgname, manifest.get("anchor_date"), workdir))
    print("[pt] 目标函数 %s（%s）" % (OBJECTIVE, OBJECTIVE_WHY))
    # 解析后的路径在上面已经印过一次了，这里只当键说明的标题 —— 不要印第二遍。
    print("[pt] 网格上的键：")
    for n in notes:
        print("     %s" % n)
    print("[pt] 格子 %d 个 × 种子 %d 个 + holdout 1 次 = 共 **%d 次训练**，"
          "约 %.0f 分钟（按每次 %.0f 秒估）"
          % (len(cells), len(seeds), runs_planned,
             runs_planned * a.sec_per_run / 60.0, a.sec_per_run))
    print("[pt] 并行 %d。注意：并行不改种子，但会抢同一块 GPU、让「耗时」失去可比性。"
          % a.jobs)
    print("[pt] holdout：从 train 窗末尾切 %d 天，搜索全程既不当训练数据也不当目标；"
          "搜索完用推荐参数在上面跑一次，与**同窗的现役模型**比。" % holdout_days)

    # 母本守卫：**第一次写盘之前**
    try:
        C.guard_master_write(workdir, cfg.get("master_roots"), a.allow_master_write, "创建工作目录")
    except C.MasterGuardError as exc:
        return stop("拒绝写入：目标在母本根之下", str(exc),
                    "母本是生产真身，误写会污染线上包。",
                    "把 --out 换成一个**不在** %s 之下的空目录。"
                    % cfg.get("master_roots", C.DEFAULT_CONFIG["master_roots"]), C.RC_BLOCKED)

    g = C.gpu_info()
    if not g.get("cuda_available"):
        return stop("拒绝搜索：没有可用的 GPU",
                    "本机没有检测到可用的 CUDA 设备（%s）。" % g.get("note", "torch 未装或 CUDA 不可用"),
                    "搜索就是反复真训练，CPU 上跑出来的模型不算数（这也是一条硬规矩）。",
                    "换一台有 GPU 的机器，或先跑 --mode smoke 验证链路。", C.RC_BLOCKED)

    if not a.commit:
        print("\n" + "=" * 74)
        print("这是**计划**，一次都没跑（默认 dry-run）。上面全部校验已通过。")
        print("要真跑，把 --commit 加上：")
        # 建议的命令必须能**直接粘**：给了 --grid 才印它，否则印出去的是
        # `--grid  --jobs 1 --commit`，粘回去会把 --jobs 当成网格文件名。
        _g = "--grid %s " % a.grid if a.grid else ""
        print("    python pt.py search --bundle %s %s--jobs %d --commit"
              % (a.bundle, _g, a.jobs))
        print("=" * 74)
        return C.RC_OK

    # ── 真跑 ────────────────────────────────────────────────────────────
    workdir.mkdir(parents=True, exist_ok=True)
    batch = time.strftime("%Y%m%d_%H%M%S")
    disp_obj, disp_path = C.load_dispatch(bundle, a.dispatch)
    if disp_path:
        print("[pt] DISPATCH: %s" % disp_path)

    # 输入缓存先单线程物化一份，避免多格同时复制同一份 X/Y 打架
    T.ensure_local_inputs(bundle, workdir, manifest)

    threads = a.allow_threads if a.allow_threads is not None else cfg.get("threads", 1)
    jobs = max(1, int(a.jobs))
    base_args = ast.literal_eval(_find_literal_value(ast.parse(entry_text), ARGS_LITERAL))

    def run_cell(cell, seed):
        tag = "%s_search_%s_%s_s%s" % (pkgname, batch, cell["cell_id"], seed)
        patch = make_entry_patcher(cell["knobs"], None, entry_text)
        rec = T.run_one(bundle, workdir, tag, "week", seed, cfg, dispatch=disp_obj,
                        threads=a.allow_threads, allow_master_write=a.allow_master_write,
                        timeout=a.timeout, entry_patch=patch)
        rec["cell_id"] = cell["cell_id"]
        for d in dims:
            rec["knob:" + d] = cell["knobs"][d]
        return rec

    jobs_list = [(c, s) for c in cells for s in seeds]
    print("\n[pt] 第一批开跑：%d 次，并行 %d" % (len(jobs_list), jobs))
    trials = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(run_cell, c, s) for c, s in jobs_list]
        for i, f in enumerate(futs, 1):
            rec = f.result()
            trials.append(rec)
            print("  [%d/%d] %s seed=%s rc=%s obj=%s  (%.0fs)"
                  % (i, len(futs), rec["cell_id"], rec.get("seed"), rec.get("rc"),
                     _num((rec.get("version_row") or {}).get(OBJECTIVE)),
                     rec.get("elapsed_sec") or 0))

    rows, n_failed = build_surface(cells, dims, grid, trials)

    tol, tol_src = tolerance(rows, _read_noise_floor(workdir))
    if a.region_tol is not None:
        tol, tol_src = float(a.region_tol), ["命令行 --region-tol 指定 %.6f" % a.region_tol]
    reg = region_pool(rows, dims, grid, tol)
    flat0, flat_tol = _knob_effects(rows, dims, tol)

    # 推荐值：只有稳定区成立、且这次搜索没有缺口时才给
    rec_out, verdict, reasons = _recommend(reg, rows, dims, n_failed)
    if flat0:
        reasons.append(
            "这几维**沿它自己走一遍，目标值一点都不动**（数值完全相同）：%s。"
            "要么这个包这一版根本没接上它，要么入口里的接法变了。"
            "别读成「这个参数不重要」—— 本工具在扫之前验过它的接入方式，"
            "跑出来却毫无差别，这本身就是一件要报回母本侧的事。" % ", ".join(flat0))
    if flat_tol:
        reasons.append(
            "这几维动了、但幅度没超过容差：%s。在这个包这段数据上分辨不出来 —— "
            "判「分辨不出」，不判「没影响」。" % ", ".join(flat_tol))

    holdout = {"status": "skipped", "why": "没有形成稳定区，没有可复核的参数。"}
    if rec_out and verdict in ("region-ok", "region-holdout-pending"):
        holdout = _holdout_run(bundle, workdir, cfg, pkgname, batch, seeds[0],
                               rec_out["params_flat"], base_args, holdout_days,
                               entry_text, disp_obj, a, dims)
        if holdout.get("verdict") == "FAIL":
            verdict = "region-holdout-failed"
            reasons.append("稳定区在独立 holdout 上**没跑赢现役模型** ⇒ 推荐值不成立。")
        elif holdout.get("verdict") == "UNKNOWN":
            reasons.append("holdout 没算出可用结论（原因见下），推荐值只能算**未经复核**。")
        elif holdout.get("verdict") == "PASS":
            verdict = "region-ok"

    for r in rows:
        rid = (reg.get("cell_region_id") or {}).get(r["cell_id"])
        r["region_id"] = rid
        r["in_region"] = (rid == reg.get("region_id")) if rid is not None else False
        r["is_peak"] = bool(reg.get("ok") and r["cell_id"] == reg["peak"]["cell_id"])

    tcols = (["batch", "cell_id"] + ["knob:" + d for d in dims]
             + ["seed", "mode", "rc", "status", "got_objective", "live_val_ic",
                "cand_test_ic", "live_test_ic", "decision", "reason",
                "elapsed_sec", "threads", "tag", "log",
                "entry_sha1_before", "entry_sha1_after", "patch_error"])
    trows = []
    for t in trials:
        vr = t.get("version_row") or {}
        ep = t.get("entry_patch") or {}
        trows.append({
            "batch": batch, "cell_id": t["cell_id"], "seed": t.get("seed"),
            "mode": t.get("mode"), "rc": t.get("rc"),
            "status": "ok" if t.get("rc") == 0 and _num(vr.get(OBJECTIVE)) is not None
                      else ("failed" if t.get("rc") != 0 else "no_result"),
            "got_objective": vr.get(OBJECTIVE),
            "live_val_ic": vr.get("live_val_ic"),
            "cand_test_ic": vr.get("cand_test_ic"), "live_test_ic": vr.get("live_test_ic"),
            "decision": t.get("decision"), "reason": t.get("reason"),
            "elapsed_sec": t.get("elapsed_sec"), "threads": t.get("threads"),
            "tag": t.get("tag"), "log": t.get("log"),
            "entry_sha1_before": ep.get("entry_sha1_before"),
            "entry_sha1_after": ep.get("entry_sha1_after"),
            "patch_error": t.get("error") or t.get("blocked"),
            **{("knob:" + d): t.get("knob:" + d) for d in dims},
        })
    scols = (["cell_id"] + ["knob:" + d for d in dims]
             + ["n_runs", "n_ok", "obj_mean", "obj_sd", "obj_min", "obj_max",
                "test_ic_mean", "gate_pass", "status", "in_region", "region_id", "is_peak"])
    _write_csv(workdir / "trials.csv", trows, tcols)
    _write_csv(workdir / "surface.csv", rows, scols)

    reg_out = {
        "generated_at": C.now_iso(), "batch": batch,
        "pkg": pkgname, "seg": manifest.get("seg"),
        "anchor_date": manifest.get("anchor_date"),
        "bundle": str(bundle),
        "objective": OBJECTIVE, "objective_why": OBJECTIVE_WHY,
        "grid_path": str(a.grid), "grid": grid, "dims": dims,
        "seeds": seeds, "jobs": jobs, "threads": threads,
        "dispatch": disp_path,
        "runs": {"cells": len(cells), "runs": len(trials),
                 "ok": sum(1 for t in trials if t.get("rc") == 0),
                 "failed": n_failed},
        "tolerance": {"value": round(tol, 6), "sources": tol_src,
                      "what": "两格的目标值差多少以内算「同一个区」。自动估："
                              "同格多种子的离散度 + 指纹噪声底，取大者，兜底 %.4f。"
                              % DEFAULT_IC_TOL},
        "region": ({"found": True, "size": reg["size"], "fraction": round(reg["fraction"], 4),
                    "stable": reg["size"] >= MIN_REGION_CELLS,
                    "thin": reg["fraction"] < THIN_REGION_FRACTION,
                    "cells": reg["cells"], "coords": reg["coords"],
                    "obj_mean": reg["obj_mean"], "obj_min": reg["obj_min"],
                    "obj_max": reg["obj_max"], "n_kept": reg["n_kept"],
                    "n_cells": reg["n_cells"],
                    "components": reg["components"],
                    "params": reg["params"],
                    "params_flat": {d: reg["params"][d]["central"] for d in dims}}
                   if reg.get("ok") else {"found": False, "why": reg.get("why")}),
        "best_obj": reg.get("best_obj"), "peak": reg.get("peak"),
        "knob_effects": {"unchanged": flat0, "below_tol": flat_tol,
                         "what": "沿这一维走一遍目标值动不动。unchanged = 数值完全相同"
                                 "（多半没接上）；below_tol = 动了但没超过容差（分辨不出）。"},
        "recommendation": {"verdict": verdict,
                           # emit 跟着**最终**裁决走（含 holdout 打回来的那一步）：
                           # holdout 已经否掉了还写 emit=true，只读 emit 的下游会用上它。
                           "emit": bool(rec_out) and verdict in ("region-ok",
                                                                 "region-holdout-pending"),
                           "params": (rec_out or {}).get("params_flat"),
                           "params_usable": bool(rec_out) and verdict in ("region-ok",
                                                                          "region-holdout-pending"),
                           "reasons": reasons},
        "holdout": holdout,
        "limitations": LIMITATIONS_BASE + [
            "本次并行 %d：并行不改随机种子，但会改变每次训练的墙钟，"
            "也让「同机重复跑」的耗时不再可比。" % jobs,
            "搜索只在包内 X/Y 这一份快照上做。换锚点后同一组参数的相对位置可能变。",
        ],
    }
    _write_json(workdir / "region.json", reg_out)

    print("\n" + "=" * 74)
    if reg.get("ok"):
        print("稳定区：%d 格（占 %d 格的 %.0f%%）  目标均值 %.6f  区间 [%.6f, %.6f]"
              % (reg["size"], reg["n_cells"], 100 * reg["fraction"], reg["obj_mean"],
                 reg["obj_min"], reg["obj_max"]))
        print("  容差 %.6f 来源：%s" % (tol, "；".join(tol_src)))
        print("  峰值格 %s obj=%.6f（**仅记录，不作为推荐**）"
              % (reg["peak"]["cell_id"], reg["peak"]["obj"]))
        if verdict in ("region-ok", "region-holdout-failed", "region-holdout-pending"):
            print("  %s（区域内中心值，不是峰值格）："
                  % ("推荐参数" if verdict != "region-holdout-failed"
                     else "区域中心值 —— **holdout 已否掉，不可当推荐用**"))
            for d in dims:
                p = reg["params"][d]
                print("    %-24s %s   （区域内出现过：%s）"
                      % (d, p["central"], p["values"]))
        if flat0:
            print("  **目标值一点都不动的维度**：%s  —— 别读成「不重要」，"
                  "多半是没接上，见 region.json 的 knob_effects" % ", ".join(flat0))
        if flat_tol:
            print("  动得没超过容差的维度：%s —— 判「分辨不出」" % ", ".join(flat_tol))
    else:
        print("没有形成稳定区：%s" % reg.get("why"))
    print("裁决：%s" % verdict)
    for r in reasons:
        print("  - %s" % r)
    if holdout.get("status") == "ran":
        print("holdout：%d 天，cand %.6f vs 现役 %.6f（同窗）  Δ=%+.6f  %s"
              % (holdout["days"], holdout["cand_ic"], holdout["live_ic"],
                 holdout["delta"], holdout["verdict"]))
    else:
        print("holdout：没做（%s）" % holdout.get("why"))
    print("产物：%s" % "、".join(str(workdir / n) for n in SEARCH_FILES))
    print("下一步：python pt.py eval --bundle %s --workdir %s" % (a.bundle, workdir))
    print("=" * 74)

    if n_failed and verdict == "unknown":
        return C.RC_FAIL
    if verdict in ("region-ok",):
        return C.RC_OK
    if verdict in ("region-holdout-failed",):
        return C.RC_FAIL
    return C.RC_WARN


def _read_noise_floor(workdir: Path):
    p = Path(workdir) / "fingerprint.json"
    if not p.is_file():
        return None
    try:
        return (json.loads(p.read_text(encoding="utf-8")) or {}).get("noise_floor")
    except Exception:
        return None


def _knob_effects(rows, dims, tol):
    """这一维到底动没动目标值。返回 (完全没动, 动得没超过容差)。

    判法是**沿这一维**比，不是全表比：把其它维固定住，看这一维从这头走到那头目标值变不变。
    全表比会漏（别的维在动，全表当然不平），也说不清是"哪一维没接上"。
    这是上面白名单静验的**动态兜底**：静验看的是源码里有没有接上，
    这里看的是跑出来到底有没有差别 —— 两边都过才算这个键真的扫过了。
    """
    identical, below = [], []
    for j, d in enumerate(dims):
        groups = {}
        for r in rows:
            if r["obj_mean"] is None or "idx" not in r:
                continue
            other = tuple(v for i, v in enumerate(r["idx"]) if i != j)
            groups.setdefault(other, []).append(r["obj_mean"])
        spreads = [max(g) - min(g) for g in groups.values() if len(g) >= 2]
        if not spreads:
            continue                        # 这一维只有 1 个取值，或没有可比的对子 —— 判不了，不报
        s = max(spreads)
        if s == 0.0:
            identical.append(d)
        elif s < tol:
            below.append(d)
    return identical, below


def _recommend(reg, rows, dims, n_failed):
    reasons = []
    if not reg.get("ok"):
        return None, "unknown", [reg.get("why") or "没有可用目标值"]
    if n_failed:
        reasons.append("有 %d 次训练没跑成：稳定区的**形状**因此无从判断（缺口可能把它切断）。"
                       "要拿推荐值，请把这两次补跑成再算。" % n_failed)
        return None, "unknown", reasons
    if len(rows) < 2:
        reasons.append("总共只有 %d 格 —— 一格谈不上「邻域稳定区」。" % len(rows))
        return None, "single-cell", reasons
    if reg["size"] < MIN_REGION_CELLS:
        reasons.append("容差内的连通块只有 %d 格 = **一根针尖，不是一个稳定区**。"
                       "峰值最容易被选择偏差挑中，本工具不报它。" % reg["size"])
        return None, "peak-only", reasons
    out = {"params_flat": {d: reg["params"][d]["central"] for d in dims},
           "params_detail": reg["params"]}
    v = "region-holdout-pending"
    if reg["fraction"] < THIN_REGION_FRACTION:
        reasons.append("区域只占 %d 格的 %.0f%%（薄）。推荐值照给，但置信度按薄区读。"
                       % (reg["n_cells"], 100 * reg["fraction"]))
    return out, v, reasons


def _holdout_run(bundle, workdir, cfg, pkgname, batch, seed, params_flat, base_args,
                 holdout_days, entry_text, disp_obj, a, dims):
    """用推荐参数在 holdout 上跑一次。

    做法：TRAIN_ARGS 改成 train_days=base-H, val_days=base_val, test_days=base_test+H。
    窗口自面板末端切分，于是 val 窗正好落在原 train 窗末尾的 H 天上 —— 搜索全程没见过它。
    """
    t = int(base_args.get("train_days", 252))
    v = int(base_args.get("val_days", 21))
    te = int(base_args.get("test_days", 21))
    args_over = {"train_days": t - holdout_days, "val_days": v, "test_days": te + holdout_days}
    tag = "%s_searchh_%s_holdout_s%s" % (pkgname, batch, seed)
    patch = make_entry_patcher(params_flat, args_over, entry_text)
    rec = T.run_one(bundle, workdir, tag, "week", seed, cfg, dispatch=disp_obj,
                    threads=a.allow_threads, allow_master_write=a.allow_master_write,
                    timeout=a.timeout, entry_patch=patch)
    vr = rec.get("version_row") or {}
    cand, live = _num(vr.get("cand_val_ic")), _num(vr.get("live_val_ic"))
    out = {"status": "ran", "tag": tag, "seed": seed, "days": holdout_days,
           "train_days": args_over["train_days"], "val_days": args_over["val_days"],
           "test_days": args_over["test_days"],
           "params": params_flat, "rc": rec.get("rc"),
           "entry_sha1_after": (rec.get("entry_patch") or {}).get("entry_sha1_after"),
           "log": rec.get("log"),
           "why_val_is_holdout":
               "val 窗 = 原 train 窗末尾的 %d 天。搜索期所有格子都把这段时间当训练数据、"
               "从不把它当目标；这一次的训练窗**不含**它，故它既没被搜过也没被训过。" % holdout_days,
           "caveat": "这一次的 test 窗是 %d 天（原 val 窗 + 原 test 窗），"
                     "它的 decision 与搜索期不可比，本工具不采信。" % args_over["test_days"]}
    if rec.get("rc") != 0 or cand is None or live is None:
        out.update({"status": "failed", "verdict": "UNKNOWN",
                    "why": "holdout 那一次没跑成或没算出 IC（rc=%s，cand=%s，live=%s）"
                           % (rec.get("rc"), cand, live)})
        return out
    out.update({"cand_ic": cand, "live_ic": live, "delta": round(cand - live, 6)})
    if cand > live:
        out["verdict"] = "PASS"
        out["note"] = ("在搜索没碰过的这段上，候选仍跑赢现役。这是**同窗**比较，"
                       "不涉及跨窗的 IC 水平差异。")
    else:
        out["verdict"] = "FAIL"
        out["note"] = ("在搜索没碰过的这段上，候选没跑赢现役 ⇒ 稳定区的推荐值大概率是"
                       "选择偏差的产物，不要据此上线。")
    return out
