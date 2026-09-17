"""attest 的打包与回传（本机侧 pack + 异地侧 collect）。

两个子命令，服务同一条凭据链的两端：

    pack     本机侧：把 Z 包封成 zip + DISPATCH.json          （本机跑，异地不用）
    collect  异地侧：把训练/评估产物封成 zip 发回本机         （`pt.py collect` 走这里）

口径（哥 2026-09-17 定，勿凭记忆改）：
- **sha1 一律实算，不抄 MANIFEST**。DISPATCH 里的 `inputs` 是本机读盘算出来的值；
  MANIFEST 的声明值只作为 `declared_sha1` 并排留档，两者不一致时 preflight 判不通过。
- **锚点两个来源都要记**：`anchor_from_manifest` 与 `anchor_from_ymeta`。两者不一致
  就是「半拉子刷新」（MANIFEST 已经是新锚、Y_meta 还在旧锚）—— 这种包**不许发**，
  因为它到了异地会带着一份自相矛盾的元数据跑，出问题时无从判断是哪一天的输入。
- 这是**包内自洽性**检查，不是跨机一致性判断。本工具不产生、不报告
  「某 Z 包与他机 D 包不一致」这类结论（职责边界 0916）。
- 打 zip 用**流式**：源包全程只读，DISPATCH.json 用 writestr 直接写进压缩包，
  不在源包目录里落任何文件（母本勿写）。
- 回传**不回写 Z**：collect 只产出一个 zip 加一个 .sha256 边车。

zip 内的目录结构：顶层就是包名目录（异地解压后直接得到 `S03_alphalgbm_csi800/`），
`DISPATCH.json` 放在这个目录里。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pt_common as C  # noqa: E402

sys.dont_write_bytecode = True

SCHEMA_DISPATCH = "portable-train/dispatch@1"

# 随包走的搜索网格在 zip 里的固定名字。放包根（与 DISPATCH.json 同级）：
# 异地 `pt.py search --bundle .` 不带 --grid 时按这个名字找，路径可预测、不靠记。
GRID_IN_ZIP = "SEARCH_GRID.json"
SCHEMA_COLLECT = "portable-train/collect@1"

# 打包时恒定排除：版本库、字节码（Z 侧铁律：禁落 __pycache__）
ALWAYS_SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "_cache"}
ALWAYS_SKIP_SUFFIX = (".pyc", ".pyo")

# 回传时的大件：默认不带（搜索场景只要指标），--with-models 才带。
# 两类分开记：模型权重（可能要上线）与输入副本（X/Y，本机本来就有那一份）。
WEIGHT_SUFFIX = (".pt", ".pth", ".pkl", ".joblib", ".onnx", ".h5", ".ckpt", ".bin", ".npy")
WEIGHT_NAMES = {"lgb.txt", "xgb.json"}
INPUT_NAMES = {"x.npy", "y_target.npy", "y.npy"}

# 依赖口径：MANIFEST.env 声明了哪些，就量哪些；另外固定补这几个训练真实用到的
DEP_NAMES = ("numpy", "pandas", "lightgbm", "xgboost", "torch", "scipy", "sklearn", "pyarrow")


# ------------------------------------------------------------------ 共用

def _pkg_name(bundle: Path) -> str:
    return Path(bundle).name


_SHA1_CACHE = {}


def _axis_span(path):
    """读一份「一行一个日期」的轴文件，返回 (首个, 末个, 行数)；读不到返回 None。
    用来在 DISPATCH 里记「这份 X 的数据覆盖到哪天」——MANIFEST 没有 input_anchor 时，
    这是异地能拿到的唯一可核对凭据（覆盖到哪天 != 落地时刻，两件事，别混）。"""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        lines = [x.strip() for x in C.read_text_any(p).splitlines()]
        lines = [x for x in lines if x]
        if not lines:
            return None
        return lines[0], lines[-1], len(lines)
    except Exception:
        return None


def _git_head(bundle):
    """包内 .git 的 HEAD。没有 .git 就返回 None —— 不猜、不编。"""
    try:
        g = Path(bundle) / ".git"
        head = (g / "HEAD")
        if not head.is_file():
            return None
        txt = C.read_text_any(head).strip()
        if txt.startswith("ref:"):
            ref = txt.split(":", 1)[1].strip()
            f = g / ref
            if f.is_file():
                return C.read_text_any(f).strip()
            pk = g / "packed-refs"
            if pk.is_file():
                for line in C.read_text_any(pk).splitlines():
                    line = line.strip()
                    if line.endswith(" " + ref) and not line.startswith("#"):
                        return line.split()[0]
            return None
        return txt or None
    except Exception:
        return None


def _sha1_16(p: Path) -> str:
    """每个文件只算一次。

    实测：X.npy 那种百兆级文件在 UNC 上读一遍是秒级，读三遍就是十几秒；
    下面 preflight 与文件清单都要 sha1，缓存一下省两次全量读。
    """
    key = str(p)
    if key not in _SHA1_CACHE:
        _SHA1_CACHE[key] = C.sha1_file(p)
    return _SHA1_CACHE[key][:16]


def _rows_from_shape(text):
    """Y_meta 的 X 是自由文本, 两种写法都要认: "=[312290, 30]" 与 "=(2767, 113, 124)"。

    只锚 `=[` 会让 S02 族整族判成「行数判不了」(假阳性); 放宽成"抠第一个数字"会抠到
    因子数或 NaN 占比。所以要锚住 `=` 后面紧跟的那个括号。
    """
    m = re.search(r"=\s*[\[\(]\s*(\d+)\s*,", str(text or ""))
    return int(m.group(1)) if m else None


def _iter_files(root: Path, skip_models: bool):
    """按相对路径排序遍历，排除版本库/字节码；skip_models 时排除 models/。"""
    out = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in ALWAYS_SKIP_DIRS for part in rel.parts):
            continue
        if p.suffix.lower() in ALWAYS_SKIP_SUFFIX:
            continue
        if skip_models and rel.parts and rel.parts[0] == "models":
            continue
        out.append((rel, p))
    return out


def _zip_it(pairs, out_zip: Path, extra: dict):
    """pairs=[(arcname, disk_path)] 流式写入；extra={arcname: text} 用 writestr 写入。"""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for arc, disk in pairs:
            zf.write(disk, arc)
        for arc, text in sorted(extra.items()):
            zf.writestr(arc, text)
    return out_zip


def _write_sidecar(zpath: Path) -> str:
    h = C.sha256_file(zpath)
    side = zpath.with_suffix(zpath.suffix + ".sha256")
    side.write_text("%s  %s\n" % (h, zpath.name), encoding="utf-8")
    return h


def _out_guard(out: Path, cfg, allow: bool) -> bool:
    """母本守卫。返回 True 表示被拦下（调用方应返回 RC_BLOCKED）。"""
    try:
        C.guard_master_write(out, cfg.get("master_roots"), allow, "写出产物")
    except C.MasterGuardError as exc:
        print(C.human_block(
            "拒绝写入：目标落在母本里",
            str(exc),
            "母本是线上包的所在地，往里写会污染在跑的包。这是防手滑的硬拦。",
            "把 --out 换成一个本机普通目录（例如 D:\\pt_out 或异地的某个工作目录），再跑一次。",
            cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
        return True
    return False


# ------------------------------------------------------------------ pack（本机）

def cmd_pack(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="pt_pack.py pack", add_help=True,
                                 description="把训练包封成 zip + DISPATCH.json（本机侧）")
    ap.add_argument("--bundle", required=True, help="要封的包目录（不许改它，只读）")
    ap.add_argument("--out", required=True, help="zip 落盘目录")
    ap.add_argument("--no-models", action="store_true", help="不带 models/（包里会写明排除了什么）")
    ap.add_argument("--note", default="", help="随包附一句话，会写进 DISPATCH.json")
    ap.add_argument("--grid", default="",
                    help="要随包发过去的搜索网格（本机侧验一遍再装进包里，异地不必自己写）")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="sha1 或锚点对不上时仍然打包（不推荐，会记进 DISPATCH）")
    ap.add_argument("--allow-master-write", action="store_true",
                    help="允许把 zip 写进母本根（默认拒绝）")
    a = ap.parse_args(argv)

    C.setup_console()
    cfg = C.load_config()
    contact = cfg.get("contact", C.DEFAULT_CONFIG["contact"])
    bundle = C.normalize_path(a.bundle)
    out_dir = C.normalize_path(a.out)

    if not bundle.is_dir():
        print(C.human_block(
            "找不到这个包目录", "路径 %s 不是一个存在的目录。" % bundle,
            "没有目录就没有输入，打包无从谈起。",
            "确认路径拼写，或用文件浏览器找到那个含 MANIFEST.json 的目录再复制路径。", contact))
        return C.RC_FAIL
    if _out_guard(out_dir, cfg, a.allow_master_write):
        return C.RC_BLOCKED

    meta = C.load_bundle_meta(bundle)
    manifest = meta.get("manifest") or {}
    ymeta = meta.get("y_meta") or {}
    if not isinstance(manifest, dict) or not manifest:
        print(C.human_block(
            "这个包里没有可用的 MANIFEST.json",
            "%s 下读不到 MANIFEST.json，或它不是合法 JSON。" % bundle,
            "MANIFEST 是异地判断「我拿到的是哪一份、锚点哪天」的唯一依据，缺了它异地无法自证。",
            "确认发的是训练包目录而不是父目录；若是真缺，回本机重新导出。", contact))
        return C.RC_FAIL

    pkg = _pkg_name(bundle)
    # ---- 锚点两个来源都读（半拉子刷新在这里现形）
    a_man = manifest.get("anchor_date") or manifest.get("anchor_raw")
    a_ym = ymeta.get("anchor_date") or ymeta.get("anchor_raw")
    a_man_s = C.parse_datetime_loose(a_man)
    a_ym_s = C.parse_datetime_loose(a_ym)

    # ---- 输入件：sha1 实算 vs 声明值
    inputs, msgs, hard = {}, [], []
    for section, key, entry in C.collect_manifest_entries(manifest):
        name = entry.get("dst") or entry.get("path") or ""
        if not name:
            continue
        p = C.resolve_entry_path(bundle, entry)
        rel = str(name).replace("\\", "/").lstrip("/")
        if p is None or not p.is_file():
            msgs.append("(不通过) 声明了但磁盘上没有：%s" % rel)
            hard.append(rel)
            continue
        rec = {"path": str(p.relative_to(bundle)).replace("\\", "/"),
               "bytes": p.stat().st_size,
               "sha1": _sha1_16(p),
               "declared_sha1": (entry.get("sha1") or "")[:16]}
        try:
            h = C.npy_header(p)
            rec["shape"] = list(h["shape"])
            rec["dtype"] = C.dtype_name(h["dtype"])
        except Exception:
            pass
        if rec["declared_sha1"]:
            if not C.hash_matches(_SHA1_CACHE[str(p)], rec["declared_sha1"]):
                msgs.append("(不通过) 账物不符 %s：声明 %s 实算 %s"
                            % (rec["path"], rec["declared_sha1"], rec["sha1"]))
                hard.append(rec["path"])
            else:
                msgs.append("(通过) %s sha1 与 MANIFEST 声明一致" % rec["path"])
        else:
            msgs.append("(注记) %s 无声明 sha1，已实算记入 DISPATCH（不得读作一致）" % rec["path"])
        # key 已经带着 section 前缀（"X[0]" / "Y.target[0]"），不要再拼一次 ——
        # 拼了就成了 "Y.Y.target[0]"，与 --export-xy 的 dst 双前缀是同一类坑。
        inputs[key] = rec

    # ---- X 行数：MANIFEST 声明 vs Y_meta 声明
    rows_man = None
    try:
        node = manifest["X"]
        node = node[0] if isinstance(node, list) else node
        rows_man = int(node["shape"][0])
    except Exception:
        pass
    rows_ym = _rows_from_shape(ymeta.get("X"))

    # ---- 判定
    anchor_ok = None
    if a_man_s is None or a_ym_s is None:
        msgs.append("(判不了) 锚字段缺失 (MANIFEST=%s Y_meta=%s) —— 判不了不等于通过"
                    % (a_man, a_ym))
    elif a_man_s != a_ym_s:
        anchor_ok = False
        msgs.append("(不通过) 锚点不一致 MANIFEST=%s Y_meta=%s —— 半拉子刷新" % (a_man_s, a_ym_s))
        hard.append("anchor")
    else:
        anchor_ok = True
        msgs.append("(通过) 锚点一致 %s" % a_man_s)

    rows_ok = None
    if rows_man is not None and rows_ym is not None:
        rows_ok = rows_man == rows_ym
        if not rows_ok:
            msgs.append("(不通过) X 行数不一致 MANIFEST=%s Y_meta=%s" % (rows_man, rows_ym))
            hard.append("rows")
        else:
            msgs.append("(通过) X 行数一致 %s" % rows_man)

    # ---- 本机实测环境 + 依赖固定
    env_expected = C.declared_env(manifest)
    dep_names = list(DEP_NAMES)
    for k in env_expected:
        base = str(k).split("==")[0].strip().lower()
        if base and base not in dep_names:
            dep_names.append(base)
    env_measured = {"python": "%s.%s.%s" % sys.version_info[:3],
                    "python_executable": sys.executable}
    deps = {}
    for n in dep_names:
        v = C.import_version(n)
        if v:
            deps[n] = v
    env_measured["packages"] = deps

    # ---- 搜索网格（可选）：网格随包走，不在异地现编
    #
    # 为什么非要在打包时装进去：要扫的键是**点号路径**（lgb.num_leaves 这类），
    # 而"哪些键在这个包上扫得动"完全取决于入口怎么写 —— 这是本机才知道的事。
    # 装之前先按**这个包**验一遍：扫不动的键在这里就被挡住。
    # 带病出门比不带更糟：扫一个没接上的键，异地会得到一张每格数字都一样的表，
    # 那看起来像"这个参数不重要"，其实那个键根本没进计算。
    grid_src, grid_rec = None, None
    if a.grid:
        gp = Path(a.grid)
        if not gp.is_file():
            print(C.human_block(
                "找不到要随包的网格文件", "`%s` 不在。" % a.grid,
                "网格是搜索的全部输入。打包时点了它却读不到，发出去的包会是个没有网格的搜索包。",
                "确认路径拼写；或者去掉 --grid，这个包就只做周训、不做搜索。", contact))
            return C.RC_FAIL
        if (bundle / GRID_IN_ZIP).is_file():
            print(C.human_block(
                "包里已经有一个 %s 了" % GRID_IN_ZIP,
                "%s 下已经存在这个文件，再带一份会有两个同名件。" % bundle,
                "解压后哪一份算数取决于顺序 —— 这种事不该留给人猜。",
                "先确认包里那份是不是该留（或是旧的一次遗留），再决定要不要 --grid。", contact))
            return C.RC_FAIL
        # 延迟 import：pack 不该因为搜索模块出问题就用不了。
        import ast as _ast

        import pt_search as S
        try:
            gobj, grid_decl = S.load_grid(gp)
        except Exception as exc:
            print(C.human_block(
                "网格文件读不了", "%s: %s" % (type(exc).__name__, exc),
                "读不了就没法在打包时替异地验一遍，所以不能发。",
                "照本工具根目录的 grid.example.json 的格式改。", contact))
            return C.RC_FAIL
        entryp = bundle / S.ENTRY_NAME
        if not entryp.is_file():
            print(C.human_block(
                "包里没有训练入口", "找不到 %s。" % entryp,
                "搜索是在这个入口的副本上改参数再跑的，没有它就没法在打包时验网格。",
                "确认 --bundle 指的是完整训练包（应含 %s、X/、Y/）。" % S.ENTRY_NAME, contact))
            return C.RC_FAIL
        etext = entryp.read_bytes().decode("utf-8")   # 不要 read_text：它会吞掉 CRLF
        try:
            node = S._find_literal_value(_ast.parse(etext), S.CFG_LITERAL)
            seg = _ast.literal_eval(node) if node is not None else None
        except Exception as exc:
            print(C.human_block(
                "训练入口解析不了", str(exc),
                "入口本身有问题，或者它不是本工具认识的那种三件套入口。",
                "把包发回母本侧确认导出格式。", contact))
            return C.RC_FAIL
        if not isinstance(seg, dict):
            print(C.human_block(
                "这个包的入口不是本工具认识的形状",
                "在 %s 里找不到模块级的 %s 字面量。" % (S.ENTRY_NAME, S.CFG_LITERAL),
                "搜索靠改写入口**副本**里的这个字面量来换参数；形状不同就不能盲改。",
                "把包发回母本侧确认导出格式。", contact))
            return C.RC_FAIL
        problems, gnotes = S.check_knobs(grid_decl, seg, etext)
        if problems:
            print(C.human_block(
                "这份网格不能跟着包发出去",
                "拿**这个包**验出 %d 处问题：\n    %s"
                % (len(problems), "\n    ".join(problems)),
                "网格里写了扫不动的键时，异地跑完会得到一张每格数字都一样的表 —— "
                "它看起来像结论，其实那个键根本没进计算。这种结果比没有结果更坏。",
                "按上面的说明改掉那些键再打一次；要新开一个可扫的键，"
                "须先在 pt_search.py 的 KNOB_EVIDENCE 里写明它进了哪一步计算。", contact))
            return C.RC_FAIL
        # build_cells 返回的是 (dims, cells) 两个值 —— 只接第一个会把元组当格子清单，
        # len() 恒为 2，于是格数永远报成 2（实测踩过：4 格的网格报成 2 格）。
        _dims, cells = S.build_cells(grid_decl)
        seeds_n = len(list(gobj.get("seeds") or [1]))
        grid_rec = {
            "archive": GRID_IN_ZIP,
            "sha1": _sha1_16(gp),
            "bytes": gp.stat().st_size,
            "objective": gobj.get("objective", S.OBJECTIVE),
            "holdout_days": gobj.get("holdout_days", S.DEFAULT_HOLDOUT_DAYS),
            "seeds": gobj.get("seeds"),
            "cells": len(cells),
            # +1 = 搜索结束后那次 holdout 复核（结构上必然有，不是估算）
            "runs_planned": len(cells) * seeds_n + 1,
            "validated_on": {"pkg": pkg, "entry_sha1": _sha1_16(entryp),
                             "knob_notes": gnotes},
        }
        grid_src = gp

    # ---- 全量文件清单（sha1 实算，异地解压后可逐件核对）
    files = _iter_files(bundle, a.no_models)

    # 包根若已经躺着一份 DISPATCH.json（上次打包留下的），不纳入清单、也不写进 zip：
    # 本次打包会重新生成一份权威的。两份同名件同时进 zip，解压后谁算数取决于顺序 ——
    # 不留这种看运气的口子，尤其旧那份正是"锚点与清单都可能过期"的那一份。
    # 典型来源：包被解压后又重新打了一次。这里只记一笔，不当硬矛盾（重打包是正当动作）。
    stale_dispatch = [rel for rel, _ in files
                      if str(rel).replace("\\", "/") == "DISPATCH.json"]
    if stale_dispatch:
        files = [(rel, p) for rel, p in files
                 if str(rel).replace("\\", "/") != "DISPATCH.json"]
        msgs.append("(注记) 包根带了一份旧的 DISPATCH.json（上次打包留下的）："
                    "本次不纳入清单、也不写进 zip，下面另行生成权威的一份。")
    inv, total = {}, 0
    for rel, p in files:
        rel_s = str(rel).replace("\\", "/")
        inv[rel_s] = {"sha1": _sha1_16(p), "bytes": p.stat().st_size}
        total += p.stat().st_size

    verdict = C.FAIL if hard else (C.UNKNOWN if anchor_ok is None else C.PASS)
    dispatch = {
        "schema": SCHEMA_DISPATCH,
        "pkg": pkg,
        "anchor_date": a_man,
        "anchor_from_manifest": a_man,
        "anchor_from_ymeta": a_ym,
        "anchor_agree": anchor_ok,
        # 数据锚：MANIFEST 有就照抄（它才是哥 0916 定义的「同日落地数据锚」）；
        # 没有就记一个可核对的弱凭据：X 的数据覆盖到哪天。两者语义不同，不要互相顶替。
        "input_anchor": manifest.get("input_anchor"),
        "data_coverage": {"x": _axis_span(bundle / "X" / "row_date.txt"),
                          "dates": _axis_span(bundle / "X" / "dates.txt")},
        "git_head": _git_head(bundle),
        "manifest_git_head": manifest.get("git_head"),
        # 旧版 doctor 只认这个键；留着当兼容，权威是下面的 files（全量逐位）
        "xy_sha1": {k: v.get("sha1") for k, v in inputs.items()},
        "x_rows_from_manifest": rows_man,
        "x_rows_from_ymeta": rows_ym,
        "source": {"bundle": str(bundle), "host": C.host_info(),
                   "packed_at": C.now_iso(), "packed_by": "attest/scripts/pt_pack.py"},
        "inputs": inputs,
        "env_expected_from_manifest": env_expected,
        "env_measured_at_pack": env_measured,
        "deps_pinned": deps,
        # 有网格才非空。异地 search 不带 --grid 时按 archive 这个名字在包里找。
        "search_grid": grid_rec,
        "files": inv,
        "file_count": len(inv),
        "total_bytes": total,
        "excluded": (["models/（--no-models）"] if a.no_models else [])
                    + [".git", "__pycache__", "*.pyc"]
                    + (["包根旧的 DISPATCH.json（已由本次的取代）"] if stale_dispatch else []),
        "preflight": {"verdict": verdict, "messages": msgs,
                      "allow_mismatch": bool(a.allow_mismatch)},
        "note": a.note,
        "contact": contact,
    }

    if hard and not a.allow_mismatch:
        print(C.human_block(
            "这个包自己就不自洽，不能发",
            "打包前的自检在 %d 处发现硬矛盾：\n    %s"
            % (len(hard), "\n    ".join(m for m in msgs if m.startswith("(不通过)"))),
            "发出去的包会被异地当成「某一天的完整快照」来训练。若 MANIFEST 与 Y_meta 说的不是同一天，"
            "异地跑出来的东西无法归因，事后也复现不了 —— 这正是要在这里拦住的原因。",
            "先在本机把包补齐（回本机用导出线重建该包），再重新打包。"
            "确知后果仍要发，加 --allow-mismatch，它会记进 DISPATCH.json。", contact))
        return C.RC_FAIL

    if a_ym is None:
        msgs.append("(注记) 本包没有 Y/Y_meta.json —— 不是三件套包，锚点与行数都只能靠 MANIFEST 单边声明。")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    zpath = out_dir / ("%s_%s_%s.zip" % (pkg, str(a_man_s or a_man or "NA").replace("-", ""), stamp))
    pairs = [(str(Path(pkg) / str(rel).replace("\\", "/")), p) for rel, p in files]
    if grid_src is not None:
        pairs.append((str(Path(pkg) / GRID_IN_ZIP), grid_src))
        # 清单也要补上：不然异地按 files 逐件核对时会把它当成"包里多出来的东西"。
        inv[GRID_IN_ZIP] = {"sha1": grid_rec["sha1"], "bytes": grid_rec["bytes"]}
        total += grid_rec["bytes"]
        dispatch["file_count"] = len(inv)
        dispatch["total_bytes"] = total
    _zip_it(pairs, zpath, {str(Path(pkg) / "DISPATCH.json"):
                           json.dumps(dispatch, ensure_ascii=False, indent=2)})
    zh = _write_sidecar(zpath)

    print("=" * 74)
    print("打包完成：%s" % zpath)
    print("  包名        : %s" % pkg)
    print("  锚点        : MANIFEST=%s  Y_meta=%s  %s"
          % (a_man, a_ym, "一致" if anchor_ok else ("不一致" if anchor_ok is False else "判不了")))
    print("  X 行数      : MANIFEST=%s  Y_meta=%s" % (rows_man, rows_ym))
    print("  文件        : %d 个，%.1f MB（压缩前）" % (len(inv), total / 1048576.0))
    print("  zip sha256  : %s" % zh)
    print("  边车        : %s" % zpath.with_suffix(zpath.suffix + ".sha256"))
    print("  依赖固定    : %s" % json.dumps(deps, ensure_ascii=False))
    print("  数据锚      : %s" % (json.dumps(dispatch["input_anchor"], ensure_ascii=False)
                                  if dispatch["input_anchor"] else
                                  "(包内没有 input_anchor —— 异地只能凭数据覆盖日佐证)"))
    _cov = dispatch["data_coverage"]["x"] or dispatch["data_coverage"]["dates"]
    print("  数据覆盖    : %s" % ("%s ~ %s（%d 行）" % _cov if _cov else "(读不到日期轴)"))
    print("  包内版本    : %s" % (dispatch["git_head"] or "无 .git（这份包没有版本号可对）"))
    if grid_rec is not None:
        print("  搜索网格    : %s（%d 格 × %d 种子 + 1 次 holdout = %d 次训练）"
              % (GRID_IN_ZIP, grid_rec["cells"], seeds_n, grid_rec["runs_planned"]))
        for n in grid_rec["validated_on"]["knob_notes"]:
            print("                " + n)
    print("  自检        : %s" % verdict)
    print("-" * 74)
    for m in msgs:
        print("  " + m)
    print("=" * 74)
    print("下一步：把 zip 与 .sha256 两个文件一起发给异地，附 README_FIRST.md 与 attest/。")
    return C.RC_OK if verdict != C.FAIL else C.RC_WARN


# ------------------------------------------------------------------ collect（异地）

# 工作目录根上必须有的（前面的每一节各产一份）。
EXPECTED_ROOT = ("doctor.json", "fingerprint.json", "eval.json", "report.md")
# 训练入口把版本表写在**每个试次的 models/ 下**，不在工作目录根 —— 两处都算数，
# 只说"在试次目录里"，不要报成缺（实测报过一次假缺口）。
EXPECTED_TRIAL = ("model_versions.csv",)
# 搜索场景（阶段3）才有的产物：没跑搜索就不该列成"缺"，那是本来就不该有。
EXPECTED_SEARCH = ("surface.csv", "region.json", "trials.csv")


def _is_weight(rel: str) -> bool:
    p = Path(rel)
    return p.suffix.lower() in WEIGHT_SUFFIX or p.name.lower() in WEIGHT_NAMES


def cmd_collect(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="pt_pack.py collect", add_help=True,
                                 description="把训练/评估产物封成 zip 发回本机（异地侧）")
    ap.add_argument("--workdir", required=True, help="训练工作目录（里面有 fingerprint.json）")
    ap.add_argument("--out", default=".", help="zip 落盘目录，默认当前目录")
    ap.add_argument("--bundle", default="", help="包目录（只为在包名里带上它，可不填）")
    ap.add_argument("--with-models", action="store_true",
                    help="连模型权重一起回传（周训要上线时才用；搜索只看指标就别带）")
    ap.add_argument("--allow-master-write", action="store_true",
                    help="允许写进母本根（默认拒绝）")
    a = ap.parse_args(argv)

    C.setup_console()
    cfg = C.load_config()
    contact = cfg.get("contact", C.DEFAULT_CONFIG["contact"])
    wd = C.normalize_path(a.workdir)
    out_dir = C.normalize_path(a.out)

    if not wd.is_dir():
        print(C.human_block(
            "找不到工作目录", "%s 不是一个存在的目录。" % wd,
            "训练与体检的产物都落在这个目录里，目录不存在说明前面那几节没跑成。",
            "先跑体检和训练，再跑这一节；若确实跑过，请检查路径是否写错。", contact))
        return C.RC_FAIL
    if _out_guard(out_dir, cfg, a.allow_master_write):
        return C.RC_BLOCKED

    fp = wd / "fingerprint.json"
    ev = wd / "eval.json"
    dj = wd / "doctor.json"
    pkg = Path(a.bundle).name if a.bundle else ""
    verdict = "NO_VERDICT"
    # 结论有两个可能的来源：跑了裁决就取 eval，没跑到就退回体检。
    # 名字里那个词到底出自哪一份，必须跟着印出来 —— 否则操作者转述"结论是 X"，
    # 本机无从判断 X 说的是裁决还是体检，只能再猜一次。
    verdict_src = "（工作目录里既没有 eval.json 也没有 doctor.json）"
    for _p, _label, key in ((ev, "eval.json（裁决）", "verdict"),
                            (dj, "doctor.json（体检）", "verdict")):
        if _p.is_file():
            try:
                obj = C.read_json_any(_p)
                if isinstance(obj, dict) and obj.get(key):
                    verdict = str(obj[key])
                    verdict_src = _label
                    break
            except Exception:
                pass
    if not pkg and fp.is_file():
        try:
            obj = C.read_json_any(fp)
            pkg = str(obj.get("pkg") or "")
        except Exception:
            pass
    pkg = pkg or wd.parent.name or "unknown"

    # 收集：整棵树（除去 _cache），权重类按开关决定
    got, skipped = {}, []
    for rel, p in _iter_files(wd, skip_models=False):
        rel_s = str(rel).replace("\\", "/")
        if Path(rel_s).parts and Path(rel_s).parts[0] == "_cache":
            continue
        if _is_weight(rel_s) and not a.with_models:
            kind = "输入副本" if Path(rel_s).name.lower() in INPUT_NAMES else "模型权重"
            skipped.append({"path": rel_s, "bytes": p.stat().st_size, "kind": kind})
            continue
        got[rel_s] = {"sha256": C.sha256_file(p), "bytes": p.stat().st_size}

    missing = [n for n in EXPECTED_ROOT if not (wd / n).is_file()]
    trial_hits = []
    for n in EXPECTED_TRIAL:
        if (wd / n).is_file():
            continue
        hits = sorted(wd.glob("trials/*/models/%s" % n))
        if hits:
            trial_hits.append("%s x%d（在试次目录里，不在工作目录根）" % (n, len(hits)))
        else:
            missing.append(n)
    # 搜索产物只在**确实跑过搜索**时才期待：判据是有没有表面/区域这类搜索痕迹。
    # 没跑搜索就说"没跑"，不列成缺件。
    search_missing = []
    ran_search = any((wd / n).is_file() for n in EXPECTED_SEARCH)
    if ran_search:
        search_missing = [n for n in EXPECTED_SEARCH if not (wd / n).is_file()]
    if not fp.is_file():
        first = "体检" if dj.is_file() else "体检与训练"
        print(C.human_block(
            "没有训练指纹，这次回传是不完整的",
            "工作目录 %s 里没有 fingerprint.json（%s那一节没跑成）。" % (wd, first),
            "没有指纹，本机就无法确认这批产物对应哪份输入、哪个种子、哪套线程 —— "
            "没有它，结果能不能用判不了。",
            "把这份包发回来仍然有价值（至少让本机看到体检报告），"
            "但请把红字原文一起发回，本机要按缺指纹处理。", contact))

    ts = time.strftime("%Y%m%d_%H%M%S")
    box = out_dir / ("pt_backup_%s_%s_%s" % (pkg, verdict, ts))
    box.mkdir(parents=True, exist_ok=True)
    for rel_s in got:
        dst = box / rel_s
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((wd / rel_s).read_bytes())

    manifest = {
        "schema": SCHEMA_COLLECT,
        "pkg": pkg,
        "verdict": verdict,
        "workdir": str(wd),
        "collected_at": C.now_iso(),
        "host": C.host_info(),
        "collected": got,
        "file_count": len(got),
        "total_bytes": sum(v["bytes"] for v in got.values()),
        "with_models": bool(a.with_models),
        "skipped": skipped,
        "missing_expected": missing,
        "found_elsewhere": trial_hits,
        "search_expected": list(EXPECTED_SEARCH),
        "search_missing": search_missing,
        "search_ran": bool(ran_search),
        "note": "只回传判定所需；不回写 Z 母本，上线由本机独断。",
        "contact": contact,
    }
    (box / "COLLECT.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zpath = Path(shutil.make_archive(str(box), "zip", str(box)))
    zh = _write_sidecar(zpath)

    print("=" * 74)
    print("回传包已生成：%s" % zpath)
    print("  包名        : %s" % pkg)
    print("  体检/裁决   : %s（读自 %s）" % (verdict, verdict_src))
    print("  内容        : %d 个文件，%.2f MB" % (len(got) + 1, manifest["total_bytes"] / 1048576.0))
    print("  zip sha256  : %s" % zh)
    if skipped:
        by_kind = {}
        for s in skipped:
            k = s.get("kind", "其他")
            by_kind[k] = by_kind.get(k, 0) + 1
        print("  未含大件    : %s（共 %.1f MB）—— 已在 COLLECT.json 逐件列明，不是丢失"
              % ("、".join("%s %d 个" % (k, v) for k, v in sorted(by_kind.items())),
                 sum(s["bytes"] for s in skipped) / 1048576.0))
    if trial_hits:
        print("  位置说明    : %s" % "；".join(trial_hits))
    if missing:
        print("  缺预期产物  : %s（已在 COLLECT.json 列明）" % ", ".join(missing))
    if not ran_search:
        print("  搜索产物    : 本包没跑搜索，故没有 %s —— 这不是缺件"
              % "、".join(EXPECTED_SEARCH))
    elif search_missing:
        print("  搜索产物缺  : %s（跑过搜索但这一件没落盘，请在 COLLECT.json 里核对）"
              % ", ".join(search_missing))
    else:
        # 全在就说全在。不说的话，操作者从"什么都没有"里读不出任何信息，
        # 而这一节恰恰是他唯一能确认搜索跑成功了的地方。
        print("  搜索产物    : %s 齐全" % "、".join(EXPECTED_SEARCH))
    print("=" * 74)
    print("把 zip 与 .sha256 发回本机。不要自己判断结果好不好，也不要往任何生产目录覆盖东西。")
    if verdict in ("FAIL", "NO_VERDICT"):
        print("注意：这一轮没有可用的裁决结论，包里的东西只够本机判断「为什么没跑成」。")
    return C.RC_OK


# ------------------------------------------------------------------ 入口

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0 if argv else C.RC_USAGE
    verb, rest = argv[0], argv[1:]
    if verb == "pack":
        return cmd_pack(rest)
    if verb == "collect":
        return cmd_collect(rest)
    print("[pt_pack] 不认识的子命令：%s（只支持 pack / collect）" % verb)
    return C.RC_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
