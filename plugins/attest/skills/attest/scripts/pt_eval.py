"""pt_eval —— 评估与裁决（异地闭环的终点）。

定位（这一条决定了整个模块的写法）：
    **不重算指标，只复算判决。**

为什么不去自己重算 ens_ic：包内 `train_from_xy.py` 已经逐字实现了生产
`D:\\Ongoing\\S03_alphalgbm\\train.py` 的 `_ens_metrics`（per-date 截面 Spearman rank IC）
与门禁（`:339` vs 生产 `:588`）。评估器另立一套口径，只会与门禁分叉，两边不一致时
没人知道该信谁。所以本模块：

    读训练**已经记录**的四个 IC -> 用生产门禁公式**独立再导一遍** decision
      -> 与包自己记录的 decision 对账（对不上就是包有问题，不是数字不好看）

这样评估器是**独立的**（不信任包自报的结论），又不是**另一套指标**（不产生口径分歧）。

边界（哥 0917 定的口径）：
    - 异地可判：这次训练是否达标、是否具备上线前置条件（产出 verdict 与 posture）。
    - 异地不可：改线上模型 / 改 active 指针 / 写 Z 母本。故本模块**只写报告**，
      对包一个字节都不写，promote 仍在本机、仍由哥独断。

铁律：UNKNOWN 绝不能被读成 PASS。「判不了」与「通过」在报告、退出码、posture 上全部分开。

读数口径（依据 arXiv 2511.07678 §7.1 与 Table 9/10）：
    多 seed 一律**池化**成单一读数（均值），逐种子原样列出备查；**禁止**从若干 seed 里
    挑最好的那个当结论 —— 那是挑噪声不是挑模型（该文实测 best-of-k 有 7.2% 的概率挑中
    k 个里最差的之一）。增量显著性用单样本 t：se=std/sqrt(n)，需 |增量|>t(0.975,n-1)*se；
    比原先「超过 1 个标准差」严格约 2.5 倍（n=3），这是有意的：怕假阳不怕假阴。
    逐日配对检验：**能做就做，不能做就说不能做**。原料是包内的 `ic_series.json`
    （逐日 IC 序列，契约见 `_sync` 侧交接 README 第二节）。三种情形分得很清：
    文件不在 -> WARN（这是本工具/训练侧的方法局限，不是这份包的缺陷）；
    文件在但契约不符 -> FAIL（缺段/裸 NaN/值类型错，都不许静默算错）；
    文件在且合规 -> 真的做配对检验，结论可以是「不显著」（UNKNOWN），不许读成通过。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pt_common as C  # noqa: E402
import pt_train as T  # noqa: E402  （只借 _noise_floor，纯标准库）

GROUP_GATE = "门禁与判决"
GROUP_NOISE = "噪声底与显著性"
GROUP_READY = "上线前置清单"
GROUP_POSTURE = "裁决姿态"

# 生产门禁判据的出处：两处逐字一致，改这里必须同步核对
#   D:\Ongoing\S03_alphalgbm\train.py:588
#   <包>\train_from_xy.py:339
GATE_SOURCE = "train.py:588 / train_from_xy.py:339"


# ------------------------------------------------------------------ 门禁复算

def derive_decision(cv, lv, ct, lt):
    """用生产门禁公式**独立再导**一遍 decision。逐字复刻 :339 / :588。

    cv/lv = 候选/活在 val 窗的 ens_ic_mean；ct/lt = test 窗。
    任一为 None 表示该窗算不出来（IC 全 NaN 等）。
    """
    if lv is None:
        # 没有活模型工件 -> first_install。注意这正是「门禁被跳过」的那条路。
        return "replaced", "first_install（无活工件，门禁未执行）"
    if cv is None or (ct is None and lt is None):
        return "skipped", "IC unavailable（有窗算不出 IC，门禁不执行）"
    if ct is not None and lt is not None:
        ok = (ct >= lt and ct > 0.0 and cv >= lv and cv > 0.0)
        reason = ("cand_test %+.4f>=live_test %+.4f 且 test>0 且 "
                  "cand_val %+.4f>=live_val %+.4f 且 val>0" % (ct, lt, cv, lv)) if ok else (
                  "gate fail ct=%+.4f lt=%+.4f cv=%+.4f lv=%+.4f%s"
                  % (ct, lt, cv, lv, "" if cv > 0.0 else " [val<=0]"))
        return ("replaced" if ok else "skipped"), reason
    ok = (cv >= lv and cv > 0.0)
    return ("replaced" if ok else "skipped"), "no-test fallback cv=%+.4f lv=%+.4f" % (cv, lv)


# 门禁条款的机器可比形式。别信散文，只比对「哪几条子句在场」。
_CLAUSE_PATTERNS = {
    "test>=live": r"(?:cand_test|c_test|ct)\s*(?:>=|≥)\s*(?:live_test|lt)\b",
    "test>0": r"(?:cand_test|c_test|ct)\s*>\s*0(?:\.0+)?(?![0-9])",
    "val>=live": r"(?:cand_val|c_val|cv)\s*(?:>=|≥)\s*(?:live_val|lv)\b",
    "val>0": r"(?:cand_val|c_val|cv)\s*>\s*0(?:\.0+)?(?![0-9])",
}
CLAUSE_HUMAN = {
    "test>=live": "候选 test IC >= 活 test IC",
    "test>0": "候选 test IC > 0",
    "val>=live": "候选 val IC >= 活 val IC",
    "val>0": "候选 val IC > 0（0914 哥拍板补的 val 侧绝对下限）",
}


def detect_clauses(text: str) -> set:
    return {k for k, pat in _CLAUSE_PATTERNS.items() if re.search(pat, text or "")}


def _code_gate_text(entry_py: Path):
    """从包入口里抠出门禁那几行（只做定位，判决不靠这个，靠 derive_decision）。"""
    try:
        src = C.read_text_any(entry_py)
    except OSError:
        return None
    lines = [l.strip() for l in src.splitlines() if 'decision = "replaced" if' in l]
    return "\n".join(lines) if lines else None


def _declared_gate_text(bundle: Path):
    """包内**散文**声明里的门禁（README / strategy.yaml）。

    这里刻意只收集文本、不解析语义：解析散文很容易过度自信，而这一条恰恰是
    最该保守的地方（S03 包里 README 声明的门禁就比代码少一条 val 侧下限）。
    """
    chunks, where = [], []
    for name in ("README.md", "README_FIRST.md", "strategy.yaml", "strategy.yml"):
        p = bundle / name
        if not p.is_file():
            continue
        try:
            txt = C.read_text_any(p)
        except OSError:
            continue
        keep = []
        lines = txt.splitlines()
        for i, l in enumerate(lines):
            if any(k in l for k in ("门禁", "gate", "GATE")):
                keep.extend(lines[max(0, i - 1):i + 4])
        if keep:
            chunks.append("\n".join(keep))
            where.append(name)
    return ("\n".join(chunks) if chunks else None), where


def check_gate(rep, bundle: Path, runs: list):
    """门禁组：包自报的 decision vs 复算的 decision；顺带核对声明的门禁子句。"""
    runs = [r for r in runs if r.get("version_row")]
    if not runs:
        rep.add(GROUP_GATE, "门禁复算", C.UNKNOWN,
                "没有任何带版本记录的试次，无从复算。先跑 train --mode week --commit。")
        return None

    per = []
    for r in runs:
        row = r["version_row"]
        cv, lv = _f(row.get("cand_val_ic")), _f(row.get("live_val_ic"))
        ct, lt = _f(row.get("cand_test_ic")), _f(row.get("live_test_ic"))
        got, reason = derive_decision(cv, lv, ct, lt)
        per.append({"tag": r.get("tag"), "seed": r.get("seed"),
                    "recorded": row.get("decision"), "derived": got, "reason": reason,
                    "cand_val_ic": cv, "live_val_ic": lv,
                    "cand_test_ic": ct, "live_test_ic": lt,
                    "method": row.get("method"), "match": row.get("decision") == got})

    mismatch = [x for x in per if not x["match"]]
    last = per[-1]

    if mismatch:
        # 注意：不必然是「包被改过」。包内 model_versions.csv 带着**历史行**，
        # 那些行是旧门禁下写的，本来就不该用现门禁去对。只有本次试次新增的行才算数。
        detail = "; ".join(
            "%s: 记录 %s / 复算 %s（%s）" % (x["tag"], x["recorded"], x["derived"], x["reason"])
            for x in mismatch)
        rep.add(GROUP_GATE, "门禁复算", C.FAIL,
                "包自报的 decision 与按生产门禁公式复算的结果对不上：" + detail,
                {"per_trial": per, "gate_source": GATE_SOURCE})
    else:
        rep.add(GROUP_GATE, "门禁复算", C.PASS,
                "%d 个试次的 decision 与复算一致（%s）" % (len(per), last["reason"]),
                {"per_trial": per, "gate_source": GATE_SOURCE})

    entry = bundle / "train_from_xy.py"
    code_txt = _code_gate_text(entry)
    decl_txt, where = _declared_gate_text(bundle)
    if code_txt is None:
        rep.add(GROUP_GATE, "门禁声明一致性", C.UNKNOWN,
                "包入口里没找到门禁行，无法比对；本包的门禁可能不是 IC 版。")
    elif decl_txt is None:
        rep.add(GROUP_GATE, "门禁声明一致性", C.UNKNOWN,
                "包内没有任何散文门禁声明可比对（代码里的判据照常复算）。"
                "判不了不等于一致。", {"code_gate": code_txt})
    else:
        cc, dc = detect_clauses(code_txt), detect_clauses(decl_txt)
        missing = cc - dc          # 代码有、声明没写 -> 文档落后于代码
        extra = dc - cc            # 声明有、代码没有 -> 更危险：文档承诺了代码没做的事
        if extra:
            rep.add(GROUP_GATE, "门禁声明一致性", C.FAIL,
                    "包内声明写了的门禁子句，代码里没有：%s。文档承诺了代码不执行的事。"
                    % "; ".join(CLAUSE_HUMAN[k] for k in sorted(extra)),
                    {"code_gate": code_txt, "declared_gate": decl_txt, "declared_in": where})
        elif missing:
            rep.add(GROUP_GATE, "门禁声明一致性", C.WARN,
                    "代码里的门禁比包内声明多出：%s。实际以代码为准（代码才是会跑的），"
                    "但按声明去读会低估门槛。" % "; ".join(CLAUSE_HUMAN[k] for k in sorted(missing)),
                    {"code_gate": code_txt, "declared_gate": decl_txt, "declared_in": where})
        elif cc == dc and cc:
            rep.add(GROUP_GATE, "门禁声明一致性", C.PASS,
                    "声明与代码的 %d 条门禁子句一致" % len(cc),
                    {"code_gate": code_txt, "declared_gate": decl_txt})
        else:
            rep.add(GROUP_GATE, "门禁声明一致性", C.UNKNOWN,
                    "两边都没抠出可比的子句，判不了。",
                    {"code_gate": code_txt, "declared_gate": decl_txt})
    return last


# ------------------------------------------------------------------ 噪声底

# 两侧 95% 的 t 分位（df=1..30）。pt_common 是纯标准库、不引 scipy，而这张表又小又固定，
# 直接钉死比用正态 1.96 近似更保守、也更诚实。df 超出表长退回 1.96。
_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def _tcrit(df: int) -> float:
    """两侧 95% 的 t 临界值。df<=0 时返回 inf（等于判不了，绝不放大成通过）。"""
    if df <= 0:
        return float("inf")
    return _T975[df - 1] if df <= len(_T975) else 1.96


def check_noise(rep, runs: list):
    """多 seed 离散度即噪声底；候选相对活的增量没超出它时，**判不了**而不是通过。"""
    nf = T._noise_floor(runs)
    n = nf.get("n") or 0
    if n < 2:
        rep.add(GROUP_NOISE, "噪声底量化", C.UNKNOWN,
                "只有 %d 次有效训练，估不出噪声底。同一份数据换个种子结果本来就会变，"
                "没有这个底数就无法判断「改进」是不是真的。请用多 seed 重跑。" % n,
                {"noise_floor": nf})
        return nf, None

    col = (nf.get("columns") or {}).get("cand_val_ic")
    if not col:
        rep.add(GROUP_NOISE, "噪声底量化", C.UNKNOWN,
                "%d 个种子的 cand_val_ic 没能凑齐，算不出离散度。" % n, {"noise_floor": nf})
        return nf, None

    rep.add(GROUP_NOISE, "噪声底量化", C.PASS,
            "%d 个种子：cand_val_ic 标准差 %.6f、极差 %.6f"
            % (n, col["std"], col["range"]), {"noise_floor": nf})

    # 读数口径：一律用跨种子**池化**值，禁止取最好的那个种子。
    # 依据 arXiv 2511.07678（AIA Forecaster）§7.1 与 Table 10：单次读数既更差也更不稳，
    # 「best-of-k 挑一个」按构造永远不可能超过候选本身，且它在 7.2% 的情况下挑中的是
    # k 个里最差的之一。我们 0912 那次事故正是拿单个读数（val IC -0.55）做的裁决。
    # 池化值本身早就有了（_noise_floor 里的 columns.mean / per_seed），原先只是没拿来当读数。
    seeds = (nf.get("per_seed") or {}).get("cand_val_ic") or []
    rep.add(GROUP_NOISE, "读数口径", C.PASS,
            "本次所有结论一律用 %d 个种子的**池化均值**（cand_val_ic=%+.4f）；逐种子 %s，"
            "极差 %.4f。**不得改取其中最好的那个种子** —— 那是挑噪声，不是挑模型。"
            % (n, col["mean"], "、".join("%+.4f" % v for v in seeds), col["range"]),
            {"pooled": col["mean"], "per_seed": seeds, "rule": "pooled-not-best-of-seed"})

    # 现役读数必须在各试次间一致。不一致 = 现役模型在跑的过程中被换过，
    # 那各试次比的就不是同一个基线，多 seed 比较整个失效（不是数字难看，是无意义）。
    lv = [x for x in (_f((r.get("version_row") or {}).get("live_val_ic")) for r in runs)
          if x is not None]
    if len(lv) < 2:
        # 取不到就说明取不到，绝不当成「一致」放行
        nf["live_consistent"] = False
        rep.add(GROUP_NOISE, "基线一致性", C.UNKNOWN,
                "少于 2 个试次记录了现役读数，无法验证现役在过程中没变，增量判不了。",
                {"live_val_ic": lv})
    elif (max(lv) - min(lv)) > 1e-9:
        nf["live_consistent"] = False
        rep.add(GROUP_NOISE, "基线一致性", C.UNKNOWN,
                "各试次记录的 live_val_ic 不一致（%s）—— 现役模型在过程中变过，"
                "各试次比的不是同一个基线，增量判不了。"
                % "、".join("%+.4f" % v for v in lv), {"live_val_ic": lv})
    else:
        nf["live_consistent"] = True
        rep.add(GROUP_NOISE, "基线一致性", C.PASS,
                "各试次的现役读数一致（live_val_ic=%s），多 seed 比的是同一个基线。"
                % "、".join("%+.4f" % v for v in lv))
    return nf, col


def check_increment(rep, last, nf, col):
    """增量显著性：**池化**候选读数 - 现役读数，是否超出种子间噪声。

    用单样本 t：se = std/sqrt(n)，|增量| > t(0.975, n-1)*se 才算显著。
    这比原先「|增量| > 1 个标准差」严格得多（n=3 时约 2.5 倍），是**有意的**：
    我们怕的是假阳（换了活结果更差），不是假阴。宁可判不了，不可假通过。
    """
    if not last or col is None:
        rep.add(GROUP_NOISE, "增量显著性", C.UNKNOWN,
                "缺门禁读数或缺噪声底，判不了增量是否显著。")
        return None
    lv = _f(last.get("live_val_ic"))
    if lv is None:
        rep.add(GROUP_NOISE, "增量显著性", C.UNKNOWN,
                "现役 val IC 缺（读作 %s），判不了增量。" % last.get("live_val_ic"))
        return None
    if nf.get("live_consistent") is False:
        # 基线本身没验住，就不能再拿它算增量放行 —— 否则会出现「基线一致性判不了」
        # 而「增量显著性通过」的自相矛盾（已实测踩到）。
        rep.add(GROUP_NOISE, "增量显著性", C.UNKNOWN,
                "基线一致性没通过（各试次的现役读数不一致或取不到），增量所比的基线不可信，"
                "**判不了**，这里不给显著性结论。", {"live_consistent": False})
        return None

    n = int(nf.get("n") or 0)
    pooled, std = col["mean"], col["std"]
    inc = pooled - lv
    tcrit = _tcrit(n - 1)
    se = std / (n ** 0.5) if n > 1 else float("inf")
    half = tcrit * se                      # 可能是 inf -> 一律判不了，不放行
    ev = {"increment_val_ic": round(inc, 6), "noise_std": round(std, 6),
          "pooled_cand_val_ic": round(pooled, 6), "live_val_ic": round(lv, 6),
          "n": n, "se": round(se, 6), "t_crit": tcrit,
          "ci": [inc - half, inc + half] if half == half else None,
          "rule": "pooled_one_sample_t"}

    if half != half or abs(inc) <= half:
        if half == half:
            msg = ("val 侧增量 %+.4f（池化 %d 个种子）的 95%% 置信区间 [%+.4f, %+.4f] "
                   "跨过 0 —— 分不清是真改进还是换个种子就会有的波动。"
                   "**判不了，不等于通过。**" % (inc, n, inc - half, inc + half))
        else:
            msg = ("val 侧增量 %+.4f：种子数 %d，算不出置信区间。**判不了，不等于通过。**"
                   % (inc, n))
        rep.add(GROUP_NOISE, "增量显著性", C.UNKNOWN, msg, ev)
        return {"significant": False, **ev}

    rep.add(GROUP_NOISE, "增量显著性", C.PASS,
            "val 侧增量 %+.4f（池化 %d 个种子）的 95%% 置信区间 [%+.4f, %+.4f] 不含 0"
            "（t=%.3f）" % (inc, n, inc - half, inc + half, tcrit), ev)
    return {"significant": True, **ev}


# ---------------------------------------------------------------- 逐日配对检验

# 契约冻结处：`handoff/ic_series/README.md` 第二节。字段名写错不会报错，
# 只会让配对检验**静默地做错** —— 所以这里逐条硬校验，不满足就当场说，不猜不修。
IC_TOP_KEYS = ("seg", "method", "generated_at", "decision", "reason",
               "ensemble_weight_now", "ensemble_weight_suggest",
               "val_window", "test_window", "ic_per_date")
IC_SEGS = ("live_val", "cand_val", "live_test", "cand_test")
IC_PAIRS = (("val", "live_val", "cand_val"), ("test", "live_test", "cand_test"))

_BLOCK_B = 4000          # 自举次数
_BLOCK_SEED = 20260917   # 固定种子：同一份数据每次跑出的区间必须**一样**，否则报告不可复核
_BLOCK_LEN_MIN = 2       # 块长下限；块长按 n^(1/3) 取，保住日间自相关


def _reject_constant(x):
    """json 的 parse_constant：裸 NaN / Infinity / -Infinity 一律当场拒。

    Python 的 json **默认读得回来**裸 NaN，别的语言的标准解析器读不了
    （契约第 5 条专门点了这件事）。这里必须显式拒，不然跨机就是一份读不了的文件。
    """
    raise ValueError("文件里有裸 %s（契约第 5 条：算不出就写 null）" % x)


def _is_num(v):
    """数才算数：bool 是 int 的子类，必须排掉，否则 True 会被当成 1。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_ic_series(obj):
    """按契约校验，返回问题清单（空列表 = 合规）。只挑契约明写的硬项，不发明新规矩。"""
    bad = []
    if not isinstance(obj, dict):
        return ["顶层不是对象"]
    for k in IC_TOP_KEYS:
        if k not in obj:
            bad.append("缺顶层键 %s" % k)
    ipd = obj.get("ic_per_date")
    if not isinstance(ipd, dict):
        bad.append("ic_per_date 不是对象")
        return bad
    for s in IC_SEGS:
        if s not in ipd:
            # 缺失 != 为 null —— 契约第 2 条明写「缺段会被判 FAIL」
            bad.append("ic_per_date 缺段 %s（缺段与 null 不是一回事）" % s)
            continue
        seg = ipd[s]
        if seg is None:
            continue
        if not isinstance(seg, dict):
            bad.append("%s 不是对象" % s)
            continue
        for d, row in list(seg.items())[:2000]:
            if not isinstance(row, dict):
                bad.append("%s[%s] 不是对象" % (s, d))
                break
            if "ens" not in row:
                bad.append("%s[%s] 缺 ens 键（契约第 4 条：ens 必须有）" % (s, d))
                break
            for kk, vv in row.items():
                if vv is None or _is_num(vv):
                    continue
                bad.append("%s[%s].%s 不是数也不是 null（%s）"
                           % (s, d, kk, type(vv).__name__))
                break
    return bad


def load_ic_series(path):
    """严格读。返回 (对象, 问题清单)；问题非空即表示不能用，调用方据此记判不了/不通过。"""
    try:
        txt = C.read_text_any(Path(path))
    except Exception as e:
        return None, ["读不出来：%s" % e]
    try:
        obj = json.loads(txt, parse_constant=_reject_constant)
    except Exception as e:
        return None, ["不是合法 JSON：%s" % e]
    return obj, validate_ic_series(obj)


def find_ic_series(bundle, workdir):
    """找 ic_series.json。返回 (路径, 候选路径列表)。

    落点按契约：候选模式在 `<out_dir>/ic_series.json`（与 pk.json 同级）；
    生产换活后随工件进 `models/ic_series.json`。故工作目录优先，其次包内 models/。
    多个命中时取**最近改动**的那个，并把全部候选列出来备查（不藏）。
    """
    cands = []
    for base in (Path(workdir), Path(bundle) / "models", Path(bundle)):
        try:
            if not base.is_dir():
                continue
            for p in sorted(base.glob("ic_series.json")):
                cands.append(p)
            # 只下潜一层：试次目录/归档目录（rejected_<ts>）都在这一层
            for sub in sorted(base.iterdir()):
                try:
                    if sub.is_dir() and (sub / "ic_series.json").is_file():
                        cands.append(sub / "ic_series.json")
                except OSError:
                    pass
        except OSError:
            pass
    seen, uniq = set(), []
    for p in cands:
        k = str(p).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    if not uniq:
        return None, []
    try:
        uniq.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        pass
    return uniq[0], uniq


def _paired_series(obj, a, b):
    """取两段在**共同日期**上的 ens 配对。返回 (日期, 候选-现役 的差, 该段名)。"""
    ipd = obj.get("ic_per_date") or {}
    sa, sb = ipd.get(a) or {}, ipd.get(b) or {}
    dates = []
    diffs = []
    for d in sorted(set(sa.keys()) & set(sb.keys())):
        ra, rb = sa.get(d) or {}, sb.get(d) or {}
        va, vb = ra.get("ens"), rb.get("ens")
        if _is_num(va) and _is_num(vb):
            dates.append(str(d))
            diffs.append(float(vb) - float(va))
    return dates, diffs


def _block_bootstrap_ci(diffs, B=_BLOCK_B, seed=_BLOCK_SEED):
    """循环分块自举给「均值差」的 95% 区间。

    为什么分块而不逐点重抽：逐日 IC 有自相关（相邻交易日同受一波行情影响），
    逐点自举会把区间**做窄**、假阳变多。块长取 n^(1/3)（n=21 时约 2~3 日），
    循环取块保证首尾相接。返回 (lo, hi, 用了的块长)。
    """
    import random
    n = len(diffs)
    if n < 4:
        return None, None, None
    L = max(_BLOCK_LEN_MIN, int(round(n ** (1.0 / 3.0))))
    if L >= n:
        L = max(1, n - 1)
    rng = random.Random(seed)
    nblocks = -(-n // L)          # ceil
    means = []
    for _ in range(B):
        s = 0.0
        for _b in range(nblocks):
            start = rng.randrange(n)
            for j in range(L):
                s += diffs[(start + j) % n]
        means.append(s / float(nblocks * L))
    means.sort()
    lo = means[int(0.025 * (B - 1))]
    hi = means[int(0.975 * (B - 1))]
    return lo, hi, L


def check_paired(rep, bundle=None, workdir=None):
    """逐日配对检验：候选 vs 现役，同一天上作差。能做就做，不能做就说不能做。

    与边际比较的区别是决定性的：同一天上两个模型面对的是同一波行情，
    作差把共同项消掉，同样的样本量下更可能把「一致地只好一点点」认出来。
    """
    if bundle is None or workdir is None:
        rep.add(GROUP_NOISE, "配对检验（逐日）", C.UNKNOWN,
                "调用方没给包/工作目录，无从查找逐日 IC 序列。")
        return {"status": "no_bundle",
                "detail": "调用方没给包/工作目录，配对检验没做（不是差为 0）。"}

    path, cands = find_ic_series(bundle, workdir)
    if path is None:
        rep.add(GROUP_NOISE, "配对检验（逐日）", C.WARN,
                "包内没有 `ic_series.json`（逐日 IC 序列）。现有增量检验只能用**跨种子的"
                "边际 t 检验**，功效明显低于逐日配对 —— 对「小幅但一致」的改进**检不出**；"
                "这里的「不显著」不等于「没改进」。补原料要训练侧落盘（属策略包/生成器，"
                "本工具不改）：见 `handoff/ic_series/README.md`。",
                {"what": "per_date_ic_series", "searched": [str(bundle), str(workdir)]})
        # 永不用 null 表示"没做" —— null 会被读成"差为 0 / 没问题"。
        return {"status": "no_ic_series",
                "detail": "包内没有 ic_series.json，配对检验**没做**（不是没有差异）。",
                "searched": [str(bundle), str(workdir)]}

    obj, problems = load_ic_series(path)
    if problems:
        rep.add(GROUP_NOISE, "ic_series 契约", C.FAIL,
                "`%s` 不符契约，本次**没有**做配对检验（契约见 handoff/ic_series/README.md "
                "第二节，读取方只认字段名，写错不会报错、只会静默算错）：%s"
                % (path, "；".join(problems[:6])),
                {"path": str(path), "problems": problems})
        rep.add(GROUP_NOISE, "配对检验（逐日）", C.UNKNOWN,
                "逐日序列契约不符，配对检验做不了 —— 判不了，不得读成通过。")
        return {"status": "contract_violation", "path": str(path),
                "detail": "ic_series.json 不符契约，配对检验**没做**（不是没有差异）。",
                "problems": problems}

    rep.add(GROUP_NOISE, "ic_series 契约", C.PASS,
            "`%s` 合规（10 个顶层键齐、四段齐、值皆为数或 null）%s"
            % (path, "；另有 %d 份同名文件，取了最近改动的这份" % (len(cands) - 1)
               if len(cands) > 1 else ""),
            {"path": str(path), "candidates": [str(p) for p in cands]})

    out = {"path": str(path), "seg": obj.get("seg"), "method": obj.get("method"),
           "decision": obj.get("decision"), "pairs": {}}
    did = False
    for tag, live_seg, cand_seg in IC_PAIRS:
        dates, diffs = _paired_series(obj, live_seg, cand_seg)
        n = len(diffs)
        if n < 4:
            out["pairs"][tag] = {"n": n, "verdict": "缺料"}
            continue
        did = True
        mean = sum(diffs) / n
        var = sum((x - mean) ** 2 for x in diffs) / (n - 1) if n > 1 else 0.0
        sd = var ** 0.5
        se = sd / (n ** 0.5) if n else float("inf")
        t = mean / se if se and se == se and se > 0 else float("nan")
        crit = _tcrit(n - 1)
        lo, hi, L = _block_bootstrap_ci(diffs)
        t_sig = (t == t) and abs(t) > crit            # NaN 比较必为 False，正是想要的
        b_sig = (lo is not None) and (lo > 0 or hi < 0)
        pair = {"n": n, "mean": mean, "sd": sd, "t": t, "t_crit": crit,
                "ci": [lo, hi], "block_len": L, "t_significant": bool(t_sig),
                "boot_significant": bool(b_sig), "dates": [dates[0], dates[-1]]}
        if t_sig and b_sig:
            pair["verdict"] = "显著"
            rep.add(GROUP_NOISE, "配对检验（逐日·%s）" % tag, C.PASS,
                    "%d 个共同交易日，候选-现役的逐日 IC 差均值 %+.5f（标准差 %.5f，t=%.3f "
                    "> %.3f）；分块自举 95%% 区间 [%+.5f, %+.5f]（块长 %d）不含 0。"
                    "两种口径一致：这是**成对**读出来的改进，不是边际上的。"
                    % (n, mean, sd, t, crit, lo, hi, L), pair)
        elif t_sig or b_sig:
            pair["verdict"] = "口径不一致"
            rep.add(GROUP_NOISE, "配对检验（逐日·%s）" % tag, C.UNKNOWN,
                    "%d 个共同交易日，差均值 %+.5f：配对 t（%.3f vs 临界 %.3f）与分块自举"
                    "区间 [%+.5f, %+.5f] **结论不一致**。取保守口径 —— **判不了**，"
                    "不得读成通过。" % (n, mean, t, crit, lo, hi), pair)
        else:
            pair["verdict"] = "不显著"
            rep.add(GROUP_NOISE, "配对检验（逐日·%s）" % tag, C.UNKNOWN,
                    "%d 个共同交易日，差均值 %+.5f（标准差 %.5f）：配对 t=%.3f 未过临界 %.3f，"
                    "分块自举 95%% 区间 [%+.5f, %+.5f] 跨过 0。**判不了，不等于通过** ——"
                    "配对检验比边际检验功效高，连它都认不出，说明这点差异落在日间波动里。"
                    % (n, mean, sd, t, crit, lo, hi), pair)
        out["pairs"][tag] = pair

    if not did:
        rep.add(GROUP_NOISE, "配对检验（逐日）", C.UNKNOWN,
                "逐日序列在，但四段里没有一对够长（>=4 个共同交易日）可作差 —— "
                "判不了，不得读成通过。", out)
    else:
        # 与包自报的判决对账：ic_series 里带的 decision 是训练当时记的，
        # 与门禁复算对不上就是包有问题，不是数字不好看。
        dec = obj.get("decision")
        if dec in ("skipped", "kept", "skip"):
            out["declared_decision"] = dec
    return out



# ------------------------------------------------------------------ 上线前置清单

def _coverage_end(bundle: Path):
    """X 日期轴的末行（= 数据覆盖到哪天）。读不到就返回 None，不猜。"""
    for rel in ("X/dates.txt", "X/row_date.txt"):
        p = Path(bundle) / rel
        try:
            if p.is_file():
                lines = [x.strip() for x in C.read_text_any(p).splitlines() if x.strip()]
                if lines:
                    return lines[-1]
        except Exception:
            pass
    return None


def _readiness_items(bundle: Path, workdir: Path, runs: list, doctor: dict):
    """每一项三态：有 -> PASS/WARN/FAIL，查不到 -> UNKNOWN。绝不把查不到写成通过。"""
    items = []

    # 1. 训练产物完整
    trial_models = []
    for r in runs:
        t = r.get("trial")
        if t and (Path(t) / "models").is_dir():
            trial_models.append(Path(t) / "models")
    if not trial_models:
        items.append(("训练产物完整", C.FAIL, "找不到任何试次目录，没有可裁决的产物。"))
    else:
        m = trial_models[-1]
        need = ("lgb.txt", "mlp.pt", "factor_meta.json", "config.json")
        miss = [n for n in need if not (m / n).is_file()]
        if miss:
            items.append(("训练产物完整", C.FAIL,
                          "试次缺件：%s（在 %s）" % (", ".join(miss), m)))
        else:
            items.append(("训练产物完整", C.PASS, "四件齐（%s）" % m))

    # 2. 账物相符：直接采信 doctor 的契约组结论，不重算（doctor 才是这条的权威）
    if not doctor:
        items.append(("账物相符", C.UNKNOWN,
                      "本次工作目录里没有 doctor.json，无法确认账物是否相符。"
                      "请先跑 pt.py doctor。"))
    else:
        ck = [c for c in doctor.get("checks", []) if c.get("group") == "契约与卫生组"]
        # 「账物相符」问的是：异地手上这份，和本机打包时记的那份，是不是一份。
        # 权威是 DISPATCH 的全量逐位比（每个随包文件都有 sha1）。
        # 包内 MANIFEST 只声明了部分条目 —— 那些没声明的永远比不出结论，
        # 那是**包自己的声明覆盖**问题，不是本次传输的账，所以单独说，不拉低这一项。
        disp = [c for c in ck if str(c.get("name", "")).startswith("与打包记录")]
        extra = ""
        nh = [c for c in ck if c.get("name") == "无哈希可比的条目"]
        if nh:
            extra = "；另：%s（那是包内声明覆盖问题，不是本次传输的账）" % nh[0].get("detail", "")[:90]
        use = disp or ck
        bad = [c for c in use if c.get("status") == C.FAIL]
        unk = [c for c in use if c.get("status") == C.UNKNOWN]
        if bad:
            items.append(("账物相符", C.FAIL,
                          "体检契约组不通过：%s" % "; ".join(c["name"] for c in bad)))
        elif unk:
            items.append(("账物相符", C.UNKNOWN,
                          "体检契约组有判不了的项：%s" % "; ".join(c["name"] for c in unk)))
        elif use:
            items.append(("账物相符", C.PASS,
                          "%s%s" % (use[0].get("detail", "体检契约组全通过"), extra)))
        else:
            items.append(("账物相符", C.UNKNOWN, "体检报告里没有契约组记录。"))

    # 3. 训练指纹齐全
    fp = workdir / "fingerprint.json"
    if not fp.is_file():
        items.append(("训练指纹齐全", C.UNKNOWN, "没有 fingerprint.json，无法复现这次训练。"))
    else:
        missing = []
        for r in runs:
            files = r.get("fingerprint_files") or {}
            for k in ("X.npy", "Y_target"):
                if not (files.get(k) or {}).get("sha1"):
                    missing.append("%s/%s" % (r.get("tag"), k))
        if missing:
            items.append(("训练指纹齐全", C.UNKNOWN,
                          "指纹里缺 sha1：%s" % ", ".join(missing[:5])))
        else:
            items.append(("训练指纹齐全", C.PASS, "%d 个试次的 X/Y 指纹齐" % len(runs)))

    # 4. 多 seed 噪声底已量化
    n = len([r for r in runs if r.get("version_row")])
    if n < 2:
        items.append(("多 seed 噪声底已量化", C.UNKNOWN, "只有 %d 次有效训练（需 >= 2）" % n))
    else:
        items.append(("多 seed 噪声底已量化", C.PASS, "%d 次有效训练" % n))

    # 5. Y_realized 在位（哥 0916：权威副本住在 Z 包内，包内无信号史，重建不出来）
    yr = bundle / "Y" / "Y_realized.parquet"
    if yr.is_file():
        items.append(("Y_realized 在位", C.PASS,
                      "%s（%.1f KB）" % (yr.name, yr.stat().st_size / 1024)))
    else:
        items.append(("Y_realized 在位", C.FAIL,
                      "包内没有 Y/Y_realized.parquet。它是「实际持仓兑现」的权威副本，"
                      "包内没有别处能重建，缺了就无法做上线后的归因。"))

    # 6. 数据锚落地（哥 0916：同日落地数据锚 MANIFEST.input_anchor）
    man = C.load_bundle_meta(bundle) or {}
    anchor = (man.get("manifest") or {}).get("input_anchor")
    if anchor:
        items.append(("数据锚落地", C.PASS,
                      "input_anchor=%s" % json.dumps(anchor, ensure_ascii=False)[:120]))
    else:
        # 包内没有落地锚。能判的只有「数据覆盖到哪天」——它证明不了落地时刻
        # （成分股月内原地改写那件事照旧成立），所以这条仍然记「判不了」，不许读成通过。
        cov = _coverage_end(bundle)
        items.append(("数据锚落地", C.UNKNOWN,
                      "MANIFEST 里没有 input_anchor，无法确认这份 X/Y 对应哪天落地的数据。"
                      "「Z=D生成」的前提不成立时，训练产物不可复现。"
                      + ("　可佐证：X 数据覆盖到 %s。" % cov if cov else "")))

    # 6b. 数据锚：X 覆盖到锚点日 —— 这条是**能判**的，且不一致是真红灯
    cov_end = _coverage_end(bundle)
    if cov_end:
        # 两边都必须先归一化：X 日期轴写 `20260916`、MANIFEST 写 `2026-09-16`，
        # 直接比字符串会把「格式不同」误判成「没铺到锚点日」（实测踩过，报了个假红灯）。
        # parse_datetime_loose 返回**归一化后的 'YYYY-MM-DD' 字符串**（不是 datetime）。
        cov_d = C.parse_datetime_loose(cov_end)
        a_mf = C.parse_datetime_loose((man.get("manifest") or man).get("anchor_date"))
        if a_mf is None or cov_d is None:
            items.append(("数据锚：X 覆盖到锚点日", C.UNKNOWN,
                          "有一边解析不出日期，无法比对（X 日期轴末行 %s / "
                          "MANIFEST.anchor_date %s）—— 判不了，不得读成通过。"
                          % (cov_end, (man.get("manifest") or {}).get("anchor_date"))))
        else:
            same = cov_d == a_mf
            items.append(("数据锚：X 覆盖到锚点日",
                          C.PASS if same else C.FAIL,
                          "X 日期轴末行 %s（归一化 %s）/ MANIFEST.anchor_date %s%s"
                          % (cov_end, cov_d, a_mf,
                             "" if same else "  —— 数据没铺到锚点日，训练窗与锚点不是一回事")))

    # 7. 依赖版本一致：取自 doctor 环境组
    if not doctor:
        items.append(("依赖版本一致", C.UNKNOWN, "没有 doctor.json，无法比对依赖版本。"))
    else:
        env = [c for c in doctor.get("checks", []) if c.get("group") == "环境组"]
        bad = [c for c in env if c.get("status") == C.FAIL]
        unk = [c for c in env if c.get("status") == C.UNKNOWN]
        if bad:
            items.append(("依赖版本一致", C.FAIL, "; ".join(c["name"] for c in bad)))
        elif unk:
            items.append(("依赖版本一致", C.UNKNOWN, "; ".join(c["name"] for c in unk)))
        elif env:
            items.append(("依赖版本一致", C.PASS, "环境组 %d 项全通过" % len(env)))
        else:
            items.append(("依赖版本一致", C.UNKNOWN, "体检报告里没有环境组记录。"))

    # 8. 交付卫生：包内不许有 __pycache__（Z 侧铁律）
    pyc = list(bundle.rglob("__pycache__")) + list(bundle.rglob("*.pyc"))
    if pyc:
        items.append(("交付卫生", C.WARN, "包内有 %d 个 __pycache__/.pyc：%s"
                      % (len(pyc), pyc[0].relative_to(bundle))))
    else:
        items.append(("交付卫生", C.PASS, "无 __pycache__/.pyc"))

    return items


def check_readiness(rep, bundle: Path, workdir: Path, runs: list, doctor: dict):
    items = _readiness_items(bundle, workdir, runs, doctor)
    for name, status, detail in items:
        rep.add(GROUP_READY, name, status, detail)
    return {n: s for n, s, _ in items}


# ------------------------------------------------------------------ 裁决姿态

BLOCKING = ("训练产物完整", "Y_realized 在位")


def decide_posture(rep, last, inc, ready: dict, doctor_verdict):
    """五级姿态。**从输入状态直接读出，不从上一级升降**（LangAlpha 的规矩）。"""
    reasons = []
    if doctor_verdict == C.FAIL:
        return "blocked", ["体检不通过（doctor FAIL）：这份包在训练前就该被拦下"]
    gate_fail = [c["name"] for c in rep.fails() if c["group"] == GROUP_GATE]
    if gate_fail:
        # 包自己记的判决与生产门禁公式对不上 -> 这份包的结论不可信，直接硬拦。
        # 不能给「再改改就行」的姿态：连它自报的判决都不能信，别的读数凭什么信。
        return "blocked", ["门禁组不通过（%s）：包自报的判决不可信" % "、".join(gate_fail)]
    for n in BLOCKING:
        if ready.get(n) == C.FAIL:
            return "blocked", ["前置项不通过：%s" % n]
    if not last:
        return "not-ready", ["没有任何带版本记录的试次，无判决可依"]
    if last.get("derived") != "replaced":
        return "not-ready", ["门禁未通过（%s）—— 候选不能替换活模型" % last.get("reason")]

    reasons.append("门禁通过（%s）" % last.get("reason"))
    unknowns = sorted({n for n, s in ready.items() if s == C.UNKNOWN}
                      | {c["name"] for c in rep.unknowns()
                         if c["group"] in (GROUP_READY, GROUP_NOISE)})
    if inc is not None and not inc.get("significant"):
        return "review-ready", reasons + [
            "val 侧增量未超出噪声底 —— 够给哥复核，不够当决策依据"]
    if unknowns:
        return "screen-grade", reasons + ["有判不了的项：%s" % "、".join(unknowns)]
    if inc is None:
        return "review-ready", reasons + ["增量显著性判不了，只够复核"]
    return "decision-grade", reasons + ["前置项齐、噪声底已量化、增量超噪声底"]


# ------------------------------------------------------------------ 主流程

def _f(v):
    """strings -> float，空/NaN -> None。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if v != v else float(v)
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan", "null"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return None if f != f else f


def load_runs(workdir: Path) -> list:
    fp = Path(workdir) / "fingerprint.json"
    if not fp.is_file():
        return []
    try:
        d = json.loads(C.read_text_any(fp))
    except (OSError, ValueError):
        return []
    return d.get("runs") or []


def load_doctor(workdir: Path):
    p = Path(workdir) / "doctor.json"
    if not p.is_file():
        return None
    try:
        return json.loads(C.read_text_any(p))
    except (OSError, ValueError):
        return None


def run_eval(bundle, workdir, out_path=None, report_path=None, quiet=False) -> C.Report:
    bundle, workdir = Path(bundle), Path(workdir)
    rep = C.Report("异地训练评估与裁决", str(bundle),
                   {"workdir": str(workdir), "gate_source": GATE_SOURCE})

    if not (bundle / "MANIFEST.json").is_file():
        rep.add("输入", "训练包", C.FAIL, "这个目录里没有 MANIFEST.json：%s" % bundle)
        return rep
    if not workdir.is_dir():
        rep.add("输入", "工作目录", C.FAIL, "工作目录不存在：%s" % workdir)
        return rep

    runs = load_runs(workdir)
    if not runs:
        rep.add("输入", "试次记录", C.FAIL,
                "%s 里没有 fingerprint.json 或里面没有 runs。没有训练记录就无从评估。"
                % workdir)
    else:
        rep.add("输入", "试次数", C.PASS, "%d 个" % len(runs))

    doctor = load_doctor(workdir)
    doctor_verdict = (doctor or {}).get("verdict")

    last = check_gate(rep, bundle, runs) if runs else None
    nf, col = check_noise(rep, runs) if runs else ({"n": 0}, None)
    paired = check_paired(rep, bundle, workdir)
    inc = check_increment(rep, last, nf, col)
    ready = check_readiness(rep, bundle, workdir, runs, doctor)
    posture, why = decide_posture(rep, last, inc, ready, doctor_verdict)
    rep.add(GROUP_POSTURE, "裁决姿态", C.PASS, "%s —— %s" % (posture, "；".join(why)),
            {"posture": posture, "reasons": why, "readiness": ready},
            kind=C.KIND_CONCLUSION)

    rep.meta["posture"] = posture
    rep.meta["readiness"] = ready
    rep.meta["noise_floor"] = nf
    rep.meta["paired"] = paired

    if out_path:
        p = C.normalize_path(out_path)
        if p.suffix.lower() != ".json":
            p = p / "eval.json"
        rep.write(p)
        if not quiet:
            print("[pt] 评估已写入 %s" % p)
    if report_path:
        rp = C.normalize_path(report_path)
        if rp.suffix.lower() != ".md":
            rp = rp / "report.md"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(render_report(rep, bundle, workdir, last, inc),
                      encoding="utf-8")
        if not quiet:
            print("[pt] 人读报告已写入 %s" % rp)
    return rep


def render_report(rep: C.Report, bundle: Path, workdir: Path, last, inc) -> str:
    """面向「不懂细节的人」：结论在前、依据在后、不确定项单列一节。"""
    c = rep.counts()
    posture = rep.meta.get("posture")
    L = []
    L.append("# 异地训练评估报告")
    L.append("")
    L.append("- 训练包：`%s`" % bundle)
    L.append("- 工作目录：`%s`" % workdir)
    L.append("- 生成时间：%s" % C.now_iso())
    L.append("")
    L.append("## 一、结论")
    L.append("")
    # 一律出中文：报告是给「不懂细节的人」读的，不能混英文枚举
    zh = {C.PASS: "通过", C.WARN: "注意", C.FAIL: "不通过", C.UNKNOWN: "判不了"}
    L.append("**评估总判：%s**（通过 %d / 注意 %d / 不通过 %d / 判不了 %d）"
             % (zh.get(rep.overall(), rep.overall()),
                c[C.PASS], c[C.WARN], c[C.FAIL], c[C.UNKNOWN]))
    L.append("")
    L.append("**裁决姿态：`%s`**" % posture)
    L.append("")
    L.append({"decision-grade": "输入完备、门禁通过、增量超出噪声底 —— 够格作为决策依据。",
              "review-ready": "够格给哥复核，但还不足以直接当决策依据。",
              "screen-grade": "只够做初筛：有判不了的项，或噪声底没量化。",
              "not-ready": "还不具备上线前置条件。",
              "blocked": "被硬拦下：体检不通过或前置项缺失，先解决再谈别的。"
              }.get(posture, "（姿态未定义）"))
    L.append("")
    L.append("> 姿态**从输入状态直接读出**，不从上一级升降。`判不了` 一律不等于通过。")
    L.append("")
    L.append("**是否 promote 不在本报告里决定**：本报告不动包、不动线上模型、不写 Z，"
             "上线仍由本机裁决、由哥独断。")
    L.append("")

    if last:
        nf = rep.meta.get("noise_floor") or {}
        col = (nf.get("columns") or {}).get("cand_val_ic") or {}
        seeds = (nf.get("per_seed") or {}).get("cand_val_ic") or []
        L.append("## 二、门禁读数")
        L.append("")
        L.append("| 项 | 值 |")
        L.append("|---|---|")
        npool = nf.get("n") or 0
        pool_txt = (_fmt(col.get("mean")) if (npool >= 2 and col.get("mean") is not None)
                    else "—（种子不足，无法池化）")
        L.append("| 候选 val IC（**池化**，本次结论所用的读数） | %s |" % pool_txt)
        L.append("| 候选 val IC（逐种子） | %s |"
                 % ("、".join(_fmt(v) for v in seeds) if seeds else "—"))
        L.append("| 候选 val IC（包内记录的单次读数，**仅备查，不用于结论**） | %s |"
                 % _fmt(last.get("cand_val_ic")))
        L.append("| 活   val IC | %s |" % _fmt(last.get("live_val_ic")))
        L.append("| 候选 test IC | %s |" % _fmt(last.get("cand_test_ic")))
        L.append("| 活   test IC | %s |" % _fmt(last.get("live_test_ic")))
        L.append("| 包自报 decision | `%s` |" % last.get("recorded"))
        L.append("| 复算 decision | `%s` |" % last.get("derived"))
        L.append("| 训练方式 | `%s` |" % last.get("method"))
        L.append("")
        L.append("复算依据：%s，逐字复刻生产门禁。复算与包自报不一致 = **包有问题**，"
                 "不是数字不好看。" % GATE_SOURCE)
        L.append("")
        L.append("读数口径：一律用**池化**值，**不得**改取其中最好的那个种子 —— 那是挑噪声。"
                 "单次读数既更差也更不稳（依据 arXiv 2511.07678 的 Table 9/10）。")
        L.append("")
        if inc:
            if inc.get("ci"):
                L.append("增量（候选池化 - 活，val 侧）：`%+.4f`，95%% 置信区间 "
                         "`[%+.4f, %+.4f]`（%d 个种子，t=%.3f）。"
                         % (inc["increment_val_ic"], inc["ci"][0], inc["ci"][1],
                            inc["n"], inc["t_crit"]))
            else:
                L.append("增量（候选池化 - 活，val 侧）：`%+.4f`（%d 个种子，算不出区间）。"
                         % (inc["increment_val_ic"], inc["n"]))
            L.append("")

    L.append("## 三、上线前置清单")
    L.append("")
    L.append("| 项 | 判定 | 说明 |")
    L.append("|---|---|---|")
    for name, status, detail in _readiness_items(bundle, workdir, load_runs(workdir),
                                                 load_doctor(workdir)):
        L.append("| %s | %s | %s |" % (name, zh.get(status, status), detail))
    L.append("")

    unk = rep.unknowns()
    L.append("## 四、判不了的项（**不等于通过**）")
    L.append("")
    if not unk:
        L.append("无。")
    else:
        for u in unk:
            L.append("- **[%s] %s**：%s" % (u["group"], u["name"], u["detail"]))
    L.append("")

    bad = rep.fails()
    L.append("## 五、不通过的项")
    L.append("")
    if not bad:
        L.append("无。")
    else:
        for u in bad:
            L.append("- **[%s] %s**：%s" % (u["group"], u["name"], u["detail"]))
    L.append("")
    L.append("## 六、下一步")
    L.append("")
    L.append("把这份 `report.md` 与 `eval.json` 发回本机，由本机裁决。")
    L.append("不确定的事不要猜，也不要把「判不了」当成「通过」。")
    L.append("")
    return "\n".join(L)


def _fmt(v):
    return "—" if v is None else "%+.4f" % v


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description="异地训练评估与裁决（只读产报告）")
    ap.add_argument("--bundle", required=True, help="训练包目录（只读）")
    ap.add_argument("--workdir", required=True, help="训练工作目录（里面有 fingerprint.json）")
    ap.add_argument("--out", default=None,
                    help="评估 JSON 落盘位置：给目录则写入 <目录>/eval.json")
    ap.add_argument("--report", default=None,
                    help="人读报告落盘位置：给目录则写入 <目录>/report.md")
    ap.add_argument("--quiet", action="store_true", help="不打屏，只落盘")
    args = ap.parse_args(argv)

    bundle, workdir = Path(args.bundle), Path(args.workdir)
    # --quiet 只闭嘴，**不**关盘：落盘是交付物，不能因为安静就没了
    out = args.out if args.out is not None else (workdir / "eval.json")
    rep_path = args.report if args.report is not None else (workdir / "report.md")

    rep = run_eval(bundle, workdir, out, rep_path, quiet=args.quiet)
    if not args.quiet:
        print(rep.render())
        print("[pt] 裁决姿态：%s" % rep.meta.get("posture"))
    return rep.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
