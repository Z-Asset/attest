"""pt_doctor —— 异地落地后的强制体检。

四组：环境 / 输入 / 标签 / 契约与卫生。任一组出现 FAIL 即退出码 1 并停。

设计原则（哥 0917 定）：
- 「判不了」绝不等于「通过」：判不了的项记 UNKNOWN，单独列出，不许并进 PASS。
  （先例：S13_csi500seq×7 那次 21 条没有 sha1 的条目被报成「逐位相等」。）
- 检查项的名字必须能让不懂细节的人看懂，不许只写内部术语。
- 一切文件名都从 MANIFEST.json / Y_meta.json **发现**，不许硬编码
  （S03 的 Y 叫 Y_target.npy，S11 叫 Y_target_up.npy + Y_target_ret.npy；
  行索引一个用 row_date.txt 一个用 row_date.npy）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pt_common as C  # noqa: E402

try:
    import numpy as np
except Exception:  # 只有 numpy 缺失也不许崩，降级为「判不了」
    np = None


G_ENV = "环境组"
G_IN = "输入组"
G_LAB = "标签组"
G_CON = "契约与卫生组"

# FAIL 时的四段式话术：按**检查项名字**匹配（不匹配 detail，否则会张冠李戴），
# 顺序即优先级，越具体的写在越前面。
HINTS = [
    ("文件头", ("这个文件声称是数组，但格式不是合法的 .npy。",
                "不要继续训练。把包删掉，向本机重新索取一份。")),
    ("哈希不符", ("磁盘上的数据和元数据里登记的不是同一份，说明文件被改过或传坏了。",
                  "不要继续训练。把这个包删掉，向本机重新索取一份。")),
    ("缺文件条目", ("包本身不完整，训练必然报错。",
                    "重新解压 / 重新传输这个包，确认传输过程没有中断。")),
    ("文件存在", ("包本身不完整，训练必然报错。",
                  "重新解压 / 重新传输这个包，确认传输过程没有中断。")),
    ("标签退化", ("标签已经不是一个有效的预测目标了（比如退化成价格、涨跌方向，"
                  "或者几乎全是同一个值）。在这种标签上训练，指标会很好看但模型是废的。"
                  "这正是 2026-09-12 S11_mom 出过的事故。",
                  "停止训练，把本报告发回本机核对标签构造；不要自行修改 Y。")),
    ("与特征重叠", ("标签和某个输入特征几乎完全一样，模型学会的只是复制那一列。",
                    "停止训练，把本报告发回本机核对标签与特征的构造。")),
    ("整列塌零", ("某个因子列在有限值里全是 0，等于这个因子完全没起作用。"
                  "这通常是整列被 fillna(0) 之后没有还原（S01_index_enhance 出过："
                  "双头实际是单头，且一路不报错）。",
                  "先确认这一列是不是本来就该全零（例如某个哑变量）；拿不准就把本报告发回本机。")),
    ("行数对齐", ("X 和 Y 的行数对不上，样本和标签错位。",
                  "不要继续训练，把报告发回本机核对。")),
    ("行日期轴", ("面板的行顺序不是「按标的分段后段内日期递增」，说明样本行可能被打乱过。",
                  "不要继续训练，把报告发回本机核对导出流程。")),
    ("元数据自洽", ("包内两份元数据自己就打架（例如行数、锚点日期对不上）。"
                    "这说明打包时只更新了一部分文件，你拿到的不是一份自洽的快照。",
                    "不要训练。把报告发回本机重新导出这个包。")),
    ("形状", ("数据维度和元数据登记的不一致，训练会在中途崩。",
              "不要继续，把报告发回本机重新导出这个包。")),
    ("Python 版本", ("异地这台机器的 Python 版本离本包要求的差得太多，"
                      "训练会在中途因为语法/库行为变化崩掉，不是数据的问题。",
                      "装一个符合要求的 Python（通常是 3.10 或更高）再重跑本体检；"
                      "拿不准就把报告发回本机。")),
    ("requirements 下限", ("包声明的依赖下限（requirements.txt）在这台机器上没有满足："
                           "要么相关的库根本没装，要么装的版本低于包能接受的最低要求。"
                           "真训练会当场崩，不会给出模型。",
                           "按报告里列出的库与下限装齐（例如 pip install -r requirements.txt），"
                           "再重跑本体检。装不上就把报告发回本机。")),
    ("数组条目", ("包内登记的这个数组文件在磁盘上缺失或读不出来，训练必然报错。",
                  "重新解压 / 重新传输这个包，确认传输过程没有中断；仍是缺的就发回本机。")),
    ("全 NaN 列", ("某个因子列在有限值里一个数都没有，这一列对模型毫无信息。"
                   "通常意味着上游数据没铺到这段窗口（这类静默退化最容易被漏掉）。",
                   "先确认这一段窗口本来是不是就没有这个因子的数据；拿不准就把报告发回本机。")),
    (" 存在", ("这一件输入/轴文件在包里找不到，训练必然报错。",
               "重新解压 / 重新传输这个包，确认传输过程没有中断。")),
    ("与打包记录不符（DISPATCH）", ("磁盘上的文件与本机打包时实算的 sha1 对不上："
                                    "传输过程中坏了，或者手里这份不是本机导出的那一份。"
                                    "在这种包上训练出来的模型不可信。",
                                    "把这份包删掉，向本机重新索取一份（并核对 zip 的 sha256）。")),
    ("DISPATCH 登记但磁盘上没有", ("打包记录里列了这件文件，但磁盘上没有 —— 解压不全。",
                                   "重新解压这个包；仍是缺的就把报告发回本机重新打包。")),
    ("包目录", ("给的路径不是一个训练包目录。",
                "确认 --bundle 指的是解压后的**包目录本身**，不是它的上一层。")),
    ("MANIFEST.json", ("这个目录里没有 MANIFEST.json，它就不是一个训练包。",
                       "确认 --bundle 指的是解压后的**包目录本身**，不是它的上一层。")),
    ("依赖版本", ("异地装的库版本与本机导出时不一致，同一个包会算出不同的数。"
                  "LightGBM 改版会改分箱，torch 改版会改数值。",
                  "按本报告给出的版本号重装；或让本机重新导出一份匹配你环境的包。")),
    ("GPU", ("本次要真训练，但没有可用的 CUDA 设备。",
             "确认显卡驱动与 CUDA 已装好；若本来就想用 CPU，请让本机确认这是允许的。")),
]


def hint_for(name: str):
    for prefix, pair in HINTS:
        if prefix in name:
            return pair
    # 兜底也必须诚实且可操作：说清"这是本工具没预料到"，而不是含糊过去，
    # 并明确把原始数据交回本机 —— 不懂细节的操作者不该被要求自己判断。
    return ("这一项本工具没有预先写好的说明 —— 这不是「没问题」，"
            "是本工具没预料到会卡在这里（属于工具的缺口，得补）。",
            "把本报告发回本机（报告里带这一项的原始数据），由本机判断下一步。")


# ==================================================================== 数组统计

def _stats_scalar(a, absmax_cap=None):
    """整体统计。a 为已加载的 numpy 块（会被摊平）。"""
    flat = a.reshape(-1)
    fin = np.isfinite(flat)
    n = flat.size
    n_fin = int(fin.sum())
    out = {"count": n, "finite": n_fin, "nan": int(np.isnan(flat).sum()),
           "inf": int(np.isinf(flat).sum()),
           "finite_ratio": (n_fin / n) if n else 0.0,
           "absmax": 0.0, "min": None, "max": None, "mean": None, "std": None,
           "zero": 0, "zero_ratio": 0.0,
           "unique_sampled": None}
    if n_fin == 0:
        return out
    v = flat[fin]
    out["absmax"] = float(np.max(np.abs(v)))
    out["min"] = float(v.min())
    out["max"] = float(v.max())
    out["mean"] = float(v.mean())
    out["std"] = float(v.std())
    z = int((v == 0).sum())
    out["zero"] = z
    out["zero_ratio"] = z / n_fin
    return out


def array_stats(path: Path, want_cols: bool, chunk_budget_elems: int = 8_000_000):
    """流式统计 npy，内存安全。

    返回 (overall, per_col 或 None, note)。
    per_col 只在 ndim == 2 时给；其他维度**明确返回 None 并附 note**，
    免得「没检查」被读成「检查通过」。
    """
    if np is None:
        return None, None, "numpy 未安装，无法读取数组"
    a = np.load(str(path), mmap_mode="r")
    shape = tuple(int(x) for x in a.shape)
    note = ""
    per_row = 1
    for d in shape[1:]:
        per_row *= d
    chunk = max(1, min(shape[0], chunk_budget_elems // max(1, per_row)))

    total = 0
    n_nan = n_inf = n_fin = n_zero = 0
    absmax = 0.0
    vmin, vmax = None, None
    s = ss = 0.0
    col_nan = col_fin = col_zero = None
    col_min = col_max = None
    if want_cols and a.ndim == 2:
        ncol = shape[1]
        col_nan = np.zeros(ncol, dtype=np.int64)
        col_fin = np.zeros(ncol, dtype=np.int64)
        col_zero = np.zeros(ncol, dtype=np.int64)
        col_min = np.full(ncol, np.inf)
        col_max = np.full(ncol, -np.inf)

    for i in range(0, shape[0], chunk):
        blk = a[i:i + chunk]
        blk = np.asarray(blk)
        if blk.dtype.kind not in "fiu":
            note = "非数值 dtype，已跳过数值统计"
            break
        b64 = blk.astype(np.float64, copy=False) if blk.dtype.kind == "f" else blk.astype(np.float64)
        fin = np.isfinite(b64)
        n_fin_blk = int(fin.sum())
        total += b64.size
        n_fin += n_fin_blk
        n_nan += int(np.isnan(b64).sum())
        n_inf += int(np.isinf(b64).sum())
        if n_fin_blk:
            v = b64[fin]
            am = float(np.max(np.abs(v)))
            if am > absmax:
                absmax = am
            lo, hi = float(v.min()), float(v.max())
            vmin = lo if vmin is None else min(vmin, lo)
            vmax = hi if vmax is None else max(vmax, hi)
            n_zero += int((v == 0).sum())
            s += float(v.sum())
            ss += float(np.square(v).sum())
        if col_fin is not None:
            if b64.ndim == 2:
                col_fin += fin.sum(axis=0)
                col_nan += (~fin).sum(axis=0)
                with np.errstate(invalid="ignore"):
                    masked = np.where(fin, b64, np.nan)
                    col_zero += (masked == 0).sum(axis=0)
                    cur_min = np.nanmin(masked, axis=0)
                    cur_max = np.nanmax(masked, axis=0)
                col_min = np.fmin(col_min, cur_min)
                col_max = np.fmax(col_max, cur_max)

    overall = {
        "shape": list(shape), "count": total, "finite": n_fin,
        "nan": n_nan, "inf": n_inf,
        "finite_ratio": (n_fin / total) if total else 0.0,
        "absmax": absmax, "min": vmin, "max": vmax,
        "mean": (s / n_fin) if n_fin else None,
        "std": (max(0.0, ss / n_fin - (s / n_fin) ** 2) ** 0.5) if n_fin else None,
        "zero": n_zero,
        "zero_ratio": (n_zero / n_fin) if n_fin else 0.0,
    }

    per_col = None
    if col_fin is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            nan_ratio = col_nan / np.maximum(col_fin + col_nan, 1)
            zero_ratio = col_zero / np.maximum(col_fin, 1)
        per_col = {
            "n_cols": int(shape[1]),
            "const_cols": int(np.sum(col_min == col_max)),
            "all_nan_cols": int(np.sum(col_fin == 0)),
            "max_col_nan_ratio": float(np.nanmax(nan_ratio)) if shape[1] else 0.0,
            "cols_nan_gt_50": int(np.sum(nan_ratio > 0.5)),
            "cols_zero_gt_99": int(np.sum(zero_ratio > 0.99)),
            "cols_zero_gt_90": int(np.sum(zero_ratio > 0.90)),
            "zero_ratio_of_cols": [float(x) for x in zero_ratio[:200]],
            "all_zero_cols": int(np.sum((zero_ratio >= 1.0) & (col_fin > 0))),
        }
    elif want_cols:
        note = (note + " " if note else "") + (
            "该数组是 %d 维，不做逐列检查（只报整体）—— 逐列项一律记「判不了」，不记通过。"
            % a.ndim)
    return overall, per_col, note.strip()


def _finite_values(path: Path, cap: int = 3_000_000):
    if np is None:
        return None
    a = np.load(str(path), mmap_mode="r", allow_pickle=False)
    flat = np.asarray(a).reshape(-1)
    if flat.dtype.kind not in "fiu":
        return None
    flat = flat.astype(np.float64, copy=False)
    fin = flat[np.isfinite(flat)]
    if fin.size > cap:
        idx = np.linspace(0, fin.size - 1, cap).astype(np.int64)
        fin = fin[idx]
    return fin


def spearman(a: "np.ndarray", b: "np.ndarray") -> float:
    """秩相关（numpy only）。"""
    n = a.size
    if n < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra * ra).sum()) * np.sqrt((rb * rb).sum()))
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


# ==================================================================== 条目发现

def pick_entries(meta: dict):
    """挑出要检查的数组条目，并**发现**它们的磁盘路径。

    返回 (x_entries, y_entries)：每项 = dict(key, path, entry, declared_shape, declared_dtype)
    """
    manifest = meta.get("manifest") or {}
    bundle = Path(meta["bundle"])
    xs, ys = [], []
    for section, key, entry in C.collect_manifest_entries(manifest):
        p = C.resolve_entry_path(bundle, entry)
        rec = {
            "key": key, "section": section, "path": p, "entry": entry,
            "is_npy": bool(p is not None and str(p).lower().endswith(".npy")),
            "tried": [str(x) for x in C.entry_candidates(bundle, entry)],
            "declared_shape": entry.get("shape"), "declared_dtype": entry.get("dtype"),
            "declared_sha1": entry.get("sha1"), "declared_bytes": entry.get("bytes"),
            "note": entry.get("note"),
        }
        if section == "X":
            xs.append(rec)
        elif section == "Y":
            ys.append(rec)
    # 兜底：MANIFEST 没给路径时，按目录扫（但仍把「无声明」如实标出）
    if not xs and (bundle / "X").is_dir():
        for f in sorted((bundle / "X").glob("*.npy")):
            xs.append({"key": "X/" + f.name, "section": "X", "path": f, "entry": {},
                       "is_npy": True, "declared_shape": None, "declared_dtype": None,
                       "declared_sha1": None, "declared_bytes": None,
                       "note": "MANIFEST 未登记，按目录发现"})
    if not ys and (bundle / "Y").is_dir():
        for f in sorted((bundle / "Y").glob("*.npy")):
            ys.append({"key": "Y/" + f.name, "section": "Y", "path": f, "entry": {},
                       "is_npy": True, "declared_shape": None, "declared_dtype": None,
                       "declared_sha1": None, "declared_bytes": None,
                       "note": "MANIFEST 未登记，按目录发现"})
    return xs, ys


# ==================================================================== 四组检查

def check_env(rep: C.Report, cfg: dict, meta: dict, dispatch: dict | None):
    g = G_ENV
    manifest = meta.get("manifest") or {}
    here = C.host_info()
    rep.add(g, "Python 可执行文件", C.PASS, here["python_executable"])
    rep.add(g, "CPU 核数", C.PASS, str(here["cpu_count"]))

    # 期望版本：DISPATCH 优先（本机导出时实测），否则 MANIFEST.env
    want = {}
    _dv = None
    if dispatch:
        # pack 写的键是 deps_pinned；早期版本读的是 dep_versions —— 两个都认，
        # 否则「以本机实测的精确版本为准」这条从来没生效过（实测踩到）。
        _dv = dispatch.get("dep_versions") or dispatch.get("deps_pinned")
    if isinstance(_dv, dict) and _dv:
        want = dict(_dv)
        src = "DISPATCH.json（本机打包时实测的精确版本）"
    elif C.declared_env(manifest):
        want = dict(C.declared_env(manifest))
        src = "MANIFEST.json 的 env 字段（生产导出时的环境）"
    else:
        src = None

    if not want:
        rep.add(g, "依赖版本比对", C.UNKNOWN,
                "包里没有声明期望版本（DISPATCH.json 的 deps_pinned 与 MANIFEST.env 都缺），"
                "无法判断你机器上的库版本是否与产出这份包的环境一致。")
    else:
        rep.add(g, "依赖版本期望来源", C.PASS, src)
        py_want = want.get("python")
        if py_want:
            got = here["python"]
            ok = ".".join(got.split(".")[:2]) == ".".join(str(py_want).split(".")[:2])
            rep.add(g, "Python 版本", C.PASS if ok else C.FAIL,
                    "期望 %s / 实际 %s（按前两位比对）" % (py_want, got))
        for mod in ("numpy", "pandas", "lightgbm", "torch"):
            if mod not in want:
                continue
            exp = str(want[mod])
            got = C.import_version(mod)
            if got is None:
                status = C.FAIL
                detail = "期望 %s / 实际 未安装" % exp
            elif _ver_compatible(got, exp):
                status = C.PASS
                detail = "期望 %s / 实际 %s" % (exp, got)
            else:
                status = C.FAIL
                detail = "期望 %s / 实际 %s" % (exp, got)
            rep.add(g, "依赖版本 %s" % mod, status, detail)

    # requirements.txt 下限（只在版本低于下限时 FAIL，避免与上面重复报警）
    req = meta.get("requirements")
    if req:
        lows = []
        for line in req.splitlines():
            line = line.split("#", 1)[0].strip()
            if ">=" in line:
                name, ver = line.split(">=", 1)
                lows.append((name.strip(), ver.strip()))
        bad, missing = [], []
        for name, ver in lows:
            got = C.import_version(name)
            if got is None:
                missing.append("%s 未安装（下限 %s，根本没得比）" % (name, ver))
            elif not _ver_compatible(got, ver, floor=True):
                bad.append("%s 实际 %s < 下限 %s" % (name, got, ver))
        # 没装的**不能**算满足：那是「判不了」，报成「通过」就是假 PASS
        if bad or missing:
            rep.add(g, "requirements 下限", C.FAIL, "; ".join(bad + missing))
        elif lows:
            rep.add(g, "requirements 下限", C.PASS, "%d 项下限声明均满足" % len(lows))
        else:
            rep.add(g, "requirements 下限", C.PASS, "无下限声明")

    gpu = C.gpu_info()
    if gpu.get("cuda_available") is None:
        rep.add(g, "GPU / CUDA", C.WARN,
                "%s；真训练若需要 GPU 会失败。" % gpu.get("note", "无法判定"))
    elif gpu["cuda_available"]:
        rep.add(g, "GPU / CUDA", C.PASS,
                "CUDA %s / %s / 设备数 %s" % (gpu.get("cuda_version"),
                                              gpu.get("device_name"), gpu.get("device_count")))
    else:
        rep.add(g, "GPU / CUDA", C.FAIL,
                "torch 已装 %s，但 CUDA 不可用。" % C.import_version("torch"))


def _ver_compatible(got: str, want: str, floor: bool = False) -> bool:
    """floor=True 时判 got >= want，否则判前两位相同。"""
    def nums(s):
        out = []
        for part in str(s).replace("+", ".").split("."):
            d = "".join(ch for ch in part if ch.isdigit())
            if d == "":
                break
            out.append(int(d))
        return out
    a, b = nums(got), nums(want)
    if not a or not b:
        return True  # 解析不了就不误报
    if floor:
        n = max(len(a), len(b))
        a += [0] * (n - len(a))
        b += [0] * (n - len(b))
        return a >= b
    return a[:2] == b[:2]


def check_inputs(rep: C.Report, meta: dict, xs: list):
    g = G_IN
    bundle = Path(meta["bundle"])
    if not xs:
        rep.add(g, "X 数组条目", C.FAIL, "MANIFEST.json 里没有任何 X 数组条目，X/ 目录也没有 .npy")
        return
    for rec in xs:
        tag = rec["key"]
        p = rec["path"]
        if p is None or not Path(p).is_file():
            rep.add(g, "%s 文件存在" % tag, C.FAIL,
                    "找不到文件。MANIFEST 登记的是 %s；按以下几种解释都找不到：%s"
                    % (rec["entry"].get("dst") or rec["entry"].get("path") or "未登记",
                       "；".join(rec.get("tried") or ["(无)"])))
            continue
        p = Path(p)
        size = p.stat().st_size
        if not rec.get("is_npy"):
            # 非 npy（factors.txt / row_date.txt / row_code.txt / dates.txt 等）：
            # 只核存在性与哈希（哈希在契约组），不做数组检查，也**不记通过数组检查**。
            rep.add(g, "%s 文件存在" % tag, C.PASS,
                    "%s（%.1f KB，非 npy，不做数组检查；哈希在契约组）" % (p.name, size / 1e3))
            continue
        try:
            hdr = C.npy_header(p)
        except Exception as exc:
            rep.add(g, "%s 文件头" % tag, C.FAIL, "不是合法的 .npy：%s" % exc)
            continue
        rep.add(g, "%s 文件存在" % tag, C.PASS, "%s（%.1f MB，npy 头 %d 字节）"
                % (p.name, size / 1e6, hdr["header_len"]))

        want_shape = rec["declared_shape"]
        if want_shape is None:
            rep.add(g, "%s 形状" % tag, C.UNKNOWN,
                    "MANIFEST 未登记 shape，实际 %s；无法判断是否与产出时一致" % (list(hdr["shape"]),))
        else:
            ws = [int(x) for x in want_shape]
            ok = list(hdr["shape"]) == ws
            rep.add(g, "%s 形状" % tag, C.PASS if ok else C.FAIL,
                    "声明 %s / 实际 %s" % (ws, list(hdr["shape"])))
        want_dt = rec["declared_dtype"]
        if want_dt:
            ok = C.dtype_matches(hdr["dtype"], want_dt)
            rep.add(g, "%s dtype" % tag, C.PASS if ok else C.WARN,
                    "声明 %s / 实际 %s" % (want_dt, C.dtype_name(hdr["dtype"])))

        overall, per_col, note = array_stats(p, want_cols=True)
        if overall is None:
            rep.add(g, "%s 数值统计" % tag, C.UNKNOWN, note)
            continue
        fr = overall["finite_ratio"]
        st = C.PASS if fr >= 0.50 else (C.WARN if fr >= 0.20 else C.FAIL)
        rep.add(g, "%s 有限值比例" % tag, st,
                "%.6f   NaN %d   Inf %d   零值率 %.4f"
                % (fr, overall["nan"], overall["inf"], overall["zero_ratio"]),
                {"finite_ratio": fr, "absmax": overall["absmax"],
                 "min": overall["min"], "max": overall["max"]})
        if per_col is None:
            rep.add(g, "%s 逐列检查" % tag, C.UNKNOWN, note)
        else:
            if per_col["all_nan_cols"]:
                rep.add(g, "%s 全 NaN 列" % tag, C.FAIL,
                        "%d / %d 列全为 NaN（列号见 QUALITY 报告）"
                        % (per_col["all_nan_cols"], per_col["n_cols"]))
            else:
                rep.add(g, "%s 全 NaN 列" % tag, C.PASS, "0 列")
            if per_col["const_cols"]:
                rep.add(g, "%s 常量列（无变化）" % tag, C.WARN,
                        "%d / %d 列取值恒定，对模型无贡献"
                        % (per_col["const_cols"], per_col["n_cols"]))
            else:
                rep.add(g, "%s 常量列（无变化）" % tag, C.PASS, "0 列")
            # 整列塌零：专治 fillna(0) 后整列被 clip 成 0 的静默退化
            n_zero99 = per_col["cols_zero_gt_99"]
            if per_col["all_zero_cols"]:
                rep.add(g, "%s 整列塌零" % tag, C.FAIL,
                        "%d / %d 列在有限值里 100%% 为零 —— 该因子完全失效，"
                        "疑似整列被填 0 后未还原（S01_index_enhance 出过同类静默退化）"
                        % (per_col["all_zero_cols"], per_col["n_cols"]),
                        {"cols_zero_gt_90": per_col["cols_zero_gt_90"]})
            elif n_zero99:
                rep.add(g, "%s 整列塌零" % tag, C.WARN,
                        "%d / %d 列的零值率 > 99%%，接近失效" % (n_zero99, per_col["n_cols"]))
            else:
                rep.add(g, "%s 整列塌零" % tag, C.PASS,
                        "零值率 > 99%% 的列：0（最高列零值率 %.4f）"
                        % max(per_col["zero_ratio_of_cols"] or [0.0]))
            mx = per_col["max_col_nan_ratio"]
            rep.add(g, "%s 最高列 NaN 率" % tag,
                    C.PASS if mx <= 0.5 else C.WARN,
                    "%.4f（NaN 率 > 50%% 的列共 %d 个）" % (mx, per_col["cols_nan_gt_50"]))

    # 行索引 / 日期轴
    _check_row_axis(rep, bundle)


def _read_str_axis(f: Path):
    """把行索引文件读成字符串数组（.npy 或纯文本）。"""
    if np is None:
        return None
    if f.suffix == ".npy":
        a = np.asarray(np.load(str(f), mmap_mode="r")).reshape(-1)
        return a.astype(str)
    return np.array([ln.strip() for ln in C.read_text_any(f).splitlines() if ln.strip()])


def _check_row_axis(rep: C.Report, bundle: Path):
    """行顺序检查。

    已实测的正确形态（S03）：面板是**先码后日**布局 —— 同一个标的的 399 个交易日连续成段，
    段与段之间日期从末尾跳回首日，故**全局非单调是正常的**。拿全局单调当判据会误杀。
    真正该判的是「按 row_code 分段后，每段内日期单调不减」。
    """
    g = G_IN
    date_file = None
    for name in ("X/row_date.npy", "X/row_date.txt", "X/dates.txt", "X/row_date.csv"):
        f = bundle / name
        if f.is_file():
            date_file = f
            break
    if date_file is None:
        rep.add(g, "行日期轴", C.UNKNOWN,
                "X/ 下找不到行日期文件（试过 row_date.npy/.txt、dates.txt、row_date.csv）")
        return None
    if np is None:
        rep.add(g, "行日期轴", C.UNKNOWN, "numpy 缺失，无法读取 %s" % date_file.name)
        return None
    try:
        dates = _read_str_axis(date_file)
    except Exception as exc:
        rep.add(g, "行日期轴", C.UNKNOWN, "%s 读取失败：%s" % (date_file.name, exc))
        return None
    if dates is None or dates.size == 0:
        rep.add(g, "行日期轴", C.UNKNOWN, "%s 读不出内容" % date_file.name)
        return None

    n = int(dates.size)
    uniq = np.unique(dates)
    global_mono = bool(np.all(dates[1:] >= dates[:-1]))

    code_file = None
    for name in ("X/row_code.npy", "X/row_code.txt"):
        f = bundle / name
        if f.is_file():
            code_file = f
            break
    if code_file is None:
        rep.add(g, "行日期轴", C.WARN,
                "%s   行数 %d   不同日期 %d   [%s .. %s]   %s；"
                "但 X/ 下没有行标码文件，无法判断「按标的分段后组内是否有序」"
                % (date_file.relative_to(bundle).as_posix(), n, uniq.size, uniq[0], uniq[-1],
                   "全局单调不减" if global_mono else "全局非单调（先码后日布局时这是正常的）"))
        return n

    try:
        codes = _read_str_axis(code_file)
    except Exception as exc:
        rep.add(g, "行日期轴", C.UNKNOWN, "%s 读取失败：%s" % (code_file.name, exc))
        return n
    if codes is None or codes.size != n:
        rep.add(g, "行日期轴", C.UNKNOWN,
                "%s(%s 行) 与 %s(%s 行) 长度不一致，无法判断行顺序"
                % (code_file.name, None if codes is None else codes.size, date_file.name, n))
        return n

    brk = np.nonzero(codes[1:] != codes[:-1])[0]
    starts = np.concatenate(([0], brk + 1))
    ends = np.concatenate((brk + 1, [n]))
    bad = 0
    for s, e in zip(starts, ends):
        seg = dates[s:e]
        if seg.size > 1 and not np.all(seg[1:] >= seg[:-1]):
            bad += 1
    detail = ("%s / %s   行数 %d   不同日期 %d   段数 %d   [%s .. %s]   %s"
              % (date_file.relative_to(bundle).as_posix(), code_file.name, n, uniq.size,
                 len(starts), uniq[0], uniq[-1],
                 "全局单调" if global_mono else "先码后日布局（全局非单调属正常）"))
    if bad == 0:
        rep.add(g, "行日期轴", C.PASS, detail + "；按标的分段后，每段内日期单调不减")
    else:
        rep.add(g, "行日期轴", C.FAIL,
                detail + "；其中 %d 段的日期不是单调递增 —— 面板可能被打乱过" % bad)
    return n


def check_labels(rep: C.Report, meta: dict, xs: list, ys: list):
    g = G_LAB
    if not ys:
        rep.add(g, "Y 数组条目", C.UNKNOWN,
                "MANIFEST.json 里没有 Y 数组条目。若这是纯规则型策略可忽略；"
                "否则说明包不完整，不能训练。")
        return
    x_rows = None
    for rec in xs:
        if rec.get("is_npy") and rec["path"] and Path(rec["path"]).is_file():
            try:
                x_rows = C.npy_header(Path(rec["path"]))["shape"][0]
                break
            except Exception:
                pass

    primary = None
    for rec in ys:
        if rec.get("is_npy") and "target" in rec["key"]:
            primary = rec
            break
    if primary is None:
        for rec in ys:
            if rec.get("is_npy"):
                primary = rec
                break
    if primary is None:
        rep.add(g, "标签主列", C.UNKNOWN,
                "Y 里没有任何 .npy 数组（只有 %s）—— 无法做标签退化检查"
                % "、".join(r["key"] for r in ys))
        return

    y_rows = None
    if primary["path"] and Path(primary["path"]).is_file():
        try:
            a = np.load(str(primary["path"]), mmap_mode="r")
            y_rows = int(a.shape[0])
            if x_rows is not None and y_rows != x_rows:
                rep.add(g, "X / Y 行数对齐", C.FAIL,
                        "X 有 %d 行，%s 有 %d 行 —— 样本与标签错位"
                        % (x_rows, Path(primary["path"]).name, y_rows))
            else:
                rep.add(g, "X / Y 行数对齐", C.PASS if x_rows is not None else C.UNKNOWN,
                        "X %s 行 / %s %s 行" % (x_rows, Path(primary["path"]).name, y_rows))
        except Exception as exc:
            rep.add(g, "X / Y 行数对齐", C.UNKNOWN, "读取 Y 失败：%s" % exc)

    for rec in ys:
        tag = rec["key"]
        p = rec["path"]
        if p is None or not Path(p).is_file():
            if str(p).endswith(".parquet") or (rec["entry"] or {}).get("dst", "").endswith(".parquet"):
                rep.add(g, "%s 存在" % tag, C.PASS, "非 npy 账本文件，本组不做数值检查")
            else:
                rep.add(g, "%s 存在" % tag, C.FAIL, "找不到文件 %s" % p)
            continue
        p = Path(p)
        if p.suffix != ".npy":
            rep.add(g, "%s 类型" % tag, C.PASS, "非 npy（%s），跳过数值检查" % p.suffix)
            continue
        overall, per_col, note = array_stats(p, want_cols=False)
        if overall is None:
            rep.add(g, "%s 数值统计" % tag, C.UNKNOWN, note)
            continue
        fr = overall["finite_ratio"]
        rep.add(g, "%s 有限值比例" % tag,
                C.PASS if fr >= 0.80 else (C.WARN if fr >= 0.50 else C.FAIL),
                "%.6f   NaN %d" % (fr, overall["nan"]))
        rep.add(g, "%s 分布" % tag, C.PASS,
                "mean %.6g  std %.6g  min %.6g  max %.6g  absmax %.6g"
                % (overall["mean"], overall["std"], overall["min"],
                   overall["max"], overall["absmax"]))
        if rec is primary:
            _degeneracy(rep, p, overall, xs, y_rows)


def _degeneracy(rep: C.Report, ypath: Path, overall: dict, xs: list, y_rows=None):
    """标签退化检测 —— 对应 S11_mom 那次事故。"""
    g = G_LAB
    if np is None:
        rep.add(g, "标签退化检查", C.UNKNOWN, "numpy 未安装")
        return
    v = _finite_values(ypath)
    if v is None or v.size < 100:
        rep.add(g, "标签退化检查", C.UNKNOWN, "样本太少或非数值，无法判定")
        return

    # 1) 取值种类
    u = np.unique(v)
    if u.size <= 5:
        rep.add(g, "标签退化：取值种类", C.FAIL,
                "有限值里只有 %d 种取值（%s）—— 回归目标退化成离散/符号标签"
                % (u.size, np.array2string(u[:5], precision=6)),
                {"unique": [float(x) for x in u[:20]]})
    elif u.size <= 64:
        rep.add(g, "标签退化：取值种类", C.WARN,
                "有限值只有 %d 种取值，疑似离散标签（若本策略确实是分类目标可忽略）" % u.size)
    else:
        rep.add(g, "标签退化：取值种类", C.PASS, "%d 种" % u.size)

    # 2) 方差
    if overall["std"] is None or overall["std"] <= 1e-12:
        rep.add(g, "标签退化：方差", C.FAIL, "标准差 ≈ 0，标签无变化，模型学不到东西")
    else:
        skew = abs(overall["mean"]) / overall["std"]
        rep.add(g, "标签退化：均值/标准差", C.PASS if skew <= 3 else C.WARN,
                "|mean|/std = %.4f %s" % (skew, "" if skew <= 3 else "（分布极度偏斜）"))

    # 3) 非零率
    nz = 1.0 - overall["zero_ratio"]
    rep.add(g, "标签退化：非零率", C.PASS if nz >= 0.10 else C.FAIL,
            "%.4f（零值占 %.4f）%s" % (nz, overall["zero_ratio"],
                                      "" if nz >= 0.10 else " —— 标签几乎恒零，不可训练"))

    # 4) 与 X 各列的秩相关（抓「标签就是抄了某一列」）
    xpath = None
    for rec in xs:
        if rec.get("is_npy") and rec["path"] and Path(rec["path"]).is_file():
            xpath = Path(rec["path"])
            break
    if xpath is None:
        rep.add(g, "标签与特征重叠", C.UNKNOWN, "找不到可用的 X 数组，无法做重叠检查")
        return
    try:
        X = np.load(str(xpath), mmap_mode="r")
        yfull = np.asarray(np.load(str(ypath), mmap_mode="r"), dtype=np.float64).reshape(-1)
        n_rows = int(y_rows if y_rows is not None else yfull.size)
        if X.ndim != 2:
            rep.add(g, "标签与特征重叠", C.UNKNOWN,
                    "X 是 %d 维数组，不做逐列重叠检查（记「判不了」，不记通过）" % X.ndim)
            return
        if X.shape[0] != yfull.size:
            rep.add(g, "标签与特征重叠", C.UNKNOWN,
                    "X %d 行与 Y %d 行不一致，无法做逐列重叠检查（记「判不了」，不记通过）"
                    % (X.shape[0], yfull.size))
            return
        yfin = np.isfinite(yfull)
        best, best_col, skipped = 0.0, -1, 0
        ncol = min(X.shape[1], 400)
        for j in range(ncol):
            col = np.asarray(X[:, j], dtype=np.float64)
            m = np.isfinite(col) & yfin
            if int(m.sum()) < 100:
                skipped += 1
                continue
            rho = spearman(col[m], yfull[m])
            if rho == rho and abs(rho) > abs(best):
                best, best_col = rho, j
        note = "；%d 列有效值不足已跳过" % skipped if skipped else ""
        if best_col < 0:
            rep.add(g, "标签与特征重叠", C.UNKNOWN, "没有可用列，未做检查")
        elif abs(best) >= 0.98:
            rep.add(g, "标签与特征重叠", C.FAIL,
                    "标签与 X 第 %d 列（dim_index=%d）的秩相关 = %.6f，几乎完全重合 —— "
                    "模型只需要复制这一列。这通常是标签构造的 join 键写错了"
                    "（S11_mom 事故：标签退化成「今天收阴」）。" % (best_col, best_col, best)
                    + note,
                    {"col": best_col, "rho": best, "n_rows": n_rows})
        elif abs(best) >= 0.90:
            rep.add(g, "标签与特征重叠", C.WARN,
                    "标签与 X 第 %d 列的秩相关 = %.6f（阈值 0.98），偏高，请人工确认不是泄漏"
                    % (best_col, best) + note)
        else:
            rep.add(g, "标签与特征重叠", C.PASS,
                    "全部 %d 列中最高秩相关 %.6f 出现在第 %d 列（阈值 0.98）%s"
                    % (ncol, best, best_col, note))
    except Exception as exc:
        rep.add(g, "标签与特征重叠", C.UNKNOWN, "重叠检查执行失败：%s" % exc)


def check_contract(rep: C.Report, meta: dict, xs: list, ys: list,
                   dispatch: dict | None):
    g = G_CON
    bundle = Path(meta["bundle"])
    all_recs = xs + ys
    nohash, matched, mismatched, missing = [], [], [], []

    for rec in all_recs:
        p = rec["path"]
        if p is None or not Path(p).is_file():
            missing.append(rec["key"])
            continue
        p = Path(p)
        decl = rec["declared_sha1"]
        if not decl:
            nohash.append(rec["key"])
            continue
        try:
            actual = C.sha1_file(p)
        except Exception as exc:
            rep.add(g, "%s 哈希" % rec["key"], C.UNKNOWN, "读取失败：%s" % exc)
            continue
        if C.hash_matches(actual, decl):
            matched.append(rec["key"])
        else:
            mismatched.append((rec["key"], decl, actual[:len(decl)]))

    # ---- 与打包记录逐位比：DISPATCH.files 覆盖**全部**随包文件，是本机打包时实算的。
    # 这一条才是「这份货有没有被动过」的权威（MANIFEST 只声明了部分条目，
    # 没声明的那些永远比不出结论，见下面「无哈希可比的条目」）。
    if dispatch:
        dm = dispatch.get("files") or {}
        legacy = False
        if not dm:
            # 旧版 DISPATCH 只写了 xy_sha1，退化成按文件名比 X/Y 两件
            legacy = True
            dm = {}
            for k, v in (dispatch.get("xy_sha1") or {}).items():
                nm = k.split(".")[-1]
                dm[nm] = v
        ok_n, bad_n, gone_n = 0, [], []
        for rel, rec_d in sorted(dm.items()):
            if legacy:
                hits = [r for r in all_recs if r["path"] and Path(r["path"]).is_file()
                        and (r["key"] == rel or Path(r["path"]).name == Path(rel))]
                p = Path(hits[0]["path"]) if hits else None
            else:
                p = bundle / rel
            decl = rec_d.get("sha1") if isinstance(rec_d, dict) else rec_d
            if p is None or not Path(p).is_file():
                gone_n.append(rel)
                continue
            if not decl:
                continue
            actual = C.sha1_file(Path(p))
            if C.hash_matches(actual, decl):
                ok_n += 1
            else:
                bad_n.append("%s（记录 %s / 实算 %s）" % (rel, decl, actual[:len(str(decl))]))
        if bad_n:
            rep.add(g, "与打包记录不符（DISPATCH）", C.FAIL,
                    "；".join(bad_n[:6]) + "  —— 传输过程损坏，或磁盘上的不是本机导出的那一份")
        if gone_n:
            rep.add(g, "DISPATCH 登记但磁盘上没有", C.FAIL, "、".join(gone_n[:8]))
        if ok_n and not bad_n and not gone_n:
            rep.add(g, "与打包记录一致（DISPATCH）", C.PASS,
                    "%d 个文件逐位一致（本机打包时实算的 sha1）" % ok_n)
        elif not ok_n:
            rep.add(g, "与打包记录一致（DISPATCH）", C.UNKNOWN,
                    "DISPATCH 里没有可比的 sha1 记录，本次**没有比对**")
        src = dispatch.get("source") or {}
        packed = src.get("packed_at") or dispatch.get("packed_at")
        if packed:
            _host = (src.get("host") or {})
            # pack 写的键是 hostname（早期草稿里叫 node，两个都认）
            _hn = _host.get("hostname") or _host.get("node") or "未知主机"
            rep.add(g, "DISPATCH 打包时间", C.PASS,
                    "%s（打包机 %s，锚点 %s）" % (packed, _hn, dispatch.get("anchor_date")))
        # 数据锚：只有 input_anchor 才算数；数据覆盖日是弱凭据，单列，不顶替
        if dispatch.get("input_anchor"):
            rep.add(g, "DISPATCH 数据锚", C.PASS,
                    json.dumps(dispatch["input_anchor"], ensure_ascii=False)[:160])
        else:
            cov = ((dispatch.get("data_coverage") or {}).get("x")
                   or (dispatch.get("data_coverage") or {}).get("dates"))
            rep.add(g, "DISPATCH 数据锚", C.UNKNOWN,
                    "打包时包内就没有 input_anchor（只能在数据覆盖日上佐证%s）"
                    % ("：%s ~ %s" % (cov[0], cov[1]) if cov else "，而日期轴也读不到"))

    if missing:
        rep.add(g, "缺文件条目", C.FAIL, "、".join(missing))
    if mismatched:
        detail = "；".join("%s 声明 %s / 实算 %s" % (k, d, a) for k, d, a in mismatched)
        rep.add(g, "哈希不符（整文件 sha1）", C.FAIL,
                detail + "  —— 账物不符，数据与元数据不是同一份")
    elif matched:
        rep.add(g, "哈希相符（整文件 sha1）", C.PASS,
                "%d 个条目逐位一致" % len(matched))
    if nohash:
        rep.add(g, "无哈希可比的条目", C.UNKNOWN,
                "%d 个条目在 MANIFEST 里没有 sha1，本次**没有比对**：%s"
                % (len(nohash), "、".join(nohash)) +
                "  —— 记「判不了」，不得读作「一致」")

    # 元数据自洽：Y_meta 与 MANIFEST 的锚点/行数
    ym = meta.get("y_meta") or {}
    mf = meta.get("manifest") or {}
    if isinstance(ym, dict) and ym and isinstance(mf, dict) and mf:
        a_ym = C.parse_datetime_loose(ym.get("anchor_date"))
        a_mf = C.parse_datetime_loose(mf.get("anchor_date"))
        if a_ym and a_mf and a_ym != a_mf:
            rep.add(g, "元数据自洽：锚点日期", C.FAIL,
                    "Y/Y_meta.json 说 %s，MANIFEST.json 说 %s —— 打包时只更新了一部分文件"
                    % (a_ym, a_mf))
        elif a_ym and a_mf:
            rep.add(g, "元数据自洽：锚点日期", C.PASS, "%s（两份一致）" % a_mf)
        else:
            rep.add(g, "元数据自洽：锚点日期", C.UNKNOWN,
                    "Y_meta (%s) 或 MANIFEST (%s) 缺 anchor_date"
                    % (ym.get("anchor_date"), mf.get("anchor_date")))
        # 行数：Y_meta 里声明 X 形状的第一维 vs 实际
        decl_rows = _ym_declared_rows(ym)
        if decl_rows and xs and xs[0]["path"] and Path(xs[0]["path"]).is_file():
            actual = C.npy_header(Path(xs[0]["path"]))["shape"][0]
            rep.add(g, "元数据自洽：X 行数",
                    C.PASS if int(decl_rows) == int(actual) else C.FAIL,
                    "Y_meta 声明 %s / 磁盘实际 %s" % (decl_rows, actual))
        elif decl_rows:
            rep.add(g, "元数据自洽：X 行数", C.UNKNOWN, "声明 %s，但找不到 X 数组比对" % decl_rows)

    # 交付卫生
    pyc = meta.get("pycache") or []
    if pyc:
        rep.add(g, "交付卫生：__pycache__", C.WARN,
                "包内有 %d 处 __pycache__（%s）—— 违反 Z 侧「禁落 pycache」铁律，"
                "通常说明导出机器上没设 PYTHONDONTWRITEBYTECODE=1" % (len(pyc), "、".join(pyc[:3])))
    else:
        rep.add(g, "交付卫生：__pycache__", C.PASS, "无")
    gh = meta.get("git_head") or ((dispatch or {}).get("git_head"))
    if gh:
        rep.add(g, "包内 git HEAD", C.PASS, "%s" % gh)
    else:
        rep.add(g, "包内 git HEAD", C.UNKNOWN,
                "包内没有 .git，DISPATCH 也没记到版本号 —— 无法判断这是哪一版")


def _ym_declared_rows(ym: dict):
    for key in ("X", "X_rows", "x"):
        v = ym.get(key)
        if isinstance(v, str):
            import re as _re
            m = _re.search(r"\[\s*(\d+)\s*,", v)
            if m:
                return int(m.group(1))
        if isinstance(v, (int, float)):
            return int(v)
    return None


# ==================================================================== 入口

def run_doctor(bundle, cfg: dict | None = None, dispatch_path=None,
               out_path=None, quiet=False) -> C.Report:
    cfg = cfg or C.load_config()
    bundle = C.normalize_path(bundle)
    if not bundle.is_dir():
        rep = C.Report("异地训练体检", str(bundle))
        rep.add(G_CON, "包目录", C.FAIL, "目录不存在：%s" % bundle)
        return rep

    # 显式 --dispatch 优先；没给就自动在包根找（漏传不许变成安静的"什么都没比"）
    dispatch = None
    if dispatch_path:
        dp = Path(dispatch_path)
        if dp.is_file():
            dispatch = C.read_json_any(dp)
    else:
        _dp = C.find_dispatch(bundle)
        if _dp is not None:
            dispatch = C.read_json_any(_dp)

    meta = C.load_bundle_meta(bundle)
    rep = C.Report("异地训练体检", str(bundle),
                   {"bundle": str(bundle), "config": cfg.get("_config_path"),
                    "host": C.host_info()})

    if meta.get("manifest") is None:
        rep.add(G_CON, "MANIFEST.json", C.FAIL,
                "包根缺少 MANIFEST.json —— 这不是一份可移植训练包，无法核对任何东西")
    else:
        rep.add(G_CON, "MANIFEST.json", C.PASS,
                "包名 %s / 锚点 %s / 导出 %s"
                % (meta["manifest"].get("pkg"), meta["manifest"].get("anchor_date"),
                   meta["manifest"].get("exported_at")))

    xs, ys = pick_entries(meta)
    check_env(rep, cfg, meta, dispatch)
    check_inputs(rep, meta, xs)
    check_labels(rep, meta, xs, ys)
    check_contract(rep, meta, xs, ys, dispatch)

    if not quiet:
        print(rep.render())
        if rep.fails():
            print(_failure_block(rep, cfg))
    if out_path:
        # --out 与 train 的 --out 目录语义对齐：给目录就落到里面的 doctor.json。
        # 只有显式写成 *.json 才当文件，避免「以为给的是目录、结果生成一个同名文件」。
        p = C.normalize_path(out_path)
        if p.suffix.lower() != ".json":
            p = p / "doctor.json"
        rep.write(p)
        if not quiet:
            print("[pt] 报告已写入 %s" % p)
    return rep


def _failure_block(rep: C.Report, cfg: dict) -> str:
    lines = []
    for rec in rep.fails():
        why, nxt = hint_for(rec["name"])
        lines.append(C.human_block(
            "体检未通过，请先不要训练：%s" % rec["name"],
            rec["detail"] or rec["name"], why, nxt,
            cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
    if rep.unknowns():
        lines.append("\n以下 %d 项**判不了**（不是通过，请人工确认）：" % len(rep.unknowns()))
        for rec in rep.unknowns():
            lines.append("   - %s: %s" % (rec["name"], (rec["detail"] or "")[:160]))
    return "\n".join(lines)


def main(argv=None) -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description="attest 体检（异地落地后第一步）")
    ap.add_argument("--bundle", required=True, help="解压后的训练包目录")
    ap.add_argument("--dispatch", default=None, help="DISPATCH.json（本机打包时生成）")
    ap.add_argument("--config", default=None, help="pt_config.json 路径")
    ap.add_argument("--out", default=None,
                    help="体检报告落盘位置：给目录则写入 <目录>/doctor.json；"
                         "以 .json 结尾才当文件（与 train --out 同为目录语义）")
    ap.add_argument("--workdir", dest="workdir_alias", default=None,
                    help="--out 的同义写法（与 eval/collect 一致）。两个都给了又不一致时会拒绝。")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    a.out, _alias_err = C.resolve_out_alias(a.out, a.workdir_alias)
    if _alias_err:
        print(C.human_block(
            "两个开关指的不是同一个位置",
            _alias_err,
            "本工具不替你挑一个用 —— 挑错了报告会落到你以为不在的地方。",
            "把其中一个删掉再跑一次。",
            cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
        return C.RC_USAGE
    rep = run_doctor(a.bundle, cfg, a.dispatch, a.out)
    return rep.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
