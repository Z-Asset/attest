"""pt_train —— 标准化训练执行器。

用途：把「可移动三件套」在一个**只读母本**之上跑训练，所有细节由脚本钉死，
操作者不需要知道任何环境变量、种子、线程数或包内部结构。

已核实的包侧事实（Z:\\Ongoing\\S03_alphalgbm_csi800\\train_from_xy.py，逐行读过）：
- `--pkg <dir>` 设定包根（:475 `_pkg_root`），X/Y 从 `<dir>/X`、`<dir>/Y` 读（:511 `_load_arrays`），
  模型写 `<dir>/models`（:500 `seg_model_dir`）。故**零改动包**即可构造试次目录。
- `_load_arrays` 硬要 `X/{X.npy,factors.txt,row_date.txt,row_code.txt}` 与 `Y/Y_target.npy`，
  并把 `X/factors.txt` 与入口内嵌 FACTORS 逐字比对（"包被改过"）。故试次目录必须整份 X/、Y/。
- `--save` 需要 GPU（:634 `_train` 里 `torch.cuda.is_available()` 为假则直接抛，不产 CPU 模型）。
- `--smoke` 是唯一的 CPU 路径，且把活目录整份搬到系统临时目录再跑（:606），包内零写盘。
- 门禁在 :339：`ct >= lt and ct > 0.0 and cv >= lv and cv > 0.0`，与生产
  `D:\\Ongoing\\S03_alphalgbm\\train.py:588` 逐字一致。
- **关键**：:331 无活工件时 `decision="replaced", reason="first_install"` —— 门禁被跳过。
  所以试次目录**必须**带上母本的 models/，否则「周训复现」是假的（必然 replaced）。
"""

from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pt_common as C  # noqa: E402

# 自设环境变量：操作者不需要知道这些，也不允许他忘
BASE_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",      # Z 侧铁律：禁落 __pycache__
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONIOENCODING": "utf-8",         # 已实测 GBK 控制台会 UnicodeEncodeError
    "PYTHONHASHSEED": "0",
}

# 模型工件：门禁要求活工件在位，故试次必须继承母本的这几个文件
MODEL_ARTIFACTS = ("lgb.txt", "mlp.pt", "xgb.json", "factor_meta.json", "config.json",
                   "model_versions.csv")


# ------------------------------------------------------------------ 目录联接

DRIVE_REMOTE = 4
DRIVE_FIXED = 3


def _drive_is_local(p) -> bool:
    """判断路径是否在本地固定盘上 —— mklink /J 只能联到本地 NTFS 卷。

    实测：GetDriveTypeW("C:")=3(固定) / ("Z:")=4(远程映射)。
    注意**不要**给盘符加 `\\\\?\\` 前缀，那样返回 1(未知)，本地盘会被误判成远程
    （已实测踩到：加了前缀后 trial 全走 copytree，白拷几十 MB）。
    """
    if os.name != "nt":
        return True
    s = str(p)
    if s.startswith("\\\\") or s.startswith("//"):
        return False                      # UNC 一律不算本地
    if len(s) > 1 and s[1] == ":":
        try:
            return ctypes.windll.kernel32.GetDriveTypeW(s[:2]) == DRIVE_FIXED
        except Exception:
            return True
    return True


def link_or_copy_dir(src: Path, dst: Path) -> str:
    """把整个目录挂到目标位置，优先 junction（免管理员、免复制），失败退化为复制。

    返回实际采用的方式：junction / symlink / copy。
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _drive_is_local(src) and _drive_is_local(dst):
        try:
            os.symlink(str(src), str(dst), target_is_directory=True)
            return "symlink"
        except Exception:
            pass
        cmdline = 'cmd /c mklink /J "%s" "%s"' % (dst, src)
        try:
            subprocess.run(cmdline, shell=False, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60)
        except Exception:
            pass
        if dst.exists():
            return "junction"
    shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    return "copy"


def ensure_local_inputs(bundle: Path, workdir: Path, manifest: dict) -> dict:
    """保证 X/ Y/ 有一份**本地**来源，多个试次共用，避免每次重复复制大数组。

    母本在本地固定盘 -> 直接用母本（试次里挂 junction，零拷贝）。
    母本在映射盘/UNC -> 先在 workdir/_cache 下物化一份，之后所有试次挂它。
    """
    bundle = Path(bundle)
    if _drive_is_local(bundle):
        return {"X": bundle / "X", "Y": bundle / "Y", "X_mode": "direct",
                "Y_mode": "direct", "cache": None}
    # 缓存键必须是**内容**的，不能是字节数：同锚点、同字节数的修正包会被
    # 静默复用旧缓存，训练出来的是上一版数据 —— 正是本工具要防的那类静默错误。
    # 实算 sha1 而不是抄 MANIFEST（账物相符）；37MB 哈希约 0.1s，可忽略。
    def _sig1(p: Path) -> str:
        try:
            return C.sha1_file(p)[:16] if p.is_file() else "missing"
        except OSError:
            return "unreadable"

    sig = "%s_%s_%s_%s" % (bundle.name, manifest.get("anchor_date", "na"),
                           _sig1(bundle / "X" / "X.npy"), _sig1(bundle / "Y" / "Y_target.npy"))
    cache = Path(workdir) / "_cache" / sig
    out = {"X": cache / "X", "Y": cache / "Y", "X_mode": None, "Y_mode": None, "cache": cache}
    if not (cache / "_READY").exists():
        cache.mkdir(parents=True, exist_ok=True)
        for sub in ("X", "Y"):
            src = bundle / sub
            if not src.is_dir():
                continue
            shutil.copytree(str(src), str(cache / sub), dirs_exist_ok=True)
        (cache / "_READY").write_text(C.now_iso(), encoding="utf-8")
    out["X_mode"] = out["Y_mode"] = "cache_copy"
    return out


# ------------------------------------------------------------------ 试次构造

def build_trial(bundle: Path, workdir: Path, tag: str, inputs: dict,
                models_src: Path | None, cfg: dict, allow_master_write: bool) -> dict:
    """构造试次目录。母本全程只读。"""
    workdir = Path(workdir)
    trial = workdir / "trials" / tag
    C.guard_master_write(trial, cfg.get("master_roots"), allow_master_write, "创建试次目录")
    if trial.exists():
        shutil.rmtree(trial, ignore_errors=True)
    trial.mkdir(parents=True, exist_ok=True)

    info = {"trial": str(trial), "links": {}}

    entry = bundle / "train_from_xy.py"
    if not entry.is_file():
        raise FileNotFoundError("包内没有 train_from_xy.py：%s" % entry)
    shutil.copy2(str(entry), str(trial / "train_from_xy.py"))
    info["entry"] = "train_from_xy.py"
    info["entry_bytes"] = (trial / "train_from_xy.py").stat().st_size

    for sub in ("X", "Y"):
        src = inputs.get(sub)
        if src is None or not Path(src).is_dir():
            raise FileNotFoundError("本地输入缺少 %s/（来源 %s）" % (sub, src))
        info["links"][sub] = link_or_copy_dir(Path(src), trial / sub)

    # 活工件：门禁要拿候选与活比，缺了就直接 first_install，等于门禁没执行
    models = trial / "models"
    models.mkdir(parents=True, exist_ok=True)
    copied = []
    if models_src and Path(models_src).is_dir():
        for name in MODEL_ARTIFACTS:
            s = Path(models_src) / name
            if s.is_file():
                shutil.copy2(str(s), str(models / name))
                copied.append(name)
    info["models_src"] = str(models_src) if models_src else None
    info["models_copied"] = copied
    return info


# ------------------------------------------------------------------ 入口能力探测

# --help 的探测是**跑一次子进程**，不是读源码正则：很多入口的 usage 是 argparse 生成的，
# 正则抠 add_argument 会漏掉只在交互相里出现的写法。以它自己打印的 usage 为准，
# 才是"它真的认"。
FLAG_PROBE_TIMEOUT = 90
_FLAG_RE = re.compile(r"--[A-Za-z][A-Za-z0-9-]*")


def probe_entry_flags(entry, timeout: int = FLAG_PROBE_TIMEOUT) -> dict:
    """问入口自己认哪些开关。返回 {"_ok", "flags", "note"}。

    _ok=False 表示**判不了**（入口不存在 / --help 跑不起来 / 输出里一个开关都没有）。
    判不了时调用方必须停，不许按"默认接受全部"或"默认一个都不接受"往下走。
    """
    entry = Path(entry)
    if not entry.is_file():
        return {"_ok": False, "flags": {}, "note": "入口不存在：%s" % entry}
    # 起进程**必须**走 C.run_subprocess：它在 os.environ 之上再叠加给定的那几项。
    # 手写 subprocess.run(env=BASE_ENV) 会**整个替换**掉环境，入口连 import pandas 都过不去
    # （已实测：rc=1 死在 pandas/_config），于是好包被判成"驱动不了"。
    # 这条纪律代码里本来就有正确实现，这里不再手写第二遍。
    rc, _out, _err = C.run_subprocess(
        [sys.executable, str(entry), "--help"],
        cwd=entry.parent, env=BASE_ENV, timeout=timeout)
    text = (_out or "") + "\n" + (_err or "")
    flags = {}
    for m in _FLAG_RE.finditer(text):
        flags[m.group(0)] = True
    if not flags:
        # "跑不起来"和"跑起来了但没有开关"是两种完全不同的故障，不能印成同一句话。
        tail = " / ".join(x.strip() for x in text.splitlines() if x.strip())[-300:]
        if rc != 0:
            return {"_ok": False, "flags": {}, "_rc": rc,
                    "note": "这个入口跑 --help 就退出了（返回码 %s）：%s" % (rc, tail)}
        return {"_ok": False, "flags": {}, "_rc": rc,
                "note": "--help 输出里一个 --开关都没有（这个入口可能不是 argparse 形态）：%s" % tail}
    return {"_ok": True, "flags": flags, "_rc": rc,
            "note": "探明 %d 个开关" % len(flags)}


# 每个模式跑起来**必须**具备的开关。
MODE_REQUIRED_FLAG = {"dry": "--pkg", "smoke": "--smoke", "week": "--save"}


def check_entry_capability(cap: dict, mode: str, seeds: list):
    """能力不满足时返回一句人话（缺哪条 + 为什么缺它不行），满足返回 None。"""
    if not cap.get("_ok"):
        return "没能问出这个入口认哪些开关（%s）。本工具不猜它认什么，宁可停在这里。" % cap.get("note")
    fl = cap.get("flags") or {}
    need = MODE_REQUIRED_FLAG.get(mode)
    if need and need not in fl:
        return "这个包的入口没有 %s 开关。它认的是：%s" % (need, " ".join(sorted(fl)))
    if len([s for s in seeds if s is not None]) > 1 and "--seed" not in fl:
        return ("要跑多个种子，但这个包的入口没有 --seed 开关（它认的是：%s）。"
                "没有它，几次跑的是同一个随机流，算出来的离散度是假的。"
                % " ".join(sorted(fl)))
    return None


def entry_seed_note(cap: dict) -> dict:
    """入口不认 --seed 时把这件事写进指纹，让"这次随机性钉不住"自己说话。"""
    fl = (cap or {}).get("flags") or {}
    return {"seed_controllable": bool("--seed" in fl)}


# ------------------------------------------------------------------ 跑一次

def run_one(bundle, workdir, tag, mode, seed, cfg, dispatch=None,
            threads=None, models_src=None, allow_master_write=False,
            timeout=None, entry_patch=None, capture_tail=4000,
            entry_flags=None) -> dict:
    """跑一次训练。mode: dry / smoke / week。

    dry  = `--pkg`（不落盘，验证入口能读到 X/Y）
    smoke= `--smoke`（CPU 小窗口，只写临时目录，验证整条代码路径可跑）
    week = `--save`（真训练，需 GPU，写试次目录的 models/）
    """
    bundle = C.normalize_path(bundle)
    workdir = C.normalize_path(cfg.get("workdir") or workdir)
    manifest = (C.load_bundle_meta(bundle).get("manifest") or {})
    n_threads = int(threads if threads is not None else cfg.get("threads", 1))
    threads_overridden = threads is not None and int(threads) != int(cfg.get("threads", 1))

    # 守卫必须**在第一次写盘之前**：ensure_local_inputs 会在 workdir 下落缓存，
    # 若先物化再判母本，就已经污染了（试出来的窟窿，别挪到后面）。
    try:
        C.guard_master_write(workdir, cfg.get("master_roots"), allow_master_write, "创建工作目录")
    except C.MasterGuardError as exc:
        return {"tag": tag, "mode": mode, "seed": seed, "started": C.now_iso(),
                "bundle": str(bundle), "rc": C.RC_BLOCKED, "blocked": str(exc),
                "finished": C.now_iso()}

    # 入口能力：**判在这一层**。调用方（笔记本第 5/6 节）是直接调 run_one 的，
    # 闸放在 cmd_train 里保护不了那条路 —— 而那条路正是异地操作者走的路。
    if entry_flags is None:
        entry_flags = probe_entry_flags(bundle / "train_from_xy.py")
    _cap_err = check_entry_capability(entry_flags, mode, [seed])
    if _cap_err:
        return {"tag": tag, "mode": mode, "seed": seed, "started": C.now_iso(),
                "bundle": str(bundle), "rc": C.RC_USAGE, "capability_fail": True,
                "error": _cap_err,
                "entry_flags": sorted((entry_flags or {}).get("flags") or {}),
                "finished": C.now_iso()}

    inputs = ensure_local_inputs(bundle, workdir, manifest)
    if models_src is None:
        cand = bundle / "models"
        models_src = cand if cand.is_dir() else None

    started = C.now_iso()
    t0 = time.time()
    rec = {"tag": tag, "mode": mode, "seed": seed, "started": started,
           "bundle": str(bundle), "pkg": manifest.get("pkg"), "seg": manifest.get("seg"),
           "anchor_date": manifest.get("anchor_date"),
           "workdir": str(workdir), "config": cfg.get("_config_path")}

    try:
        info = build_trial(bundle, workdir, tag, inputs, models_src, cfg, allow_master_write)
    except C.MasterGuardError as exc:
        rec.update({"rc": C.RC_BLOCKED, "blocked": str(exc), "finished": C.now_iso()})
        return rec
    except Exception as exc:
        rec.update({"rc": C.RC_FAIL, "error": "%s: %s" % (type(exc).__name__, exc),
                    "finished": C.now_iso()})
        return rec
    trial = Path(info["trial"])
    rec.update(info)

    # 试次目录里的入口**副本**可按需改写（搜索要扫超参，超参就在入口的 SEG_CFG 里）。
    # 只改副本：母本、Z 包、_sync 生成器一律不动。改写前后都留 sha1，
    # 让「这次跑的到底是哪一份入口」可追。
    # 回调由调用方给：它负责「只动该动的那一段」，并把 before/after 回报出来。
    # 必须在起子进程**之前** —— 迟一步，跑的还是原参数。
    if entry_patch is not None:
        try:
            rec["entry_patch"] = entry_patch(trial)
        except Exception as exc:
            rec.update({"rc": C.RC_FAIL,
                        "error": "改写试次入口失败：%s: %s" % (type(exc).__name__, exc),
                        "finished": C.now_iso()})
            return rec

    # 输入指纹：整文件 sha1（与 MANIFEST 同一口径，已实测确认）
    xf = trial / "X" / "X.npy"
    yf = None
    for name in ("Y_target.npy", "Y_target_ret.npy", "Y_target_up.npy"):
        if (trial / "Y" / name).is_file():
            yf = trial / "Y" / name
            break
    fp_files = {}
    for label, f in (("X.npy", xf), ("Y_target", yf)):
        if f and Path(f).is_file():
            st = Path(f).stat()
            fp_files[label] = {"name": Path(f).name, "bytes": st.st_size,
                               "sha1": C.sha1_file(Path(f))}
    if (trial / "X" / "factors.txt").is_file():
        fp_files["factors.txt"] = {
            "sha1": C.sha1_file(trial / "X" / "factors.txt"),
            "n": len([l for l in C.read_text_any(trial / "X" / "factors.txt").splitlines() if l.strip()])}

    env = dict(BASE_ENV)
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)
    env["OPENBLAS_NUM_THREADS"] = str(n_threads)
    env["NUMEXPR_NUM_THREADS"] = str(n_threads)

    # 只追加它**认**的开关。盲追加的后果不是报错，是**假归因**：argparse 以 2 退出，
    # 被读成"这一轮训练失败"，操作者被支去发日志、不敢重跑，而重跑一百次都一样。
    # 调用方没给就自己探一次：宁可多花一点时间，也不让任何一个调用点忘了探。
    if entry_flags is None:
        entry_flags = probe_entry_flags(trial / "train_from_xy.py")
    if not entry_flags.get("_ok"):
        rec.update({"rc": C.RC_USAGE, "finished": C.now_iso(),
                    "error": "问不出试次入口认哪些开关：%s" % entry_flags.get("note")})
        return rec
    _fl = entry_flags["flags"]
    argv = [sys.executable, str(trial / "train_from_xy.py")]
    if "--pkg" in _fl:
        argv += ["--pkg", str(trial)]
    if mode == "smoke" and "--smoke" in _fl:
        argv.append("--smoke")
    elif mode == "week" and "--save" in _fl:
        argv.append("--save")
    if seed is not None and mode in ("smoke", "week") and "--seed" in _fl:
        argv += ["--seed", str(seed)]
    # 要了种子、却没能传给它 —— 这件事必须能被下游看见，否则"多 seed 的离散度"
    # 是拿同一条随机流算出来的假数。
    _seed_applied = seed if ("--seed" in _fl and seed is not None
                             and mode in ("smoke", "week")) else None

    log = workdir / "logs" / ("%s.log" % tag)
    rc, out, err = C.run_subprocess(argv, cwd=trial, env=env, log_path=log, timeout=timeout)

    rec.update({
        "rc": rc, "elapsed_sec": round(time.time() - t0, 1), "argv": argv,
        "log": str(log), "threads": n_threads, "threads_overridden": threads_overridden,
        "env": env,
        "fingerprint_files": fp_files,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "deps": {m: C.import_version(m) for m in ("numpy", "pandas", "lightgbm", "torch")},
        "gpu": C.gpu_info(),
        "host": C.host_info(),
        "bundle_git_head": C.load_bundle_meta(bundle).get("git_head"),
        "entry_flags": sorted(_fl),
        "entry_seed": entry_seed_note(entry_flags),
        "entry_pkg_flag_omitted": "--pkg" not in _fl,
        "seed_requested": seed,
        "seed_applied": _seed_applied,
        "finished": C.now_iso(),
        "stdout_tail": out[-capture_tail:],
        "stderr_tail": err[-capture_tail:],
    })

    # 摘出这次的关键结果
    m = re.search(r"metrics:\s*(\{.*\})", out)
    if m:
        blob = m.group(1)
        try:
            rec["metrics"] = ast.literal_eval(blob)
        except Exception:
            try:
                rec["metrics"] = json.loads(blob.replace("'", '"'))
            except Exception:
                rec["metrics_raw"] = blob
    for key in ("decision", "reason", "ensemble_weight"):
        mm = re.search(r"%s=([^\s]+)" % key, out)
        if mm:
            rec[key] = mm.group(1)
    rec["reason"] = rec.get("reason") or _reason_from_out(out)
    # 门禁的三个数不在 metrics 里，只在 model_versions.csv 上
    mv = trial / "models" / "model_versions.csv"
    if mv.is_file():
        txt = C.read_text_any(mv)
        lines = [l for l in txt.splitlines() if l.strip()]
        rec["model_versions_tail"] = lines[-3:]
        try:
            rows = list(csv.DictReader(io.StringIO(txt)))
            if rows:
                rec["version_row"] = rows[-1]
        except Exception:
            pass
    rec["models_out"] = sorted(p.name for p in (trial / "models").iterdir()) \
        if (trial / "models").is_dir() else []
    rec["rejected_dirs"] = sorted(p.name for p in trial.glob("models/rejected_*"))
    return rec


# ------------------------------------------------------------------ 入口

def cmd_train(argv=None) -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description="attest 标准化训练（默认不写盘）")
    ap.add_argument("--bundle", required=True, help="训练包目录（只读）")
    ap.add_argument("--out", default=None, help="工作目录（试次/日志/指纹落这里）")
    ap.add_argument("--workdir", dest="workdir_alias", default=None,
                    help="--out 的同义写法（与 eval/collect 一致）。两个都给了又不一致时会拒绝。")
    ap.add_argument("--config", default=None, help="pt_config.json 路径")
    ap.add_argument("--dispatch", default=None, help="DISPATCH.json 路径")
    ap.add_argument("--mode", default="dry", choices=["dry", "smoke", "week"],
                    help="dry=只验证入口 / smoke=CPU 冒烟（只写临时目录）/ week=真训练（需 GPU）")
    ap.add_argument("--commit", action="store_true",
                    help="确认真跑。week 模式必须显式给出；dry/smoke 不需要")
    ap.add_argument("--seed", type=int, default=None, help="固定 RNG 种子")
    ap.add_argument("--seeds", default=None, help="多 seed，逗号分隔，例如 1,2,3（量化噪声底）")
    ap.add_argument("--tag", default=None, help="试次名（默认自动：包名_模式_时间）")
    ap.add_argument("--allow-threads", type=int, default=None,
                    help="覆盖线程数。线程数是实验条件不是环境细节，改了会记进指纹")
    ap.add_argument("--models-from", default=None,
                    help="活工件来源目录（默认取包内 models/；门禁要跟它比）")
    ap.add_argument("--timeout", type=int, default=None, help="单次超时（秒）")
    ap.add_argument("--allow-master-write", action="store_true",
                    help="允许把试次写进母本根之下。默认禁止，防手滑污染 Z")
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
    manifest = (C.load_bundle_meta(bundle).get("manifest") or {})
    pkgname = manifest.get("pkg") or bundle.name

    if a.mode == "week" and not a.commit:
        print(C.human_block(
            "拒绝真训练：缺少 --commit",
            "你选了 --mode week（真训练），但没有加 --commit。",
            "真训练会占用 GPU、写入模型文件，必须由你显式确认一次，防止误触发。",
            "如果确实要训练，把命令改成：\n"
            "    python pt.py train --bundle %s --mode week --commit" % a.bundle,
            cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
        return C.RC_USAGE

    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    elif a.seed is not None:
        seeds = [a.seed]
    elif a.mode == "week":
        # 默认种子数只认 pt_config.json（兜底在 pt_common.DEFAULT_CONFIG）——
        # 这里不再另写一份字面量：写两份，将来改一处就会留下一个不报错的旧值。
        seeds = list(cfg.get("seeds") or C.DEFAULT_CONFIG["seeds"])
    else:
        seeds = [None]

    print("[pt] 包 %s   锚点 %s   模式 %s   种子 %s"
          % (pkgname, manifest.get("anchor_date"), a.mode, seeds))
    print("[pt] 工作目录 %s" % cfg["workdir"])
    if a.mode == "week":
        g = C.gpu_info()
        if not g.get("cuda_available"):
            print(C.human_block(
                "拒绝真训练：没有可用的 GPU",
                "本机没有检测到可用的 CUDA 设备（%s）。" % g.get("note", "torch 未装或 CUDA 不可用"),
                "模型训练必须在 GPU 上跑是硬规矩，CPU 上跑出来的模型不算数。",
                "改跑 --mode smoke（CPU 冒烟，只验证代码路径，不产模型）；"
                "要真训练请换一台有 GPU 的机器。",
                cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
            return C.RC_BLOCKED

    # 入口能力：**在拷 X/Y 之前**问。这个包的入口若不认本模式要用的开关，
    # 那是"驱动不了"，不是"这一轮跑坏了"——必须在花掉几百 MB 拷贝之前就说清楚。
    cap = probe_entry_flags(bundle / "train_from_xy.py")
    print("[pt] 入口能力：%s" % cap["note"])
    _cap_err = check_entry_capability(cap, a.mode, seeds)
    if _cap_err:
        print(C.human_block(
            "这个包驱动不了（不是这一轮跑坏了）",
            "%s\n包：%s   模式：%s" % (_cap_err, pkgname, a.mode),
            "本工具按入口自己声明的开关来跑，不替它猜。缺的开关补不上，这一轮就起不来；重跑多少次都是同一句话 —— 所以**不要重跑**。",
            "把上面这一整段发回本机（连同包名）。本机侧要么换一个能驱动的包，要么按这个入口的 CLI 面补齐 —— 这属于打包侧的事。",
            cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
        return C.RC_USAGE
    if "--seed" not in (cap.get("flags") or {}):
        print("(注记) 这个包的入口没有 --seed：本工具不会给它追加 --seed，本次的随机性钉不住，指纹里记 seed_controllable=false。")

    # DISPATCH：显式优先，否则自动在包根找 —— 它记着本机打包时实算的 sha1 与精确依赖版本，
    # 漏读等于两条最硬的核对都没发生。
    disp_obj, disp_path = C.load_dispatch(bundle, a.dispatch)
    if disp_path:
        print("[pt] DISPATCH: %s" % disp_path)
    results = []
    batch_ts = time.strftime("%Y%m%d_%H%M%S")
    for seed in seeds:
        tag = "%s_%s_s%s_%s" % (a.tag or pkgname, a.mode,
                                "x" if seed is None else seed, batch_ts)
        print("\n[pt] >>> 试次 %s" % tag)
        rec = run_one(bundle, cfg["workdir"], tag, a.mode, seed, cfg,
                      dispatch=disp_obj,
                      threads=a.allow_threads,
                      models_src=C.normalize_path(a.models_from) if a.models_from else None,
                      allow_master_write=a.allow_master_write, timeout=a.timeout,
                      entry_flags=cap)
        results.append(rec)
        if "blocked" in rec:
            print(C.human_block(
                "拒绝写入：目标在母本根之下",
                rec["blocked"], "母本是生产真身，误写会污染线上包。",
                "把 --out 换成一个**不在** %s 之下的空目录"
                "（例如你自己的用户目录下新建一个）。"
                % cfg.get("master_roots", C.DEFAULT_CONFIG["master_roots"]),
                cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
            return C.RC_BLOCKED
        if "error" in rec:
            # 兜底路径（run_one 自己探入口失败）也要说成"驱动不了"，不能说成"训练崩了"——
            # 归因错一次，操作者就多等一轮，而且会去改本来没坏的东西。
            _capfail = str(rec["error"]).startswith("问不出")
            print(C.human_block(
                "这个包驱动不了（不是这一轮跑坏了）" if _capfail else "训练未能启动",
                rec["error"],
                "本工具按入口自己声明的开关来跑，不替它猜；重跑多少次都是同一句话，不要重跑。" if _capfail else "试次目录没建起来，训练根本没开始。",
                "把这一整段发回本机（连同包名）。" if _capfail else "把上面的报错连同本报告发回本机。",
                cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
            return C.RC_USAGE if _capfail else C.RC_FAIL
        print("[pt] rc=%s  耗时 %ss  decision=%s"
              % (rec["rc"], rec.get("elapsed_sec"), rec.get("decision", "-")))
        if rec["rc"] != 0:
            print("[pt] 日志尾部：")
            print((rec.get("stderr_tail") or rec.get("stdout_tail") or "")[-1500:])
            print(C.human_block(
                "训练执行失败（返回码 %s）" % rec["rc"],
                "训练程序在跑的过程中非正常退出。完整输出在 %s" % rec.get("log"),
                "这一轮的结果不可用，不要拿它做判断。",
                "把 %s 发回本机，不要自己重跑或改包。" % rec.get("log"),
                cfg.get("contact", C.DEFAULT_CONFIG["contact"])))
            return C.RC_FAIL

    # 落指纹
    wd = C.normalize_path(cfg["workdir"])
    summary = {"bundle": str(bundle), "pkg": pkgname, "anchor_date": manifest.get("anchor_date"),
               "mode": a.mode, "seeds": seeds, "runs": results,
               "noise_floor": _noise_floor(results), "generated_at": C.now_iso()}
    fp = wd / "fingerprint.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("训练完成。指纹已写入 %s" % fp)
    for rec in results:
        d = (rec.get("fingerprint_files") or {}).get("X.npy", {}).get("sha1", "")[:16]
        print("  %s  seed=%s  decision=%s  X.npy sha1=%s" % (rec["tag"], rec["seed"],
                                                             rec.get("decision", "-"), d))
    nf = summary["noise_floor"]
    if nf:
        print("  噪声底（多 seed 的 val IC 离散度）：%s" % json.dumps(nf, ensure_ascii=False))
    print("  下一步：python pt.py eval --bundle %s --workdir %s" % (a.bundle, wd))
    print("=" * 74)
    return C.RC_OK


def _as_float(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def _reason_from_out(out: str):
    m = re.search(r"\[门禁\]\s*decision=\S+\s*\((.*?)\)", out)
    return m.group(1) if m else None


def _noise_floor(results: list) -> dict:
    """多 seed 的 val/test IC 离散度即噪声底。单 seed 时不报（判不了）。"""
    cols = ("cand_val_ic", "cand_test_ic")
    series = {c: [] for c in cols}
    for rec in results:
        row = rec.get("version_row") or {}
        for c in cols:
            v = _as_float(row.get(c))
            if v is not None and v == v:
                series[c].append(v)
    n = max((len(v) for v in series.values()), default=0)
    # 种子根本没传进进程时，跑几遍都是同一条随机流 —— 那样算出来的"离散度"不是小，
    # 是**假的**。宁可报判不了，也不能给一个会让人以为"噪声底很低、改进可信"的数。
    _unseeded = [r.get("tag") for r in results
                 if r.get("seed_requested") is not None and r.get("seed_applied") is None]
    if _unseeded:
        return {"n": 0, "seeds_not_applied": _unseeded,
                "note": "%d 次训练的种子**没有真正传进训练程序**（这个包的入口不认 --seed）。"
                        "这几次跑的是同一条随机流，算出来的离散度不是噪声底，是假的 ——"
                        "所以这里不出数。要噪声底请换一个入口认 --seed 的包。" % len(_unseeded)}
    if n < 2:
        return {"n": n, "note": "只有 %d 次有效结果，无法估计噪声底；"
                                "要量化噪声底请用 --seeds 1,2,3（至少 2 个不同种子）。" % n}
    out = {"n": n, "per_seed": {c: series[c] for c in cols}, "columns": {}}
    for c in cols:
        v = series[c]
        if len(v) < 2:
            continue
        mean = sum(v) / len(v)
        var = sum((x - mean) ** 2 for x in v) / (len(v) - 1)
        out["columns"][c] = {"mean": round(mean, 6), "std": round(var ** 0.5, 6),
                             "range": round(max(v) - min(v), 6),
                             "min": round(min(v), 6), "max": round(max(v), 6)}
    out["note"] = ("同包同数据、只换随机种子的离散度。候选相对基线的增量若不超出这个范围，"
                   "就不算改进 —— 判 UNKNOWN，不判 PASS。")
    return out


if __name__ == "__main__":
    raise SystemExit(cmd_train())
