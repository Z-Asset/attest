"""attest 共用层。

设计约束（哥 2026-09-17 定）：
- 本文件与整个 scripts/ 内**零绝对路径、零盘符假设**；所有根目录来自
  pt_config.json -> 环境变量 -> CLI 参数 三级覆盖。
- 操作者是「不懂细节的人」：失败必须说人话，且「判不了」绝不等于「通过」。
- 只依赖标准库 + numpy；真训练时才需要 lightgbm/torch。本模块在只有 numpy 的环境里也不能崩。

关键口径（已实测确认，勿凭记忆改）：
- MANIFEST.json 里每个 npy 条目的 `sha1` = **整文件 sha1**（前 16 位即为声明值）。
- 同一批条目的 `bytes` = **去掉 npy 头后的载荷字节数**（S03 实测 size-bytes == 128）。
  两者不是同一个东西：sha1 判等用整文件，bytes 只能当参考，不能当一致性判据。
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

# 铁律：禁止落 __pycache__（Z 侧既有规矩，本工具自己也要守）
sys.dont_write_bytecode = True

# ---------------------------------------------------------------- 三态与退出码

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

# 有些项**不是检查项**，是结论（例如"裁决姿态"）。它的状态恒为 PASS ——
# "姿态算出来了"这件事确实成立 —— 但印成「[通过] 裁决姿态 blocked」会让人把
# blocked 读成通过。kind="CONCLUSION" 的项改印 [结论]，且不计入合计的"通过"。
KIND_CONCLUSION = "CONCLUSION"

# 退出码：0 干净通过 / 2 有 WARN 可继续 / 1 有 FAIL 必须停 / 3 用法错 / 4 被守卫拦下
RC_OK = 0
RC_FAIL = 1
RC_WARN = 2
RC_USAGE = 3
RC_BLOCKED = 4


def setup_console() -> None:
    """把 stdout/stderr 切成 utf-8。

    Windows 控制台默认 GBK，打印非 ASCII 会 UnicodeEncodeError 直接崩
    （2026-09-17 本机实测撞到）。这是「操作者不懂细节」场景下最没必要的失败。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


# ---------------------------------------------------------------- 基础工具

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def host_info() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "cwd": str(Path.cwd()),
    }


def read_text_any(path: Path) -> str:
    """按 utf-8 -> utf-8-sig -> gbk 顺序试读，全失败则用 replace 兜底。"""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_json_any(path: Path):
    return json.loads(read_text_any(path))


def sha1_file(path: Path, chunk: int = 1 << 20) -> str:
    """整文件 sha1（流式，内存安全）。"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """整文件 sha256（流式）。zip 边车与回传清单用它，比 sha1 更适合跨机传输校验。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_matches(actual_full: str, declared: str) -> bool:
    """声明值可能是 16 位前缀也可能是 40 位全量；一律按前缀比对。"""
    d = (declared or "").strip().lower()
    if not d:
        return False
    return actual_full.lower().startswith(d)


def npy_header(path: Path) -> dict:
    """只读 npy 头，不加载数据。返回 shape/dtype/fortran_order/header_len。"""
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("not a .npy file: %s" % path)
        major = f.read(1)
        f.read(1)  # minor
        if major == b"\x01":
            hlen = struct.unpack("<H", f.read(2))[0]
        else:
            hlen = struct.unpack("<I", f.read(4))[0]
        header_len = 6 + 2 + (2 if major == b"\x01" else 4) + hlen
        raw = f.read(hlen)
    text = raw.decode("latin1").strip()
    info = ast.literal_eval(text)
    return {
        "shape": tuple(info.get("shape", ())),
        "dtype": str(info.get("descr")),
        "fortran_order": bool(info.get("fortran_order", False)),
        "header_len": header_len,
    }


_DTYPE_ALIAS = {
    "f2": "float16", "f4": "float32", "f8": "float64", "f16": "float128",
    "i1": "int8", "i2": "int16", "i4": "int32", "i8": "int64",
    "u1": "uint8", "u2": "uint16", "u4": "uint32", "u8": "uint64",
    "b1": "bool", "?": "bool",
}


def dtype_name(descr) -> str:
    """把 npy 头里的描述串（'<f4'）换成正规名（'float32'）。

    numpy 头里存的是 kind+size 的编码，不是 'float32' 这种名字；
    直接字符串比对会把 float32 判成不一致（已实测踩到）。
    """
    s = str(descr).strip()
    if s in _DTYPE_ALIAS:
        return _DTYPE_ALIAS[s]
    core = s.lstrip("<>=|")
    if core in _DTYPE_ALIAS:
        return _DTYPE_ALIAS[core]
    try:
        import numpy as _np
        return _np.dtype(s).name
    except Exception:
        return s


def dtype_matches(descr, declared) -> bool:
    d = str(declared or "").strip().lower()
    if not d:
        return True
    return dtype_name(descr).lower() == d or str(descr).lower() == d


def import_version(mod_name: str):
    """返回模块版本号；装没装都不抛。"""
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return None
    v = getattr(mod, "__version__", None)
    if v:
        return str(v)
    try:
        import importlib.metadata as md
        return md.version(mod_name)
    except Exception:
        return None


def gpu_info() -> dict:
    """尽力拿 CUDA 信息；torch 缺失或无 GPU 都不抛。"""
    try:
        import torch  # noqa
    except Exception:
        return {"torch_available": False, "cuda_available": None,
                "note": "torch 未安装，无法判定 GPU"}
    out = {"torch_available": True, "cuda_available": bool(torch.cuda.is_available()),
           "cuda_version": getattr(torch.version, "cuda", None)}
    if out["cuda_available"]:
        try:
            out["device_count"] = torch.cuda.device_count()
            out["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- 路径守卫

def normalize_path(p) -> Path:
    try:
        return Path(p).expanduser().resolve()
    except Exception:
        return Path(p).expanduser().absolute()


def resolve_out_alias(out_val, alias_val):
    """把 `--workdir` 别名归一到 `--out`。返回 (值, 错误文本或 None)。

    doctor/train/search 用 `--out`，eval/collect 用 `--workdir`（那两个的 --out
    是报告/zip 的落点，另有含义）。同一个概念两套写法，对不懂细节的人是最容易错的一处。
    故前者收一个同义别名 —— 但**两个都给了且不是同一个位置时拒绝**：
    「后者覆盖前者」是静默替人做决定，本工具不干这个。
    """
    if not alias_val:
        return out_val, None
    if out_val and normalize_path(alias_val) != normalize_path(out_val):
        return None, ("--out 与 --workdir 都给了，但指的不是同一个位置：\n"
                      "    --out     = %s\n    --workdir = %s" % (out_val, alias_val))
    return (out_val or alias_val), None


def is_under(child, parents) -> bool:
    """child 是否落在任一父目录之下（大小写不敏感，Windows 友好）。"""
    c = str(normalize_path(child)).rstrip("\\/").lower()
    for parent in parents or []:
        par = str(normalize_path(parent)).rstrip("\\/").lower()
        if not par:
            continue
        if c == par or c.startswith(par + os.sep) or c.startswith(par + "/"):
            return True
    return False


class MasterGuardError(Exception):
    """目标落在母本根之下且未获显式授权。"""


def guard_master_write(target, master_roots, allow: bool, what: str = "写入") -> None:
    """防「不懂细节的人」手滑污染母本。"""
    if allow:
        return
    if is_under(target, master_roots):
        raise MasterGuardError(
            "拒绝%s：目标 %s 落在母本根 %s 之下。" % (what, target, list(master_roots))
        )


# ---------------------------------------------------------------- 配置

DEFAULT_CONFIG = {
    "master_roots": ["Z:/Ongoing", "D:/Ongoing"],
    "workdir": "",              # 空 = 用 --out / 环境变量 / 当前目录下的 _pt_work
    "threads": 1,
    "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "expected_deps": [],        # 空 = 以包内 MANIFEST.env 为准
    "contact": "（未配置联系人：请在 pt_config.json 的 contact 里填）",
}


def find_dispatch(bundle):
    """在包里找 DISPATCH.json，找不到返回 None。

    异地操作者不该被要求记住 --dispatch：打包时它就在包根，漏传就等于整条契约核对**空跑**
    （不是报错，是安静地什么都没比 —— 这个坑已经踩过一次）。环境变量 PT_DISPATCH 可覆盖。
    """
    cands = []
    envp = os.environ.get("PT_DISPATCH")
    if envp:
        cands.append(Path(envp))
    b = Path(bundle)
    cands += [b / "DISPATCH.json", b.parent / "DISPATCH.json"]
    for c in cands:
        try:
            if c.is_file():
                return c
        except OSError:
            pass
    return None


def load_dispatch(bundle, explicit=None):
    """显式给的优先；没给就自动找。返回 (对象, 路径)；读不出来就 (None, None)，不猜。"""
    p = Path(explicit) if explicit else find_dispatch(bundle)
    if p is None:
        return None, None
    try:
        return read_json_any(p), str(p)
    except Exception:
        return None, None


def find_config(cli_config=None):
    """三级解析：CLI -> 环境变量 PT_CONFIG -> 脚本同级目录。"""
    candidates = []
    if cli_config:
        candidates.append(Path(cli_config))
    env = os.environ.get("PT_CONFIG")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parent
    candidates.append(here.parent / "pt_config.json")
    candidates.append(here / "pt_config.json")
    for c in candidates:
        try:
            if c.is_file():
                return c
        except Exception:
            continue
    return None


def load_config(cli_config=None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = find_config(cli_config)
    if cfg_path is not None:
        try:
            user = read_json_any(cfg_path)
            if isinstance(user, dict):
                cfg.update(user)
        except Exception as exc:
            print("[pt] 警告：配置文件 %s 读取失败（%s），改用默认配置。" % (cfg_path, exc))
    cfg["_config_path"] = str(cfg_path) if cfg_path else None
    return cfg


# ---------------------------------------------------------------- 三态报告

# 渲染顺序：先看环境能不能跑，再看数据对不对，最后看账物是否相符
_GROUP_ORDER = ["环境组", "输入组", "标签组", "契约与卫生组"]

class Report:
    """一个检查器 = 一份三态报告。

    铁律：UNKNOWN 绝不能被读成 PASS。overall() 的优先级是 FAIL > UNKNOWN > WARN > PASS，
    其中 UNKNOWN 单独可查（`unknowns()`），供调用方决定是否放行。
    """

    def __init__(self, title: str, subject: str = "", meta: dict | None = None):
        self.title = title
        self.subject = subject
        self.meta = dict(meta or {})
        self.checks: list[dict] = []
        self.started = now_iso()

    def add(self, group: str, name: str, status: str, detail: str = "", evidence=None,
            kind: str = ""):
        rec = {"group": group, "name": name, "status": status, "detail": detail}
        if kind:
            rec["kind"] = kind
        if evidence is not None:
            rec["evidence"] = evidence
        self.checks.append(rec)
        return rec

    def counts(self) -> dict:
        c = {PASS: 0, WARN: 0, FAIL: 0, UNKNOWN: 0}
        for rec in self.checks:
            # 结论类不算检查项：它恒为 PASS，计进"通过"会让合计虚高一项
            if rec.get("kind") == KIND_CONCLUSION:
                continue
            c[rec["status"]] = c.get(rec["status"], 0) + 1
        return c

    def fails(self):
        return [r for r in self.checks if r["status"] == FAIL]

    def warns(self):
        return [r for r in self.checks if r["status"] == WARN]

    def unknowns(self):
        return [r for r in self.checks if r["status"] == UNKNOWN]

    def overall(self) -> str:
        c = self.counts()
        if c[FAIL]:
            return FAIL
        if c[UNKNOWN]:
            return UNKNOWN
        if c[WARN]:
            return WARN
        return PASS

    def exit_code(self) -> int:
        o = self.overall()
        if o == FAIL:
            return RC_FAIL
        if o == UNKNOWN:
            # UNKNOWN 不放行为「干净通过」，但也不是 FAIL；用 WARN 码提示需人看
            return RC_WARN
        if o == WARN:
            return RC_WARN
        return RC_OK

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "subject": self.subject,
            "started": self.started,
            "finished": now_iso(),
            "verdict": self.overall(),
            "counts": self.counts(),
            "meta": self.meta,
            "checks": self.checks,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    # -------------------------------------------------- 人读渲染

    def render(self) -> str:
        icon = {PASS: "[通过]", WARN: "[注意]", FAIL: "[不通过]", UNKNOWN: "[判不了]"}
        lines = []
        lines.append("=" * 74)
        lines.append("%s   对象: %s" % (self.title, self.subject))
        lines.append("=" * 74)
        order = [FAIL, UNKNOWN, WARN, PASS]
        label = {FAIL: "不通过（必须停下来处理）", UNKNOWN: "判不了（信息不足，不能当作通过）",
                 WARN: "注意（可以继续，但请看一眼）", PASS: "通过"}
        groups = []
        for rec in self.checks:
            if rec["group"] not in groups:
                groups.append(rec["group"])
        groups.sort(key=lambda x: (_GROUP_ORDER.index(x)
                                   if x in _GROUP_ORDER else len(_GROUP_ORDER)))
        for g in groups:
            lines.append("")
            lines.append("-- %s" % g)
            for st in order:
                for rec in self.checks:
                    if rec["group"] != g or rec["status"] != st:
                        continue
                    lines.append("   %s %s" % ("[结论]" if rec.get("kind") == KIND_CONCLUSION
                                              else icon[st], rec["name"]))
                    if rec["detail"]:
                        lines.append("        %s" % rec["detail"])
        c = self.counts()
        lines.append("")
        lines.append("-" * 74)
        lines.append("合计：不通过 %d / 判不了 %d / 注意 %d / 通过 %d    =>  总判: %s"
                     % (c[FAIL], c[UNKNOWN], c[WARN], c[PASS], label[self.overall()]))
        for st in (FAIL, UNKNOWN, WARN, PASS):
            if c[st]:
                lines.append("   %s: %s" % (label[st], ", ".join(
                    r["name"] for r in self.checks
                    if r["status"] == st and r.get("kind") != KIND_CONCLUSION)))
        return "\n".join(lines)


def human_block(headline: str, what: str, why: str, next_step: str, contact: str) -> str:
    """给「不懂细节的人」看的四段式。任何 FAIL 都必须走这里，不许只丢一句报错。"""
    return (
        "\n" + "!" * 74 + "\n"
        "%s\n" % headline +
        "!" * 74 + "\n"
        "发生了什么：\n    %s\n"
        "为什么是问题：\n    %s\n"
        "下一步做什么：\n    %s\n"
        "拿不准找谁：\n    %s\n" % (what, why, next_step, contact) +
        "!" * 74 + "\n"
    )


# ---------------------------------------------------------------- 包发现

def entry_candidates(bundle: Path, entry: dict):
    """把一个 dst/path 声明展开成若干候选磁盘路径（按可信度排序）。

    已知坑（两条）：
    - `dst` 是 Z_ROOT 相对**且带包名前缀**（"S03_alphalgbm_csi800\\X\\X.npy"），
      直接 join 会多套一层；S13 那族还写成双反斜杠。
    - 但**不能只靠"目录名 == 包名"来剥** —— 操作者从浏览器下载解压常拿到
      "S03_alphalgbm_csi800 (1)" 这种目录名，那样每个条目都会假报缺失。
      所以这里只生成候选，最终按**哪个真的存在**决定，并把试过的一并记下来。
    """
    bundle = Path(bundle)
    raw = entry.get("dst") or entry.get("path")
    if not raw:
        return []
    s = re.sub(r"/{2,}", "/", str(raw).replace("\\", "/")).strip("/")
    if not s:
        return []
    parts = s.split("/")

    stripped = bundle.joinpath(*parts[1:]) if len(parts) > 1 else None
    asis = bundle.joinpath(*parts)
    tail2 = bundle.joinpath(*parts[-2:]) if len(parts) > 2 else None

    order = []
    if parts[0] == bundle.name and stripped is not None:
        order = [stripped, asis, tail2]        # 目录名==包名：按惯例先剥
    else:
        order = [asis, stripped, tail2]        # 目录名被改过：先按原样找

    out = []
    for p in order:
        if p is not None and p not in out:
            out.append(p)
    return out


def resolve_entry_path(bundle: Path, entry: dict):
    """取第一个真实存在的候选；都不存在则返回惯例路径（供报错文案使用）。"""
    cands = entry_candidates(bundle, entry)
    if not cands:
        return None
    for c in cands:
        try:
            if c.exists():
                return c
        except OSError:
            continue
    return cands[0]


def collect_manifest_entries(manifest: dict):
    """把 MANIFEST 里的 X / Y 条目摊平成 (section, key, entry) 三元组。

    实测两种形态：
      X = [ {shape,dtype,bytes,sha1,src,dst,note}, ... ]        （list）
      Y = { "target": [ {...}, ... ], "settle": [...] }         （dict of list）
    另有 `files` 键在部分包是空 dict。
    """
    out = []
    for section in ("X", "Y", "files"):
        node = manifest.get(section)
        if node is None:
            continue
        if isinstance(node, list):
            for i, e in enumerate(node):
                if isinstance(e, dict):
                    out.append((section, "%s[%d]" % (section, i), e))
        elif isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, list):
                    for i, e in enumerate(v):
                        if isinstance(e, dict):
                            out.append((section, "%s.%s[%d]" % (section, k, i), e))
                elif isinstance(v, dict):
                    out.append((section, "%s.%s" % (section, k), v))
    return out


def load_bundle_meta(bundle: Path) -> dict:
    """读包的元数据。缺文件不抛，只记 None / 标记。"""
    bundle = Path(bundle)
    meta = {"bundle": str(bundle), "manifest": None, "y_meta": None,
            "quality": None, "factors": None, "requirements": None,
            "git_head": None, "pycache": None}
    for key, rel in (("manifest", "MANIFEST.json"), ("y_meta", "Y/Y_meta.json"),
                     ("quality", "QUALITY.json"), ("factors", "FACTORS.json")):
        f = bundle / rel
        if f.is_file():
            try:
                meta[key] = read_json_any(f)
            except Exception as exc:
                meta[key] = {"_read_error": str(exc)}
    req = bundle / "requirements.txt"
    if req.is_file():
        meta["requirements"] = read_text_any(req)
    # 交付卫生：__pycache__ 违反 Z 侧「禁落 pycache」铁律
    pyc = sorted(p for p in bundle.rglob("__pycache__") if p.is_dir())
    meta["pycache"] = [str(p.relative_to(bundle)) for p in pyc]
    # git HEAD（有就用，没有不算错）
    head = bundle / ".git" / "HEAD"
    if head.is_file():
        try:
            txt = read_text_any(head).strip()
            if txt.startswith("ref:"):
                ref = bundle / ".git" / txt.split(":", 1)[1].strip()
                meta["git_head"] = read_text_any(ref).strip() if ref.is_file() else txt
            else:
                meta["git_head"] = txt
        except Exception:
            pass
    return meta


def declared_env(manifest: dict) -> dict:
    """MANIFEST.env 是生产导出时的真实环境（S03 实测：
    python 3.10.20 / numpy 2.2.6 / pandas 2.3.3 / lightgbm 4.6.0 / torch 2.6.0+cu124）。
    异地比对依赖版本就该用这一份，而不是 requirements.txt 的下限声明。"""
    if isinstance(manifest, dict):
        env = manifest.get("env")
        if isinstance(env, dict):
            return env
    return {}


def parse_datetime_loose(s):
    """只取 YYYY-MM-DD 段用于比较锚点；解析不出返回 None。"""
    if not s:
        return None
    m = re.search(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", str(s))
    return "%s-%s-%s" % m.groups() if m else None


def run_subprocess(argv, cwd=None, env=None, log_path=None, timeout=None):
    """跑子进程并把 stdout/stderr 同时落到日志文件与内存。

    返回 (returncode, stdout_text, stderr_text)。不抛（超时也返回码）。
    """
    t0 = time.time()
    full_env = dict(os.environ)
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    try:
        proc = subprocess.run(
            [str(a) for a in argv], cwd=str(cwd) if cwd else None, env=full_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = -9
        out = exc.stdout or b""
        err = (exc.stderr or b"") + b"\n[TIMEOUT]\n"
    except Exception as exc:
        rc, out, err = -1, b"", ("[启动失败] %s\n" % exc).encode("utf-8")
    out_s = out.decode("utf-8", errors="replace")
    err_s = err.decode("utf-8", errors="replace")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 74 + "\n")
            f.write("$ %s   (cwd=%s)\n" % (" ".join(str(a) for a in argv), cwd))
            f.write("返回码 %s   耗时 %.1fs\n" % (rc, time.time() - t0))
            f.write("-" * 74 + "\n[stdout]\n" + out_s + "\n[stderr]\n" + err_s + "\n")
    return rc, out_s, err_s
