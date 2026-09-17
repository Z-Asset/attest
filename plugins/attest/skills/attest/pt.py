"""attest —— 可移动三件套的异地标准化训练入口。

五个动词，一个动作一条命令；所有环境变量、种子、线程数、路径拼接都由脚本钉死，
操作者不需要知道包内部结构。

    python pt.py doctor --bundle .                                  体检（必须第一步）
    python pt.py train  --bundle . --mode smoke                      CPU 冒烟（只写临时目录）
    python pt.py train  --bundle . --mode week --commit              真周训（需 GPU）
    python pt.py search --bundle . --grid grid.json --jobs 4         超参搜索
    python pt.py eval   --bundle . --workdir _pt_work                评估与裁决
    python pt.py collect --out <dir>                                 打包回传

盘符不假设：所有根目录来自 pt_config.json -> 环境变量 PT_* -> 命令行 三级覆盖。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pt_common as C  # noqa: E402

VERBS = ("doctor", "train", "search", "eval", "collect")


def usage() -> str:
    return (
        "attest —— 可移动三件套的异地标准化训练\n"
        "\n"
        "用法：python pt.py <动词> [参数...]\n"
        "\n"
        "  doctor   体检：环境 / 输入 / 标签 / 契约四组，任一组不通过就停\n"
        "              python pt.py doctor --bundle .\n"
        "  train    训练：默认不写盘。先 smoke 验证链路，再 week 真训\n"
        "              python pt.py train --bundle . --mode smoke\n"
        "              python pt.py train --bundle . --mode week --commit\n"
        "  search   超参搜索：网格随包发来，不用自己写；默认只看计划\n"
        "              python pt.py search --bundle . --jobs 2\n"
        "              python pt.py search --bundle . --jobs 2 --commit\n"
        "  eval     评估与裁决：复算门禁 + 噪声底 + 上线前置清单 + 裁决姿态\n"
        "              python pt.py eval --bundle . --workdir _pt_work\n"
        "  collect  回传打包：把体检/训练/评估的产物封成 zip 发回本机\n"
        "              python pt.py collect --workdir _pt_work --out .\n"
        "\n"
        "第一次用请先看 README_FIRST.md。只有 Jupyter 的机器请直接开 ATTEST.ipynb，\n"
        "那台机器上不需要敲任何命令。\n"
        "\n"
        "本机侧另有一个 pack 动作（把 Z 包封成 zip + DISPATCH.json），\n"
        "它不属于异地操作者的五步，直接调：python scripts/pt_pack.py pack --help\n"
    )


def main(argv=None) -> int:
    C.setup_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0 if argv else C.RC_USAGE
    verb, rest = argv[0], argv[1:]

    if verb not in VERBS:
        print("[pt] 不认识的动词：%s\n" % verb)
        print(usage())
        return C.RC_USAGE

    if verb == "doctor":
        from pt_doctor import main as run
        return run(rest)
    if verb == "train":
        from pt_train import cmd_train as run
        return run(rest)
    if verb == "search":
        from pt_search import cmd_search as run
        return run(rest)
    if verb == "eval":
        from pt_eval import main as run
        return run(rest)
    if verb == "collect":
        from pt_pack import cmd_collect as run
        return run(rest)

    print(C.human_block(
        "这个动词没有对应的实现：%s" % verb,
        "`%s` 在 VERBS 里，但没有分派到任何脚本 —— 这是本工具自己的缺陷。" % verb,
        "现在用它只会得到空结果，拿不到可用的判定。",
        "把这条报回本机（这不是你操作错）。",
        C.load_config().get("contact", C.DEFAULT_CONFIG["contact"])))
    return C.RC_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
