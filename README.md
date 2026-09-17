# attest — 训练鉴证

把可移动三件套(`X/` + `Y/` + `train_from_xy.py`)的异地标准化训练,钉成一条**可复算、可归因、可审计**的裁决链。

## 它是什么

attest 不训练模型、不改一行策略代码。它做的事,是让一次训练**可信到可以上线**:

- 把 Z 母本包封成 `zip + DISPATCH.json`(sha1 实算,不抄声明)发到异地;
- 异地(只有 Jupyter、操作者不懂内部细节)跑 `doctor → train → eval → collect`;
- 结果封回本机,由**人**独断 promote。

四道闸硬拦四类事故:

| 闸 | 拦的事故 |
|---|---|
| 锚点一致性 | 半拉子刷新(MANIFEST 与 Y_meta 不是同一天) |
| sha1 账物对账 | 账物不符(声明哈希与实算不符) |
| 三态口径 | 「判不了」被读成「通过」 |
| 母本守卫 | 手滑污染 Z/D 线上目录 |

铁律:**「判不了」绝不等于「通过」**。这是全套防呆里最容易被人绕过去的一条。

## 结构(双轨单源)

本包同时承载两套结构,SKILL.md 只维护一份(`plugins/attest/skills/`):

- **Claude Code 插件** — `.claude-plugin/marketplace.json` + `plugins/`
- **DSH/npm 插件** — `package.json` + `cordis.patch.yml` + `src/index.ts`

## 安装

```bash
dsh plugin --profile <name> add @zasset/attest
```

## 用法

异地侧五个动词(详见 `plugins/attest/skills/attest/README_FIRST.md`):

```bash
python pt.py doctor --bundle <包目录>                      # 体检,不通过就停
python pt.py train  --bundle <包目录> --mode smoke          # CPU 冒烟
python pt.py train  --bundle <包目录> --mode week --commit  # 真训练(要 GPU)
python pt.py eval   --bundle <包目录> --workdir <空目录>     # 评估与裁决
python pt.py collect --workdir <空目录> --out .             # 打包回传
```

本机侧打包发出:

```bash
python scripts/pt_pack.py pack --bundle <Z包目录> --out <本地目录>
```

## 边界

不写 Z 母本、不改任何策略包、不改 `_sync` 生成器、不改 `D:\ZML\trading\*`。promote 只在本机、由人独断。
