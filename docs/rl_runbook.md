# SearchRL 上机手册（AutoDL 单卡 4090D，半天方案）

本手册对应 `agentos/rl/`：多跳问答检索环境 + verl 多轮 GRPO + LoRA。所有命令都在仓库根目录执行。

> 规则：**所有选择只在 dev 上做，test 只在最后评测一次。** 失败的运行也保留在 `results/` 里。

## 0. 租机前检查

| 项目 | 要求 | 原因 |
|---|---|---|
| GPU | RTX 4090 / 4090D，24GB | 1.5B + LoRA + vLLM 采样 |
| **最高支持 CUDA 版本** | **≥ 13.0**（驱动 ≥ 580） | verl 0.9.1 固定 torch 2.11，PyPI 上的 torch 2.11 是 CUDA 13 版本 |
| 数据盘 | ≥ 50GB（`~/autodl-tmp`） | 虚拟环境约 10GB、模型约 4GB、检查点约 20GB |
| 镜像 | 任意带 Python ≥ 3.10 的基础镜像 | 环境由脚本单独创建 |

AutoDL 选机页面会写出每台机器的"最高CUDA版本"。低于 13.0 的机器，`autodl_setup.sh` 第一步就会退出。

系统盘通常只有 30GB。安装脚本会把 pip、ModelScope、Hugging Face 的缓存都放到数据盘；训练中途可以用 `df -h / ~/autodl-tmp` 看两块盘的剩余空间。每次运行保留最近 2 个检查点，冒烟和不再需要的检查点可以删掉：`rm -rf ~/autodl-tmp/checkpoints/smoke`。

## 1. 时间表（约 6 小时）

| 时间 | 内容 | 命令（见下文） |
|---|---|---|
| 0:00–0:50 | 装环境、下模型、准备数据、算稠密向量、冒烟 5 步 | 第 2、3 节 |
| 0:50–1:15 | RL 前评测：dev 格式检查 + test 3 seed | 第 4 节 |
| 1:15–3:00 | 主训练 `grpo_v11`（每步约 70 秒） | 第 5 节 |
| 3:00–4:00 | 导出 LoRA，RL 后评测 test + OOD | 第 6 节 |
| 之后 | 可选的消融实验 | 第 7 节 |
| 最后 | 分析曲线和 reward hacking，取回结果 | 第 8 节 |

冒烟时会实测每步耗时；单卡、每个 micro-batch 1 条时约 70 秒/步。按 dev 验证成绩选检查点，不必跑满设定的步数。

## 2. 上传代码并装环境（约 30–40 分钟）

### 方法 A：从本机直接上传（推荐）

在 AutoDL 控制台复制实例的登录命令，格式是 `ssh -p <端口> root@<主机名>`，从中拿到主机名和端口。先确认 AutoDL 上有 rsync（多数镜像自带）：

```bash
ssh -p <端口> root@<主机名> "which rsync || (apt-get update && apt-get install -y rsync)"
```

然后在**本机**仓库根目录运行（会提示输入 AutoDL 的密码）：

```bash
bash scripts/sync_to_autodl.sh <主机名> <端口> --dry-run   # 先看会传哪些文件
bash scripts/sync_to_autodl.sh <主机名> <端口>             # 正式上传，约 83MB
```

会上传：代码、git 历史、本机已经生成好的 `data/rl`（划分、检索池、语料、verl parquet）。
**不会上传：** `.env`（API key）、虚拟环境、缓存、原始下载文件、数据库，以及本地 `.git/info/exclude` 里列出的个人文件。

上传中断了直接重跑同一条命令，`rsync` 只补传没传完的部分。之后在本机改了代码，也是重跑这条命令同步。

### 方法 B：从 GitHub 克隆

先在本机 `git push`，然后在 AutoDL 上：

```bash
cd ~/autodl-tmp && git clone https://github.com/zibinghou742-gif/agentos.git
```

这种方式没有 `data/rl`，安装脚本会自动下载原始数据并重新生成。

### 安装

登录 AutoDL 之后，**一定要在 tmux 里运行**，否则 SSH 一断，安装就中止：

```bash
tmux new -s setup
cd ~/autodl-tmp/agentos && bash scripts/autodl_setup.sh   # 不编译 flash-attn，训练时用 ATTN=sdpa
```

SSH 断了之后，重新登录，用 `tmux attach -t setup` 回到原来的窗口。

安装脚本可以**放心重跑**：已经装好的部分会跳过，已经下载的文件在缓存里，不会重新下载。如果某个镜像总是断线，可以换一个镜像：

```bash
INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/autodl_setup.sh
```

装完后激活环境：`source ~/autodl-tmp/venv/bin/activate`

脚本会依次完成：检查驱动 → 创建 venv 并安装 `verl[vllm]==0.9.1` → 从 ModelScope 下载 `Qwen2.5-1.5B-Instruct` 和 `bge-small-en-v1.5` → 准备数据（`data/rl` 已上传就直接用，否则下载并生成）→ 生成 verl parquet → 计算稠密向量 → 运行环境单元测试。

**数据一致性检查（只有方法 B 需要）：** 重新生成的 `results/rl_data_report.json`，除了 `generated_at` 以外应该和仓库里的完全一致（原始文件 sha256 也一致）。

```bash
git diff results/rl_data_report.json | grep '^[-+] ' | grep -v generated_at
```

这条命令应该没有输出。


## 3. 冒烟 5 步（约 10 分钟）

```bash
EXP=smoke STEPS=5 SAVE_FREQ=-1 TEST_FREQ=-1 ATTN=sdpa bash scripts/rl_train_grpo.sh trainer.val_before_train=False
python scripts/rl_analyze_rollouts.py results/rl_train/smoke --group 8
```

检查四件事：

1. `results/rl_train/smoke/rollouts/1.jsonl` 里的 `output` 含有 `<tool_call>` 和 `<tool_response>`，`tool_calls` 字段大于 0。如果 `tool_calls` 全是 0，但 `output` 里能看到工具结果，说明解码时特殊 token 被去掉了，**先停下来排查**，否则奖励全部算错。
2. `train.log` 里每步的 `timing_s/step`。用它重新估算 150 步要多久；如果单步超过 2 分钟，就把主训练改成 `STEPS=100` 或 `BATCH=6`。
3. 显存没有 OOM。如果 OOM，依次尝试：`GPU_UTIL=0.35`、`GROUP=6`、`BATCH=6`。
4. `zero_variance_groups` 不是 1.0。如果几乎所有组的奖励都相同（比如全是 0），说明模型还不会调用工具，需要先看第 4 节的格式检查。

## 4. RL 前评测（约 20 分钟）

```bash
# dev 上的格式检查：50 题、1 个 seed，看工具调用是否合法、会不会给出 <answer>
MODEL=~/autodl-tmp/models/Qwen2.5-1.5B-Instruct TAG=pre_rl_check SPLITS=dev LIMIT=50 SEEDS=0 bash scripts/rl_serve_and_eval.sh

# 正式的 RL 前基线：test 500 题 × 3 个 seed
MODEL=~/autodl-tmp/models/Qwen2.5-1.5B-Instruct TAG=pre_rl SPLITS="test ood_2wiki ood_musique" bash scripts/rl_serve_and_eval.sh

# 闭卷对照：不给工具，检查模型是否本来就记得答案
MODEL=~/autodl-tmp/models/Qwen2.5-1.5B-Instruct TAG=pre_rl_closed_book SPLITS=test EXTRA=--closed-book bash scripts/rl_serve_and_eval.sh
```

如果 `valid_call_rate` 很低或者 `no_answer_rate` 很高，RL 很可能学不起来。这时只能调**系统提示词**（`agentos/rl/env.py`），而且只能根据 dev 的结果去调。

## 5. 主训练

```bash
EXP=grpo_v11 ATTN=sdpa bash scripts/rl_train_grpo.sh
```

- 默认奖励是 `v11`（格式示例 + 检索加分 + 前 2 次调用免费）。
- 每 25 步保存检查点，并在 dev 上验证一次（贪心解码）；训练开始前也会先验证一次，得到 RL 前的 dev 分数。最多保留 `MAX_CKPT` 个检查点（每个约 6GB），注意数据盘空间。
- 断了直接重跑同一条命令（`EXP` 不变），会从最近的检查点继续；日志会追加写入。
- 所有参数都记录在 `results/rl_train/grpo_v11/run_config.env`。
- 训练中可以随时用 `python scripts/rl_analyze_rollouts.py results/rl_train/grpo_v11 --group 8` 看曲线，重点看工具调用次数有没有掉到 0。

## 6. 导出 LoRA 并评测

先看 dev 验证成绩，选最好的一步（下例为第 50 步）：

```bash
for s in 0 25 50; do grep "step:$s " results/rl_train/grpo_v11/train.log | tr ' ' '\n' | grep "^val-core"; done
python -m verl.model_merger merge --backend fsdp \
  --local_dir ~/autodl-tmp/checkpoints/grpo_v11/global_step_50/actor \
  --target_dir ~/autodl-tmp/merged/grpo_v11_step50
```

**注意：** 合并工具把训练好的 LoRA 存在 `lora_adapter/` 里，同目录下的 `model.safetensors` 是**未合并**的原模型。所以评测时要让 vLLM 加载"原模型 + LoRA"：

```bash
M=~/autodl-tmp/models/Qwen2.5-1.5B-Instruct
MODEL=$M TAG=pre_rl SPLITS="test ood_2wiki ood_musique" bash scripts/rl_serve_and_eval.sh
MODEL=$M LORA=~/autodl-tmp/merged/grpo_v11_step50/lora_adapter TAG=post_rl SPLITS="test ood_2wiki ood_musique" bash scripts/rl_serve_and_eval.sh
MODEL=$M TAG=pre_rl_closed_book SPLITS=test SEEDS=0 EXTRA=--closed-book bash scripts/rl_serve_and_eval.sh
```

确认 LoRA 生效：RL 后的 `no_answer_rate` 应接近 0。评测和训练使用完全相同的工具文本（`prompt_tool_schemas()`），不要改动它。

## 7. 可选的消融实验

以后有更多 GPU 时间时，可以跑的其他消融：

| 实验 | 命令变量 |
|---|---|
| outcome-only 奖励 | `REWARD_PRESET=outcome_only` |
| 去掉成本扣分 | `REWARD_PRESET=v11_no_cost` |
| DAPO 动态采样 | `DYNAMIC_SAMPLING=1` |
| sequence-mean 聚合 | `LOSS_AGG=seq-mean-token-mean` |
| 去掉超轮数软惩罚 | `REWARD_PRESET=no_overlong` |

## 8. 分析、取回结果并提交

在 **AutoDL** 上生成分析结果：

```bash
python scripts/rl_analyze_rollouts.py results/rl_train/grpo_v11 --group 8
```

在**本机**仓库根目录把结果取回来。AutoDL 上没有 GitHub 登录，所以在本机提交和推送：

```bash
rsync -az --progress -e "ssh -p <端口>" --exclude "rollouts/" --exclude "tensorboard/" root@<主机名>:/root/autodl-tmp/agentos/results/ ./results/
```

```bash
git add results && git commit -m "Add SearchRL training and evaluation results"
```

关机前，如果想保留 rollout 原始记录，用于以后挑 reward hacking 的例子，就去掉上面的 `--exclude "rollouts/"` 再取一次。它们不进 git，但留在本机。

会提交的内容：评测轨迹和 `summary.json`、训练曲线 `curves.csv` / `verl_metrics.csv` / `curves.png`、`hacking.json`、`run_config.env`。不会提交的内容：rollout 原始记录、tensorboard、日志、检查点（都已在 `.gitignore` 里）。

**reward hacking：** `hacking.json` 里列出了四类可疑行为（狂调工具、不搜就答、答案里列多个候选、连续非法调用），每类保存 3 条示例。这些只是**疑似**，要人工读完轨迹再下结论。

## 9. 大模型 prompt-only 对照（用 API，不占 GPU）

```bash
python scripts/rl_eval.py --split test --limit 200 --model qwen3.8-max --enable-thinking false \
  --dense ~/autodl-tmp/models/bge-small-en-v1.5 --seeds 0 --tag large_prompt_only
```

API key 只放在 `.env` 里，不要写进命令或提交。实测 qwen3.8-flash 每题约 7K token，200 题 × 1 个 seed 大约 140 万 token。是否加到 3 个 seed，要先看预算。

## 10. 常见问题

| 现象 | 处理 |
|---|---|
| `autodl_setup.sh` 提示 CUDA < 13.0 | 换一台"最高CUDA版本 ≥ 13.0"的机器 |
| flash-attn 导入失败 | 训练时加 `ATTN=sdpa` |
| vLLM 显存不足 | `GPU_UTIL=0.35`，或减小 `GROUP` / `BATCH` |
| 奖励一直是 0、零方差组接近 100% | 先看第 4 节格式检查，在 dev 上调整提示词；或者用 `DYNAMIC_SAMPLING=1` |
| `rollouts` 里 `tool_calls` 全是 0 | 解码丢掉了 `<tool_call>`，见第 3 节第 1 条 |
| 训练中断 | 重跑同一条命令，自动从检查点继续 |
