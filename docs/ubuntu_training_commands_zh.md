# Ubuntu 训练、恢复、评估与消融命令手册

更新：2026-09-14。命令已按当前仓库入口核对，使用 **Ubuntu Bash** 语法。本文整理运行方法，不代表已经在你的 Ubuntu 服务器上完成部署验证。

**真实配置现已生成：** `config/category_progressive_real/main.json`。它基于仓库实际五数据集清单，采用用户确认的四阶段划分。无需手写 JSON/CSV，上传与目录布局见 [真实数据流上传说明](real_data_stream_upload_zh.md)。以下 `REID_STREAM` 已默认指向这份文件。

## 1. 先准备路径、环境和数据

以下路径需要替换为服务器实际路径。后续命令默认在同一个 Bash 会话、仓库根目录执行：

```bash
export REID_PROJECT="/workspace/GuangjinOuyang/code"
export REID_STREAM="/workspace/GuangjinOuyang/code/config/category_progressive_real/main.json"
export REID_CLIP="/workspace/GuangjinOuyang/code/data/pretrained/ViT-B-16.pt"
export REID_RUN_ROOT="/workspace/GuangjinOuyang/code/experiments/ecpm_pgca"



在80G服务器上，参数是这样的: 
export REID_PROJECT="/home/haichao/ouyang/CVPR2026-VLADR-main"
export REID_STREAM="/home/haichao/ouyang/CVPR2026-VLADR-main/config/category_progressive_real/main.json"
export REID_CLIP="/home/haichao/ouyang/CVPR2026-VLADR-main/data/pretrained/ViT-B-16.pt"
export REID_RUN_ROOT="/home/haichao/ouyang/CVPR2026-VLADR-main/experiments/ecpm_pgca"


cd "$REID_PROJECT"
mkdir -p "$REID_RUN_ROOT/console"
export CUDA_VISIBLE_DEVICES=0
```

`CUDA_VISIBLE_DEVICES=0` 表示只暴露物理 GPU 0，程序内使用 `--device cuda`。当前入口是单进程单 GPU，不能用 `torchrun` 或 `DataParallel` 直接替代启动方式。精确恢复保持同样的 GPU 可见性、软件版本、线程配置、代码和数据。

上传时要包含新增的 `train_category_progressive.py`、`reid/`、`lreid_dataset/`、`tools/` 和数据配置等文件。如果使用 Git，不要遗漏尚未跟踪的新文件。

在已激活的 Python 环境中检查依赖：

```bash
python --version
nvidia-smi
python -m pip check

python - <<'PY'
import importlib.metadata as md
import torch
import torchvision
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('PyTorch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
for package in ('finch-clust', 'numpy', 'scipy', 'scikit-learn', 'Pillow'):
    print(package, md.version(package))
PY

python train_category_progressive.py --help
```

本项目 FINCH 实现要求 `finch-clust==0.2.3`。仓库 `requirements.txt` 还不是经过 Ubuntu 验证的完整环境锁文件：PyTorch/torchvision 安装项被注释，部分固定版本与本机实际验证环境不同。需先根据服务器驱动和 Python 环境配好依赖；本手册不提供未经验证的一键安装组合。环境成功后可记录实际版本：

```bash
python -m pip freeze > "$REID_RUN_ROOT/environment.freeze.txt"
```

数据和权重要求：

- `REID_STREAM` 指向真实的数据流 JSON，引用真实的 train/query/gallery CSV。仓库 example 中的图片路径是占位符。
- 类别可再次出现，同一真实训练身份不能跨阶段重复；训练与 validation/test 身份分离。
- 使用 `--evaluate` 时，所选 split 必须覆盖所有会出现的类别，不能只准备 person 的评估集。
- JSON 的相对数据根目录、manifest 路径按配置解析；请使用 Linux 路径，确认大小写一致。消融基础 JSON 的 `stream_config` 则相对命令工作目录解析，本文统一用绝对路径。
- `REID_CLIP` 是本地 OpenAI CLIP ViT-B/16 权重。入口不会自动下载。

先审计数据流和文件存在性：

```bash
python tools/validate_category_stream.py \
  --stream-config "$REID_STREAM" \
  --check-images \
  --output "$REID_RUN_ROOT/stream_audit.json"
```

`--check-images` 只检查路径是否存在，不解码图片；不能据此确认图片未损坏。首次服务器验收仍需要短训练。

## 2. 先做服务器短训练与恢复检查

这条命令使用真实配置的完整阶段列表，但每阶段只执行 `2×3=6` 次优化器更新。每类别 P=2、K=2，适合先确认流程：

```bash
python -u train_category_progressive.py \
  --stream-config "$REID_STREAM" \
  --reference-checkpoint "$REID_CLIP" \
  --output-dir "$REID_RUN_ROOT/server_check" \
  --device cuda --amp \
  --epochs 2 --iterations-per-epoch 3 \
  --batch-size 4 --num-instances 2 \
  --prototype-batch-size 32 --eval-batch-size 32 \
  --workers 0 --seed 42 \
  --evaluate --eval-split test --eval-gallery both \
  --checkpoint-every 1 --max-updates 2
```

它先保存两次更新后退出，再执行：

```bash
python -u train_category_progressive.py \
  --output-dir "$REID_RUN_ROOT/server_check" \
  --resume
```

检查最终状态为 `phase=complete`、`pending_evaluation=null`，评估数量等于阶段数。`evaluation_summary.json` 应包含完整阶段序列。

**短训练不等于小数据提取。** 每阶段 ECPM 仍读取全部当前训练图像并聚类，评估也读取完整 query/gallery；如果数据很大，先准备单独的小规模审计通过的数据流配置。短训练只验证工程，不用于决定超参数或声称精度。

也可以使用自动生成图像的真实 CLIP 工程检查：

```bash
python tools/smoke_test_progressive_evaluation.py \
  --device cuda \
  --output "$REID_RUN_ROOT/generated_smoke.json"
```

这个工具使用默认权重位置 `~/.cache/clip/ViT-B-16.pt`，不接受 `--reference-checkpoint`。如需使用任意权重路径，采用前面的真实配置短训练命令即可。

## 3. 完整方法训练命令

通过服务器验收后，新建一个独立实验。以下超参数是当前实现默认值的明确写法，不是已经验证的最优参数：

```bash
python -u train_category_progressive.py \
  --stream-config "$REID_STREAM" \
  --reference-checkpoint "$REID_CLIP" \
  --output-dir "$REID_RUN_ROOT/full_seed42" \
  --device cuda --amp \
  --epochs 10 --iterations-per-epoch 100 \
  --batch-size 32 --num-instances 4 \
  --prototype-batch-size 128 --eval-batch-size 128 \
  --workers 0 --seed 42 \
  --adapter-lr 0.0003 --head-lr 0.0003 --weight-decay 0.0001 \
  --lambda-tri 1.0 --triplet-margin 0.3 \
  --consistency drift --lambda-con 1.0 --gamma 1.0 \
  --init-mode similarity --alpha 0.5 --delta 0.5 \
  --control-summary ecpm --routing-summary ecpm --beta 0.5 \
  --evaluate --eval-split test --eval-gallery both \
  --checkpoint-every 1
```

新实验要求输出目录不存在或为空。不要用短训练目录恢复后修改 `epochs`、批大小等参数来冒充完整训练；完整训练使用新目录。

### 3.1 训练参数

| 参数 | 默认值 | 含义/约束 |
|---|---|---|
| `--stream-config` | 新运行必填 | 类别交错、身份不重复的数据流 JSON |
| `--output-dir` | 必填 | 该实验独立的输出目录 |
| `--reference-checkpoint` | `~/.cache/clip/ViT-B-16.pt` | 新运行使用的本地参考权重 |
| `--device` | `cpu` | GPU 训练显式写 `cuda` |
| `--amp` | 关闭 | 出现此开关即启用 CUDA AMP；不写 `--amp True` |
| `--epochs` | 10 | 每阶段的训练 epoch 数 |
| `--iterations-per-epoch` | 100 | 每个 epoch 固定的优化器更新次数 |
| `--batch-size` | 32 | 每个当前类别、每次更新的图像数 B，不是所有类别合计 |
| `--num-instances` | 4 | 每个身份的图像数 K；P=B/K，要求 P≥2、K≥2 |
| `--workers` | 0 | 当前精确恢复入口只接受 0 |
| `--seed` | 42 | 模型初始化、训练采样及随机来源对照的种子 |
| `--adapter-lr` | 0.0003 | 当前类别 Adapter 学习率 |
| `--head-lr` | 0.0003 | 当前阶段临时身份分类器学习率 |
| `--weight-decay` | 0.0001 | AdamW 权重衰减 |
| `--lambda-tri` | 1.0 | Triplet 损失系数 |
| `--triplet-margin` | 0.3 | 类别内 batch-hard Triplet margin |
| `--max-grad-norm` | 不裁剪 | 可选正数；例如 `--max-grad-norm 1.0` |

每次更新，每个当前类别各取一个 P×K 批次，依次反向传播，再执行一次优化器更新。例如阶段有 3 类、B=32，该更新共使用 96 张图片，但不是把它们合并成跨类别 Triplet。

每阶段更新数为 `epochs × iterations_per_epoch`。这里一个 epoch 是固定数量的采样更新，不保证恰好遍历全部图像一次；采样轮次耗尽后会重新采样。每个阶段的每个类别都必须至少有 P 个身份，B=32、K=4 就要求至少 8 个身份。

### 3.2 ECPM 与 PGCA 参数

| 参数 | 默认值 | 含义/可选值 |
|---|---|---|
| `--prototype-batch-size` | 128 | 当前身份参考原型提取批大小，独立于训练 B |
| `--finch-chunk-size` | 256 | FINCH 精确余弦近邻的距离计算分块行数 |
| `--consistency` | `drift` | `off` 无蒸馏；`fixed` 固定系数；`drift` 漂移调节 |
| `--lambda-con` | 1.0 | 固定系数，或漂移模式的基准系数，非负 |
| `--gamma` | 1.0 | 漂移敏感度，非负 |
| `--init-mode` | `similarity` | `default`、`similarity`、`random`、`source` |
| `--alpha` | 0.5 | 初始化匹配中类别中心相似度的权重，[0,1] |
| `--delta` | 0.5 | 初始化选源阈值，[-1,1]；分数≥阈值才接受 |
| `--transfer-source` | 无 | `init-mode=source` 时指定历史类别，主要用于诊断 |
| `--control-summary` | `ecpm` | `ecpm` 或 `identity_mean`；同时影响漂移和迁移所用的控制摘要 |

熟悉类别的蒸馏系数为 `lambda_con × exp(-gamma × drift)`；陌生类别没有旧同类别教师，不加这个损失。`fixed` 使用 `lambda_con`，不随漂移变化。

相似性选源分数为 `alpha × 类别中心余弦相似度 + (1-alpha) × 新模式到旧模式的平均最佳匹配相似度`。无历史或低于阈值则默认初始化。`random` 有历史时随机选一个历史来源，不使用阈值过滤；`source` 强制使用指定的历史来源。二者是显式对照，不是完整方法的默认行为。

`finch-chunk-size` 控制临时距离块，算法仍是精确近邻，总计算量仍可能随类别累计身份数呈平方增长。它不是原型预算，也不改变保存全部身份原型的设定。

### 3.3 评估参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--evaluate` | 关闭 | 每阶段提交后评估所有已见类别 |
| `--eval-split` | `test` | `validation` 或 `test`，须有全类别覆盖 |
| `--eval-gallery` | `per_dataset` | 原图库 `per_dataset`、混合图库 `mixed` 或分别报告 `both` |
| `--eval-batch-size` | 128 | 推理批大小 |
| `--routing-summary` | `ecpm` | `ecpm` 或 `identity_mean`，独立于训练控制摘要 |
| `--beta` | 0.5 | 路由中类别中心项权重，[0,1] |

`score(c)=beta×cos(z,m_c)+(1-beta)×max_k cos(z,g_c,k)`。硬路由候选始终是所有已见类别，不用真实类别选 Adapter。每次评估同时输出正式 `prototype` 路由和诊断用 `oracle`。

应先在 validation 确定超参数，再固定配置评估 test。不要根据 test 指标选择来源、权重、alpha、beta 或阈值。本文 test 命令示范的是固定配置后的评估运行。

## 4. 中断恢复与主动暂停

恢复同一个实验，不必重新写训练参数：

```bash
python -u train_category_progressive.py \
  --output-dir "$REID_RUN_ROOT/full_seed42" \
  --resume
```

恢复后再运行 100 次更新，然后保存退出：

```bash
python -u train_category_progressive.py \
  --output-dir "$REID_RUN_ROOT/full_seed42" \
  --resume --max-updates 100
```

完成本次调用中的一个阶段提交与评估后退出：

```bash
python -u train_category_progressive.py \
  --output-dir "$REID_RUN_ROOT/full_seed42" \
  --resume --max-stages 1
```

| 控制参数 | 默认值 | 说明 |
|---|---|---|
| `--resume` | 关闭 | 从该输出目录的 `latest.pt` 恢复 |
| `--checkpoint-every` | 1 | 每多少次完整更新保存；调大可减少写盘，但异常时需重做更多更新 |
| `--max-updates` | 无限制 | 本次调用最多新增多少次更新，不是全实验累计上限 |
| `--max-stages` | 无限制 | 本次调用最多新增多少个已提交阶段 |

主动暂停、阶段最后一次更新和阶段提交会强制保存。若正好停在阶段最后一次更新之后，断点可能还是 `training`，ECPM 尚未提交；恢复后会完成提交，不需要手工处理。

精确恢复以最后成功落盘的完整优化器更新为边界，保存参数、分类头、教师、AdamW、scaler、RNG、采样位置及 ECPM 候选。它不保存一次反向传播中途的计算图。

必须保留整个输出目录，特别是 `reference.pt`、`latest.pt` 和日志前缀。恢复时不能改变训练超参数、是否启用评估或评估协议；也不能把 Windows 的未完成断点直接作为 Ubuntu 同环境精确恢复。旧的完整阶段可以用独立评估工具读取，但这与续训是不同用途。

## 5. 后台运行与查看进度

断开 SSH 后继续运行的 Bash 示例，使用另一个新实验目录：

```bash
nohup python -u train_category_progressive.py \
  --stream-config "$REID_STREAM" \
  --reference-checkpoint "$REID_CLIP" \
  --output-dir "$REID_RUN_ROOT/full_seed42_bg" \
  --device cuda --amp --workers 0 --seed 42 \
  --epochs 10 --iterations-per-epoch 100 \
  --batch-size 32 --num-instances 4 \
  --evaluate --eval-split test --eval-gallery both \
  > "$REID_RUN_ROOT/console/full_seed42_bg.log" 2>&1 &

tail -f "$REID_RUN_ROOT/console/full_seed42_bg.log"
```

日志重定向放在独立的 `console/` 中。不要先往新实验输出目录写 `console.log`，否则“新运行目录必须为空”的检查会拒绝启动。每个实验目录只运行一个进程。

后台恢复示例：

```bash
nohup python -u train_category_progressive.py \
  --output-dir "$REID_RUN_ROOT/full_seed42_bg" --resume \
  >> "$REID_RUN_ROOT/console/full_seed42_bg.log" 2>&1 &
```

确认原训练进程已经退出后再恢复；不要同时启动两个操作同一断点目录的进程。

## 6. 单独评估已完成阶段

不重训，评估当前保存的最后一个已完成阶段：

```bash
python tools/evaluate_category_progressive.py \
  --run-dir "$REID_RUN_ROOT/full_seed42" \
  --output "$REID_RUN_ROOT/full_seed42/eval_ecpm_test.json" \
  --device cuda --split test --gallery both \
  --summary ecpm --beta 0.5 --batch-size 128
```

同一检查点改用简单身份均值路由，隔离路由机制的影响：

```bash
python tools/evaluate_category_progressive.py \
  --run-dir "$REID_RUN_ROOT/full_seed42" \
  --output "$REID_RUN_ROOT/full_seed42/eval_mean_test.json" \
  --device cuda --split test --gallery both \
  --summary identity_mean --beta 0.5 --batch-size 128
```

独立评估工具的参数名是 `--split`、`--gallery`、`--summary`、`--batch-size`；训练入口对应的是 `--eval-split`、`--eval-gallery`、`--routing-summary`、`--eval-batch-size`，不要混用。

只接受已提交阶段，拒绝训练中的学生配旧原型。单独评估末阶段不能重建被覆盖的历史模型；完整终身矩阵来自逐阶段评估历史。完整训练最终会覆盖此前的 `latest.pt`，要保留特定阶段模型，应在 `--max-stages` 停止后另行保存整个实验目录。

## 7. 分别运行各项消融

在同一个 Bash 会话先定义共同设置；下面的机制参数没有放入数组，避免被重复指定：

```bash
REID_COMMON=(
  --stream-config "$REID_STREAM"
  --reference-checkpoint "$REID_CLIP"
  --device cuda --amp
  --epochs 10 --iterations-per-epoch 100
  --batch-size 32 --num-instances 4
  --prototype-batch-size 128 --eval-batch-size 128
  --workers 0
  --adapter-lr 0.0003 --head-lr 0.0003 --weight-decay 0.0001
  --lambda-tri 1.0 --triplet-margin 0.3
  --evaluate --eval-split test --eval-gallery both
  --checkpoint-every 1
)
```

每条命令都会新建独立实验，预算和种子一致；不要与第 3 节的已有输出目录混用：

```bash
# 1. 完整方法
python -u train_category_progressive.py "${REID_COMMON[@]}" \
  --seed 42 --output-dir "$REID_RUN_ROOT/manual_full_s42"

# 2. 基础持久 Adapter：无一致性、默认初始化、身份均值路由
python -u train_category_progressive.py "${REID_COMMON[@]}" \
  --seed 42 --output-dir "$REID_RUN_ROOT/manual_baseline_s42" \
  --consistency off --init-mode default \
  --control-summary identity_mean --routing-summary identity_mean

# 3. 固定蒸馏：其余沿用完整方法
python -u train_category_progressive.py "${REID_COMMON[@]}" \
  --seed 42 --output-dir "$REID_RUN_ROOT/manual_fixed_s42" \
  --consistency fixed --lambda-con 1.0

# 4. 新类别默认初始化
python -u train_category_progressive.py "${REID_COMMON[@]}" \
  --seed 42 --output-dir "$REID_RUN_ROOT/manual_default_init_s42" \
  --init-mode default

# 5. 随机历史来源初始化
python -u train_category_progressive.py "${REID_COMMON[@]}" \
  --seed 42 --output-dir "$REID_RUN_ROOT/manual_random_init_s42" \
  --init-mode random

# 6. 仅类别中心相似度选源
python -u train_category_progressive.py "${REID_COMMON[@]}" \
  --seed 42 --output-dir "$REID_RUN_ROOT/manual_global_init_s42" \
  --init-mode similarity --alpha 1.0

# 7. 只替换训练控制摘要
python -u train_category_progressive.py "${REID_COMMON[@]}" \
  --seed 42 --output-dir "$REID_RUN_ROOT/manual_mean_control_s42" \
  --control-summary identity_mean --routing-summary ecpm

# 8. 只替换路由摘要
python -u train_category_progressive.py "${REID_COMMON[@]}" \
  --seed 42 --output-dir "$REID_RUN_ROOT/manual_mean_routing_s42" \
  --control-summary ecpm --routing-summary identity_mean

# 9. 同时替换控制和路由摘要
python -u train_category_progressive.py "${REID_COMMON[@]}" \
  --seed 42 --output-dir "$REID_RUN_ROOT/manual_mean_both_s42" \
  --control-summary identity_mean --routing-summary identity_mean
```

每个实验已经同时输出 oracle 和 prototype，无需为了 oracle 再训练一次。第 8 个路由消融也可以用第 6 节的独立评估在同一检查点完成，减少重复训练；如果只保留末阶段模型，只能补该阶段的路由对照。

身份均值是“每个身份原型等权平均后归一化”，不是每张图片等权。为了分开比较控制和路由，当前对照仍计算并保留 FINCH 记忆；不要把其耗时/内存当作完全删除 FINCH 的结果。

单个手动实验中断后，对对应输出目录执行 `--resume`，不要再次执行那条新建命令。

## 8. 自动执行完整消融套件

相比逐条运行，套件会自动写计划、按实验顺序运行，并汇总终身矩阵。基础配置文件使用 **下划线参数名**，布尔值使用 JSON 的 `true`/`false`。

用第 1 节已导出的路径生成服务器配置：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

settings = {
    'stream_config': os.environ['REID_STREAM'],
    'reference_checkpoint': os.environ['REID_CLIP'],
    'device': 'cuda', 'amp': True,
    'epochs': 10, 'iterations_per_epoch': 100,
    'batch_size': 32, 'num_instances': 4,
    'prototype_batch_size': 128, 'eval_batch_size': 128,
    'workers': 0, 'seed': 42,
    'adapter_lr': 0.0003, 'head_lr': 0.0003, 'weight_decay': 0.0001,
    'lambda_tri': 1.0, 'triplet_margin': 0.3,
    'lambda_con': 1.0, 'gamma': 1.0, 'alpha': 0.5, 'delta': 0.5,
    'eval_split': 'test', 'eval_gallery': 'both', 'beta': 0.5,
}
path = Path('config/ecpm_pgca_server.json')
with path.open('x', encoding='utf-8') as handle:
    json.dump(settings, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(path.resolve())
PY
```

文件已存在会报错以避免覆盖；这时直接检查、编辑现有 JSON 即可。基础配置需保持完整方法的默认模式；具体消融由套件覆盖。不要把 `output_dir`、`resume`、`checkpoint_every`、`max_updates` 或 `max_stages` 写进这个 JSON，它们不是该工具接受的基础训练配置项。套件默认每更新保存一次并自动启用评估。

只生成计划，不训练：

```bash
python tools/run_category_ablations.py \
  --base-config config/ecpm_pgca_server.json \
  --output-dir "$REID_RUN_ROOT/suite_seed42"
```

检查 `suite_seed42/ablation_plan.json` 后，执行全部 9 个实验：

```bash
python -u tools/run_category_ablations.py \
  --base-config config/ecpm_pgca_server.json \
  --output-dir "$REID_RUN_ROOT/suite_seed42" \
  --execute
```

中断后仍执行上面同一条命令：已完成实验不会重新训练，未完成实验从自己的 `latest.pt` 恢复。

只执行指定子集时，从一开始使用独立套件目录：

```bash
python -u tools/run_category_ablations.py \
  --base-config config/ecpm_pgca_server.json \
  --output-dir "$REID_RUN_ROOT/suite_subset_seed42" \
  --experiments full fixed_consistency default_initialization \
  --execute
```

可选名称为 `full`、`persistent_baseline`、`fixed_consistency`、`default_initialization`、`random_historical_source`、`global_only_initialization`、`mean_control_only`、`mean_routing_only`、`mean_control_and_routing`。

同一个套件目录不能在恢复时更改基础参数或实验列表；改变计划请新建目录。输出 `ablation_results.json` 汇总完成实验的终身性能/遗忘矩阵，各实验子目录保留完整日志。

## 9. 多个随机种子

在已经定义第 7 节 `REID_COMMON` 的会话中，完整方法按三个示例种子顺序运行：

```bash
for reid_seed in 42 43 44; do
  python -u train_category_progressive.py "${REID_COMMON[@]}" \
    --seed "$reid_seed" \
    --output-dir "$REID_RUN_ROOT/multiseed_full_s${reid_seed}" || break
done
```

这是新建实验循环，某次失败后先单独恢复该实验，再处理剩余种子，不能直接从头重跑这段循环。消融套件则为每个种子生成独立基础 JSON 和输出目录，分别执行同样的配方。比较各方法时对齐阶段配置、训练预算、种子和评估协议。

一次完整套件有 9 个训练实验，三个种子共有 27 个；不要把单次实验耗时当成整个套件耗时。

## 10. 验证集机制诊断命令

这些诊断研究实际迁移收益、实际蒸馏效果，不只是画权重公式。先保留目标阶段之前的边界。例如下面训练完成 T1 后退出，接下来诊断 T2：

```bash
python -u train_category_progressive.py \
  --stream-config "$REID_STREAM" \
  --reference-checkpoint "$REID_CLIP" \
  --output-dir "$REID_RUN_ROOT/diagnostic_boundary" \
  --device cuda --amp --workers 0 --seed 42 \
  --epochs 10 --iterations-per-epoch 100 \
  --batch-size 32 --num-instances 4 \
  --max-stages 1
```

目标阶段需要满足：来源诊断有“新类别＋历史来源”；漂移诊断有“再次出现的熟悉类别”。所有已见类别还需要 validation 集，只有 test 不够。

枚举历史来源、默认初始化和相似性初始化：

```bash
python tools/diagnose_pgca_validation.py \
  --run-dir "$REID_RUN_ROOT/diagnostic_boundary" \
  --output "$REID_RUN_ROOT/source_diagnosis_s42.json" \
  --device cuda --kind sources
```

比较动态漂移权重与固定权重：

```bash
python tools/diagnose_pgca_validation.py \
  --run-dir "$REID_RUN_ROOT/diagnostic_boundary" \
  --output "$REID_RUN_ROOT/drift_diagnosis_s42.json" \
  --device cuda --kind drift --weights 0 0.1 1 10
```

每个变体使用同一个阶段前模型、候选原型、采样/增强 RNG 和训练预算，原始边界不被修改。工具只用 validation，没有切换到 test 的参数。

输出 `empirical_rows` 可比较来源相似度与实际 mAP 增益，或漂移/有效权重与实际 mAP、Rank1。来源诊断对同阶段多个新类别同时采用该来源方案；如需单个新类别的独立研究，可设计只引入一个新类别的诊断阶段。

中断后重跑同一诊断命令会跳过已完成变体；当前未完成变体从共同起点重跑。这是变体边界恢复，不是主训练入口那种每更新恢复。

## 11. 结果文件与常见问题

| 文件 | 用途 |
|---|---|
| `reference.pt` | 冻结参考主干和配置，恢复必需 |
| `latest.pt` | 唯一权威最新状态：模型、ECPM、优化器/教师、RNG、采样游标和评估历史等 |
| `run_config.json` | 便于查看的参数和环境记录 |
| `progress.jsonl` | 准备、提交、评估、暂停、恢复及资源记录 |
| `stage_0000.jsonl` 等 | 各阶段 CE、Triplet、蒸馏、迁移及采样日志 |
| `evaluation.jsonl` | 每阶段完整检索与路由诊断 |
| `evaluation_summary.json` | 类别性能矩阵、宏平均、遗忘和阶段报告 |
| `stage_results.md` | 逐阶段可读表格：训练类别、已见类别、数据量、mAP、R1 和旧类别遗忘 |
| `stage_results.json` | 新版结构化逐阶段报告；区分全部已见类别平均与本阶段训练类别平均 |
| `stage_metrics.csv` | 可用 Excel 查看的逐阶段、逐类别明细 |
| `*.recovered-*.jsonl` | 崩溃后未被有效断点包含的日志尾部归档，不再次计入正式统计 |

逐阶段输出、遗忘口径和历史 JSON 转换命令见 [逐阶段结果使用说明](stage_reporting_usage_zh.md)。新代码在每阶段评估完成后立即打印并保存结果，仍需开启 `--evaluate`。本次修改改变源码指纹，旧版本训练不要直接跨版本恢复；已完成结果可用转换工具整理，无需重新训练。

常见处理：

- **每类别身份数少于 P：** 新实验减小训练 B 或调整 K，使 B/K 不超过任何阶段类别的身份数，同时保持 P≥2、K≥2。
- **CUDA OOM：** 确认发生在训练、原型提取还是评估，分别调整对应批大小。训练批大小需满足 P×K；若要改已保存的运行配置，需新建实验，不能假装精确恢复。
- **缺少评估集：** 补齐所选 split 的全部类别 query/gallery。只想检查训练时可新建不带 `--evaluate` 的实验。
- **输出目录非空：** 新实验换目录；已有有效训练使用 `--resume`。不要为了绕过检查删除有效断点。
- **一阶段训练结束后终端长时间无输出：** 新版会打印 `evaluation_progress`，包含特征提取批次、已完成 query 数和指标。上传文件及日志含义见 [评估进度说明](evaluation_progress_zh.md)。当前已启动的旧进程不会自动加载修改。
- **堆栈停在评估的 Path.resolve / os.lstat：** 已修复配对内反复解析路径的问题。T1 已提交且首次评估未完成时，可以用 [评估提速与断点接续](fast_evaluation_recovery_zh.md) 的受限迁移工具复制到新目录，保留训练状态后继续评估和 T2。
- **AMP 报梯度 non-finite：** 已修复在 GradScaler 降低倍率前直接终止的问题；上传新版 `reid/trainer_category_progressive.py` 后换新目录重跑。具体步骤、重试日志和 FP32 排查见 [AMP 修复说明](amp_gradient_fix_zh.md)。源码变化后不能对旧版本断点做精确恢复。
- **参考权重不存在：** 核对 `--reference-checkpoint`；本地默认缓存路径与服务器路径是两回事。
- **恢复报版本、代码、路径或指纹不一致：** 核对原运行环境和数据配置，不要删除校验代码绕过。
- **训练日志的 accuracy 很高：** 它是临时分类头训练准确率，不等于检索 Rank1；正式性能看评估报告。
- **身份路由一致性很高：** 可能全部路由到同一个错误类别，应同时看路由准确率、混淆矩阵和 oracle/prototype 差距。

训练运行、验证集调参、正式测试三者应分别记录。生成图像冒烟检查只能证明工程链路，不能作为论文效果或创新性结论。

进一步说明见 [第 8 步恢复说明](progressive_training_usage_zh.md)、[第 9 步评估与消融说明](progressive_evaluation_usage_zh.md)。
