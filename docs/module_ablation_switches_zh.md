# ECPM / PGCA 四个独立消融开关

## 1. 默认行为

四个新参数默认均为 `on`。你原来的完整训练命令不用增加参数，仍使用原来的 FINCH 聚类、累计原型演化、漂移蒸馏和相似性迁移。数据划分、阶段数、CE、Triplet、类别 Adapter、评估协议及逐阶段报告流程均保留。

| 消融点 | 新参数 | 默认 | 设为 off 后 |
|---|---|---|---|
| ECPM：Category clustering | `--category-clustering on/off` | on | 不运行 FINCH，每类别只构建一个身份均值原型 |
| ECPM：Prototype evolution | `--prototype-evolution on/off` | on | 再次出现的类别只用本阶段身份构建摘要，不合并历史身份参与构建 |
| PGCA：Prototype drift-guided recurring adaptation | `--recurring-adaptation on/off` | on | 关闭熟悉类别蒸馏分支，保留 CE＋Triplet |
| PGCA：Prototype similarity-guided emerging transfer | `--emerging-transfer on/off` | on | 新类别使用默认 Adapter 初始化，不选择或复制历史来源 |

开关可以任意组合。关闭某个模块不会隐式关闭其他开关；例如关闭聚类以后，PGCA 仍可根据单中心原型计算漂移和相似性。

## 2. 每个消融的准确含义

### 2.1 关闭 Category clustering

身份原型仍按原方法从冻结参考编码器提取。设参与该类别摘要构建的身份原型为 \(p_i\)，关闭聚类时：

\[
g_c=m_c=\operatorname{Norm}\left(\frac{1}{N_c}\sum_i p_i\right).
\]

这时模式原型集合只有一个元素。没有 FINCH 调用，也不分配 FINCH 成对距离矩阵。身份是等权的，不按每个身份图片数量加权。

PGCA 和自动路由都会使用这个单中心摘要；不是只改变路由端。此时全局项和模式项通常等价，所以 `alpha`、`beta` 对两者加权的作用会退化，这是该消融的自然结果。

旧的 `--control-summary identity_mean`、`--routing-summary identity_mean` 仍可用，但它们原来只替换使用端视图，不跳过 ECPM 聚类。新增开关会真正跳过聚类计算。

### 2.2 关闭 Prototype evolution

采用 **仅最新一次出现阶段的身份原型（latest-stage-only）** 对照：

\[
\text{on: }\mathcal P_c^{\le t}=\bigcup_{s\le t}\mathcal P_c^s,
\qquad
\text{off: }\mathcal P_c^{\text{latest}}=\mathcal P_c^{\tau_c(t)},
\]

其中 \(\tau_c(t)\) 是类别 \(c\) 截至当前最近一次参与训练的阶段。

例如 person 在 T1、T2 出现：on 时 T2 用两阶段身份共同构建模式和中心；off 时 T2 只用 T2 新身份。若 T3 没有 person，仍保留 T2 摘要。

这一开关关闭的是**跨阶段累计身份融合和重建**，不是“首次出现后永久冻结摘要”。因此 off 时仍可以比较上次中心与本次中心、计算非零漂移，不会顺带取消漂移蒸馏。论文消融建议标为 “w/o cumulative prototype evolution (latest-stage-only)”，避免让读者误解对照含义。

完整身份原型账本继续保留，用于身份覆盖、断点和一致性校验，但历史向量不参与关闭演化后的当前摘要、路由和 PGCA 控制。该对照不用于证明身份记忆存储下降。摘要里 `category_modes[c].identities` 是实际参与摘要的身份数；顶层累计 `categories[c]` 仍是历史身份总数。

旧的 identity_mean 控制/路由视图也遵循此开关，不会通过旧视图重新混入历史身份。

### 2.3 关闭 recurring adaptation

有效模式变成 `consistency=off`：熟悉类别的特征蒸馏权重为零，不创建或前向计算历史教师，当前类别仍训练 CE＋Triplet。

这验证的是整个熟悉类别自适应分支的作用。如果希望单独验证“漂移引导”是否优于普通蒸馏，应另做 `--consistency fixed`：

| 对照 | 参数 | 实际权重 |
|---|---|---|
| 完整漂移蒸馏 | `--recurring-adaptation on --consistency drift` | `lambda_con * exp(-gamma * drift)` |
| 固定权重蒸馏 | `--recurring-adaptation on --consistency fixed` | `lambda_con` |
| 关闭整个分支 | `--recurring-adaptation off` | 0 |

### 2.4 关闭 emerging transfer

有效模式变成 `init_mode=default`：新类别不做来源相似性评分、不复制历史 Adapter，按原有默认方式创建自己的 Adapter。旧类别 Adapter 保留，身份分类头继续按原流程建立。

这验证迁移分支相对默认初始化的收益。若要进一步区分“相似性选源”与“任意迁移”的贡献，可保留 `--emerging-transfer on`，另用原有 `--init-mode random`。

### 2.5 新开关与旧参数的优先级

`off` 覆盖对应 PGCA 分支模式。例如同时写 `--consistency drift --recurring-adaptation off`，实际不做蒸馏；同时写 `--init-mode similarity --emerging-transfer off`，实际不做迁移。

`on` 保留旧参数的含义：`on + consistency fixed` 仍是固定蒸馏，`on + init-mode default` 仍是默认初始化。原有数值参数合法性校验保留，不会因为关闭模块就接受非法数值。

## 3. 按你给出的预算运行五组实验

在 Ubuntu 项目根目录、原训练 Python 环境执行。先定义公共参数，数值与本次提供的完整命令一致。这里是 Bash 数组，不是 Windows PowerShell 语法。

```bash
REID_MODULE_COMMON=(
  --stream-config "$REID_STREAM"
  --reference-checkpoint "$REID_CLIP"
  --device cuda --amp
  --epochs 10 --iterations-per-epoch 100
  --batch-size 32 --num-instances 4
  --prototype-batch-size 128 --eval-batch-size 128
  --workers 0 --seed 42
  --adapter-lr 0.0003 --head-lr 0.0003 --weight-decay 0.0001
  --lambda-tri 1.0 --triplet-margin 0.3
  --consistency drift --lambda-con 1.0 --gamma 1.0
  --init-mode similarity --alpha 0.5 --delta 0.5
  --control-summary ecpm --routing-summary ecpm --beta 0.5
  --evaluate --eval-split test --eval-gallery both
  --checkpoint-every 1
)
```

以下每条命令单独执行；每组使用不同的全新目录。完整组使用新名称，避免覆盖你之前的 `full_seed42`。也可以继续使用原完整命令，只需确保输出目录为空。

```bash
# 完整方法，四项默认开启
python -u train_category_progressive.py "${REID_MODULE_COMMON[@]}" \
  --output-dir "$REID_RUN_ROOT/modules_full_seed42"

# 1. 关闭 Category clustering
python -u train_category_progressive.py "${REID_MODULE_COMMON[@]}" \
  --output-dir "$REID_RUN_ROOT/modules_no_clustering_seed42" \
  --category-clustering off

# 2. 关闭累计 Prototype evolution
python -u train_category_progressive.py "${REID_MODULE_COMMON[@]}" \
  --output-dir "$REID_RUN_ROOT/modules_no_evolution_seed42" \
  --prototype-evolution off

# 3. 关闭熟悉类别自适应分支
python -u train_category_progressive.py "${REID_MODULE_COMMON[@]}" \
  --output-dir "$REID_RUN_ROOT/modules_no_recurring_seed42" \
  --recurring-adaptation off

# 4. 关闭新类别迁移分支
python -u train_category_progressive.py "${REID_MODULE_COMMON[@]}" \
  --output-dir "$REID_RUN_ROOT/modules_no_transfer_seed42" \
  --emerging-transfer off
```

如需固定第 2 号可见 GPU，可在每条 `python` 命令前加 `CUDA_VISIBLE_DEVICES=2`，与原先使用方式一致。

同时关闭四项的额外对照：

```bash
python -u train_category_progressive.py "${REID_MODULE_COMMON[@]}" \
  --output-dir "$REID_RUN_ROOT/modules_all_off_seed42" \
  --category-clustering off --prototype-evolution off \
  --recurring-adaptation off --emerging-transfer off
```

四项全关仍保留冻结参考编码器、持久类别 Adapter、身份原型提取、CE＋Triplet 和单中心路由，不等于删除整个模型结构。

现有消融套件也增加了四个可选配方：`no_category_clustering`、`no_prototype_evolution`、`no_recurring_adaptation`、`no_emerging_transfer`，可通过 `tools/run_category_ablations.py --experiments` 显式选择。未指定实验名称时仍使用原来的九组配方，不自动扩大旧套件。

## 4. 如何确认开关确实生效

- `run_config.json`：`settings` 保留用户参数，`module_switches` 记录四项开关，`effective_pgca` 记录最终使用的 PGCA 模式。
- 终端 `stage_training_start`：显示开关及 `effective_consistency`、`effective_initialization`。
- `progress.jsonl`：运行开始、阶段准备事件记录开关；ECPM 提交摘要记录两个 ECPM 开关。
- `evaluation_summary.json` / `stage_results.json` / `stage_results.md`：每阶段记录开关和 PGCA 实际模式，仍输出各类别 mAP、R1、旧类别遗忘。

关闭聚类时 `finch_runtime=null`，每类别只有一个模式，距离矩阵字节数为零。关闭演化时再次出现类别的摘要身份数只包含最近阶段，但累计身份总数继续增加。

已有验证诊断工具会保留非目标分支的开关；如果显式要求诊断 sources 或 drift，它会按诊断计划重新设置目标分支模式，这是诊断实验本身的目的。

## 5. 恢复与结果比较

本次源码更新改变精确恢复指纹。旧代码启动的未完成训练不要直接切换到新代码恢复，也不要删除检查绕过；本版本新建的实验可以按原方式恢复：

```bash
python -u train_category_progressive.py \
  --output-dir "$REID_RUN_ROOT/modules_no_evolution_seed42" --resume
```

恢复会读取已保存开关，不必重写原命令。恢复时更改任一开关会报错，避免同一实验中途更换方法。要做另一组消融，必须使用独立输出目录，从相同参考权重和数据流重新训练。

比较相同阶段、相同类别、相同图库和路由协议的结果。主要看 `per_dataset / prototype` 下全部已见类别 mAP、R1、旧类别平均遗忘，同时看逐类别变化。T1 没有历史类别，因此演化、熟悉类别蒸馏和历史迁移的差异主要要在后续阶段判断。

模块间有依赖：更换原型会改变 PGCA 控制和类别路由，单项关闭测量的是该部分在完整系统中的总贡献，不是独立因果效应。仅比较完整蒸馏与完全关闭蒸馏，不能证明漂移权重优于固定权重。

## 6. 实现位置与进度

本次只修改代码、测试和说明，未生成服务器上传补丁包。

| 文件 | 改动 |
|---|---|
| `train_category_progressive.py` | 四项 CLI 开关、默认值、PGCA 有效配置、日志与恢复校验 |
| `reid/memory/finch_modes.py` | 关闭聚类时的单中心聚合 |
| `reid/memory/ecpm_modes.py` | 累计/仅最近阶段摘要选择，开关绑定、候选及断点验证 |
| `reid/memory/prototype_views.py` | 旧 identity_mean 使用端也遵循演化开关 |
| `reid/evaluation/stage_reporting.py` | 阶段可读报告记录开关及实际模式 |
| `tools/run_category_ablations.py` | 增加四个具名配方，保持旧默认套件 |
| `tools/diagnose_pgca_validation.py` | 诊断时遵循非目标 PGCA 分支开关 |
| `tests/test_module_ablations.py` | 默认等价、真实分支跳过、四项单关/全关、恢复及评估测试 |

开发记录：实现及验证完成，共 103 项相关测试通过。其中新增开关测试 9 项、诊断分支兼容测试 1 项、原有功能回归 93 项。包含循环验证四项分别关闭和全关、四种 ECPM 组合、默认与显式全开数值一致，以及中断恢复后的模型参数和检索结果一致。测试使用本地小模型/合成图片，证明已覆盖场景的工程行为，不代表实际数据上的消融收益。

测试记录：`module_ablation_switch_tests.txt`、`module_ablation_diagnostic_tests.txt`、`module_ablation_regression_tests.txt`；参数帮助：`module_ablation_cli_help.txt`。
