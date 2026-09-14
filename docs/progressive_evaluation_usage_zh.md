# 第 9 步：硬路由、终身检索评估与消融

日期：2026-09-14。

## 1. 实际推理在做什么

对一张测试图片，先用冻结参考编码器提取 `z=Norm(F0(x))`，对所有已见类别计算：

```text
score(c) = beta * cos(z, m_c) + (1-beta) * max_k cos(z, g_c,k)
predicted_category = argmax_c score(c)
descriptor = Norm(F(x; Adapter[predicted_category]))
```

类别中心与每个模式分别做余弦匹配，模式项取当前类别内的最大值。完全相同的分数按类别键排序打破平局。一个 batch 可以预测出多个类别，代码按预测分组执行 Adapter，再恢复原图像顺序。

`RoutedEncoder(images)` 只接收图像张量，不接收类别、身份、来源数据集或路径。query 与 gallery 独立路由；即便某个评估集只有行人，候选仍然包含所有已见类别。当前是两次视觉前向：参考路由一次、所选 Adapter 一次，没有把最终参考向量直接送进内部 Adapter。

路由代码：`reid/evaluation/prototype_router.py`。正式结果为 `prototype`；`oracle` 使用真实类别选 Adapter，只用来诊断表征质量，不作为无类别标签方法的主结果。

## 2. 启动逐阶段评估

在第 8 步入口上增加 `--evaluate`：

```powershell
python train_category_progressive.py --stream-config "D:\ReIDData\stream.json" --output-dir "logs\full_run" --device cuda --amp --batch-size 32 --num-instances 4 --workers 0 --evaluate --eval-split test --eval-gallery both --eval-batch-size 128 --beta 0.5
```

`--evaluate` 默认关闭，以保持原训练入口行为。启用后，每次完成阶段训练并提交 ECPM，都会先评估当前所有已见类别，再开始下一阶段。配置恢复沿用第 8 步：

```powershell
python train_category_progressive.py --output-dir "logs\full_run" --resume
```

每个最终会到达的类别都需要在所选 split 下配置固定的 query/gallery。缺少评估集时，入口会在训练前报错，不会把少评的类别从宏平均里悄悄删掉。第 1 步审计继续保证训练身份与评估身份不重叠、query 有有效 gallery 正样本。示例 JSON 的图像仍是占位路径，正式使用前需替换为实际数据。

评估失败时，断点已保存该阶段的训练结果和原型提交，同时记录 `pending_evaluation`。重新 `--resume` 只重评该阶段，不重训、不重复追加身份。评估日志先写入而断点保存失败时，第 8 步日志事务会归档多余尾部并回滚。评估前后保存/恢复所有 RNG，避免 DataLoader 构造或评估操作改变下一阶段训练。

## 3. 两种 gallery 协议

| `--eval-gallery` | 定义 |
|---|---|
| `per_dataset` | 每个 EvaluationView 保留自己的原始 gallery，类别内多个数据集也不自动合并 |
| `mixed` | 合并当前已见类别、当前 split 的全部 gallery，按实际图像路径去重；每个 query 仍沿用所属评估集的摄像头排除规则 |
| `both` | 分别输出以上两种结果 |

两种协议都对每张图像从全部已见类别中路由。`per_dataset` 使用数据集固有的候选图库范围，但不会用类别标签替正式路由选 Adapter。

检索使用归一化描述符的余弦相似度，按 query 逐行计算，避免分配完整 Q×G 距离矩阵。相同图片从 gallery 排除；`cross_camera` 还排除同身份同摄像头图像，`exclude_self` 只排除自身。并列相似度保持 gallery 原顺序。不同数据源局部 PID 不会被误当作同一身份，匹配使用已审计的完整身份键。

混合 gallery 指标能反映不同 Adapter 输出是否具有可比性。不要把按原始 gallery 得到的 mAP 写成混合类别检索结果。

## 4. 输出怎么读

完整输出目录继续保留 `reference.pt`、`latest.pt` 和训练日志，新增：

- `evaluation.jsonl`：每阶段完整评估事件，参与断点日志回滚。
- `evaluation_summary.json`：便于读取的阶段报告及终身矩阵，从有效 checkpoint 中的评估历史生成。
- `latest.pt` 内的 `evaluations` 和 `pending_evaluation`：恢复所用的权威状态。

每阶段的 `retrieval` 按 gallery 协议区分，再分别列出 oracle/prototype 的数据集指标、类别指标和类别宏平均。每个类别内多个评估集按 query 数量加权；不同类别之间等权平均。mAP 和 Rank1 均为百分数。

`oracle_minus_prototype` 是逐类别的检索差距，可辅助判断退化是否伴随路由错误。它是诊断差值，不强制非负；oracle 不被代码假定为每个实验中的严格数学上界。

`routing` 提供：

- 路由准确率和逐类别准确率。
- 混淆矩阵：行是真实类别，列是预测类别，类别顺序单独给出。
- 同一测试身份的多张图像是否全部路由一致，以及多数预测所占比例的身份平均值。

路由诊断将同一路径的 query/gallery 图像去重；跨图像一致性只统计至少有两张不同图片的身份。全都选错同一个类别也可以有 100% 一致性，所以一致性不能代替准确率。

终身矩阵使用 `performance[t][c]` 保存各阶段各类别性能；类别尚未出现时为 JSON `null`，不是 0。遗忘计算为：

```text
forgetting[t,c] = max_{s<=t, c已出现} performance[s,c] - performance[t,c]
```

mAP、Rank1、oracle、prototype、两种 gallery 分别统计。新类别首次出现时遗忘为 0；宏平均只覆盖该阶段已见类别。评估历史必须从 T1 连续到当前阶段，不能把不同 split、beta 或路由原型设置拼在一起。

资源报告包含全量身份原型、模式/类别中心及历史快照字节、Adapter 张量字节、各类别 FINCH 耗时与距离块大小、oracle/prototype 前向耗时、评估总耗时和 CUDA 峰值已分配内存。CUDA 峰值包含该评估进程现有模型张量，并非纯路由临时内存；CPU/native FINCH 峰值未直接测量，距离块大小也不等于进程峰值。计时需在相同硬件和批大小下比较。

## 5. 单独评估一个已完成阶段的检查点

```powershell
python tools/evaluate_category_progressive.py --run-dir "logs\full_run" --output "logs\full_run\evaluation_extra.json" --device cuda --gallery both --split test --summary ecpm --beta 0.5
```

该工具不继续训练，也不修改原 checkpoint。只接受完成阶段后的状态，拒绝尚在训练的学生 bank，避免用更新中的学生配旧原型做“该阶段正式结果”。它可读取第 8 步保存的已完成连续阶段检查点；报告记录新的评估环境，不宣称跨代码版本继续训练的精确性。

把 `--summary` 改为 `identity_mean`，即可在同一个检查点上比较简单类别均值路由。单个末阶段检查点只能重评该阶段，不能重建已经覆盖掉的历史模型；完整终身矩阵应使用启用了逐阶段评估的训练过程。

## 6. 九个可执行消融配方

`tools/run_category_ablations.py` 为每个实验创建独立输出目录，固定数据、种子、训练预算及基础超参数。每个实验都输出 oracle 和 prototype，所以不需要为 oracle 再训练一次。

| 实验名 | 相对完整方法的改动 |
|---|---|
| `full` | ECPM＋漂移蒸馏＋相似性初始化＋完整原型路由 |
| `persistent_baseline` | 无一致性、默认初始化、身份均值控制/路由 |
| `fixed_consistency` | 将漂移权重改为固定 `lambda_con` |
| `default_initialization` | 新类别从默认零残差 Adapter 开始 |
| `random_historical_source` | 有历史时随机复制一个历史 Adapter，不使用相似度阈值选源 |
| `global_only_initialization` | 设 `alpha=1`，初始化只按类别中心相似度选源，仍使用同一阈值 |
| `mean_control_only` | 学习控制用身份均值，路由仍用 ECPM |
| `mean_routing_only` | 学习控制仍用 ECPM，只替换路由 |
| `mean_control_and_routing` | 控制与路由均用身份均值 |

身份均值定义是 `Norm(mean(所有已到达身份原型))`，每身份等权，不按图像数量加权；该均值作为一个模式。控制对照会同时改变漂移和迁移匹配所使用的摘要。实现通过原始、已验证 ECPM 候选派生控制视图，不修改基础记忆，不重复读取图片。

为支持控制/路由分离，上述对照仍计算和保存 FINCH 记忆。因此这是机制消融，不能将这些运行的实际内存/耗时声称为删除 FINCH 后的最小开销。

基础 JSON 示例为 `config/category_ablation_example.json`。其中 `stream_config` 路径相对执行命令的工作目录；样例 P=2、K=2 只是配合教程的两个身份。请按真实数据调整，并先在验证集确定超参数。

先生成实验计划：

```powershell
python tools/run_category_ablations.py --base-config config/category_ablation_example.json --output-dir logs/ablation_suite
```

正式执行全部实验，或中断后继续：

```powershell
python tools/run_category_ablations.py --base-config config/category_ablation_example.json --output-dir logs/ablation_suite --execute
```

可用 `--experiments full fixed_consistency` 指定子集。同一输出目录的计划不可更改；已存在的实验从自身 `latest.pt` 恢复。套件的 `ablation_results.json` 汇总完成实验的终身矩阵，计划写在 `ablation_plan.json`。

## 7. 两项机制诊断：用验证集测真实效果

在目标阶段开始前保留一个阶段边界，例如先运行到 T1 提交：

```powershell
python train_category_progressive.py --stream-config "D:\ReIDData\stream.json" --output-dir logs/diagnostic_boundary --device cuda --amp --batch-size 4 --num-instances 2 --max-stages 1
```

来源诊断枚举该边界的所有历史来源，并与默认初始化和相似性选择比较：

```powershell
python tools/diagnose_pgca_validation.py --run-dir logs/diagnostic_boundary --output logs/source_diagnosis.json --device cuda --kind sources
```

漂移诊断比较动态权重和固定权重组：

```powershell
python tools/diagnose_pgca_validation.py --run-dir logs/diagnostic_boundary --output logs/drift_diagnosis.json --device cuda --kind drift --weights 0 0.1 1 10
```

每个变体从完全相同的历史 Adapter、旧 ECPM、当前候选、采样种子、增强 RNG 和训练预算开始。工具只评估 `validation`，没有切换到 test 的参数；缺少已见类别验证集会报错。原始边界 checkpoint 不被修改。

来源输出包含真实来源、原型相似度、验证集 oracle mAP/Rank1，以及相对默认初始化的 mAP 增益；权重输出包含实际漂移、有效蒸馏权重及训练后的验证指标。它们用来检查“高相似度是否伴随更好迁移”“不同漂移对应的较好权重是否有规律”，不以公式曲线代替实验证据。

若同阶段有多个新类别，来源变体会同时对这些新类别使用该历史来源，并分别报告结果；需要孤立研究单个新类别时，应设计只引入一个新类别的小规模验证流。固定权重变体同样同时作用于当前熟悉类别。观察结论应跨阶段和种子重复验证。

诊断按“已完成变体”保存输出；重复命令会跳过已完成变体，当前尚未完成的变体从共同起点重跑，不沿用主训练任务的阶段内断点。不要把这个输出当作可用于正式测试调参的选择器。

## 8. 验收与正式实验边界

检查覆盖路由公式、混合 batch 顺序、身份均值公式、gallery 干扰、遗忘定义、无未来图像读取、评估不扰动训练、评估中断恢复、9 个消融执行与恢复、验证集来源/权重枚举及旧路径回归。

`docs/step9_regression_tests.txt` 保存完整回归记录；`docs/progressive_evaluation_smoke.json` 和对应 `_metrics.json` 保存真实 CLIP 三阶段短实验的状态及指标。生成图像只用于核验工程流程，不能据其指标判断方法有效或是否足够投稿。

目前未知类别不会被拒识，任何输入都会选择一个已见类别。固定总原型预算、类别顺序/缺席间隔/不平衡的大规模实验和正式多种子性能结果，需要在真实数据和具体实验预算确定后开展。至此九步的基础训练、推理、评估和必要消融接口已经具备。
