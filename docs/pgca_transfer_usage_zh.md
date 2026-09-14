# 第 7 步：PGCA 陌生类别的相似性迁移初始化

日期：2026-09-13。至此 ECPM 和 PGCA 两个分支已实现；第 8 步连续阶段调度/精确恢复、第 9 步正式路由/终身评估仍待完成。

## 1. 大白话理解

假设 T1 学过 person 和 vehicle，T2 首次出现 panda。系统先用 panda 当前身份的原型概括它，再与 T1 已有类别的原型比较。

如果某个历史类别与 panda 的综合相似度足够高，就把那个类别**在 T2 开始前**的 Adapter 参数复制给 panda，作为学习起点；否则使用默认零残差 Adapter。此后 panda 独立学习当前身份的 CE＋Triplet，不需要一直模仿这个来源。

允许 vehicle 这样的本阶段缺席类别成为来源，因为它已经在历史阶段学习过。不允许同在 T2 首次出现的 panda、boat 互相借用。即使 person 在 T2 也要继续学习，给 panda 的仍然是 T1 person 的版本，而不是已经在 T2 更新过的版本。

复制后两个 Adapter 初始数值相同，但参数存储独立。训练 panda 不会通过共享参数修改来源；如果来源 person 同时训练，它按自己的数据与第 6 步蒸馏目标更新。

## 2. 如何比较类别

\(m_c\) 是新类别中心，\(g_{c,k}\) 是新类别第 k 个模式；\(m_j^{old}\)、\(g_{j,l}^{old}\) 都来自阶段开始前的历史记忆。

\[
S_{mode}(c,j)=\frac{1}{K_c}\sum_k\max_l\cos(g_{c,k},g_{j,l}^{old}),
\]

\[
s(c,j)=\alpha\cos(m_c,m_j^{old})+(1-\alpha)S_{mode}(c,j).
\]

第一项看整体方向是否相似；第二项是对**每一个新类别模式**，在该历史类别中找最接近的模式，然后平均这些最高分。

这是新类别 → 历史类别的单向覆盖，不是反过来计算，也不是强制一对一匹配。多个新模式可以匹配同一个历史模式。

例如新模式为 `(1,0)`、`(0,1)`，历史只有 `(1,0)`：两个新模式的最高分为 1、0，因此覆盖得分为 0.5。若错误地反向计算，只有一个历史模式要匹配，就会得到 1。方向会改变选源结论。

\[
j^*=\arg\max_{j\in seen\_before}s(c,j),\qquad
A_c=\begin{cases}
\operatorname{copy}(A_{j^*}^{old}),&s(c,j^*)\geq\delta,\\
A_0,&\text{否则}.
\end{cases}
\]

阈值使用 **≥**。历史为空则直接回退，不对空集合求最大值。分数完全相同的来源按类别名称排序选择；不额外引入“近似相等”容差。`alpha` ∈ [0,1]，`delta` ∈ [-1,1]，包括负余弦分数的情况。评分使用 CPU FP32，并显式关闭外层 CPU autocast，避免阈值判断被降精度改变。

## 3. 实现接口与默认行为

核心位于 [reid/adaptation/pgca.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/adaptation/pgca.py)：

- `PGCATransferConfig`：`mode='default'/'similarity'`、`alpha`、`delta`。
- `prototype_similarity`：全局、模式覆盖、综合分数。
- `select_transfer_source`：纯评分/阈值选源，返回全部候选和选择原因。
- `PrototypeGuidedInitialization`：校验阶段与 ECPM 候选、保存历史 Adapter CPU 副本、准备决策、应用初始化。

| 入口 | 默认初始化模式 | 默认熟悉类别蒸馏 |
|---|---|---|
| Trainer Python API | `default` | `off` |
| 原第 6 步 `train_pgca_recurring_stage.py` | `default` | `drift` |
| 新 `train_pgca_stage.py` | `similarity` | `drift` |

`alpha=0.5`、`delta=0.5` 为当前工程默认值，不是经过真实验证集选出的最优超参数。关闭迁移应使用 `--init-mode default`，不是把阈值设成非法值；即便 delta=1，完全一致的类别仍可能通过阈值。

默认零残差沿用原实现：升维层为零，降维层按原规则初始化。它不是把所有 Adapter 参数都设为零。保留原有的排序注册和分类头初始化顺序；T1 空历史时，similarity 回退与默认基础训练的参数更新已验证逐元素一致。

## 4. Trainer 接法

```python
from reid.adaptation.pgca import PGCATransferConfig
from reid.loss.pgca import PGCAConsistencyConfig
from reid.trainer_category_progressive import CategoryProgressiveTrainer

# model、memory 为前一完整阶段的同一实验状态。
stage = stream.stage("t2")
candidate = memory.prepare_stage(model, stage, batch_size=64)
# loaders、training_config 按第 3/6 步创建；只含当前训练图像。
trainer = CategoryProgressiveTrainer(
    model, stage, loaders, training_config,
    log_path="runs/pgca/t2/training.jsonl",
    consistency=PGCAConsistencyConfig('drift', lambda_con=1., gamma=1.),
    initialization=PGCATransferConfig('similarity', alpha=.5, delta=.5),
    ecpm_memory=memory, ecpm_candidate=candidate,
)
print(trainer.initialization.metadata())
for epoch in range(trainer.config.epochs):
    trainer.train_epoch(epoch)
training_summary = trainer.finish_stage()
memory.commit_stage(candidate)
```

不要在构造此 Trainer 前手动注册新类别或提前提交 ECPM。similarity 模式要求新目标尚未注册；重复应用初始化会报错，以避免覆盖已经开始学习的 Adapter。

当前顺序是：校验输入 → 保存历史源参数/准备所有选源结果 → 冻结 recurring 教师 → 按类别排序创建/初始化 Adapter 与当前临时分类器 → 构造新阶段优化器 → 训练。

所有源快照在任何当前类别训练前就已保存。源参数复制只涉及 Adapter，不复制分类器、优化器状态或原型。新类别使用自己的 ECPM 候选和新建分类头，训练损失仍是 CE＋Triplet；第 6 步的同类别教师只用于 recurring 类别。

高级独立调用可构造 `PrototypeGuidedInitialization(...)`，先读 `metadata()`，再调用 `apply(model)`。它会使用捕获的 CPU 副本，即使活的来源 Adapter 在这两次调用之间发生变化，复制的仍是旧版本。不要对同一模型先独立 apply，再让 similarity Trainer 重复初始化；常规训练交给 Trainer 统一执行即可。

所有候选和源快照在目标修改前校验。若发生显存不足或回调异常等运行错误，仍应从前一完整检查点重建当前阶段，不把内存中的部分初始化视为可继续训练的完成状态。当前没有实现整个初始化过程的故障回滚事务。

## 5. 单阶段命令

新入口 [tools/train_pgca_stage.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/tools/train_pgca_stage.py) 默认同时开启 drift 和 similarity，复用第 6 步单阶段训练/保存实现，避免维护两套训练循环。

在仓库根目录运行，将 `config/my_stream.json` 替换成实际协议。仓库示例图片仍是占位路径。

```powershell
python tools/train_pgca_stage.py --stream-config config/my_stream.json --stage-id t1 --output-dir runs/full_pgca/t1 --device cuda --amp --alpha .5 --delta .5

python tools/train_pgca_stage.py --stream-config config/my_stream.json --stage-id t2 --previous-checkpoint runs/full_pgca/t1/completed_stage.pt --output-dir runs/full_pgca/t2 --device cuda --amp --alpha .5 --delta .5
```

T1 没有来源，自动默认初始化。后续类别判断基于协议 `seen_before`，不会因为当前类别已加入候选原型而被误判成历史类别。

也可从第 6 步完成检查点接续：

```powershell
python tools/train_pgca_stage.py --stream-config config/my_stream.json --stage-id t2 --previous-checkpoint runs/recurring/t1/completed_stage.pt --output-dir runs/full_pgca/t2 --device cuda --amp
```

如果只有第 3 步模型和第 5 步完整 ECPM，则成对导入：

```powershell
python tools/train_pgca_stage.py --stream-config config/my_stream.json --stage-id t2 --previous-checkpoint runs/baseline/t1/completed_model.pt --previous-memory runs/ecpm/t1.pt --output-dir runs/full_pgca/t2 --device cuda --amp
```

两份输入必须结束于紧邻前一阶段。新旧 PGCA 组合检查点都可以恢复；新 similarity 输出类型为 `pgca_stage`，原 default 路径保持 `pgca_recurring_stage`，两者含同样的模型/ECPM完成状态结构。

用于消融的组合：

| 设置 | 含义 |
|---|---|
| `--consistency off --init-mode default` | 基础目标和默认初始化，仍维护 ECPM |
| `--consistency drift --init-mode default` | 只开熟悉类别分支 |
| `--consistency off --init-mode similarity` | 只开陌生类别分支 |
| `--consistency drift --init-mode similarity` | 两个 PGCA 分支 |

各组合使用独立输出目录；超参数应用验证集确定，不能用测试效果反复选择来源阈值。

## 6. 日志与恢复

阶段开始/结束日志、最终 summary 和组合 checkpoint 都包含 `initialization`：

```text
config: mode / alpha / delta
historical_categories / new_categories
decisions[new_category]:
  candidates[source]: global_similarity / mode_similarity / score
  best_source / best_score
  selected_source
  reason: threshold_met / below_threshold / no_history / default_mode
  initialized_adapter_sha256 / source_adapter_sha256  # similarity 应用后记录
historical_adapter_sha256
snapshot_tensor_bytes
applied
```

`best_source` 是最高分候选，`selected_source` 是实际复制来源；阈值未通过时前者可能存在、后者为 None。不要把最高分候选误读成已执行迁移。

历史 Adapter 副本仅在初始化阶段保存在 CPU；初始化完成后释放，日志仅保留哈希和分数。无新类别时不复制源参数。`snapshot_tensor_bytes` 是初始化期间的张量载荷，不是训练期常驻内存，也不含第 6 步独立教师的内存。

保存仍使用 `completed_stage.pt`：训练成功后，把模型、对应 ECPM、蒸馏摘要、初始化来源/参数和运行配置放在一个原子文件中。输出目录必须为空，不覆盖前一完整阶段。

中断恢复仍以阶段为单位。当前阶段没有完整检查点时，保留日志供排查，加载上一完整阶段并在新输出目录重跑；不要只恢复部分新 Adapter 后再次执行迁移。epoch 内优化器、教师、RNG 和采样器精确恢复由第 8 步实现。

## 7. 本步验证与方法局限

- 14 项针对性测试通过：单向模式覆盖、alpha 两端、阈值等号/负分/并列、外层 autocast 不改变判断、旧中心选源、缺席类别源、阈值回退及空历史。
- 多个新类别的来源集合不包含彼此；反转协议内当前类别顺序，复制和回退两种情况下的初始化参数与决策均一致。
- 复制所有源参数后不共享存储；源后续改变也不影响已捕获的旧快照。新类别训练不会更新缺席源，也不调用源教师。
- 前 6 步 147 项回归通过。
- 真实 CLIP ViT-B/16、CUDA＋AMP 三阶段短训练通过（每阶段 2 epoch × 2 次更新）。合成图片中 T2 panda 对 person/vehicle 的分数同为约 0.994191，按并列规则选 person，复制的是 T1 person。源/目标初始化哈希一致、存储独立；此后正常训练并保存恢复。
- T3 没有新类别，不重新初始化历史 Adapter；参考、教师、缺席 Adapter 保持，累计身份数为 4→8→12，删除旧图片后可继续。

报告：[本步测试](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/pgca_transfer_tests.txt)、[回归测试](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/pgca_transfer_regression.txt)、[真实 CLIP 双分支验证](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/pgca_transfer_smoke.json)。逐阶段日志为 `docs/pgca_transfer_smoke_t1.jsonl`、`_t2.jsonl`、`_t3.jsonl`。

合成数据的来源选择不是“panda 应当借用 person”的语义结论；这些验证只说明公式和复制/训练流程正确。**高原型相似度不保证 Adapter 迁移有益，阈值也不保证消除负迁移。** 论文对此应保留实验验证空间，后续比较默认初始化与相似性初始化的真实效果。

下一步是第 8 步：完整阶段流程、日志与断点恢复。开发恢复位置见 [第 7 步执行记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step7_progress.md)。
