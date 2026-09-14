# 第 6 步：PGCA 熟悉类别的漂移控制特征蒸馏

日期：2026-09-13。已实现熟悉类别分支；陌生类别的相似性迁移属于第 7 步，完整阶段调度、epoch 内恢复及正式路由评估仍未实现。

## 1. 大白话解释

T1 训练过 person/a、person/b，T2 来了 person/c、person/d。虽然身份都是新的，但物体类别 person 是熟悉的。

在 T2 训练开始前，把 person Adapter 的旧版本连同参考视觉塔复制为一个冻结教师。训练时，同一张 person/c 图片分别交给旧教师和正在学习的学生：CE＋Triplet 帮助学生区分新身份，一致性损失提醒学生不要把旧的 person 表达方式改变得过快。

这里**不会拿 person/a、person/b 的旧图片回放**。旧教师看到的也是当前 person/c、person/d 的图片。旧身份原型仅用于 ECPM 计算漂移，原型没有作为样本进入蒸馏损失。

如果 ECPM 显示新旧类别中心变化较小，就给旧教师较大的约束权重；变化较大则减小约束，给学生更多适应空间。权重每阶段计算一次，整个阶段保持固定。

T2 的 panda 是新类别，本步只让它学习 CE＋Triplet，没有源类别蒸馏，也暂未实现相似类别 Adapter 迁移。

## 2. 公式和三种模式

熟悉类别 \(c\) 的特征一致性为：

\[
\mathcal L_{con}^{c}=\frac{1}{B}\sum_i\left[1-\cos\left(f_{student}^{c}(x_i),
\operatorname{stopgrad}(f_{teacher}^{c}(x_i))\right)\right].
\]

师生使用**同一个已经增强的输入张量**；不单独给教师重新随机裁剪或翻转。教师 eval、no_grad，余弦损失转换到 FP32 计算；学生保留梯度。

\[
\lambda_c=\lambda_{con}\exp(-\gamma d_c),\qquad
\mathcal L_c=\mathcal L_{CE}^{c}+\lambda_{tri}\mathcal L_{Triplet}^{c}+\lambda_c\mathcal L_{con}^{c}.
\]

| 模式 | 熟悉类别一致性权重 | 用途 |
|---|---|---|
| `off` | 0 | 原基础训练目标 |
| `fixed` | `lambda_con` | 固定蒸馏权重对照 |
| `drift` | `lambda_con * exp(-gamma * d)` | 论文中的漂移控制分支 |

新类别权重始终为 0，缺席类别不参与当前训练。`gamma=0` 使 drift 退化为 fixed；`lambda_con=0` 使其退化为基础训练。所有有效权重均为 0 时不创建教师，也不运行教师前向。

默认超参数 `lambda_con=1`、`gamma=1` 只是工程起点，尚未经验证集调参。API 默认 `off`，保持旧调用方式；新 PGCA 单阶段 CLI 默认 `drift`，避免误以为开启了完整方法但实际没启用。

## 3. 实现与教师快照

- [损失与权重配置](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/loss/pgca.py)：`FeatureConsistencyLoss`、`PGCAConsistencyConfig`、`consistency_weight`。
- [教师与阶段上下文](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/adaptation/pgca.py)：`FrozenCategoryTeacher`、`RecurringCategoryConsistency`。
- [Trainer](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/trainer_category_progressive.py)：可选接入一致性损失、权重和日志。
- [单阶段运行工具](E:/Multi_modal_Code/CVPR2026-VLADR-main/tools/train_pgca_recurring_stage.py)：准备 ECPM 候选、训练、成功后提交并保存。

所有 recurring 教师均在创建当前阶段分类头、训练任何当前类别之前冻结。学生继续使用原 Adapter 的参数对象；教师使用独立复制的参数，没有在学生活跃计算图上替换或覆盖参数。

首版使用**一份完整视觉塔与历史 Adapter bank 的独立副本**，不是每个 recurring 类别复制一个完整塔。为了实现清晰，副本包含当时已注册的 Adapter；前向仅允许 recurring 类别。所有教师参数冻结且不进入优化器，阶段结束释放教师。复制操作不消耗随机数，因此不会改变后续分类头/新 Adapter 初始化。

这会增加内存和计算：标准 ViT-B/16 本次验证中 T2 教师张量约 348 MB（十进制，包含两个历史 Adapter），另外还有临时前向工作区。日志记录真实 `tensor_bytes`；不能把本实现描述成“只增加一个小 Adapter 的内存”。教师完整状态在 epoch 前后及阶段结束做哈希核对，前向检查冻结/eval/无梯度策略。

## 4. Python 接法

```python
from reid.loss.pgca import PGCAConsistencyConfig
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig
from lreid_dataset.category_stream_loaders import build_stage_loaders

# model / memory / stream 是前一完整阶段恢复出的同一实验状态。
stage = stream.stage("t2")

# 注意：先 prepare，暂不 commit。此时 memory 仍保存阶段开始前的旧状态。
candidate = memory.prepare_stage(model, stage, batch_size=64)
train_transform, reference_transform = model.make_transforms()
loaders = build_stage_loaders(stage, batch_size=32, num_instances=4,
    train_transform=train_transform, reference_transform=reference_transform)

trainer = CategoryProgressiveTrainer(
    model, stage, loaders,
    CategoryTrainingConfig(epochs=10, iterations_per_epoch=100, amp=True),
    log_path="runs/pgca/t2/training.jsonl",
    consistency=PGCAConsistencyConfig(mode="drift", lambda_con=1., gamma=1.),
    ecpm_memory=memory, ecpm_candidate=candidate,
)
for epoch in range(trainer.config.epochs):
    trainer.train_epoch(epoch)
summary = trainer.finish_stage()
memory.commit_stage(candidate)
# 后续保存 model 与 memory；下面的 CLI 已提供对应的组合保存方式。
```

上述 AMP 示例要求 CUDA；CPU 上用 `amp=False`。训练仍要求单类别 P≥2、K≥2。

`drift` 必须提供完整旧 ECPM 及其未提交的当前候选。校验会检查参考签名、阶段、身份覆盖、旧快照、模式公式和漂移，拒绝手工随意传一个漂移数或使用已提交候选。权重在构造 Trainer 时复制为数值，之后修改外部候选或日志副本不会改变已固定的权重。

`fixed` 和 `off` 可不提供 ECPM；若提供则成对传入并校验。fixed 未提供 ECPM 时，熟悉类别的漂移日志为 None，含义是未计算，不是漂移为 0。

Trainer 本身不提交 ECPM。调用方应在训练成功后提交；训练失败后不能把当前候选当作完成状态。`validate_candidate` 是本步为 PGCA 新增的非提交校验接口。

## 5. 单阶段命令

在仓库根目录运行，替换为实际数据协议。仓库示例图片是占位路径。

```powershell
python tools/train_pgca_recurring_stage.py --stream-config config/my_stream.json --stage-id t1 --output-dir runs/pgca/t1 --device cuda --amp --consistency drift --lambda-con 1 --gamma 1

python tools/train_pgca_recurring_stage.py --stream-config config/my_stream.json --stage-id t2 --previous-checkpoint runs/pgca/t1/completed_stage.pt --output-dir runs/pgca/t2 --device cuda --amp --consistency drift --lambda-con 1 --gamma 1
```

T1 没有历史类别，自动只训练基础损失，不创建教师。T2 从前一组合检查点恢复，使用其中对应的模型和 ECPM；不允许另外混入一份 ECPM。

如果之前保存的是第 3 步基础训练模型和第 5 步 ECPM，可显式导入：

```powershell
python tools/train_pgca_recurring_stage.py --stream-config config/my_stream.json --stage-id t2 --previous-checkpoint runs/baseline/t1/completed_model.pt --previous-memory runs/ecpm/t1.pt --output-dir runs/pgca/t2 --device cuda --amp
```

两份输入必须对应同协议的紧邻前一完成阶段。第 4 步身份原型文件需要先按第 5 步升级，不能直接当作完整 ECPM。

消融时分别使用 `--consistency off`、`--consistency fixed`、`--consistency drift`，并使用独立输出目录。新 CLI 即使 off 也会维护 ECPM，以保持实验流程一致；若只想跑完全无 ECPM 的基础训练，原 `train_category_baseline_stage.py` 仍可用。

支持 `--epochs`、`--iterations-per-epoch`、`--batch-size`、`--num-instances`、`--prototype-batch-size`、`--workers`、`--seed`、两种学习率、Triplet 配置和 FINCH 分块大小。后续阶段沿用保存状态的 FINCH 设置。第一阶段可显式传本地 `--reference-checkpoint`，不会自动下载。

## 6. 日志、检查点与中断恢复

输出目录包含：

- `run_config.json`：运行参数、协议指纹及参考签名。
- `training.jsonl`：阶段开始、每步、每 epoch、结束或错误事件，逐条刷新。
- `training_summary.json`：训练统计及提交后的 ECPM 摘要。
- `completed_stage.pt`：训练成功后将模型、对应 ECPM、PGCA 配置与教师摘要一起原子保存。

启用 fixed/drift 后，每个类别日志包含 `prototype_drift`、`consistency_weight`、`consistency`、`weighted_consistency`，同时保留 CE、Triplet 和总损失。零权重/新类别不执行教师前向，日志中的一致性 0 是跳过计算的占位值，不是测量出的教师差异为零。

阶段开始事件包含 recurring 类别、全阶段固定权重、教师整体哈希和各 recurring Adapter 哈希，可以核查教师来自哪个阶段开始模型。每步总损失为 CE＋加权 Triplet＋加权一致性；各当前类别的平均损失相加，仍采用相等训练预算。

恢复只接受完整 `completed_stage.pt`，从下一阶段开始。旧检查点不会被失败的新阶段覆盖；临时文件不代表完成。若中断时只有日志，没有完整组合检查点，应保留原输出用于排查，从前一完整阶段在新的输出目录重跑当前阶段。若仅最后 JSON 摘要写失败而组合 `.pt` 已成功，可直接使用该完整检查点继续。

当前未保存优化器、RNG、教师和数据迭代器的 epoch 内恢复状态；日志中的已完成迭代数不能当作可直接恢复的游标。真实 epoch 内精确恢复属于第 8 步。API 在失败后也禁止继续使用该 Trainer，应恢复前一完整模型。

## 7. 本步验证

- 18 项本步测试通过，前 5 步 129 项回归通过。
- 手算余弦损失与权重公式、教师 stop-gradient、一致性损失单独向学生 Adapter 传递非零有限梯度均通过。
- 相同初始状态与随机种子下，零权重训练和原默认基础 Trainer 的参数更新逐元素一致；gamma=0 的 drift 与 fixed 参数更新逐元素一致。
- 师生输入张量相同；新类别不调用教师；旧图片不存在也可训练当前阶段；错误教师特征在当前优化器更新前终止，不提交 ECPM。
- 真实 CLIP ViT-B/16、CUDA＋AMP、合成三阶段，每阶段 2 epoch × 2 次更新通过。T2 教师只用于 person，T3 分别用于 person 和 vehicle；教师来自正确的阶段开始版本，教师特征和参考特征逐元素不变，缺席 Adapter 不变，当前 Adapter 更新。
- 每个阶段保存并恢复模型和 ECPM，删除旧图片后继续下一阶段；基础 checkpoint＋ECPM 导入以及组合 checkpoint 接续均验证通过。

报告：[本步测试](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/pgca_recurring_tests.txt)、[回归测试](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/pgca_recurring_regression.txt)、[真实 CLIP 短训练](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/pgca_recurring_smoke.json)。逐阶段训练日志为 `docs/pgca_recurring_smoke_t1.jsonl`、`_t2.jsonl`、`_t3.jsonl`。

这是实现正确性验证，尚不是正式 ReID 精度、防遗忘效果或漂移权重优越性的证据。第 5 步已记录中心漂移对对称新模式不敏感的局限，后续仍需 fixed/drift 消融验证其有效性。

下一步是第 7 步：PGCA 陌生类别的相似性迁移初始化。开发恢复位置见 [执行记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step6_progress.md)。
