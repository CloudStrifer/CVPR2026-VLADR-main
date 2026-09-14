# 第 3 步：类别内部 CE + Triplet 基础训练

本步骤已将前两步的数据流和类别模型接成可运行的基础训练链路。当前总损失只有 CE 与 Triplet，没有 ECPM、蒸馏、迁移选源、OCIA/OSAF 或文本/属性损失。

## 1. 实现文件

- [基础 Trainer](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/trainer_category_progressive.py)
- [类别内 Triplet](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/loss/category_triplet.py)
- [单阶段运行工具](E:/Multi_modal_Code/CVPR2026-VLADR-main/tools/train_category_baseline_stage.py)
- [Oracle 诊断评估](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/evaluation/category_oracle.py)
- [当前执行与中断记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step3_progress.md)

原有训练入口和前两步模型/数据层代码没有因本步骤改变。

## 2. 损失到底怎么算

对于某类别 c 的当前 batch：

1. 输入该类别 Adapter，得到原始投影 CLS 特征 f。
2. f 进入当前阶段、当前类别的临时线性分类头，计算平均 CE。
3. 将 f 做 L2 归一化，计算欧氏距离矩阵。
4. 对每个样本，在同类别 batch 内选距离最大的同身份样本作为 hardest positive，选距离最小的不同身份样本作为 hardest negative。
5. 计算带 margin 的平均 Triplet：

\[
\hat f_i=\frac{f_i}{\|f_i\|_2},\quad
d_{ij}=\|\hat f_i-\hat f_j\|_2,
\]

\[
\mathcal L_{\mathrm{tri}}^c
=\frac1B\sum_i\max(0,d_i^+-d_i^-+m),
\qquad
\mathcal L_c=\mathcal L_{\mathrm{ce}}^c+\lambda_{\mathrm{tri}}\mathcal L_{\mathrm{tri}}^c.
\]

排除样本自身所在 batch 位置作为正样本。若 P×K 采样对短身份进行有放回采样，另一位置恰好是同一图片时允许正样本距离为 0。

每个类别的 batch 都必须至少有两个身份、每身份至少两个样本。缺少有效正负样本会报错，不跨类别寻找负样本。

CE 使用 FP32 logits 计算；Triplet 在 FP32 下计算归一化、距离和损失。CUDA 可开启学生前向 AMP，并使用 GradScaler。非有限损失或梯度导致报错，不能把被跳过的更新记成成功。

## 3. 多类别如何共同训练

一个迭代的流程：

```text
从每个当前类别的 loader 各取一个 batch
    ↓
核对类别、阶段、train split、图片路径、真实身份键和局部标签
    ↓
person：平均 CE + 平均 Triplet → backward
vehicle：平均 CE + 平均 Triplet → backward
    ↓
所有当前类别梯度累加完毕后，统一 optimizer.step 一次
```

实现的目标是各类别平均损失之和，不再除以类别数。各类别参数彼此独立，因此可以逐类别反向传播，减少同时保留的计算图数量；测试已核对其结果与显式求和后反向传播一致。

每类别每阶段更新次数固定为：

```text
epochs × iterations_per_epoch
```

某类别 loader 较短时，从该类别当前阶段 loader 重新开始；不会切到历史或评估数据。不同类别图像量不同时，更新次数仍相同。每次重新遍历都推进 sampler 的 pass 编号，避免反复使用完全相同的一轮采样顺序。

本基线采用 AdamW，Adapter 和分类头分开设置学习率。默认参数是工程起点，不代表经实验选出的最优值：

| 参数 | 默认值 |
|---|---|
| epochs | 10 |
| iterations_per_epoch | 100 |
| adapter_lr / head_lr | 0.0003 / 0.0003 |
| weight_decay | 0.0001 |
| lambda_tri | 1.0 |
| triplet_margin | 0.3 |
| amp | False |
| max_grad_norm（Python API） | None，不裁剪但检查梯度有限性 |

`lambda_tri=0` 时实际目标退化为 CE；Triplet 仍会计算并记录，方便对照。

## 4. Python 调用方式

```python
from lreid_dataset.category_stream import load_category_stream
from lreid_dataset.category_stream_loaders import build_stage_loaders
from reid.models.wrapper import make_category_model
from reid.trainer_category_progressive import CategoryProgressiveTrainer, CategoryTrainingConfig

stream = load_category_stream('config/my_category_stream.json')
stage = stream.stage('t1')
model = make_category_model(device='cuda')
train_transform, reference_transform = model.make_transforms()
loaders = build_stage_loaders(
    stage, batch_size=4, num_instances=2,
    train_transform=train_transform, reference_transform=reference_transform,
)
trainer = CategoryProgressiveTrainer(
    model, stage, loaders,
    CategoryTrainingConfig(epochs=2, iterations_per_epoch=20, amp=True),
    log_path='runs/my_baseline_t1/training.jsonl',
)

for epoch in range(trainer.config.epochs):
    report = trainer.train_epoch(epoch)
    print(report)

summary = trainer.finish_stage()
```

`my_category_stream.json` 需要替换成实际数据配置。随机种子应像第 1 步文档一样设置；命令行工具已设置 Python/NumPy/PyTorch 种子。

Trainer 会完成当前类别注册、临时分类头创建、训练集合设置和 optimizer 创建，不需要调用者重复创建分类头。若传入模型还有上一阶段的临时头，会报错。

后续阶段可用同一个 model 创建新 Trainer，但必须先完成上一阶段 `finish_stage()`。该方法冻结类别参数、丢弃临时分类头、释放 Trainer 持有的 loader 与 optimizer；下一阶段新建临时头和 optimizer，重复类别的 Adapter 参数保留。

模型注册表必须包含所有 `seen_before` 类别，且不能带入未来类别。本步骤不凭注册表证明它们已训练过；单阶段命令行工具还会检查上阶段检查点的阶段号和数据协议指纹。

## 5. 可直接运行的单阶段工具

从仓库根目录运行，示例使用 PowerShell 换行：

```powershell
python tools/train_category_baseline_stage.py `
  --stream-config config/my_category_stream.json `
  --stage-id t1 `
  --output-dir runs/category_baseline/t1 `
  --device cuda --amp `
  --epochs 2 --iterations-per-epoch 20 `
  --batch-size 4 --num-instances 2
```

第一阶段使用第 2 步的本地 CLIP 权重，也可显式传 `--reference-checkpoint`。

完成后接续第二阶段：

```powershell
python tools/train_category_baseline_stage.py `
  --stream-config config/my_category_stream.json `
  --stage-id t2 `
  --previous-checkpoint runs/category_baseline/t1/completed_model.pt `
  --output-dir runs/category_baseline/t2 `
  --device cuda --amp `
  --epochs 2 --iterations-per-epoch 20 `
  --batch-size 4 --num-instances 2
```

后一阶段必须读取同一协议中紧邻的上一阶段完成检查点，不能从新 CLIP 权重开始，也不能跳过阶段。协议指纹包含配置与解析后的 manifest 元数据；改动正式划分或数据路径后，不会默认为同一条训练流。

输出目录必须为空，防止日志和旧实验混写。小数据身份数可能不足默认的 P=8，因此教程命令使用 P=2、K=2；真实实验应按身份数量设置。

仓库的 `category_progressive_example.json` 图片路径仍是占位路径，不能拿它直接进行正式训练。本步骤未转换或覆盖真实数据集 manifest。

## 6. 保存了什么，如何处理中断

单阶段工具生成：

| 文件 | 内容 |
|---|---|
| run_config.json | 参数、损失设置、数据协议和参考特征签名 |
| training.jsonl | 每次成功更新、每个 epoch 和异常事件；逐行写入并关闭文件 |
| training_summary.json | 完整阶段统计和各类别更新次数 |
| completed_model.pt | 已完成阶段的参考模型与全部类别 Adapter，临时分类头已丢弃 |
| oracle_evaluation.json | 可选的正确类别 Adapter 诊断结果 |

每个训练步日志分别记录 CE、Triplet、加权 Triplet、总损失、当前训练身份分类准确率、正负距离、活跃 Triplet 比例、样本数和梯度范数。epoch 日志增加各类别更新次数和 loader 循环次数。

`completed_model.pt` 使用临时文件写完后原子替换，并在可选评估前保存。评估失败不会丢掉已完成阶段的模型。

**本工具支持已完成阶段之间接续，不支持阶段内精确 resume。** 中途失败的 Trainer 不允许直接重复调用同一 epoch；需要从上一完成阶段在新输出目录重跑当前阶段。第 8 步才统一处理 optimizer、临时分类头、RNG、采样位置及完整终身状态恢复。

这与开发过程的额度中断是两件事：本次开发代码、测试和进度已持续写入文件，恢复额度后按执行记录接续即可，不需要重做已完成步骤。

## 7. Oracle 诊断评估

添加 `--eval-split validation` 或 `--eval-split test`，在训练完成后评估截至该阶段已出现类别的固定评估集合；默认不运行评估。诊断不用于自动选择最佳 epoch，也不根据测试集调整超参数。

也可调用：

```python
from reid.evaluation.category_oracle import evaluate_category_oracle

result = evaluate_category_oracle(model, stream.evaluations_at('t1')[0])
```

这里直接使用真实类别选择 Adapter，因此输出标记为 `routing="oracle"`。它用于诊断表征能力，不是论文要求的无类别标签完整结果；原型硬路由留到第 9 步。

检索使用归一化描述符的余弦排序，输出百分制 mAP/Rank-1，并保留第 1 步定义的规则：

- 始终排除 query 本身的图片。
- `cross_camera` 排除同身份同 camid 的 gallery。
- `exclude_self` 允许同 camid 的不同图片匹配。

旧通用评估入口默认相机过滤，不能直接表达所有新协议；因此新增小型按行计算的诊断实现，并以手工排名核对结果。大规模评估汇总、自动路由和完整消融仍属于第 9 步。

## 8. 本步骤验证结果

2026-09-13：

- 21 项新增损失/Trainer/运行工具/评估测试通过：[测试日志](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_training_tests.txt)。
- 第 2 步 24 项模型回归通过：[日志](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_training_model_regression.txt)。
- 第 1 步 41 项数据回归通过：[日志](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_training_data_regression.txt)。
- 使用完整本地 CLIP ViT-B/16 和临时生成图片，CUDA + AMP 跑通 T1、T2，每阶段 4 次更新；参考输出逐元素不变，T2 缺席的 vehicle 输出不变：[报告](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_training_smoke.json)。

测试另验证了：小样本 CE 下降，最后一轮当前身份分类准确率至少 95%；不同类别 loader 长度不同仍得到相同更新次数；第二类别损失异常时不会先提交第一类别的参数更新。这些是受控功能验证，不代表真实 ReID 精度。

复现测试：

```powershell
python -m unittest discover -s tests -p "test_category_training.py" -v
python tools/smoke_test_category_training.py --device cuda --output docs/category_training_smoke_repeat.json
```

冒烟脚本会在报告旁保存 `*_t1.jsonl` 和 `*_t2.jsonl`，重复运行应使用新的输出名称，避免覆盖已有日志。图片在临时目录生成，不修改真实数据。

下一步是第 4 步：利用固定参考特征建立并累计身份原型记忆。
