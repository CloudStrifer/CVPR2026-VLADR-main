# 第 2 步：冻结参考编码器与持久类别 Adapter

本步骤建立 ECPM/PGCA 的模型基础。类别内完整 Trainer、原型记忆、动态蒸馏、迁移源选择和测试路由分别属于后续步骤。

## 1. 已实现内容

- [CategoryAdapterBank](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/models/category_adapter_bank.py)：一个冻结的视觉 Transformer、按物体类别持久保存的 Adapter，以及类别参数/快照管理。
- [wrapper.make_category_model](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/models/wrapper.py)：新模型构建入口；原来的 `make_model` 及 OCIA/OSAF 训练路径保持原行为。
- [模型测试](E:/Multi_modal_Code/CVPR2026-VLADR-main/tests/test_category_adapter_bank.py)：冻结、梯度、阶段注册、深拷贝、恢复和预处理检查。
- [完整 CLIP 冒烟脚本](E:/Multi_modal_Code/CVPR2026-VLADR-main/tools/smoke_test_category_adapter_bank.py)：使用本地预训练模型和随机输入验证工程链路。

默认结构：CLIP ViT-B/16 视觉主干保持冻结，在末尾 4 个 Transformer block 中各添加瓶颈宽度 64 的残差 Adapter，scale=1。每个类别 396,544 个 Adapter 参数，视觉参考模型 86,192,640 个参数。

Adapter 直接复用已有 `DomainAdapter`：降维层正常初始化，升维层权重和偏置置零。因此新 Adapter 初始为零残差；不能将两层同时全部置零。

## 2. 构建模型

```python
from reid.models.wrapper import make_category_model

model = make_category_model(
    reference_checkpoint=r'C:\Users\Cloud\.cache\clip\ViT-B-16.pt',
    input_size=(224, 224),
    resize_mode='pad',
    last_blocks=4,
    bottleneck_dim=64,
    scale=1.0,
    device='cuda',  # 没有 GPU 时可使用 cpu
)
```

省略 `reference_checkpoint` 时只查找用户目录中的 `.cache/clip/ViT-B-16.pt`。本机已有该文件，已用于验证。文件不存在会明确报错，不会自动下载，也不会加载旧 Stage1 prompt 或旧 ReID checkpoint。

构建器支持本地 OpenAI CLIP ViT-B/16 TorchScript 或 tensor state_dict，只保留视觉塔；不创建旧的文本塔、prompt learner、BN neck 和累计身份分类器。

权重从 T1 开始就冻结，没有“第一个阶段先微调主干”的例外。新建的 Adapter 默认也冻结，必须显式指定当前更新集合。

## 3. 接续第 1 步数据流

```python
from lreid_dataset.category_stream import load_category_stream
from lreid_dataset.category_stream_loaders import build_stage_loaders

stream = load_category_stream('config/my_category_stream.json')
stage = stream.stage('t1')

for view in stage.categories:
    is_new = model.add_category(view.category)
    print(view.category, is_new)

current_categories = [view.category for view in stage.categories]
model.set_trainable_categories(current_categories)

train_transform, reference_transform = model.make_transforms()
loaders = build_stage_loaders(
    stage,
    batch_size=4,
    num_instances=2,
    train_transform=train_transform,
    reference_transform=reference_transform,
)

for view in stage.categories:
    model.create_temporary_head(view.category, len(view.identity_keys))

# 第 3 步在所有类别和分类头准备好之后，再创建 optimizer。
```

其中 `my_category_stream.json` 应替换成真实数据配置。仓库中的教程配置仍是占位图片路径；本次没有重新划分真实数据。

同类别再次到达时 `add_category('person')` 返回 False，保留原 Adapter 的参数对象和数值。类别内部实际注册键为类别名称的稳定 SHA-256 编码，允许 `giant.panda/亚洲` 等显示名称，避免 ModuleDict 对点号的限制。

类别名称区分大小写；`person` 与 `Person` 是两个类别。与第 1 步协议保持同一套类别名称。

## 4. 两个特征接口

```python
images = batch['images'].to('cuda')

z = model.encode_reference(images)
f = model.encode_category(images, 'person')
```

| 接口 | Adapter | 梯度 | 返回值 |
|---|---|---|---|
| encode_reference | 全部关闭 | no_grad | [B,512] 原始投影 CLS |
| encode_category | 指定类别 | 保留学生梯度图 | [B,512] 原始投影 CLS |

两个接口都不拼接 768 维特征，不自动归一化，不通过 BN，不输出身份 logits。普通 train/eval 切换不会改变返回值定义。

对应论文后续使用：

- ECPM 先将 `encode_reference` 的原始向量按身份求平均，再 L2 归一化。
- PGCA 教师/学生比较 `encode_category` 的特征余弦相似度。
- 最终检索描述符对 `encode_category` 输出进行 L2 归一化。

`model(images)` 等价于参考接口；`model(images, category='person')` 等价于类别接口。未知类别会报错，不根据输入自动新增类别。

输入必须是按固定预处理生成的浮点 Tensor [B,3,H,W]，尺寸与 ReferenceConfig 一致；调用方将输入放到模型所在设备。

## 5. 冻结、精度和预处理

冻结同时覆盖两个方面：

1. 共享视觉参数 `requires_grad=False`，不进入训练集合。
2. 共享模块保持 eval 行为，防止 BN running statistics 或随机层状态变化。外部父模型调用 `.train()` 也不会将参考编码器切回训练状态。

但学生 Adapter 前向仍然经过完整视觉网络并保留梯度图。不能在 Trainer 中把整个学生调用包进 `torch.no_grad()`，否则 Adapter 也学不到。

预处理由不可变的 `ReferenceConfig` 记录：权重文件 SHA-256、输入尺寸、resize 方式、mean/std。`model.make_transforms()` 使用这份配置生成训练变换和确定性参考变换；只允许额外调整训练翻转、裁剪 padding 和擦除概率。

默认 mean/std 沿用当前仓库：(0.485,0.456,0.406)/(0.229,0.224,0.225)。这是明确的工程选择，不宣称是标准 CLIP 预处理或实验最优参数。如要使用另一套归一化，必须在建模时指定，并在原型提取和参考路由中保持一致。

构建器使用 FP32 参考权重；`encode_reference` 会关闭外层 autocast，避免同一输入的参考特征随训练 AMP 上下文改变。学生可使用 autocast。原型建立后不要调用 `model.half()` 或用 dtype 转换改变参考数值；移动设备 `.to('cuda')` 可以正常使用。

`model.assert_reference_unchanged()` 会核验参考权重和缓冲区指纹；导出整个 bank/checkpoint 时也会执行检查。参考签名同时包含加载后权重、预处理配置和特征定义，用来拒绝不同参考空间下的 Adapter 恢复。

## 6. 临时分类头与训练集合

```python
model.set_trainable_categories(['person', 'panda'])
model.train()

features = model.encode_category(images, 'person')
logits = model.classify(features, 'person')

parameters = list(model.trainable_parameters())
```

`set_trainable_categories` 只接受已注册类别；缺席类别的参数被冻结，残留梯度也会清空，避免旧 optimizer 因已有梯度继续更新它们。

临时分类头为 `Linear(512, 当前新身份数, bias=False)`，类别分别维护，无 BN、无历史累计身份行。分类头同样只在所属类别位于当前训练集合时可训练。

阶段结束时：

```python
model.set_trainable_categories([])
model.discard_temporary_heads()
```

下一阶段创建新的分类头后，需要重建 optimizer。不要继续复用仍指向旧分类头的 optimizer。本步骤提供必要接口，第 3/8 步负责正式训练循环和阶段生命周期。

## 7. Adapter 快照、复制与恢复

```python
# 阶段开始前保存旧 person 的独立快照。
person_snapshot = model.export_adapter('person')

# 新 panda 从 person 的当前快照复制一份独立 Adapter。
model.copy_category('person', 'panda')

# 或显式使用某个较早阶段快照。
model.load_adapter('panda', person_snapshot)
```

快照张量 detach 后复制到 CPU，不是当前参数的引用。复制后 source/target 参数数值起初相同，但内存独立；训练 panda 不会修改 person。

`copy_category` 拒绝已存在的目标类别，避免重复类别被意外初始化。`load_adapter` 允许显式覆盖已存在的目标，因此只能用于明确的阶段边界恢复/初始化，不能在尚未反向传播的计算图中调用。

载入前检查参考签名、block 数、瓶颈维度、scale、参数名称、形状和数值有效性。复制的内容只有 Adapter，没有源分类器、优化器、原型或源类别梯度。

这里尚未实现“原型相似性选源”，也没有构建教师前向网络；快照接口供第 6/7 步使用。PGCA 仍需在阶段开始时保存上阶段版本，不能误用已更新的当前阶段参数作为历史来源。

## 8. 保存整个类别模型

推荐使用专用快照格式，而不是单独保存普通 `model.state_dict()`：动态类别注册表、参考预处理和签名也必须保存。

```python
import torch
from reid.models.category_adapter_bank import CategoryAdapterBank

# 包含冻结视觉权重和类别 Adapter，可脱离原始 CLIP 文件恢复。
torch.save(model.export_checkpoint(), 'category_model.pth')

checkpoint = torch.load('category_model.pth', map_location='cpu', weights_only=True)
restored = CategoryAdapterBank.from_checkpoint(checkpoint, device='cuda')
```

恢复后默认 eval，所有类别冻结，没有临时分类器。进入新阶段时再调用 `set_trainable_categories` 并创建当前分类头。

若外部已经持有完全相同的冻结参考编码器，也可只保存 `export_bank()`，然后向一个空类别模型调用 `load_bank(snapshot)`。整个 bank 的校验在创建类别之前完成，避免坏快照只恢复一半。

本步骤的 checkpoint 是**模型状态恢复**，不包含阶段进度、ECPM、optimizer、scheduler、数据迭代位置或随机数状态。完整训练断点恢复仍属于第 8 步，不能将本接口当成完整训练 resume。

## 9. 验证结果与接续位置

2026-09-13 已完成：

- 新模型 24 项测试通过：[完整输出](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_adapter_bank_tests.txt)。
- 旧 Adapter 5 项回归通过：[输出](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_adapter_bank_legacy_regression.txt)。
- 第 1 步 41 项数据层回归通过：[输出](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_adapter_bank_data_regression.txt)。
- 完整本地 CLIP ViT-B/16 在 CUDA + AMP 下通过冒烟验证：[结构化报告](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_adapter_bank_smoke.json)。

冒烟验证使用 2 张 224×224 随机输入，输出 2×512；person Adapter 能获得梯度并改变输出，参考特征和 vehicle 输出逐元素保持不变，保存恢复后的输出最大差异为 0。该结果只验证实现正确性，不是 ReID 精度结果。

复现命令：

```powershell
python -m unittest discover -s tests -p "test_category_adapter_bank.py" -v
python -m unittest discover -s tests -p "test_domain_adapters.py" -v
python -m unittest discover -s tests -p "test_category_stream*.py" -v
python tools/smoke_test_category_adapter_bank.py --device cuda --output docs/category_adapter_bank_smoke.json
```

中断接续先读取 [第 2 步执行记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step2_progress.md)。本步骤完成后，下一步是实现类别内部 CE + Triplet 的基础 Trainer。
