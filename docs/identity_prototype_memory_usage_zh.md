# 第 4 步：身份原型记忆使用说明

日期：2026-09-13。当前实现 ECPM 的身份摘要层；聚类、模式原型、类别中心与漂移属于第 5 步。

## 1. 现在具体做了什么

以 person/a 有两张训练图片为例：先让冻结参考编码器分别提取特征，然后求两张图的平均特征，最后把平均向量的长度变成 1。这个向量就是 person/a 的身份原型。

\[
p_{c,y}=\frac{\mu_{c,y}}{\|\mu_{c,y}\|_2},\qquad
\mu_{c,y}=\frac{1}{n_{c,y}}\sum_{x\in\mathcal D_{c,y}^{t}}F_0(x).
\]

顺序是**原始参考特征 → 求平均 → L2 归一化**。例如两张图特征为 `(3,0)`、`(0,4)`，结果是 `(0.6,0.8)`，不是先分别归一化后得到的 `(0.707,0.707)`。

T1 保存 person/a、person/b、vehicle/a、vehicle/b；T2 只计算 person/c、person/d、panda/a、panda/b 并追加。T1 的向量无需重算，vehicle 缺席也不会被更新。类别之间以及不同来源之间用完整身份键区分，不使用每阶段重新编号的 CE 标签作为历史索引；已声明的身份别名按协议合并。

## 2. 实现位置

| 文件 | 功能 |
|---|---|
| [ecpm.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/memory/ecpm.py) | FP32 聚合、覆盖检查、候选提取、整体提交、记忆保存恢复 |
| [category_stream_loaders.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/lreid_dataset/category_stream_loaders.py) | `build_stage_prototype_loaders`，独立于训练 P×K 采样 |
| [extract_identity_prototypes.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/tools/extract_identity_prototypes.py) | 每次提取一个阶段的命令行入口 |
| [test_identity_prototype_memory.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/tests/test_identity_prototype_memory.py) | 公式、协议、状态与接口测试 |
| [smoke_test_identity_prototype_memory.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/tools/smoke_test_identity_prototype_memory.py) | 真实 CLIP 权重、合成图像两阶段冒烟 |

提取器自动使用模型固定预处理，关闭参考分支的 Adapter 和梯度，不使用训练随机增强。顺序加载、保留尾批、每张当前训练图片恰好一次；一个类别只有一个身份、一张图片也可提取。提取前后检查参考权重没有变化，参考前向不会继承外层 AMP。

## 3. Python 接口

```python
from lreid_dataset.category_stream import load_category_stream
from reid.models.category_adapter_bank import build_category_model
from reid.memory.ecpm import IdentityPrototypeMemory

stream = load_category_stream("config/my_stream.json")
model = build_category_model(device="cuda")  # 本地 CLIP 缓存，或显式指定 reference_checkpoint
memory = IdentityPrototypeMemory(model, stream)

# 提取全部当前类别，但暂时不改变历史记忆。
candidate = memory.prepare_stage(model, stream.stage("t1"), batch_size=64, workers=0)
assert memory.processed_stages == ()

# 所有身份都有效才整体提交；返回数量和兼容性摘要。
summary = memory.commit_stage(candidate)
memory.save("runs/memory/t1.pt")

# 新进程中，用同一参考配置和同一协议恢复，再处理下一个阶段。
memory = IdentityPrototypeMemory.load("runs/memory/t1.pt", model, stream)
memory.update_stage(model, stream.stage("t2"), batch_size=64)  # prepare + commit
memory.save("runs/memory/t2.pt")

person_rows = memory.category_prototypes("person")  # 独立副本，不暴露可修改的内部历史
# 每行：identity_key、vector [D]、image_count、first_stage
```

候选必须按协议阶段顺序提交，不能跳阶段、重复阶段或复用过期候选。一个类别提取失败时，不会留下其余类别的部分提交。NaN、无穷值、零均值范数会报错，错误包含对应身份键。当前对象是单写者接口，不用于多个线程同时更新。

`prepare_stage` 和 `commit_stage` 分开，是为了后续第 5—8 步能先读取旧记忆并计算当前候选，再在阶段成功后提交。当前没有把新记忆插入基础训练工具，也没有实现 PGCA。由于 F0 冻结，单独验证时可在该阶段训练前或训练后提取；最终 PGCA 流程需要在当前阶段训练前得到候选统计量，不能等训练结束才计算迁移或蒸馏条件。

## 4. 命令行提取

以下命令在仓库根目录运行。`config/my_stream.json` 必须换成已经完成审计、图像确实存在的实际配置；仓库的 `category_progressive_example.json` 是占位图像示例，不能直接用于正式提取。

```powershell
python tools/extract_identity_prototypes.py --stream-config config/my_stream.json --stage-id t1 --device cuda --batch-size 64 --output-memory runs/memory/t1.pt

python tools/extract_identity_prototypes.py --stream-config config/my_stream.json --stage-id t2 --device cuda --batch-size 64 --previous-memory runs/memory/t1.pt --output-memory runs/memory/t2.pt
```

默认使用本地 `C:\Users\Cloud\.cache\clip\ViT-B-16.pt`（实际按当前用户主目录定位），也可传 `--reference-checkpoint`。不会隐式下载模型。

如果要沿用第 3 步保存的模型及其参考预处理，改用：

```powershell
python tools/extract_identity_prototypes.py --stream-config config/my_stream.json --stage-id t2 --model-checkpoint runs/baseline/t1/completed_model.pt --previous-memory runs/memory/t1.pt --output-memory runs/memory/t2.pt --device cuda
```

`--model-checkpoint` 支持同协议的当前或紧邻前一阶段已完成基础训练 checkpoint，不能同时传 `--reference-checkpoint`。即使存在训练过的 Adapter，提取也只走 F0。

工具拒绝覆盖已存在的输出文件。每阶段输出一份 `.pt`，终端打印开始事件和最终 JSON 摘要；如需持久终端记录，可用 PowerShell 重定向。显存不足可减小 `--batch-size`，不改变身份聚合规则。不同硬件或 batch 大小的浮点结果可能存在微小数值差异。

## 5. 保存内容与中断恢复

记忆文件只保存：

- 协议指纹、阶段顺序和阶段元数据摘要哈希。
- 参考权重/预处理签名、参考配置、维数和公式版本。
- 已提交阶段列表。
- 每身份完整身份键、CPU FP32 单位向量、图像数、首次阶段。

不保存图片、历史路径、逐图特征、模型权重、临时分类头或训练优化器。初始化时会读取全部协议元数据以检查顺序和身份覆盖，但不读取未来图像。恢复时依赖原协议文件；协议指纹含路径信息，移动数据根目录会造成不匹配，应保留原协议或另行设计显式迁移，不应绕过校验。

标准 512 维下，每个向量占 2048 字节；`summary()['vector_bytes']` 只统计向量载荷，文件还包含身份元数据和序列化开销。原型随历史身份数量线性增长，首版不是固定总内存方案。

保存采用同目录临时文件、刷新后原子替换。Python API 可以有意更新一个文件；CLI 要求每次使用新输出路径。如果写入/替换失败，先前已保存文件保持完整。内存 `commit_stage` 和磁盘 `save` 是两个明确步骤：提交成功后若保存失败，可重试 `save`，不能对同一内存重复提交该阶段。

中断后分两种情况：

1. 已得到当前阶段完整 `.pt`：加载它，从下一阶段继续。
2. 当前阶段只有临时文件或没有输出：加载上一完整 `.pt`，重新提取整个当前阶段；第一阶段失败则重新从空记忆提取。遗留 `.tmp` 不能当成完成检查点。

当前支持**阶段之间恢复**，不保存提取到第几张图的游标，也不提供训练 epoch 内精确恢复。当前阶段提取和保存成功前，应保留该阶段图像；后续阶段只需要其原型。

开发过程恢复见 [第 4 步执行记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step4_progress.md)。

## 6. 验证结果

- 22 项针对性测试通过：原始均值公式、尾批与图像覆盖、别名合并、单图身份、适配器绕过、错误签名、重复阶段、全阶段提交、原子保存失败、保存恢复继续追加，以及两种 CLI 模型来源。
- 前 3 步 86 项回归通过。
- 真实 CLIP ViT-B/16 + CUDA、外层 autocast、非零 Adapter 的两阶段合成图像验证通过。T1 为 4 个身份，T2 累计 8 个身份（person 4、vehicle 2、panda 2），向量载荷 16384 字节。
- 旧原型逐元素不变，每阶段保存加载后继续运行；T1 图像删除后仍能完成 T2。独立计算均值与提取结果的最大绝对差为 `1.1920928955078125e-07`。

报告：[单元/集成测试](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/identity_prototype_memory_tests.txt)、[回归测试](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/identity_prototype_memory_regression.txt)、[CLIP 冒烟 JSON](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/identity_prototype_memory_smoke.json)。这些是实现验证，不是正式数据上的识别效果或防遗忘证据。

下一步在累计身份原型上实现 FINCH 聚类、模式原型、类别概括与漂移；本步没有生成虚假的模式/类别中心占位值。
