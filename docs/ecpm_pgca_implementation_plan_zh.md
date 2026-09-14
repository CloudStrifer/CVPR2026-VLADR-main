# ECPM + PGCA 分步实施计划

日期：2026-09-14  
依据：`C:\Users\Cloud\Desktop\name2.tex` 中的方法定义，以及当前仓库代码。  
状态：第 1—9 步的基础实现与验收已完成。第 9 步硬路由、逐阶段检索/路由诊断/遗忘矩阵、评估事务恢复及 9 个消融配方已接入；190 项回归检查通过，真实 CLIP CUDA＋AMP 三阶段评估与恢复对照通过。已按用户确认方案生成五数据集四阶段真实数据划分；正式多种子性能实验尚未开展，生成图像验证不能证明方法有效。

第 1 步入口：[使用说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_stream_usage_zh.md)；[执行与中断恢复记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step1_progress.md)。

第 2 步入口：[使用说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_adapter_bank_usage_zh.md)；[执行与中断恢复记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step2_progress.md)。

第 3 步入口：[使用说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/category_training_usage_zh.md)；[执行与中断恢复记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step3_progress.md)。

第 4 步入口：[使用说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/identity_prototype_memory_usage_zh.md)；[执行与中断恢复记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step4_progress.md)。

第 5 步入口：[使用说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_modes_usage_zh.md)；[执行与中断恢复记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step5_progress.md)。

第 6 步入口：[使用说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/pgca_recurring_usage_zh.md)；[执行与中断恢复记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step6_progress.md)。

第 7 步入口：[使用说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/pgca_transfer_usage_zh.md)；[执行与中断恢复记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step7_progress.md)。

第 8 步入口：[连续训练与恢复说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/progressive_training_usage_zh.md)；[执行与中断恢复记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step8_progress.md)。

第 9 步入口：[硬路由、终身评估与消融说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/progressive_evaluation_usage_zh.md)；[执行记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step9_progress.md)。

服务器操作入口：[Ubuntu 训练、恢复、评估与消融命令手册](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ubuntu_training_commands_zh.md)。包含 Bash 命令、参数默认值、9 项消融、多种子运行、验证集诊断及常见问题。

真实数据配置：[main.json](E:/Multi_modal_Code/CVPR2026-VLADR-main/config/category_progressive_real/main.json)；[上传与使用说明](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/real_data_stream_upload_zh.md)。采用原始四阶段安排、seed=42、约 10% 原训练身份验证留出；原测试清单保持。

## 1. 目标与实施边界

目标是在现有 CLIP-ReID 工程上实现两个模块：

- **ECPM：演化式聚类类别原型记忆。** 保存历史身份摘要，归纳类别内部的外观模式，并提供类别变化、类别相似性和测试路由所需的统计量。
- **PGCA：原型引导的类别自适应。** 熟悉类别继续更新原 Adapter，利用原型变化调节蒸馏；陌生类别根据原型相似度选择历史 Adapter 初始化。

第一版忠实实现 `name2.tex`，用实验检查原型变化和迁移相似度是否有效。此前讨论的 PIRM、RMPA、IGA、RFP 等方案不属于本计划；旧的 `interleaved_identity_reid_design_zh.md` 不作为本次实现规范。

本文区分两类内容：

- **论文规定：** 已在 `name2.tex` 中写明的机制，第一版保持一致。
- **工程建议：** 论文未规定、为实现而选择的接口或默认行为，实施时应记录到配置和实验说明中。

任务约束：

1. 一个阶段可以包含多个物体类别。
2. 类别可跨阶段重复出现，同一个真实训练身份只属于一个阶段。
3. 同一身份在所属阶段内可以有多张图像，并可训练多个 epoch。
4. 阶段结束后，后续训练不读取该阶段的历史图像；允许保留身份原型、类别模式、类别 Adapter 和必要元数据。
5. 训练时类别标签已知；正式测试的特征提取与路由不读取真实类别标签。
6. 参考编码器从 T1 开始就冻结；只有当前类别的 Adapter 和临时身份分类器参与学习。
7. 首版保留全部历史身份原型。这是无历史图像回放，但不是固定总内存的方法。

旧 OCIA/OSAF 路径保留作为对照。新入口不依赖 Stage1 prompt 文件，不启用 OCIA、OSAF、属性蒸馏或额外的历史特征回放损失。

## 2. 九步总览

| 步骤 | 要完成的功能 | 完成后得到什么 | 前置依赖 |
|---|---|---|---|
| 1 | 定义类别交错、身份不重复的数据流 | 可审计的阶段配置和类别数据加载器 | 无 |
| 2 | 冻结参考编码器，建立持久类别 Adapter | 稳定的参考特征和可独立更新的类别模型 | 1 |
| 3 | 跑通类别内部 CE + Triplet | 不带两模块增强的基础训练链路 | 2 |
| 4 | 实现身份原型提取与累积记忆 | 每个历史身份一个固定摘要 | 1、2 |
| 5 | 实现 FINCH、模式原型、类别原型与漂移 | 完整 ECPM 和可读取的统计量 | 4 |
| 6 | 实现熟悉类别的动态特征蒸馏 | PGCA 的旧类别分支 | 3、5 |
| 7 | 实现陌生类别的相似性迁移初始化 | PGCA 的新类别分支 | 2、5 |
| 8 | 串联阶段流程、日志和断点恢复 | 能连续运行多个阶段的训练入口 | 1—7 |
| 9 | 实现硬路由、终身评估和必要消融 | 可判断是否有效、为什么有效的完整系统 | 8 |

建议按表中顺序实现。每一步先通过本节列出的验收，再进入下一步。第 3 步前不用等待聚类和迁移功能，可以先确认基本训练没有问题。

## 3. 当前代码可复用什么

以下是已检查的现有实现，不表示新功能已经存在。

| 现有位置 | 已有功能 | 本次处理建议 |
|---|---|---|
| [train_stage2.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/train_stage2.py) | 按 domain 顺序训练、扩展累计分类器、加载 prompt/anchor | 新建独立类别渐进入口；复用通用工具，不照搬阶段主循环 |
| [manifest_reid.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/lreid_dataset/datasets/manifest_reid.py) | 读取 manifest、映射 PID、验证 query/gallery | 复用解析思路，新增保留原始身份与阶段信息的协议层 |
| [get_data_loaders.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/lreid_dataset/datasets/get_data_loaders.py) | 图像增强、身份采样、确定性初始化 loader | 复用变换与采样器，改为阶段内按类别组织 |
| [wrapper.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/models/wrapper.py) | Adapter 创建与切换接口 | 增加清晰的参考特征/类别特征入口 |
| [make_model_clipreid.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/models/CLIP_ReID/model/make_model_clipreid.py) | CLIP 特征、分类器、BN、训练/测试返回值 | 新路径显式统一特征，避免沿用多种默认返回值 |
| [clip/model.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/models/CLIP_ReID/model/clip/model.py) | Transformer 内残差 Adapter、参数集合、关闭 Adapter | 复用现有结构，将注册键的含义改为物体类别 |
| [trainer_stage2.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/trainer_stage2.py) | CE、Triplet、旧方案辅助损失 | 参考训练组织方式，新建精简 Trainer |
| [fast_test.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/evaluation/fast_test.py) | 特征提取、分块检索和指标 | 复用检索指标，新增 ECPM 硬路由入口 |

需要特别处理的差异：

- 当前 `ManifestReID._as_tuples()` 输出的第四项为常数 `0`，不能把它当成可靠的物体类别标签。
- 当前主循环以 `domain_index` 推进，一个训练 domain 对应一次阶段更新；不直接支持“一个阶段多个类别，类别随后再次出现”。
- 当前 `get_image=True` 返回投影后的 CLS 特征，普通测试分支可能返回两种特征的拼接。新路径不能混用这些结果。
- 当前旧入口在第一阶段可能解冻视觉主干。新方法必须从 T1 起冻结，不能沿用这一特殊处理。
- 当前 LwF 是分类 logits 的 KL 蒸馏；PGCA 需要当前图像上的特征余弦一致性，两者不能直接替换名称使用。

## 4. 步骤 1：建立新的数据流协议

### 目标

准确表达“同类别的新身份再次到来”，并保证后续训练只接触当前阶段图像。

### 要做什么

1. 新增阶段配置解析器，分开定义 `stage_id`、`category`、`source_dataset` 和真实身份。
2. 按真实身份划分阶段，不能随机按图片划分；同一身份的所有训练图像进入同一阶段。
3. 身份唯一键使用 `(category, source_dataset, original_pid)`。如果多个数据源实际包含同一个身份，先建立统一映射；不能靠换数据源名称掩盖身份重叠。
4. 不把 `stage_id` 放进身份唯一键，否则同一身份跨阶段出现会被错误地当成两个身份。
5. 每个阶段按类别建立训练 loader 和确定性的原型提取 loader。
6. 每个 `(stage, category)` 单独建立从真实身份键到 `0...N-1` 的临时分类标签映射。
7. 固定训练/验证/测试身份划分。标准 ReID 主评估使用与训练身份不重叠的 query/gallery；同一测试集合跨阶段反复评估是允许的，不等于历史训练图像回放。
8. 数据审计可预先检查全量元数据，但模型训练、原型提取、源类别选择都不能提前使用未来阶段图像或统计量。

建议新增文件：`lreid_dataset/category_stream.py`、`tools/validate_category_stream.py`、`config/category_progressive_example.json`。

配置结构示意；第 1 步已实现解析，完整可审计示例见 `config/category_progressive_example.json`：

```json
{
  "schema_version": 1,
  "data_root": "../data/my_category_stream",
  "stages": [
    {
      "stage_id": "t1",
      "categories": [
        {"category": "person", "train_manifest": "manifests/t1_person.csv"},
        {"category": "vehicle", "train_manifest": "manifests/t1_vehicle.csv"}
      ]
    },
    {
      "stage_id": "t2",
      "categories": [
        {"category": "person", "train_manifest": "manifests/t2_person.csv"},
        {"category": "panda", "train_manifest": "manifests/t2_panda.csv"}
      ]
    }
  ]
}
```

每个训练 manifest 保留 `path, original_pid, source_dataset, camid` 等字段；category 和 stage 可由父配置提供，但解析后的样本必须保留这些信息。路径相对哪一层解析要写入 schema，建议 manifest 相对配置文件、图像相对显式数据根目录。

### 验收

- 输出每阶段的类别、新身份数、图像数，以及 new/recurring/absent 类别清单。
- 人为把 person A 放入 T1、T2 时，校验必须报错。
- 不同数据源的同编号 PID 不误合并，同一类别跨阶段的临时标签 `0` 不误当成同一身份。
- 原型 loader 只访问当前训练图像；训练为空时直接报错，不能沿用旧 loader 的 query/gallery 回退逻辑。
- 测试集存在有效正样本；相机过滤规则按数据集约定执行，不能随意伪造相机编号。

### 第 1 步实施记录（2026-09-13）

- 已新增 `lreid_dataset/category_stream.py`、`lreid_dataset/category_stream_loaders.py`、审计 CLI、三阶段示例及测试。
- 所有本节验收条件均有测试覆盖；41 项测试通过，包含临时真实图像上的当前阶段隔离与 Windows 多进程加载。
- 身份键可通过显式 alias 合并跨来源同一真实身份；评估集合区分 validation/test。
- 训练 loader 和原型 loader 单独组织；原型逐图一次，训练为空报错。
- 工程调整：加载器单独放在 `category_stream_loaders.py`，使纯元数据审计不依赖 PyTorch；P×K 采样保留旧规则，使用局部随机状态支持按 epoch 重建。
- 三阶段示例是占位元数据，未改动现有五数据集 manifest，也未执行真实数据集的重新分期或模型训练。
- 详细使用方式和验证记录见本文顶部链接。后续按第 2 步继续，不必重做协议层。

## 5. 步骤 2：建立冻结编码器和类别 Adapter 管理

### 目标

让类别身份固定下来：person 永远对应同一套可持续更新的 Adapter；原型参考空间始终不变。

### 要做什么

1. 复用现有 Transformer 内残差 Adapter，不重新设计网络结构。
2. 注册 Adapter 时以类别为键，建立稳定的类别名称到内部合法键映射。
3. 建立两个显式接口：

```python
encode_reference(images)                    # Adapter 全关闭，固定参考特征
encode_category(images, category)           # 指定类别 Adapter，允许学生梯度
```

4. **工程建议：** 两个接口首版都返回投影后的 CLS 原始向量，特征维数从模型读取；CE、Triplet、蒸馏和推理都围绕这个明确输出组织。原型空间和适配空间维数相同，但语义角色不同。
5. 新建临时线性身份分类头，避免依赖旧代码累计分类器和共享 BN。首版不用 BN neck；这属于实现选择，不是论文新增模块。
6. 固定参考模型预处理配置，包括输入尺寸、resize、归一化和权重标识。原型提取关闭随机裁剪、翻转和擦除。
7. 冻结主干参数以及共享状态；仅 `requires_grad=False` 不足以冻结 BN running statistics。参考和教师前向采用 eval 行为。
8. 学生 Adapter 前向必须保留梯度图：主干权重冻结，不意味着整条含 Adapter 的学生前向能包在 `no_grad()` 中。
9. 默认 Adapter 初始化复用现有实现：升维层为零初始化，使初始残差为零；不把整个 Adapter 的所有层都初始化为零。
10. 增加类别 Adapter 导出、深拷贝、载入和参数列表接口，供后面的教师与迁移使用。

建议新增 `reid/models/category_adapter_bank.py`，并为现有 wrapper 增加上述接口。Adapter 的 block 数、瓶颈宽度、scale 保持各类别一致，保证可复制；第一版可沿用当前接口默认的 4 个 block、64 维瓶颈、scale=1，作为工程起点而非最优超参数结论。

### 验收

- T1 创建 person、vehicle，T2 再遇 person 时 Adapter 总数只因 panda 增加。
- 更新 person 一步后，主干参数/缓冲区、vehicle 参数均不变。
- 更新 Adapter 前后，同一确定性输入的 `encode_reference()` 输出在数值容差内一致。
- person Adapter 能收到非零梯度；切换类别确实能选择不同参数。
- 参考和类别输出维数一致且明确，普通模型 train/eval 切换不偷偷改变新接口特征含义。

### 第 2 步实施记录（2026-09-13）

- 已新增 `CategoryAdapterBank`，使用冻结视觉塔和持久类别 Adapter；统一输出原始投影 CLS，标准 ViT-B/16 为 512 维。
- `wrapper.make_category_model` 从本地 CLIP 权重建立新路径；旧入口保持不变。
- 当前类别训练集合、临时线性分类头、Adapter 独立复制/载入和模型 checkpoint 恢复均已实现。
- 固定预处理与参考权重签名；参考前向关闭外层 autocast，学生前向保留梯度。
- 24 项新模型测试通过，旧 Adapter 5 项和第 1 步 41 项回归通过。
- 完整 CLIP CUDA + AMP 验证：参考特征及缺席类别输出逐元素不变，当前类别获得有效梯度，模型恢复后输出差异 0。
- 模型权重恢复已具备，完整训练断点仍属于第 8 步。详细调用方式见本文顶部第 2 步文档链接。

## 6. 步骤 3：先跑通类别内部基础 ReID 训练

### 目标

先建立可靠的“持久类别 Adapter + 当前身份学习”基线，再接入 ECPM/PGCA。

### 要做什么

1. 新建 `reid/trainer_category_progressive.py`，首版只实现：

\[
\mathcal L_{\mathrm{id}}^c
=\mathcal L_{\mathrm{ce}}^c+\lambda_{\mathrm{tri}}\mathcal L_{\mathrm{tri}}^c.
\]

2. 分类器只包含当前 `(stage, category)` 的新身份，阶段结束丢弃，不累计历史类别行。
3. 一个类别内部做 P×K 身份采样；Triplet 正负样本都来自这个类别。
4. **工程建议：** 用原始投影向量进入临时线性分类器；Triplet 使用 L2 归一化特征，并固定距离形式和 margin。所有基线采用同一选择。
5. 推荐每个训练迭代从每个当前类别取一个 batch，分别前向、计算类别平均损失，再累加梯度和更新。这样对应论文对类别损失求和，也避免一个大类别完全支配训练。
6. 对每类别每阶段的迭代数作显式配置；较小类别循环采样属于当前阶段重复训练，不属于历史回放。
7. 如果单类别不足两个身份或不能形成有效正负对，明确报错或记录跳过 Triplet 的策略；不能用别的类别补负样本掩盖问题。
8. 本步临时使用正确类别 Adapter 进行诊断性评估；第 9 步才接入正式无类别标签路由。

### 验收

- 小数据训练中损失有限、参数正常更新，模型能学习当前身份区分。
- person batch 的 Triplet 不使用 vehicle 样本。
- 分类头输出维数等于当前类别新身份数。
- 基础日志分别记录 CE、Triplet 和每类别更新次数。
- 确认旧 anchor、文本匹配、属性蒸馏等损失没有进入总损失。

### 第 3 步实施记录（2026-09-13）

- 新增类别内 batch-hard Triplet 与基础 Trainer；CE 使用原始投影特征，Triplet 使用归一化特征的欧氏距离，类别内挖掘。
- 每步各当前类别各取一个 batch，梯度相加、统一更新；不等长 loader 仅循环当前类别数据。
- 阶段内临时分类头、阶段结束清理、逐步 JSONL 日志与异常保护已接通。
- 新增单阶段基线运行工具，完成阶段之间可载入模型接续；不替代第 8 步完整流程和阶段内 resume。
- 新增正确类别 Adapter 的 oracle 诊断评估；第 9 步仍需实现无类别标签路由。
- 21 项新测试和模型/数据 65 项回归通过；完整 CLIP CUDA + AMP 跑通 T1、T2，参考与缺席类别输出保持不变。
- 使用方式和验证报告见顶部第 3 步链接。下一步实现身份原型记忆。

## 7. 步骤 4：实现身份原型记忆

### 目标

把当前阶段每个身份压缩为一个固定向量，后续只用这些摘要累计类别经验。

### 要做什么

1. 新建 `reid/memory/ecpm.py`；对当前训练图像进行一次无随机增强的参考特征扫描。
2. 使用 FP32 累加每身份的特征和、图像数；每张图片恰好贡献一次，不用训练 P×K sampler 提取原型。
3. 按论文先平均原始参考特征，再归一化：

\[
p_{c,y}^{t}=\operatorname{Norm}\left(
\frac{1}{n_{c,y}}\sum_x F_0(x)\right).
\]

不要擅自变成“每张图先归一化后再平均”。

4. 将新身份原型加入 `P_c`，旧身份原型保持原值。
5. 保存身份键、首次阶段、图像数和向量等必要信息，不保存历史图像或逐图特征缓存。
6. 检查零范数和非有限值；异常时明确报错并记录身份键，避免产生用于余弦匹配的无效原型。

建议状态结构：

```text
ECPMState
  schema_version
  reference_weight_id / preprocessing_hash / feature_dim
  processed_stage_ids
  categories[category]
    identity_keys
    identity_prototypes       [N_id, D]
    first_stage_per_identity
    image_counts
    mode_prototypes           [K, D]    # 第 5 步生成
    cluster_sizes             [K]
    category_prototype        [D]       # 第 5 步生成
    last_updated_stage
```

### 验收

- 用手工小向量核对平均和归一化结果。
- 原型数等于已到达的不同训练身份总数。
- T2 更新后，T1 身份向量逐项保持不变。
- 保存再加载后身份键与向量一一对应。
- 重复执行已提交阶段时拒绝再次添加，或按阶段事务恢复；不能悄悄重复计入同一身份。

### 第 4 步实施记录（2026-09-13）

- 已新增 `reid/memory/ecpm.py`：固定 F0 特征、FP32 身份累加、完整图像覆盖检查、先平均后归一化。
- 独立顺序原型 loader 不构造训练 P×K sampler，支持一个身份一张图片。
- 全阶段先准备候选、验证后整体提交；旧身份向量保持原值，重复或跳跃阶段拒绝提交。
- 状态绑定参考签名和协议，保存身份键/向量/图像数/首次阶段，无历史图片、路径或逐图特征缓存。实际以身份记录列表序列化、提供按类别读取副本；第 5 步再添加聚类层，不预填模式和类别中心。
- `tools/extract_identity_prototypes.py` 支持本地参考权重或第 3 步完成模型、上一阶段记忆、新输出文件；原子保存和阶段间恢复已验证，尚不提供提取批次内或训练 epoch 内恢复。
- 22 项针对性测试、前 3 步 86 项回归通过；完整 CLIP CUDA 合成两阶段累计 8 个身份，旧向量逐元素不变，保存恢复与删除旧图片后接续通过。
- 第 3 步训练入口仍是基础训练。第 8 步才整体串联；后续 PGCA 需要训练前准备本阶段候选统计。

## 8. 步骤 5：完成 ECPM 的聚类、类别概括与漂移

### 目标

让身份摘要形成可供 PGCA 和路由使用的多层类别记忆。

### 要做什么

1. 封装 FINCH 调用，只对当前出现类别的累计身份原型重新聚类；缺席类别原型原样保留。
2. FINCH 可能返回多个层次的划分。**工程建议：** 首版固定使用返回的第一层划分，使用余弦距离，并把划分层选择、依赖版本及实际簇数写入日志；这不是论文已指定的细节。
3. 显式处理单身份、重复向量、小样本和全体归为一簇等边界。缺少 FINCH 依赖时明确失败，不能静默换成其他算法。
4. 计算模式原型和类别原型：

\[
g_{c,k}^{t}=\operatorname{Norm}
\left(\frac{1}{|\mathcal G_{c,k}^{t}|}\sum_{p\in\mathcal G_{c,k}^{t}}p\right),
\qquad
m_c^t=\operatorname{Norm}
\left(\frac{1}{K_c^t}\sum_k g_{c,k}^{t}\right).
\]

类别原型按模式等权，不能再次按簇大小加权，否则改变了论文机制。

5. 更新前保存不可变的 `m_old`、`B_old` 快照；更新后计算：

\[
d_c^t=1-\cos(m_c^{t-1},m_c^t).
\]

6. 仅 recurring 类别计算漂移，新类别漂移记为“不适用”。余弦数值裁剪到 [-1,1] 后，漂移理论范围为 [0,2]，不是 [0,1]。
7. 同时记录累计身份数、簇数、簇大小分布、聚类时间和内存字节数。
8. 状态快照使用深拷贝或不可变数据；不能先覆盖旧中心再计算漂移。

### 验收

- 手工给定两个大小不同的簇，最终类别原型仍对两个模式等权。
- 完全相同的旧/新类别原型得到约 0 漂移。
- 缺席类别的全部记忆不变。
- 原型都为有限的单位向量，近零均值有明确异常处理。
- 加入“新模式对称变化而中心变化很小”的诊断例子，记录中心漂移的局限；这不是要求算法错误地检测出所有分布变化。

完成本步，ECPM 功能齐全，但还不能据此认定漂移信号已经能指导最优学习强度。

### 第 5 步实施记录（2026-09-13）

- 新增 `finch_modes.py` 和 `ECPMMemory`：累计身份聚类、模式等权类别中心、全体历史类别旧快照及 recurring 漂移。
- 固定官方 `finch-clust==0.2.3`、余弦第一层。通过分块精确 1-NN 加官方 `clust_rank/get_clust` 执行第一层，跳过无用高层；与官方完整调用第一层对照通过。记录依赖版本、源文件哈希、分块大小；不启用近似近邻或算法回退。
- 当前出现类别才重新聚类，缺席类别全部状态不变。候选准备不修改历史，验证全部类别后整体提交，支持原子保存与加载校验。
- `tools/update_ecpm_memory.py` 支持当前图像提取及从第 4 步已保存向量按阶段升级，后者无需历史图像。第 4 步独立接口保留。
- 模式/类别均值近零明确报错；新类别漂移为 None，recurring 漂移范围 [0,2]。保存一份最后阶段开始前的类别快照，统计其内存开销。
- 21 项新测试、108 项回归通过；真实 CLIP CUDA 合成三阶段累计 12 个身份，官方第一层对照、历史/缺席状态保持、保存恢复、删除旧图片后接续全部通过。
- 对称新增模式可使模式数 1→3 而中心漂移为 0，该局限已保留诊断并写入使用说明。未验证真实数据上的 PGCA 收益。

## 9. 步骤 6：实现 PGCA 的熟悉类别分支

### 目标

类别再次出现时，复用原 Adapter，并按原型漂移调整历史教师约束。

### 要做什么

1. 阶段开始、任何类别训练之前，保存所有 recurring 类别的 Adapter 冻结快照。
2. 学生继续使用原类别 Adapter，教师使用上阶段同类别 Adapter；不使用“上一个训练类别”作为教师。
3. 对同一批当前新身份图像、同一个增强后的输入张量，分别计算教师和学生特征。
4. 教师 eval + no_grad，学生保留梯度。实现：

\[
\mathcal L_{\mathrm{con}}^c
=\frac1B\sum_i\left[1-\cos(f_{\mathrm{new}}(x_i),
\operatorname{sg}[f_{\mathrm{old}}(x_i)])\right].
\]

5. 由本阶段 ECPM 更新计算一次类别权重：

\[
\lambda_c^t=\lambda_{\mathrm{con}}\exp(-\gamma d_c^t),
\qquad
\mathcal L_c=\mathcal L_{\mathrm{id}}^c+
\lambda_c^t\mathcal L_{\mathrm{con}}^c.
\]

权重在该阶段内保持不变；不能未经说明改为逐 batch 重新估计漂移。

6. 新增 `reid/loss/pgca.py`，实现特征一致性与权重函数。
7. 在 Trainer 中加入 `off / fixed / drift` 三种一致性模式，方便消融；完整方法使用 `drift`。
8. 首版可通过独立冻结视觉教师承载历史 Adapter；内存优化时可共享冻结权重，但必须保证教师参数不受学生更新影响。不要在尚未反向的学生计算图上原地覆盖参数来切换教师。

### 验收

- 完全相同的教师/学生特征一致性损失接近 0。
- 教师参数不变且无梯度，学生能正常学习。
- \(\gamma=0\) 时退化为固定权重；\(\lambda_{\mathrm{con}}=0\) 时退化为基础训练。
- 更大漂移对应更小权重，日志同时显示原始一致性损失、权重及加权损失。
- 后续训练只读取当前新身份图像。历史原型用于决定权重，没有进入该蒸馏损失作为 replay 样本。

### 第 6 步实施记录（2026-09-13）

- 新增 `reid/loss/pgca.py`，实现 FP32 特征余弦一致性、停止教师梯度、off/fixed/drift 权重配置。
- 新增 `reid/adaptation/pgca.py`，在任何当前类别训练前复制一份独立冻结视觉塔/历史 Adapter bank，前向只用于 recurring 类别。学生参数不覆盖、不共享教师存储；师生接收相同增强输入。
- Trainer 可选接入 PGCA；默认 API 仍 off。漂移候选在旧 ECPM 上非提交校验，权重阶段固定，新类别与零权重不执行教师前向。日志保留原始一致性、权重、漂移、加权一致性和教师哈希。
- `tools/train_pgca_recurring_stage.py` 提供单阶段验证入口：训练前 prepare，成功后 commit，并把模型/ECPM 原子保存为同一个完成检查点。可导入第 3 步模型＋第 5 步记忆；不支持 epoch 内恢复，新类别尚不执行迁移，完整调度仍属第 8 步。
- 教师使用完整视觉副本，显式记录其内存开销；不是仅复制小 Adapter。复制不消耗随机数，阶段结束释放。
- 18 项本步测试、129 项回归通过。零权重与默认基础 Trainer 更新逐元素相同，gamma=0 与 fixed 相同；教师冻结/输入一致/学生梯度/历史数据隔离/失败保护通过。
- 真实 CLIP CUDA＋AMP 三阶段短训练通过：T2 person，T3 person＋vehicle 使用各自阶段开始教师，参考/教师/缺席 Adapter 保持，当前 Adapter 更新，每阶段保存恢复后可在无旧图片情况下接续。
- 未运行正式数据上的防遗忘对照，不把工程通过解释为漂移信号优于固定蒸馏。

## 10. 步骤 7：实现 PGCA 的陌生类别分支

### 目标

为新类别找到可能有帮助的历史 Adapter 起点，并保证同阶段类别的训练顺序不影响源选择。

### 要做什么

1. 新建 `reid/adaptation/pgca.py`，读取新类别当前原型，以及阶段开始时所有历史类别的原型快照。
2. 实现模式覆盖匹配与综合相似度：

\[
S_{\mathrm{mode}}(c,j)=\frac1{K_c}\sum_k\max_l
\cos(g_{c,k},g_{j,l}),
\]

\[
s(c,j)=\alpha\cos(m_c,m_j)+(1-\alpha)S_{\mathrm{mode}}(c,j).
\]

这里是“新类别模式到历史类别模式”的单向匹配，不能擅自改成对称匹配或一对一匹配。

3. 从阶段开始前已存在的类别中选最高分源；所有比较和复制均使用 `t-1` 快照。
4. 若最高分大于等于 `delta`，深拷贝源 Adapter 全部层；否则使用默认零残差初始化。
5. T1 无历史类别时直接默认初始化；禁止对空候选集合做 argmax。
6. 同阶段新类别之间不互相借用。一个 recurring 类别即便先完成本阶段训练，也只能提供其上阶段版本给本阶段的新类别。
7. 仅复制 Adapter，不复制源身份分类器、优化器状态或源类别原型。
8. 新类别后续只训练 CE + Triplet，不额外施加源类别教师一致性。
9. 相似度并列时按稳定类别键排序决定源；记录所有候选的全局分数、模式分数、总分、最终来源与阈值回退原因。

### 验收

- 用手工相似度矩阵核对每行取 max 再平均的方向。
- 达到阈值时复制，不足阈值和历史为空时回退。
- 复制后初始参数相等，但不共享可训练存储；训练新类别不会修改源类别。
- 打乱同阶段类别遍历顺序，迁移来源与初始化参数不变。
- 相似度阈值只能过滤低相似候选，日志和说明不宣称它保证没有负迁移。

### 第 7 步实施记录（2026-09-13）

- `reid/adaptation/pgca.py` 新增配置、单向模式覆盖/综合分数、阈值选源、阶段开始 CPU 源快照及初始化器。
- 只从 seen_before 的旧类别原型选源，包含缺席类别；不使用当前 recurring 新中心，不允许同阶段新类别互借。>=delta 复制，空历史/低分回退；并列按类别键排序。
- 所有源参数在任何当前类别训练前复制保存；应用时不读活源参数。目标存储独立，只有新 Adapter 被初始化，新类别仍不加源教师蒸馏。
- Trainer 保留原有排序注册/分类头的 RNG 调用顺序；空历史 fallback 与默认基础训练更新相同。默认初始化 API/第 6 步 CLI 保持原行为。
- 新 `tools/train_pgca_stage.py` 默认 drift＋similarity，复用单阶段训练和原子组合保存。`pgca_stage` 和第 6 步 `pgca_recurring_stage` 可接续，baseline＋ECPM 仍可导入；完整连续调度待第 8 步。
- 日志/checkpoint 保存全部候选分数、最佳/实际来源、阈值原因、源/目标哈希及临时源快照内存；应用后释放源 CPU 副本。
- 14 项本步测试、147 项回归通过，真实 CLIP CUDA＋AMP 三阶段接受一次迁移并验证旧源版本/独立存储/训练/保存恢复。未运行正式 ReID 迁移效果实验，不宣称阈值保证无负迁移。

## 11. 步骤 8：串联完整阶段流程与恢复机制

### 目标

使前面的独立功能按正确时序运行，形成一个可复现的连续训练程序。

建议新增入口 `train_category_progressive.py`。现有 `train_stage1.py` / `train_stage2.py` 表示旧方案两个训练过程，与数据流 T1/T2 不是同一概念；新入口避免复用这个命名造成误解。

### 每个阶段的执行顺序

```text
载入上一个已完成阶段的状态
    ↓
固定 seen_before、历史原型快照和历史 Adapter 快照
    ↓
读取当前训练数据，确定 new / recurring / absent
    ↓
关闭 Adapter，提取当前身份原型
    ↓
生成本阶段 ECPM 候选更新状态
    ↓
recurring：准备同类别教师，计算漂移和蒸馏权重
new：用上阶段历史状态选源，创建并初始化 Adapter
    ↓
为每个当前类别创建临时分类器和新的阶段优化器
    ↓
只训练当前类别 Adapter 和临时分类器
    ↓
保存已完成阶段状态，提交 ECPM 更新，丢弃临时分类器和教师
    ↓
只用已到达类别的状态进行评估，然后进入下一阶段
```

旧/新类别判断必须基于 `seen_before`，不能基于已经加入新类别的 ECPM 字典。

### 配置与日志

最低需要这些配置组；以下是拟新增字段，并非现有命令行参数：

| 配置组 | 字段 |
|---|---|
| 数据 | stream_config、输入尺寸、预处理、随机种子 |
| Adapter | last_blocks、bottleneck_dim、scale |
| 训练 | 每类别迭代数、P/K、学习率、Triplet margin、lambda_tri |
| ECPM | 聚类方法、FINCH 层次、距离、原型 dtype |
| PGCA 旧类别 | consistency_mode、lambda_con、gamma |
| PGCA 新类别 | init_mode、alpha、delta |
| 推理 | routing_mode、beta |

`alpha`、`beta` 在 [0,1]；`delta` 按综合余弦分数范围 [-1,1] 校验；`lambda_con`、`gamma` 非负。待调参量不要写成已经验证的推荐最优值，使用验证集确定，不能用测试结果反复选阈值。

每阶段生成结构化日志，至少包含：类别状态、身份/模式数量、漂移、蒸馏权重、迁移来源、每类损失、训练步数、耗时及原型/Adapter 存储量。

### 检查点

保存：已完成阶段编号、已见类别映射、所有类别 Adapter、ECPM 状态、配置与协议摘要、参考权重标识、随机数状态。参考权重必须能可靠重建；若无法保证外部权重长期一致，则一并保存冻结主干。

**第 8 步实际实现（2026-09-14）：支持阶段边界及阶段内完整优化器更新边界恢复。** 按用户本步要求，超过原先仅阶段边界恢复的最低建议。`reference.pt` 保存不可变冻结主干；`latest.pt` 原子保存学生 bank、临时分类器、优化器、AMP scaler、采样位置、RNG、训练进度、候选 ECPM、历史教师和已执行迁移记录。当前没有学习率调度器，显式保存 `scheduler=None`。

阶段内恢复直接载入上述状态，不重做迁移或重建旧教师。默认每次更新后落盘；异常时仅重做最后有效断点之后的更新。提交原型之前保存最后一次更新，提交完成后保存阶段边界，避免身份重复累计。日志按断点字节前缀验证、归档多余尾部并回滚。精确路径限定单进程单设备、workers=0、内置增强、同一代码及软件/设备配置。命令、验证证据和限制见第 8 步使用说明。

### 验收

- 至少运行 T1(person/vehicle)、T2(person/panda)、T3(vehicle/person)，覆盖新类别、重复类别和缺席后返回。
- 完整运行与阶段边界恢复运行得到一致的身份记忆、迁移来源、权重和数值容差内的模型输出。
- 故意中断当前阶段后，重跑不会重复累计身份原型。
- 下阶段训练进程不打开历史训练图片，也不依赖保留历史训练 loader。
- 没有训练未来阶段 Adapter、提取未来原型或使用同阶段更新后的源模型。

## 12. 步骤 9：接入硬路由、评估与消融

### 目标

实现论文完整推理，并区分“类别 Adapter 学得如何”和“系统能否选对 Adapter”。

### 推理实现

建议新增 `reid/evaluation/prototype_router.py` 与 `reid/evaluation/category_progressive.py`，复用现有分块距离和排名计算。

1. 参考前向得到 \(z=\operatorname{Norm}(F_0(x))\)。
2. 只在当前已见类别中计算：

\[
r_c(x)=\beta\cos(z,m_c)+(1-\beta)\max_k\cos(z,g_{c,k}).
\]

3. 硬选择 \(\hat c=\arg\max_c r_c(x)\)。
4. 按预测类别分组，只对每组图像运行相应 Adapter，再恢复输入顺序。
5. 输出 \(f(x)=\operatorname{Norm}(F(x;F_0,A_{\hat c}))\)，query/gallery 用余弦相似度检索。
6. 两次完整视觉前向是首版可接受的实现：一次参考路由、一次类别 Adapter。由于 Adapter 插在 Transformer 内，不能直接把最终参考向量送入它来冒充第二次前向。
7. 路由不读取文件路径中的类别名称、数据集名或真实 category；类别标签仅由评估器用于统计。
8. 首版没有未知类别拒识机制。未在训练中出现过的物体类别若要评估，应列为额外泛化实验，不与“已见类别的新测试身份”混为一谈。

### 评估协议

主结果：每阶段对所有已见类别的固定、训练身份不重叠的测试集合评估；分别报告各类别 mAP/Rank-1 和类别宏平均。

至少保留两种路由结果：

- `oracle`：用正确类别 Adapter，只作为诊断，不能作为无类别标签方法的正式结果。
- `prototype`：按论文硬路由，是完整系统结果。

同时记录路由准确率、混淆矩阵、同一测试身份的跨图像路由一致性，以及 oracle 与 prototype 的检索差距。冻结的旧 Adapter 仍可能因新类别加入和原型更新而失去路由竞争，需要单独观察。

**工程建议：** 首版主评估保持各数据集/类别既有 gallery 协议，但路由候选仍是所有已见类别。另做混合类别 gallery 实验，验证不同 Adapter 输出之间的可比性。报告时明确是哪一种，不能把按类别 gallery 的指标写成全混合检索指标。

记录每类别性能矩阵 \(R_{t,c}\)，只对已出现类别统计。若类别首次出现于 \(t_c\)，可用：

\[
F_{t,c}=\max_{t_c\leq s\leq t}R_{s,c}-R_{t,c}
\]

表示截至阶段 t 相对于历史最好值的下降，mAP 与 Rank-1 分开计算。新类别第一次出现时该值为 0。

### 最低消融集合

| 实验 | 改动 | 要回答的问题 |
|---|---|---|
| 基础持久 Adapter | 默认初始化，无一致性，简单类别均值路由 | 收益是否只是参数隔离带来的？ |
| 固定蒸馏对照 | 同一 ECPM、初始化和路由，只将 drift 改为 fixed | 漂移控制优于固定权重吗？ |
| 初始化对照 | 其他保持一致，比较默认、随机历史源、仅全局相似度、完整模式匹配 | 原型匹配是否选到了更有效的迁移源？ |
| ECPM 对照 | 简单身份均值与聚类等权概括；再分开替换控制信号和路由 | 聚类帮助的是学习控制、路由，还是两者？ |
| 路由对照 | 同一检查点，oracle 与 prototype | 性能瓶颈在表征还是路由？ |

两项关键机制诊断：

1. 在验证集研究不同漂移值下固定蒸馏权重的效果，检查漂移是否能预测合适的保留强度。不要仅画“漂移越大、公式权重越小”的曲线，这只是公式本身。
2. 在小规模验证实验中枚举不同历史源，比较真实迁移收益与相似度排序。保持训练预算和重复次数可比，不用测试集选源。

首轮通过后，再做类别顺序、缺席间隔、随机种子、类别不平衡及固定原型预算实验。固定预算版本属于后续变体，需要明确采样/压缩规则，不能悄悄改变论文的全量原型累积定义。

### 验收

- 手工向量验证路由公式、max 维度、权重和 argmax。
- batch 内混合预测类别时，分组前向恢复后的图像顺序正确。
- 正式路由完全不依赖真实类别；query 与 gallery 独立决策。
- 保存/重载检查点前后，路由和最终描述符一致。
- 输出完整逐阶段指标与上述消融，能定位退化来自训练还是路由。
- 报告全量身份原型、模式、类别中心、Adapter 的字节数，以及聚类耗时、峰值内存和推理耗时。

### 第 9 步实际实现与验收（2026-09-14）

已新增 `PrototypeRouter` / `RoutedEncoder`、`evaluate_stage` / `lifelong_summary`。连续入口用 `--evaluate` 开启所有已见类别的逐阶段评估；默认关闭以兼容第 8 步的纯训练用法。评估失败保留已提交阶段，恢复后只重评，训练 RNG 不变。独立工具支持完成阶段检查点评估、9 个匹配预算的消融配方，以及从共同阶段前断点出发的来源/固定权重验证集枚举。

主指标保持各 EvaluationView 原 gallery，混合 gallery 单独输出。资源记录明确区分 CUDA 峰值、张量存储和 FINCH 距离块估计，未声称测得 CPU/native 聚类峰值。身份均值消融派生自相同已验证身份原型，仍保留 FINCH 存储以隔离控制与路由机制，不能据其耗时推断删除 FINCH 后的开销。诊断工具按已完成变体恢复；当前变体中断后从共同阶段起点重跑。

190 项回归包含 15 项本步检查，覆盖公式、gallery、数据泄漏、9 个消融执行、来源/权重验证集诊断和精确恢复。真实 CLIP CUDA＋AMP 三阶段 6 次更新与独立进程恢复后，Adapter、RNG、训练日志、路由、描述符和检索报告一致。证据为 `docs/step9_regression_tests.txt`、`docs/progressive_evaluation_smoke.json`、`docs/progressive_evaluation_smoke_metrics.json`。这些为工程验收，不是正式数据性能或顶会创新证据。

## 13. 建议文件组织

以下为最初目标布局。第 1—9 步已按实际接口完成；新增恢复、消融和诊断文件详见各步骤执行记录。

```text
train_category_progressive.py
config/category_progressive_example.json
lreid_dataset/category_stream.py
lreid_dataset/category_stream_loaders.py
reid/models/category_adapter_bank.py
reid/memory/__init__.py
reid/memory/ecpm.py
reid/adaptation/__init__.py
reid/adaptation/pgca.py
reid/loss/pgca.py
reid/trainer_category_progressive.py
reid/evaluation/prototype_router.py
reid/evaluation/category_progressive.py
tools/validate_category_stream.py
tests/test_category_stream.py
tests/test_category_stream_loaders.py
tests/test_category_adapter_bank.py
tests/test_ecpm.py
tests/test_pgca.py
tests/test_prototype_router.py
tests/test_category_progressive_resume.py
```

测试重点放在会改变实验结论的错误：身份泄漏、参考空间变化、梯度误冻结、参数共享、阶段快照错误、原型重复提交和真实类别泄漏。先用小张量与小模型验证公式/状态，再用少量真实图片跑三阶段；不用一开始就跑完整大数据实验。

## 14. 后续逐步执行方式

下一次可以直接指定：“按本文档实现第 1 步，完成验收并更新进度。”每一步完成后，在对应小节记录实际文件、验证结果和偏离建议的原因；未完成的步骤不提前打勾。

- [x] 步骤 1：数据流协议与审计（41 项测试通过，实际数据阶段划分需另行准备）
- [x] 步骤 2：冻结参考编码器与类别 Adapter（24 项新测试、46 项回归和完整 CLIP CUDA 冒烟通过）
- [x] 步骤 3：类别内基础 ReID 训练（21 项新测试、65 项回归和完整 CLIP 两阶段短训练通过）
- [x] 步骤 4：身份原型记忆（22 项测试、86 项回归、完整 CLIP CUDA 两阶段提取与恢复通过）
- [x] 步骤 5：完整 ECPM（21 项本步测试、108 项回归、真实 CLIP 三阶段及对称模式诊断通过）
- [x] 步骤 6：PGCA 熟悉类别分支（18 项本步测试、129 项回归、真实 CLIP CUDA＋AMP 三阶段短训练通过）
- [x] 步骤 7：PGCA 陌生类别分支（14 项本步测试、147 项回归、真实 CLIP CUDA＋AMP 双分支三阶段验证通过）
- [x] 步骤 8：阶段流程与检查点（完整更新边界恢复、日志回滚、13 项专项及真实 CLIP 恢复通过）
- [x] 步骤 9：路由、评估与消融（15 项本步检查，合计 190 项回归，真实 CLIP 三阶段评估与跨进程恢复通过）

**第一版完成标准：** 九步均通过验收，能够按论文定义训练和评估 ECPM + PGCA，并有足够日志检查核心假设。实现正确不等于创新已被证明，最终贡献判断依赖第 9 步的对照与机制证据。
