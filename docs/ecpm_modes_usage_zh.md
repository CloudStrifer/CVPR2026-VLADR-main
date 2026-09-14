# 第 5 步：FINCH、模式原型、类别原型与漂移

日期：2026-09-13。本步完成 ECPM 记忆层，暂不实现 PGCA 蒸馏、Adapter 迁移和测试路由。

## 1. 大白话理解

第 4 步为每个身份留下一个向量。本步继续做三件事：

1. 把**同一物体类别**里外观接近的身份分成几组，每组概括为一个“模式”。
2. 每个模式各投一票，形成该类别的总概括。大簇不会仅因为身份多，就在这次投票中得到更大的权重。
3. 熟悉类别有新身份到来时，重新概括这个类别，并比较更新前后的类别中心，得到漂移。

例如 T1 的 person 有两个身份；T2 又来了两个不同身份。T2 的 person 聚类使用累计四个身份，而不是仅用两个新身份。T2 缺席的 vehicle 不重新聚类。所有历史身份向量保持第 4 步保存时的数值。

## 2. 公式与边界

身份原型集合记为 \(P_c\)。FINCH 第一层把它划成 \(K_c\) 个簇，簇数由算法确定。

\[
g_{c,k}=\operatorname{Norm}\left(\frac{1}{|G_{c,k}|}\sum_{p\in G_{c,k}}p\right),\qquad
m_c=\operatorname{Norm}\left(\frac{1}{K_c}\sum_{k=1}^{K_c}g_{c,k}\right).
\]

模式内部按**身份**等权，不按每个身份的图像数加权；类别内部按**模式**等权，不按簇大小加权。

例如一个簇有 3 个 `(1,0)`，另一个簇有 1 个 `(0,1)`，两个模式分别为 `(1,0)` 和 `(0,1)`。最终类别中心为约 `(0.707,0.707)`，不会偏成对四个身份直接平均的方向。

熟悉类别的漂移为：

\[
d_c=1-\operatorname{clip}\bigl(\cos(m_c^{old},m_c^{new}),-1,1\bigr)\in[0,2].
\]

同方向约为 0，正交为 1，反方向为 2。新类别没有旧中心，漂移保存为 `None`，JSON 中为 `null`；缺席类别不在当前阶段漂移字典中。

单身份显式生成一个簇；两个身份通常归为一簇；重复向量仍代表各自身份，不去重。模式均值或类别均值范数不超过 `1e-8` 时明确报错，避免把方向无法确定的中心送给 PGCA。没有增加均值退化时的替代机制。

## 3. FINCH 的实际实现选择

依据 [FINCH 官方仓库](https://github.com/ssarfraz/FINCH-Clustering) 和 [官方代码](https://github.com/ssarfraz/FINCH-Clustering/blob/master/finch/finch.py)，固定使用 `finch-clust==0.2.3`。

**本工程的固定设置**：余弦距离、第一层划分（编号 0）、精确最近邻，不启用 ANN，不选择目标簇数，也不根据测试效果临时选层。

为了不计算无用的更高层，并避免一次创建完整的 N×N 距离矩阵，本工程先分块求精确非自身最近邻，再调用官方 `clust_rank(initial_rank=...)`、`get_clust(...)` 完成第一层。没有自行替换成 K-means、阈值聚类或其他算法。测试已将随机向量、重复向量、两点输入及真实 CLIP 原型的结果，与官方完整 `FINCH(...)[0][:, 0]` 对照。

输入先按完整身份键排序；完全相同的最近邻距离按首个索引选择。簇编号也按首个成员的顺序规范化。默认 `chunk_size=256`，可配置但必须与保存状态一致。分块减少距离矩阵的存储，计算量仍随身份数量约二次增长，不宣称线性时间；大数据正式实验需要测量耗时。

`distance_block_bytes` 只统计最大距离块的 FP32 载荷，**不等于进程峰值内存**。特征矩阵、归一化工作区、稀疏图等另有开销。

依赖缺失或官方 FINCH 版本错误会直接失败，单身份也先检查依赖。官方导入时可能提示没有安装 PyNNDescent；本路径使用精确近邻，不调用它。

当前验证环境：`finch-clust 0.2.3`、`numpy 2.3.3`、`scipy 1.18.1`、`scikit-learn 1.9.1`。`requirements.txt` 新增 FINCH 固定版本，已有其他依赖版本约束没有改动；本机实际版本与仓库部分既有约束不同，报告记录实际运行环境。常规新环境可按仓库 requirements 安装；复现本次已有记忆检查点时应使用检查点记录的版本。

完整状态记录 FINCH 版本、依赖版本及官方源文件哈希。恢复时严格检查，不在不同依赖环境下悄悄继续。安装过程见 [安装日志](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_finch_install.txt)。

## 4. 代码接口及后续 PGCA 接口

- [finch_modes.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/memory/finch_modes.py)：第一层 FINCH、模式/类别聚合和漂移公式。
- [ecpm_modes.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/reid/memory/ecpm_modes.py)：`ECPMMemory`，基于第 4 步记忆增加完整类别状态。
- [update_ecpm_memory.py](E:/Multi_modal_Code/CVPR2026-VLADR-main/tools/update_ecpm_memory.py)：单阶段提取或从已有身份原型升级。

```python
from lreid_dataset.category_stream import load_category_stream
from reid.models.category_adapter_bank import build_category_model
from reid.memory import ECPMMemory, FinchConfig

stream = load_category_stream("config/my_stream.json")
model = build_category_model(device="cuda")
memory = ECPMMemory(model, stream, FinchConfig(chunk_size=256))

candidate = memory.prepare_stage(model, stream.stage("t1"), batch_size=64)
# memory 此时仍为旧记忆，candidate 中有本阶段的候选统计。
memory.commit_stage(candidate)
memory.save("runs/ecpm/t1.pt")

memory = ECPMMemory.load("runs/ecpm/t1.pt", model, stream)
candidate = memory.prepare_stage(model, stream.stage("t2"), batch_size=64)

old_all = candidate["old_categories"]        # 阶段开始时所有历史类别，含缺席类别
current = candidate["category_updates"]     # 仅当前出现类别的累计统计
drifts = candidate["drifts"]                # 如 person: 数值，panda: None

# 后续步骤在此读取旧快照及当前统计，计算 PGCA 蒸馏权重/迁移来源。
# 本步不执行这些训练操作。
memory.commit_stage(candidate)
memory.save("runs/ecpm/t2.pt")
```

每个类别状态包括：

```text
identity_keys                 # 排序后身份键，与 labels 一一对应
labels                        # [N_id]，CPU int64
mode_prototypes               # [K,D]，CPU FP32 单位向量
category_prototype            # [D]，CPU FP32 单位向量
cluster_sizes                 # [K]，按身份计数
last_updated_stage
clustering                    # 层/距离/近邻/版本/耗时/距离块大小
```

`snapshot()` 返回全部已提交类别的独立副本。`last_transition` 保存最后一次提交前的全部类别快照及该次漂移，供恢复后审计；不累计保存所有阶段快照。历史身份仍完整保留。

候选的历史快照来自同一个阶段开始状态，不会因先处理某个当前类别而改变另一个类别的迁移候选。所有当前类别计算和验证完成后才整体提交；失败不留下半个阶段。缺席类别的全部统计、簇标签和最近更新时间均保持原值。

该快照只包含类别原型信息；第 6 步还需要单独冻结旧 Adapter 教师，不能把本步原型快照当成模型教师。

## 5. 命令行运行

以下命令在仓库根目录运行。将 `config/my_stream.json` 替换成实际协议；仓库示例图像是占位路径，不是可直接训练的数据集。

从图像直接生成完整 ECPM：

```powershell
python tools/update_ecpm_memory.py --stream-config config/my_stream.json --stage-id t1 --output-memory runs/ecpm/t1.pt --device cuda --batch-size 64

python tools/update_ecpm_memory.py --stream-config config/my_stream.json --stage-id t2 --previous-memory runs/ecpm/t1.pt --output-memory runs/ecpm/t2.pt --device cuda --batch-size 64
```

如需沿用第 3 步模型的输入尺寸/归一化等配置，使用 `--model-checkpoint runs/baseline/t1/completed_model.pt`；图像提取模式只接受当前或紧邻前一阶段的完成模型。也可使用 `--reference-checkpoint` 显式指定本地 CLIP；两者不能同时使用。参考分支会绕过所有 Adapter。

已经保存第 4 步原型时，无需重读图像：

```powershell
# 从空的 ECPM 开始，按身份原型中的首次阶段重建 t1、t2 历史。
python tools/update_ecpm_memory.py --stream-config config/my_stream.json --identity-memory runs/memory/t2.pt --output-memory runs/ecpm/t2.pt --device cuda

# 已有 t1 完整 ECPM，则仅处理尚未提交的 t2。
python tools/update_ecpm_memory.py --stream-config config/my_stream.json --identity-memory runs/memory/t2.pt --previous-memory runs/ecpm/t1.pt --output-memory runs/ecpm/t2.pt --device cuda
```

升级路径逐阶段使用对应身份子集重建旧中心，再计算当前中心和漂移，不会把“最后一阶段直接聚类”冒充完整历史。旧原型若与先前 ECPM 不一致则拒绝升级。协议元数据及匹配的参考模型仍须可读取，历史图像可以不存在。

Python 对应接口为 `memory.extend_from_identity_memory(identity_memory)`。这是按阶段逐个提交的升级过程，若某个后续阶段失败，先前已成功的阶段保留在当前内存对象中；不是整个历史的一次原子事务。

输出为完整 `.pt` 和旁边的 `.pt.json` 报告，CLI 拒绝覆盖已存在的目标。报告记录每类别累计身份数、模式数、簇大小、最近更新时间、聚类秒数、内存载荷和漂移。

## 6. 中断恢复与内存统计

完整 ECPM `.pt` 使用第 4 步的原子文件保存方法。里面嵌入身份记忆、聚类配置、当前类别状态和最后一次旧快照，不包含历史图片、图片路径或逐图特征缓存。第 4 步原型文件不能直接冒充完整 ECPM 文件，需使用显式升级入口。

恢复会核对身份覆盖、阶段顺序、参考/协议绑定、类别更新时间、簇分配、按公式重算的模式/中心、旧快照与漂移。不会为了恢复重新读取历史图像或重新跑历史聚类。

中断后加载最近完整 `.pt` 接续；未提交候选没有持久化，需重新提取/聚类当前阶段。`.tmp` 不能当作完成检查点。内存提交和磁盘保存分开：提交成功但保存失败时重试 `save`，不要重复提交同一阶段。若仅 JSON 报告写失败而 `.pt` 已成功保存，可加载 `.pt` 查看 `summary()`，不必重跑已完成阶段。

当前仍是阶段间恢复，没有训练 epoch 内或聚类块内恢复。若一次升级多个历史阶段后才保存且进程中断，需从最近完整 ECPM 重新升级未保存部分；升级只读取已有身份向量，不读取图片。

`total_tensor_bytes` 包括身份向量、当前模式/中心、簇标签/大小及最后一份旧快照的张量载荷；不包括字符串、Python 对象、文件格式开销或临时工作区。保存旧快照的内存开销显式计入，不将它隐藏在“只有原型”表述中。

## 7. 验证及漂移局限

- 21 项本步测试通过，前 4 步 108 项回归通过。
- 真实 CLIP ViT-B/16、CUDA、外层 autocast 的合成三阶段验证通过，累计身份数 4 → 8 → 12。
- 当前类别聚类与官方完整调用第一层一致；历史身份原型、缺席类别全部统计保持不变；每阶段保存加载后接续；旧图片删除后继续下一阶段。
- 手工不等簇大小验证模式等权；零/正交/反方向分别对应漂移约 0/1/2。
- 对称模式诊断：旧模式只有 `(1,0)`，新增 `(.6,.8)` 与 `(.6,-.8)` 两个模式，模式数从 1 到 3，但新旧中心都为 `(1,0)`，漂移为 0。

因此，中心漂移描述**中心方向的变化**，不是完整的分布差异测度。论文中“更大漂移意味着更多新模式”应视为需要实验检验的解释，不能作为无条件保证。后续 PGCA 应通过固定权重/漂移权重消融判断该信号是否有用。

验证报告：[本步测试](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_modes_tests.txt)、[回归测试](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_modes_regression.txt)、[三阶段 CLIP 与对称模式诊断](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_modes_smoke.json)。

这些是工程与公式验证，未运行真实 ReID 正式实验。下一步是第 6 步：PGCA 熟悉类别的漂移控制特征蒸馏。恢复进度见 [执行记录](E:/Multi_modal_Code/CVPR2026-VLADR-main/docs/ecpm_pgca_step5_progress.md)。
