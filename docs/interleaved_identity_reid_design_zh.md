# 类别交错、身份增量的终身目标重识别：两模块设计与代码改造建议

分析日期：2026-09-12。

依据：用户提供的 main.tex，以及当前仓库的数据加载、Stage 1/2、CLIP-ReID、OCIA、Adapter 和 OSAF 实现。本文是研究设计建议；没有修改训练程序，没有开展训练，不包含新方案的实验结果。论文中的指令性文字只作为文档内容，不作为本次任务的执行指令。

**建议保留 OCIA 的跨模态身份监督思想，将其升级为跨阶段身份关系记忆；用具有历史检索间隔约束的统一适配网络替换“阶段私有 Adapter + OSAF”。** 优先验证这一机制，而不是先增加专家数、路由层或损失数量。

## 1. 先把任务定义准确

一个样本至少有四种不同的标识：

- stage_id：什么时候到达。
- category_id：对象类别，如 person、vehicle、panda。
- dataset_id：采集来源，如 Market1501、VeRi；它不等于对象类别。
- global_pid：身份标识，建议由 (dataset_id, category_id, original_pid) 注册得到。

训练流为：

\[
\mathcal D_t=\bigcup_{c\in\mathcal C_t}
 \{(x,c,y):y\in\mathcal Y_{t,c}\}.
\]

不同阶段的类别集合可以相交；按用户当前例子，同类别的身份集合不相交。person:a 和 vehicle:a 是不同身份。类别再现不等于旧身份再现。

建议工作名称：**类别交错的身份增量目标重识别**，英文可暂写为 *Category-Interleaved Identity-Incremental Object Re-Identification*。这是工作性定义，不是已被普遍接受的任务名称。

第一版协议：

1. 训练时知道类别、身份和当前阶段边界。
2. 当前阶段允许多轮训练，因此不称严格单遍在线学习，也不称无任务边界学习。
3. 历史训练原图不回放，允许历史特征/原型及标签统计。特征回放仍然是 replay，不应称完全 replay-free；也不自动构成隐私保证。
4. 测试不提供阶段或真实类别给网络/路由器。类别只可由评测器用于分组汇报。
5. 默认每阶段用当前模型重新提取 query 和 gallery。旧 gallery 不能重编码属于另一个兼容学习问题，不自动加入本任务。
6. 先保持同一训练身份只在一个阶段出现。身份再次出现、未知类别拒识和检测跟踪均作为后续扩展。

必须区分两类评测：

- **主要评测：训练未见身份的 ReID 泛化。** 各数据集保留训练/测试身份互斥；只拆分训练身份形成学习流，固定测试 query/gallery。
- **补充评测：已注册身份的历史识别。** 若要直接观察 person:a、b 被遗忘，可为这些训练身份留出独立图像/视角，但应另列协议，不能与训练身份互斥的标准 ReID 混报。

## 2. 当前实现能保留什么，哪里不适配

| 位置 | 已核实的实现 | 新设定下的影响 |
|---|---|---|
| reid/utils/feature_tools.py，build_cross_modal_identity_anchors | 当前域均值中心化、类别文本子空间投影、融合身份文本特征 | 核心身份监督可复用；混合阶段不能共用一个阶段中心 |
| reid/trainer_stage2.py:200–236 | 只选择当前 batch 的 unique_labels 对应锚点，计算对比分类 | 不包含历史身份作为正样本或历史关系约束 |
| train_stage2.py:713，_set_domain_adapter_stage2_mode | 冻结共享参数，仅开放当前 Adapter 与分类器 | 参数隔离是现有知识保持的主要来源 |
| train_stage2.py:1027–1110 | 按 all_train_sets 循环，每项一个 name，一个 Adapter | 阶段、域和知识单元绑定 |
| lreid_dataset/datasets/manifest_reid.py:232 | 行 category 必须等于整个 spec 的 category | 直接把多类别 CSV 塞给一个训练项会报错 |
| 同文件:346 | 输出 (path,pid,camid,0) | category 与真实 source 信息未传到训练 batch |
| reid/models/CLIP_ReID/model/make_model_clipreid.py:70 | 一次构造使用一个 object_noun | 同一阶段混合身份需要按身份选择不同类别模板 |
| reid/evaluation/adapter_fusion.py:167 起 | 每个 Adapter 一个语义键，视觉分数是与其原型的最大相似度 | 同类别多阶段语义键区分力小，多类别阶段语义键表达不足 |
| reid/evaluation/fast_test.py:122 起 | 提取基础描述子，运行被选中的各 Adapter，再融合 | 即使旧参数不变，新专家加入也可能改变旧图像的最终输出 |

**OCIA 保存固定锚点不等于保存图像到锚点的映射。** 即使历史锚点仍在，若没有旧输入、功能约束或参数隔离，后续更新依然可能改变旧图像的特征。

OSAF 也不是数学上不能处理混合阶段。可以改成多个 category key 或 stage-category 专家；但参数冗余、类别再现时的知识分散、历史路由变化和推理成本会更加明显。

此外，OSAF 的 max-prototype 打分存在原型数量相关的选择效应：给某个候选集合添加原型时，其最大相似度不会下降。各候选覆盖量不同会影响路由，需要等预算或校准。该性质不等于已证明你现有结果出现路由偏差。

当前命令行默认 adapter-routing=oracle、adapter-routing-scope=unseen。若旧方法作为“全程无域标签”基线，需显式核实使用 osaf + all；不能仅根据论文措辞推断运行时设置。_osaf_descriptor 当前会对一个 batch 中被任何样本选到的每个 Adapter 运行整个 batch；所以 top-k=2 不保证每 batch 只额外做两次前向。

## 3. 值得围绕的核心问题

以 person 为例：

- t1 仅见 a、b、c，模型学习区分这些身份。
- t2 只见 d、e，类别仍然是 person，但旧人像不再可用。
- 普通当前 batch 损失只约束 d 和 e，不要求它们与历史 a、b、c 分离。
- 对“当前 d、e 图像”蒸馏旧模型，只约束旧模型在新输入上的行为，不能充分反映旧 a、b、c 的判别关系。
- 同时训练 panda 和 tiger 时，共享参数还可能破坏暂时缺席的 vehicle 内部身份间隔。

由此可形成两个明确任务：

1. **记住什么：** 同类别中哪些身份容易混淆、同一身份跨视角如何聚合，以及这些关系如何跨阶段连接。
2. **如何更新：** 允许加入新身份、改善旧关系，但抑制历史有效检索间隔被后续训练压缩。

不建议将“维护每阶段分类精度”作为主线。标准 ReID 最终面对的通常是训练未见身份，真正需要保持的是可迁移的身份度量能力。

## 4. 模块一：跨阶段身份关系记忆 PIRM

暂用英文：*Persistent Identity Relation Memory*。名字只是便于讨论，不能代替贡献证据。

### 4.1 复用 OCIA，但记忆按 category → identity 组织

每个已到达身份保存以下信息：

~~~python
IdentityMemoryEntry(
    global_pid,
    category_id,
    dataset_id,
    first_seen_stage,
    text_anchor,       # 冻结文本编码器输出，512 维
    visual_mean,       # 冻结图像编码器原型
    feature_probes,    # 少量冻结特征，保留视角差异
    counts,
    relation_edges,   # 同类别近邻身份及可靠检索关系
)
~~~

最小版本每身份保存 2–4 个特征探针，可先按真实摄像头分组求均值；无摄像头信息时使用特征聚类，并明确它只是视角近似。聚类均值不是真实图像，不能声称完整保存了身份分布。

与纯原型向量相比，多探针用来构造同身份正对；与全局蒸馏相比，同类别近邻用来聚焦细粒度混淆。无需额外引入 GNN，稀疏邻接表已经足够实现。

### 4.2 OCIA 改为类别条件，而不是阶段条件

\[
r_y=(I-\rho U_cU_c^\top)(v_y-\mu_c),\qquad
a_y=\operatorname{norm}\left[t_y+\lambda_r\operatorname{norm}(r_y)\right].
\]

- U_c 来自该类别的固定文本模板，不来自混合阶段。
- μ_c 使用截至当前已到达身份的类别统计；不允许访问未来身份。
- 小样本类别先采用软去偏，ρ 在验证集上选择；不要直接断言“类别子空间内全部都是无用信息”。车辆轮廓、动物身体结构也可能携带身份线索。
- 数值上要处理近零残差、单身份类别和退化子空间。

**必须解决中心更新造成的目标坐标不一致。** 如果 μ_c 更新，只给新身份计算新中心下的锚点、保留旧身份旧中心下的锚点，会混用不同定义。

建议：保存原始 v_y 和 t_y；每阶段开始根据当时可用的同一 μ_c 为该类别全部保留身份重建锚点，整个阶段内冻结；锚点快照记录版本号。这样目标是阶段内固定，而不是永久固定。原始冻结特征的坐标始终稳定。历史关系目标独立保存，不能因重建锚点被悄悄覆盖。动态锚点与历史关系约束可能冲突，需记录损失并做消融。

累计类别均值可通过“每身份均值之和 / 身份数”维护，避免大身份按图片数量主导；若将来支持同身份再次到达，需要更新该身份统计差量，不能作为新身份重复计数。

### 4.3 让旧身份真正参与新身份学习

当前样本 i 的身份锚定损失：

\[
\mathcal L_\mathrm{anchor}(i)=
-\log\frac{\exp(\operatorname{sim}(\hat q_i,a_{y_i})/\tau)}
{\sum_{j\in\mathcal B(c_i)}
\exp(\operatorname{sim}(\hat q_i,a_j)/\tau)}.
\]

其中 B(c_i) 是同类别的当前身份与保留历史身份，始终包含正身份。大记忆库可采样困难负身份，但要记录分母的采样策略。

这样 t2 的 person:d 不仅与 person:e 区分，也看到历史 person:a、b、c。类别标签用于训练时选择损失候选；推理模型不接受真实类别。

不要在主身份损失里让大量明显不同的跨类别负例主导梯度。跨类别区分可由共享表征和一个较轻的全局目标支持。类别内只有一个身份时，应从历史记忆取同类别负例；没有则跳过该项，并记录有效样本数。

### 4.4 保存关系，不只保存位置

使用同类别三元组 (u,p,n)：

- u、p：同一身份的不同探针。
- n：同类别、不同身份的困难近邻探针。
- 按 global_pid 判断正负，stage_id 不参与身份相等判断。

保存该关系建立时模型的可靠间隔：

\[
\Delta^\mathrm{ref}_{upn}
=s(z_u^\mathrm{ref},z_p^\mathrm{ref})
-s(z_u^\mathrm{ref},z_n^\mathrm{ref}).
\]

仅为标签支持、参考模型排序正确的关系建立“保持”目标；错误关系由监督损失纠正，不应当作必须保留的知识。对过大间隔设置上限，避免大量容量用于维持已经十分容易的三元组。

类别再现后，将新旧身份之间的困难关系接入该类别图。暂时缺席的类别仍保留探针与边，供下一模块约束。这里的潜在贡献是**跨时间出现的身份关系连接和维护**，而不是“用了一个 graph”。

## 5. 模块二：检索间隔保持适配 RMPA

暂用英文：*Retrieval-Margin-Preserving Adaptation*。

### 5.1 第一版用统一残差网络替换阶段专家

为确保历史特征能够约束所有可训练视觉参数，先采用清楚的网络边界：

\[
h_0(x)=[f_0(x)\Vert q_0(x)],\quad
\hat h_\theta=h_0+W_\mathrm{up}\sigma(W_\mathrm{down}\operatorname{LN}(h_0)),
\quad z_\theta=\operatorname{norm}(\hat h_\theta).
\]

f_0、q_0 来自冻结 CLIP；分支缩放/归一化方式在建库前固定并保存配置。ViT-B/16 在当前实现对应 768+512=1280 维。输出后 512 维可作为与文本锚点对齐的 q 分支。

- 从 t1 起冻结 CLIP，持续更新同一个残差网络。
- 升维层零初始化，使初始行为接近基础特征。
- 历史记忆保存 h_0，而不是随 θ 变化的 z_θ。
- 当前图像和历史 h_0 都进入同一个可训练网络。
- CE、Triplet、记忆关系损失都必须连接到该网络实际输出。
- 所有类别推理走相同路径，无阶段路由，无需 OSAF。

这里的残差网络仍可叫 Adapter。替换的是“按阶段私有、只在推理融合”的组织方式，不需要因为名字而彻底抛弃 Adapter 结构。

**重要限制：最终特征回放不能保护其上游可训练 Transformer Adapter。** 保存最后的 512 维原型，然后声称它能蒸馏第 7–12 层 Adapter，是梯度路径错误。

若最终特征头不足以学习细粒度信息，第二版可以冻结前缀、在后缀中使用一组共享 Adapter，记忆保存冻结前缀输出的 token 序列。回放从完全相同的切分点进入。这样能训练后层，但记忆成本显著增大；压缩后的 token 均值仅是近似输入，必须验证。

### 5.2 保护历史检索间隔，允许旧关系改善

对模块一保存的历史关系：

\[
m^\mathrm{ref}_{upn}=\min(\Delta^\mathrm{ref}_{upn},m_\mathrm{cap}),
\qquad
\mathcal L_\mathrm{keep}
=\mathbb E_{(u,p,n)\in\mathcal E_\mathrm{old}}
\left[m^\mathrm{ref}_{upn}
-\big(s(z_\theta(u),z_\theta(p))-s(z_\theta(u),z_\theta(n))\big)\right]_+.
\]

只使用 Δref>0 的可靠关系，并按类别均衡聚合。其含义：

- 旧 a 的跨视角正匹配相对旧 b 的优势若被压缩，产生惩罚。
- 新训练让这条关系变好，不要求恢复旧的准确坐标或旧相似度。
- 因而比“所有旧输出必须逐点等于教师”更允许知识迁移。

这是可检验设计，不是“绝不遗忘”的保证。约束只覆盖保存的探针与关系，且软惩罚未必优化到零；也不能直接保证真实图像上的 mAP。

不要每阶段无条件把退化后的模型间隔设成新目标，否则会逐步接受遗忘。初版保留建边时的参考间隔；如更新参考，只在可靠性检查后接纳改善，且有 m_cap 上限。关系被预算淘汰后，不再宣称它仍受到保护。

### 5.3 显式处理新旧身份冲突

除当前 batch 的类别内 Triplet 外，再加入同类别、跨到达阶段的三元组：

- 当前 d 的两视角作正对，旧 a 的探针作负例。
- 旧 a 的两探针作正对，当前 d 作负例。

两个方向都进入同一个模型，避免只把新样本推离固定旧锚点，却没有约束旧探针相对于新身份的位置。困难身份候选先在冻结空间筛选，再在当前输出空间选择困难负例；选择索引不需要求导。

将当前与跨阶段三元组一起计入 L_rank。由此主目标可以控制为：

\[
\mathcal L=
\mathcal L_\mathrm{ID}
\lambda_a\mathcal L_\mathrm{anchor}
\lambda_r\mathcal L_\mathrm{rank}
\lambda_k\mathcal L_\mathrm{keep}.
\]

其中 L_ID 为明确类别候选范围的身份 CE；可做去掉可学习分类器、仅采用锚定 CE 的简化对照。不要同时堆入 SCSD、全局 KL、EWC、正交投影和多个额外分类器后再难以解释收益。

### 5.4 与模块一的关系

- PIRM 决定要保存、回放和连接的身份知识。
- RMPA 决定共享模型如何吸收新身份且抑制旧排序间隔退化。
- PIRM 的跨阶段边提供 RMPA 的约束；RMPA 学到的新可靠关系在阶段结束后写回 PIRM。

所谓“两模块”，是这个记忆构建与受约束适配的闭环，不是把四种通用损失随意分成两组。

## 6. 主训练循环应如何改

以下为接口级伪代码，不是可直接运行的实现：

~~~python
encoder.eval()
encoder.requires_grad_(False)
memory = IdentityRelationMemory(...)
registry = IdentityRegistry(...)
head = UnifiedResidualAdapter(...)

for stage in stream:
    records = stage.current_training_records()
    registry.register(records)

    # 仅当前阶段身份。禁止提前加载未来 Stage 1 身份 prompt/原型。
    text_features = learn_current_identity_prompts(records, registry)
    current_probes = extract_frozen_probes(encoder, records)
    anchors = memory.build_stage_anchor_snapshot(
        current_probes, text_features, registry
    )

    for batch in make_category_identity_batches(records):
        with torch.no_grad():
            h_now = encoder.encode_base_descriptor(batch.images)

        # 采样历史关系时必须一并获取它的全部端点。
        replay, old_edges = memory.sample_relation_batch(
            active_categories=batch.category_ids,
            include_absent_categories=True,
        )
        z_now, q_now = head(h_now)
        z_old, q_old = head(replay.frozen_features)

        loss = category_identity_ce(z_now, batch.global_pids)
        loss += anchor_loss(q_now, anchors, batch)
        loss += category_rank_loss(
            z_now, z_old, batch, replay,
            include_cross_stage_pairs=True,
        )
        loss += historical_margin_loss(
            z_old, old_edges, reference="stored_reliable_margin"
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    memory.commit_current_identities(current_probes, text_features)
    memory.connect_new_old_relations(head, registry)
    evaluate_fixed_query_gallery(encoder, head)
    save_checkpoint(head, memory, registry, stage)
~~~

实际实现时，采样单位需要能给 absent categories 带回同身份正对与同类负身份；不能只随机取单个原型导致 keep loss 常年没有有效三元组。

记忆更新应在阶段边界或明确的在线规则下进行，不把测试集反馈用于选记忆。采样、聚类、困难负例候选选择和舍弃策略都要固定随机种子并写入配置。

## 7. 文件级改造清单

建议另加新入口，保留旧 OCIA+OSAF 作为对照。

| 文件 | 具体建议 |
|---|---|
| 新增 tools/build_interleaved_stream.py | 在每个原始数据集训练身份内分片，按类别再现日程组成混合阶段；输出 manifest、身份注册表、阶段审计报告 |
| 新增 config/interleaved_stream.json | 显式区分 categories、datasets、stages、evaluation；每阶段含多个来源及身份子集 |
| 新增 lreid_dataset/datasets/stream_manifest_reid.py | 逐行携带 category_id、dataset_id、stage_id、original_pid、global_pid；不要绕过旧 ManifestReID 的单类别检查 |
| reid/utils/data/preprocessor.py 或新 StreamPreprocessor | 返回具名字段/dict，停止依赖第四位置同时表示域/类别/阶段 |
| reid/utils/data/sampler.py | 新增 CategoryIdentityBatchSampler；按类别 → 身份 → 图像采样，至少形成同类异身份负例 |
| train_stage1.py 的逻辑 / 新 prompt 函数 | 从全数据集预训练改为当前阶段身份 prompt 学习；未来身份不出现 |
| make_model_clipreid.py 的 PromptLearner | 按 global_pid 选择对应类别的 prefix/suffix/tokenized prompt；现有逐行 buffer 机制可复用 |
| reid/utils/feature_tools.py | 类别条件的原型/子空间/锚点构造；返回原始均值与计数，支持一致快照重建 |
| 新增 reid/memory/identity_relation_memory.py | 特征探针、锚点快照、全局身份键、近邻关系和历史间隔；保存预算及版本信息 |
| 新增 reid/models/unified_residual_adapter.py | 对冻结特征的统一残差变换；输出与实际检索相同的描述子 |
| reid/models/CLIP_ReID/model/make_model_clipreid.py 或新 wrapper | 暴露 encode_base_descriptor、forward_from_features；分类器接新描述子 |
| 新增 reid/loss/identity_relation.py | 类别内锚定、跨阶段困难三元组与单侧历史间隔保持；实现空候选处理 |
| 新增 reid/trainer_interleaved.py / train_interleaved.py | stage 流、在线 prompt、记忆回放与原子 checkpoint；移除 add_num 连续切片假设 |
| reid/evaluation/fast_test.py 或新评测入口 | 统一网络单次编码；所有阶段评估同一固定 query/gallery，输出类别曲线 |

接口建议：

~~~python
batch = {
    "images": ..., "paths": ...,
    "global_pids": ..., "category_ids": ...,
    "dataset_ids": ..., "camera_ids": ..., "stage_ids": ...,
}

checkpoint = {
    "stream_hash": ...,
    "completed_stage": ...,
    "encoder_version": ...,
    "preprocess_config": ...,
    "head": ...,
    "optimizer": ...,
    "scheduler": ...,
    "identity_registry": ...,
    "category_statistics": ...,
    "anchor_snapshot_version": ...,
    "memory": ...,
    "relation_references": ...,
    "rng_states": ...,
}
~~~

当前类别切片不能用 global_pid 的连续区间代替，应用显式候选索引并重映射 CE target。新分类器若扩容，优化器必须更新参数引用；断点恢复后也要一致。

当前模型的分类 CE 使用 f 分支，Triplet/锚定使用 q 分支。新头如果只插到 q 分支，原来的 CE 可能完全训练不到新头。建议训练与评测统一基于同一新描述子，并为文本锚定单独取对齐分支。

## 8. 预算、规模与最小实验

第一版可以保留所有训练身份的少量特征以验证机制，但必须承认记忆随身份数量增长。统一头的参数量固定，不意味着总存储固定；文本向量、prompt、身份映射、关系边也要计入。

以每身份 K=4 个 1280 维 FP16 探针为例，仅探针约为 4×1280×2=10,240 字节/身份；1,000 个身份约 9.77 MiB，不包含文本、原型、边和元数据。若保留 token 序列，应重新测量，不能沿用此数字。

正式实验至少包含固定总字节预算。可按类别平衡分配，再比较 reservoir、随机身份、困难关系覆盖等策略。长尾类别最低配额可作为协议设置；类别数持续增加时，最低配额是否可行需显式限定。

只保留全部身份的可学习四 token prompt 也会增长。第一版 Stage 1 后可只存冻结 text_features，丢弃不再优化的旧 prompt 参数。若研究旧身份再次到达时更新 prompt，另列其存储成本。

建议分三个实现阶段：

1. **协议验证。** 原 CLIP 冻结检索、混合流顺序微调、原 OCIA+阶段 Adapter/OSAF。先确认新设定下旧方案究竟是模型遗忘、路由变化还是根本没有明显下降。
2. **机制验证。** 统一残差头 + 同预算随机特征回放，随后加入 PIRM 和 RMPA。第一轮可用 person+vehicle+panda，6 个阶段；按各类别可用训练身份数合理分配，不强行要求每阶段相同身份数。
3. **完整验证。** 五类别、较长流、缺席后再现、类别不平衡，多个 stream seeds；若冻结最终特征头性能不足，再换 token 接口版本。

“每类别一个 Adapter 并持续更新”可以作为低成本中间基线；它能缓解阶段膨胀，但同类别旧身份仍会遗忘，不能单靠它解决问题。

## 9. 必需的对照与判别实验

### 9.1 基线

- Frozen CLIP：没有参数遗忘，检验是否只是预训练特征足够。
- 共享网络顺序微调：观察混合流的真实遗忘。
- 原 OCIA + 阶段 Adapter + OSAF：显式无 oracle，混合阶段需先实现合理的多类别 routing entry；不能故意使用不适配的单类别键当弱基线。
- 类别 Adapter 持续更新。
- 统一头 + 类别平衡的随机 latent replay。
- 统一头 + 普通 prototype replay。
- 统一头 + 普通点对点蒸馏 / 关系 KL。
- DKP 类原型方法、SEMA 类适配器方法，在同协议下进行可复现适配。
- 使用截至当前全部原图的联合训练：作为额外数据访问条件下的参考，不作为合规主方法。

保持骨干、初始化、增强、训练更新次数和存储预算可比。第一阶段联合训练骨干的旧方案与从头冻结骨干的新头，应另有共同初始化/共同冻结条件的对照，否则收益混入初始化差异。

### 9.2 两模块消融

设公共底座为统一头+同预算随机特征回放：

- 底座。
- 底座 + PIRM：类别条件跨模态锚点与跨阶段关系采样，使用普通监督，不使用历史间隔项。
- 底座 + RMPA：相同数量随机合法三元组的历史间隔约束，不使用 PIRM 的困难关系组织和跨模态锚点增强。
- 底座 + PIRM + RMPA。

另比较“与旧输出相等”对“历史间隔不下降”，以及“只约束当前 batch”对“同类别新旧身份共同约束”。各变体必须使用同一记忆预算和相近关系计算量。

### 9.3 直接证明动机的实验

1. **参数不变时的路由遗忘。** 固定旧专家，逐步加入新专家，在固定旧测试集合记录描述子变化、路由占比和 mAP。由此检验 OSAF 的候选扩张是否实际造成遗忘。
2. **同类别跨阶段混淆。** 在同类别、不同到达批次的身份上统计距离和困难负例来源；将训练记忆诊断与未见测试身份检索分开汇报。
3. **缺席时长。** vehicle 消失 1、3、5 个阶段，比较其固定测试集 mAP 与保存关系的违约率。
4. **再现带来的迁移。** 学 vehicle 新身份后，评估原 vehicle 测试身份；只有性能确实提升时才称正向知识积累。
5. **记忆预算曲线。** 降低总字节预算，比较随机 replay 与关系记忆；若只在全量身份记忆下有效，方法的实用贡献受限。
6. **锚点对照。** 文本-only、原始 OCIA、类别历史中心、软投影；验证语义去偏是否真的有帮助。

### 9.4 指标

记 A[t,c] 为训练完阶段 t 后，在类别 c 固定测试集上的 mAP。只对截至 t 已引入类别计算宏平均，并同时报告每类别曲线：

\[
F_{T,c}=\max_{s\in[t_c,T]}A[s,c]-A[T,c].
\]

t_c 为类别首次出现的阶段。由于类别会再现，不能不加解释地沿用“一个域只出现一次”的对角性能矩阵；应记录首次学习、缺席、再次学习的位置。

跨类别 pooled gallery 可作为补充，更能检验统一检索，但必须：

- 跨数据集 PID 使用命名空间。
- 摄像头也按 dataset 注册，防止不同来源同 camid 被错误排除。
- 保留真实跨摄像头匹配规则；不能人为编摄像头 ID 制造有效匹配。
- 仍报告各类别内部 mAP，防止类别区分掩盖身份区分。

五类别各对应一个数据集时，类别变化与数据集变化高度混杂。建议至少为 person 或 vehicle 加第二个数据来源，分别测类别再现与域偏移；做不到时应将结论限制为该基准，不能声称完全解耦二者。

## 10. 新颖性边界与已核对文献

本次为定向检索，不是穷尽式系统综述。当前没有可调用的专用学术搜索 MCP；采用网页搜索并核对 arXiv、CVF、OpenReview 的原始论文页面/PDF。个别 CVF HTML 返回 403，使用可访问的论文 PDF/检索全文信息，不将失败页面当作已阅读全文。重复的预印本、会议页面和作者仓库按同一工作处理。

| 工作 | 已有内容 | 对本方案的约束 |
|---|---|---|
| [Class-Incremental Learning with Repetition, CoLLAs 2023](https://arxiv.org/abs/2301.11396) | 类别自然再现的持续学习流及评估 | 不声称首次提出类别重复出现；本任务需突出类别/身份两层和检索目标 |
| [Distribution-aware Knowledge Prototyping, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Distribution-aware_Knowledge_Prototyping_for_Non-exemplar_Lifelong_Person_Re-identification_CVPR_2024_paper.pdf) | 用身份分布原型保持、获取终身 ReID 知识 | 原型、分布和非原图记忆本身不是足够差异 |
| [SEMA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Self-Expansion_of_Pre-trained_Models_with_Mixture_of_Adapters_for_Continual_CVPR_2025_paper.pdf) | 复用/扩展 Adapter 和可学习混合路由 | 动态专家池+路由本身已有明确先例 |
| [Latent Replay for Real-Time Continual Learning](https://arxiv.org/abs/1912.01100) | 中间激活回放，稳定底层表示以保持缓存有效 | 冻结编码器+特征回放是实现基础，不能独立包装成新模块 |
| [Gradient Projection Memory, ICLR 2021](https://openreview.net/pdf?id=ZK4Cvv5Eci) | 在历史表征子空间之外更新以减轻干扰 | 给 Adapter 加正交投影并不自动产生新颖性 |
| [Learning Continual Compatible Representation for Re-indexing Free Lifelong Person Re-identification, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Cui_Learning_Continual_Compatible_Representation_for_Re-indexing_Free_Lifelong_Person_Re-identification_CVPR_2024_paper.pdf) | 不能重新编码历史图库时的兼容表示 | 本方案不自动解决新旧版本特征直接检索问题 |

建议作为待验证贡献表达：

> 面向对象类别交错再现、类别内新身份分批到达的终身 ReID，构建跨阶段身份关系记忆，显式连接不同时段到达的同类别身份；通过历史检索间隔保持的统一适配，将暂时缺席类别的判别关系纳入后续训练，并允许可靠旧关系随新知识改善。

这句话里的“构建”“提出”是设计；“优于”“显著降低”“实现正迁移”都必须等实验验证。单侧间隔、原型记忆、残差头分别都有常见技术基础，不能保证组合本身足够新。若同预算普通 latent replay 已达到相同性能，或仅靠类别平衡采样即可解释提升，应进一步收缩或重设贡献。

## 11. 哪些结果会让我改变主方案

- 若原模型在交错流中几乎不遗忘，新协议本身不能证明必须更换结构，应先分析计算与迁移问题。
- 若统一最终特征头上限明显偏低，改用冻结前缀 + 共享后缀 Adapter 的 token 回放版本。
- 若历史关系保持与普通关系蒸馏没有稳定差异，不把 RMPA 单列强创新。
- 若不允许保存任何历史特征，则本文主方案不适用，需要转为参数隔离/梯度统计等路线，并接受旧身份函数约束更弱的限制。
- 若允许少量原图回放，应先加入强的图像 replay 基线；不必为了“无图像”叙事主动放弃用户允许的信息。

就当前代码而言，最值得先做的是：**混合阶段数据协议、显式全局身份映射、类别内新旧身份共同监督，以及确实能回传到可训练网络的历史关系保持。** 这些部分能把研究问题从“调用哪个历史阶段模型”推进到“怎样在统一度量空间持续加入新身份”。
