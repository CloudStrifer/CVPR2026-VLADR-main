# 跨类别终身目标重识别：OCIA–OSAF 方法、运行与实验手册

> 文档状态：当前代码的统一说明入口  
> 最近核对：2026-08-06  
> 适用代码：本仓库当前版本  
> 核心方案：OCIA + 域私有 Adapter + OSAF；SCSD 不属于当前两创新点主方案

---

## 1. 文档目的与证据边界

本文统一整理项目当前真正实现的任务、模型、两项创新、训练和评测流程、
消融实验及未知域协议。阅读或修改方法时，应优先以本文、当前代码和单元测试
为准。`docs/` 中其余文档用于保留设计演化过程，不应把早期设想直接写成
当前实现。

项目的核心论点是：

> 在跨类别、无历史图像回放的顺序 ReID 中，OCIA 构造类别去偏且身份敏感的
> 跨模态学习目标，域私有 Adapter 隔离各训练阶段的参数更新，OSAF 则在未知域
> 推理时选择并融合历史 Adapter 中更可能跨类别迁移的身份残差。

当前证据边界必须明确：

- 五域数据、manifest、训练入口和评测入口已经接通；
- OCIA、域私有 Adapter、OSAF 的数学单元测试已经通过；
- 当前目录没有可供完整 OCIA 使用的新 Stage 1 checkpoint；
- 当前正式配置的 `test_domains` 为空，因此尚无真实未知域 OSAF 结果；
- 在获得实验结果前，只能称“提出”“设计”“实现”，不能声称 OSAF 已提升未知域性能。

---

## 2. 术语表

| 统一术语 | 英文与缩写 | 本文含义 |
|---|---|---|
| 跨类别终身目标重识别 | Cross-Category Lifelong Object ReID | 对象类别和数据域依次变化的终身实例检索 |
| 已见域 | Seen Domain | 已经提供训练图像和身份标签的域 |
| 未知域 | Unseen Domain | 从未参与训练、调参或模型选择，仅提供 Query/Gallery 的测试域 |
| 对象感知跨模态身份锚定 | Object-Aware Cross-Modal Identity Anchoring, OCIA | 创新点 1 |
| 域私有 Adapter | Domain-Private Adapter | 每个训练域独立增加的视觉残差模块 |
| OCIA 引导的语义去偏 Adapter 融合 | OCIA-guided Semantic-Debiased Adapter Fusion, OSAF | 创新点 2 |
| 基础路径 | Base Path | 关闭所有私有 Adapter 的共享视觉编码路径 |
| 完整 Adapter 路径 | Full Adapter Path | 启用某一源域 Adapter 后的视觉路径 |
| 类别公共语义子空间 | Category-Common Semantic Subspace | 由类别通用文本构造的 OCIA 正交子空间 |
| 身份视觉原型 | Visual Identity Prototype | 冻结 CLIP 对同一身份训练图像的平均投影特征 |
| SCSD | Semantic Compatibility-Aware Selective Distillation | 已实现的可选旧方案，不属于当前 OCIA–OSAF 主方法 |

---

## 3. 任务定义

训练数据以域序列 `D_1, ..., D_T` 依次到达。第 `t` 个域包含图像和域内
身份标签：

\[
\mathcal D_t=\{(x_i^t,y_i^t)\}_{i=1}^{N_t},
\qquad o_t\in\mathcal O,
\]

其中 `o_t` 是对象名词，如 person、giant panda、vehicle、tiger 或 boat。
在阶段 `t`：

1. 只能使用当前域训练图像和身份标签；
2. 历史训练图像不参与当前优化；
3. 历史 Adapter 在进入下一域后保持冻结；
4. 最终模型需要在全部已见域上保持身份检索能力；
5. 最终模型还应直接应用于训练期间从未出现的未知域。

当前正式训练顺序为：

\[
\text{Market1501}
\rightarrow
\text{iPanda50}
\rightarrow
\text{VeRi}
\rightarrow
\text{ATRW}
\rightarrow
\text{Boat}.
\]

### 3.1 当前数据规模

| 阶段 | 数据集 | 对象类别 | Train 图像 / ID | Query | Gallery |
|---:|---|---|---:|---:|---:|
| 1 | Market1501 | person | 12,936 / 751 | 3,368 | 15,913 |
| 2 | iPanda50 | giant panda | 5,179 / 35 | 408 | 1,287 |
| 3 | VeRi | vehicle | 37,778 / 576 | 1,678 | 11,579 |
| 4 | ATRW | tiger | 1,887 / 107 | 75 | 1,689 |
| 5 | Boat | boat | 4,179 / 80 | 27 | 165 |

五域共包含 98,148 张 manifest 图像和 1,549 个训练身份。输入统一为
`224 × 224`，采用等比例缩放后居中填充，避免将非行人对象强行拉伸为
行人长宽比。

### 3.2 数据配置边界

`train_domains` 中的域会参与 Stage 1 和 Stage 2。`test_domains` 中的域只会
构建 Query/Gallery loader，不需要也不允许使用该域的 Stage 1 Prompt checkpoint。

严格未知域必须满足：

- 其训练图像和训练身份标签未被使用；
- 其 Query/Gallery 标签仅用于最终计算指标；
- 不使用该域结果选择 OSAF 超参数；
- 不为该域计算身份视觉原型、类别中心或 OCIA 子空间；
- 不根据该域名称手工选择 Adapter。

### 3.3 当前两脚本执行方式

当前 `train_stage1.py` 会先遍历配置中的全部训练域，生成所有 Prompt 和固定
统计；`train_stage2.py` 再执行顺序学习。论文概念上应表述为“域到达时先完成
该域 Stage 1，再进入该域 Stage 2”。如果需要最严格的在线协议，应为每个阶段
生成只包含当前已到达域的配置，或将两个阶段改成按域交替执行，避免产生
“预先处理未来域数据”的质疑。

---

## 4. 总体框架

~~~text
当前训练域 D_t
    │
    ├── Stage 1
    │     ├── 冻结 CLIP 图像/文本编码器
    │     ├── 学习对象感知身份 Prompt
    │     ├── 提取冻结 CLIP 身份视觉原型
    │     └── 构造类别公共语义子空间
    │
    ├── OCIA
    │     └── 文本身份特征 + 类别去偏视觉残差
    │                           ↓
    │                    固定身份 Anchor
    │
    └── Stage 2
          ├── 首域：共享骨干 + 首域 Adapter + 分类器
          └── 后续域：冻结共享骨干和旧 Adapter
                      仅训练当前 Adapter + 分类器

训练完成
    ├── 默认：已见域使用该域完整 Adapter，未知域使用 OSAF
    └── routing-scope=all：已见域与未知域统一使用 OSAF
                  ↓
               逐图 Top-K 路由
                  ├── 计算 Adapter 增量
                  ├── OCIA 子空间语义去偏
                  └── 加权融合到基础特征
~~~

基础模型为 CLIP ViT-B/16。视觉编码器输出：

- 768 维原始全局视觉特征 `f(x)`；
- 512 维 CLIP 图文共享投影 `q(x)`。

训练时，身份分类使用 768 维特征，Triplet 和 OCIA 对齐使用 512 维投影。
默认评测描述子为：

\[
\mathbf z(x)=
\operatorname{norm}
\left[
\mathbf f^{768}(x)\Vert\mathbf q^{512}(x)
\right].
\]

---

## 5. Stage 1：对象感知身份 Prompt 与固定统计

### 5.1 对象感知身份 Prompt

域 `t` 中身份 `y` 的模板为：

\[
\text{“A photo of a }[X]_1[X]_2[X]_3[X]_4\;o_t\text{.”}
\]

每个身份拥有四个独立可学习 token。CLIP 图像编码器、文本编码器和其余视觉
参数保持冻结，只优化 Prompt token。文本特征为：

\[
\mathbf t_{t,y}
=
\operatorname{norm}
\left[
T(\operatorname{Prompt}_{t,y})
\right].
\]

Stage 1 使用对称的图像到文本和文本到图像监督对比损失：

\[
\mathcal L_{\mathrm{S1}}
=
\mathcal L_{\mathrm{i2t}}
+
\mathcal L_{\mathrm{t2i}}.
\]

### 5.2 身份视觉原型和域中心

Prompt 训练完成后，使用无随机增强 loader 和冻结 CLIP 图像编码器提取投影：

\[
\mathbf v_{t,y}
=
\frac{1}{|\mathcal D_{t,y}|}
\sum_{x\in\mathcal D_{t,y}}
\operatorname{norm}(I_0(x)),
\]

\[
\bar{\mathbf v}_t
=
\frac{1}{C_t}
\sum_{y=1}^{C_t}\mathbf v_{t,y}.
\]

`v_{t,y}` 是身份视觉原型，`bar(v)_t` 是当前域的平均视觉中心。

### 5.3 类别公共语义子空间

每个域配置三条类别通用文本，分别描述对象整体轮廓、基本组成结构和常见观察
视角。冻结文本编码器得到特征后，通过 reduced QR 分解构造正交基：

\[
\mathbf H_t
=
[\mathbf h_{t,1},...,\mathbf h_{t,M_g}],
\qquad
\mathbf U_t=\operatorname{orth}(\mathbf H_t).
\]

`U_t` 位于 512 维 CLIP 共享空间，其列向量表示类别公共语义方向。

### 5.4 Stage 1 checkpoint

当前 OCIA checkpoint 至少包含：

- `state_dict`、`num_class`、`n_ctx` 和 `dataset_name`；
- `object_noun`；
- `visual_prototypes` 和 `category_center`；
- `category_semantic_prompts`；
- `category_subspace_basis` 和 `category_subspace_rank`；
- `ocia_schema_version`。

旧版只有 Prompt 参数的 checkpoint 不能用于 `prototype`、`centered` 或
`ocia` 模式。

---

## 6. 创新点 1：OCIA

### 6.1 动机

纯文本身份 Prompt 具有稳定语义，但可能缺少细粒度实例差异。直接加入视觉
身份原型虽然能补充外观，却同时包含人的基本轮廓、车辆结构或动物身体形态等
类别公共信息。这些信息有助于判断类别，却不一定有助于区分同类中的身份。

OCIA 保留文本语义的稳定性和视觉原型的细粒度，同时显式移除可由类别公共
文本解释的视觉方向。

### 6.2 类别中心化与公共方向剔除

\[
\widehat{\mathbf v}_{t,y}
=
\mathbf v_{t,y}-\bar{\mathbf v}_t,
\]

\[
\mathbf r_{t,y}
=
(\mathbf I-\mathbf U_t\mathbf U_t^\top)
\widehat{\mathbf v}_{t,y}.
\]

代码使用的等价行向量形式为：

~~~python
centered = visual_prototypes - category_center
projection = (centered @ category_basis) @ category_basis.T
residual = normalize(centered - projection)
~~~

`r_{t,y}` 被解释为类别去偏的身份视觉残差。

### 6.3 跨模态身份 Anchor

\[
\mathbf a_{t,y}
=
\operatorname{norm}
\left(
\mathbf t_{t,y}
+
\lambda_r\operatorname{norm}(\mathbf r_{t,y})
\right),
\]

其中 `lambda_r` 对应 `--anchor-residual-weight`。Anchor 在当前域
Stage 2 开始前构造并 `detach`，训练期间不更新。

### 6.4 OCIA 对齐损失

\[
\mathcal L_{\mathrm{OCIA}}
=
-\frac{1}{|\mathcal B_t|}
\sum_i
\log
\frac{
\exp(\operatorname{sim}(\mathbf q(x_i),\mathbf a_{t,y_i})/\tau_a)
}{
\sum_{y\in\mathcal Y(\mathcal B_t)}
\exp(\operatorname{sim}(\mathbf q(x_i),\mathbf a_{t,y})/\tau_a)
}.
\]

当前代码分母是 batch 中出现的唯一身份，不是整个当前域的全部身份。
`tau_a` 对应 `--anchor-temperature`。

### 6.5 OCIA 的可验证主张

- 真实类别名词是否优于通用 `object`；
- 视觉身份原型是否补充纯文本 Prompt；
- 域中心化是否削弱一阶类别/数据集偏移；
- 子空间投影剔除是否优于仅中心化；
- 合理的 `lambda_r` 是否优于 `lambda_r=0`；
- OCIA 是否在不增加已见域推理文本编码成本的情况下改善检索。

---

## 7. 域私有 Adapter 与终身更新

### 7.1 Adapter 结构

每个训练域在 ViT 最后 `L` 个 Transformer Block 的 MLP 分支并联瓶颈
Adapter：

\[
A_t^\ell(h)
=
\gamma
W_{t,\mathrm{up}}^\ell
\sigma(
W_{t,\mathrm{down}}^\ell h
).
\]

上投影使用零初始化，因此新 Adapter 在创建时近似恒等残差，不会立即破坏
当前模型输出。

### 7.2 顺序训练策略

- 第一域：训练共享视觉骨干、第一域 Adapter 和身份分类器；
- 第一域结束后：冻结共享骨干和第一域 Adapter；
- 后续域：增加并激活当前域 Adapter；
- 冻结全部旧 Adapter，只训练当前 Adapter 和身份分类器；
- Prompt、CLIP 文本编码器始终在 Stage 2 冻结。

默认 `classifier_scope=current`，身份交叉熵只在当前域分类器切片上计算，
避免让车辆、动物等新身份与全部历史跨类别身份产生不必要的负类排斥。

### 7.3 参数增量

| Adapter 配置 | 每域大致参数量 | 用途 |
|---|---:|---|
| 最后 4 Block，瓶颈 64 | 0.40M | 轻量基线 |
| 最后 6 Block，瓶颈 128 | 1.19M | 当前推荐 |
| 最后 12 Block，瓶颈 128 | 2.37M | 容量上界/成本消融 |

---

## 8. 创新点 2：OSAF

### 8.1 动机和工作阶段

域私有 Adapter 能隔离历史知识，但未知域没有对应 Adapter。直接使用基础模型
无法复用已学习的细粒度知识；平均全部 Adapter 又会把无关类别结构偏置注入
未知图像。

OSAF 为每个待测图像选择最相关的 Top-K 源 Adapter，并从其增量中去除源类别
公共语义后再融合。默认只用于未知域；加入 `--adapter-routing-scope all` 后，
已见域也采用完全相同的无域标签路由。OSAF 不属于训练损失，因此路由超参数
可复用同一训练 checkpoint 重新评测。

### 8.2 基础路径和 Adapter 增量

\[
\mathbf\Delta_t(x)
=
\mathbf q_t(x)-\mathbf q_0(x),
\]

其中 `q_0` 是关闭全部 Adapter 的基础投影，`q_t` 是启用源域 `t`
完整 Adapter 后的投影。

### 8.3 路由统计与双线索得分

每个已学习域保存类别语义键 `c_t`、全部身份视觉原型 `{v_{t,y}}`
和 OCIA 子空间 `U_t`。当前代码保存全部身份原型，而不是定长聚类中心，
所以路由统计的存储量会随源域身份数增长。

\[
s_t(x)
=
\alpha\,
\operatorname{sim}(\mathbf q_0(x),\mathbf c_t)
+
(1-\alpha)
\max_y
\operatorname{sim}(\mathbf q_0(x),\mathbf v_{t,y}).
\]

语义项判断对象类别兼容性，视觉项判断源域分布接近程度。`alpha` 对应
`--adapter-semantic-weight`。

### 8.4 Top-K 选择

\[
\mathcal K(x)=\operatorname{TopK}_t s_t(x),
\]

\[
w_t(x)
=
\frac{\exp(s_t(x)/\tau_g)}
{\sum_{j\in\mathcal K(x)}\exp(s_j(x)/\tau_g)}.
\]

`K` 对应 `--adapter-topk`，`tau_g` 对应
`--adapter-routing-temperature`。

### 8.5 OCIA 引导的残差去偏

\[
\widetilde{\mathbf\Delta}_t(x)
=
(\mathbf I-\rho\mathbf U_t\mathbf U_t^\top)
\mathbf\Delta_t(x).
\]

`rho` 对应 `--adapter-debias-strength`：`rho=0` 为普通
Top-K Adapter 融合，`rho=1` 为完全删除类别公共子空间投影。

### 8.6 OSAF 路由描述子

\[
\mathbf q_u(x)
=
\mathbf q_0(x)
+
\lambda_f
\sum_{t\in\mathcal K(x)}
w_t(x)\widetilde{\mathbf\Delta}_t(x),
\]

\[
\mathbf z_u(x)
=
\operatorname{norm}
\left[
\mathbf f_0^{768}(x)
\Vert
\mathbf q_u^{512}(x)
\right].
\]

`lambda_f` 对应 `--adapter-fusion-weight`。OSAF 只融合 512 维
CLIP 投影；768 维分支保持基础路径。

### 8.7 当前边界

当前实现没有路由置信度阈值或自动回退门控。即使未知样本与所有源域都很远，
代码仍会选择 Top-K Adapter。因此必须报告未知域基础路径、不去偏 Top-K 和
完整 OSAF，并逐数据集检查正迁移与负迁移。

---

## 9. 当前训练目标与推理行为

当前两创新点主方案设置 `--attr-distill-mode none`，Stage 2 总损失为：

\[
\mathcal L
=
\mathcal L_{\mathrm{CE}}
+
\mathcal L_{\mathrm{Tri}}
+
\lambda_a\mathcal L_{\mathrm{OCIA}}.
\]

`lambda_a` 对应 `--global-loss-weight`。

| 场景 | 视觉路径 | 是否使用文本编码器 | 是否使用 OSAF |
|---|---|---:|---:|
| 当前域训练 | 当前完整 Adapter | 仅查固定 Anchor | 否 |
| 已见域 Oracle 评测 | 数据集名称对应的完整 Adapter | 否 | 否 |
| 已见域 Routed 评测 | 基础路径 + Top-K 去偏残差 | 否 | 是，需 `routing-scope=all` |
| 未知域评测 | 基础路径 + Top-K 去偏残差 | 否 | 是 |

---

## 10. 代码位置

| 功能 | 文件 |
|---|---|
| Stage 1 Prompt 学习和 checkpoint | `train_stage1.py` |
| Stage 2 顺序训练、模型扩展和参数开关 | `train_stage2.py` |
| CE、Triplet、OCIA 和可选 SCSD 损失 | `reid/trainer_stage2.py` |
| 视觉原型、QR 子空间和 OCIA Anchor | `reid/utils/feature_tools.py` |
| 域私有 Adapter | `reid/models/CLIP_ReID/model/clip/model.py` |
| OSAF 路由、Top-K 和语义去偏 | `reid/evaluation/adapter_fusion.py` |
| 已见域/未知域路径切换与评测 | `reid/evaluation/fast_test.py` |
| 五域配置 | `config/cross_category_five_domains.json` |
| Manifest 数据入口 | `lreid_dataset/datasets/manifest_reid.py` |
| 核心单元测试 | `tests/test_identity_anchors.py`、`tests/test_domain_adapters.py`、`tests/test_adapter_fusion.py` |

---

## 11. 运行前检查

以下命令按 Ubuntu Bash 编写，并从仓库根目录执行。多行命令末尾的反斜杠
`\` 后不要添加空格。

### 11.1 环境

~~~bash
conda create -n continual-clipreid python=3.9
conda activate continual-clipreid

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python setup.py develop
~~~

不建议直接使用未经完整验证的 Python 3.13 进行正式训练；项目按 Python 3.9
和指定 PyTorch 版本组织。

### 11.2 数据和 manifest 校验

~~~bash
python tools/validate_cross_category_config.py \
  --domain-config config/cross_category_five_domains.json \
  --data-dir data
~~~

预期最后输出：

~~~text
Validated 5 domains and 98148 images.
~~~

### 11.3 Loader 冒烟测试

~~~bash
python tools/smoke_test_cross_category_loaders.py \
  --domain-config config/cross_category_five_domains.json \
  --data-dir data \
  --batch-size 16 \
  --num-instances 4 \
  --workers 0
~~~

### 11.4 CPU 单元测试

~~~bash
python -m unittest discover -s tests -p "test_*.py" -v
~~~

### 11.5 真实 CLIP 小规模训练路径测试

该命令需要 CUDA，并可能在首次运行时下载 CLIP 权重：

~~~bash
python tools/smoke_test_training_losses.py \
  --data-dir data \
  --domain-config config/cross_category_five_domains.json \
  --visual-anchor-mode ocia \
  --anchor-residual-weight 1.0 \
  --attr-distill-mode none \
  --continual-update-mode domain_adapter
~~~

---

## 12. 完整训练命令

### 12.1 Stage 1：生成当前 OCIA checkpoint

现有 `_STAGE1_PROMPTS_WEIGHT` 是旧权重，不能用于当前 OCIA。建议使用新的
独立目录：

~~~bash
CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
  --data-dir data \
  --domain-config config/cross_category_five_domains.json \
  --prompt-checkpoint-source output-dir \
  --stage1-prompts-out-dir ./_PROMPTS_OCIA_CATEGORY \
  --batch-size 64 \
  --num-instances 4 \
  --workers 8 \
  --prompt-epochs 120 \
  --warmup-step 10 \
  --milestones 30 \
  --seed 1234 \
  --logs-dir ./RESULTS/prompts_ocia_category
~~~

完成后应存在：

~~~text
_PROMPTS_OCIA_CATEGORY/
  market1501_clipreid_prompt.pth
  ipanda50_clipreid_prompt.pth
  veri_clipreid_prompt.pth
  atrw_clipreid_prompt.pth
  boat_clipreid_prompt.pth
~~~

实现细节：当前 Stage 1 的 scheduler 在每个 mini-batch 后调用，因此
`--warmup-step` 和 `--milestones` 在实际代码中按迭代次数推进，而 Stage 2 的
scheduler 按 epoch 推进。正式实验记录中应如实注明，若希望 Stage 1 也按 epoch
衰减，需要先修改 scheduler 的调用位置。

### 12.2 Stage 2：完整 OCIA + 域私有 Adapter + OSAF

~~~bash
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
  --data-dir data \
  --domain-config config/cross_category_five_domains.json \
  --prompt-checkpoint-source output-dir \
  --stage1-prompts-out-dir ./_PROMPTS_OCIA_CATEGORY \
  --visual-anchor-mode ocia \
  --anchor-residual-weight 1.0 \
  --anchor-temperature 0.07 \
  --global-loss-weight 1.0 \
  --attr-distill-mode none \
  --classifier-scope current \
  --eval-descriptor raw \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 6 \
  --adapter-bottleneck-dim 128 \
  --adapter-scale 1.0 \
  --adapter-lr 0.0003 \
  --adapter-routing osaf \
  --adapter-routing-scope all \
  --adapter-topk 2 \
  --adapter-semantic-weight 0.5 \
  --adapter-routing-temperature 0.1 \
  --adapter-debias-strength 1.0 \
  --adapter-fusion-weight 1.0 \
  --stage2-base-lr 0.000005 \
  --classifier-lr-multiplier 10 \
  --batch-size 64 \
  --num-instances 4 \
  --workers 8 \
  --epochs0 80 \
  --epochs 60 \
  --warmup-step 10 \
  --milestones 30 \
  --eval-stage 0,1,2,3,4 \
  --seed 1234 \
  --other-details ocia_osaf_full \
  --logs-dir ./RESULTS/ocia_osaf_full
~~~

当前五域配置没有未知域。由于命令设置了 `--adapter-routing-scope all`，每个
评测阶段的已见域都会通过 OSAF Top-K 路由，所得 `Seen-Avg` 即
`Seen-Routed`；`UnSeen-Avg` 仍为 `NaN`。

### 12.3 加载最终 checkpoint 重新评测

`--testing` 接收包含最终 `boat_checkpoint.pth.tar` 的目录，而不是 checkpoint
文件本身。

~~~bash
LATEST_RUN=$(find ./RESULTS/ocia_osaf_full/stage2 \
  -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | \
  sort -nr | head -n 1 | cut -d' ' -f2-)
CKPT_DIR="${LATEST_RUN}/_CKPTS"

python train_stage2.py \
  --data-dir data \
  --domain-config config/cross_category_five_domains.json \
  --prompt-checkpoint-source output-dir \
  --stage1-prompts-out-dir ./_PROMPTS_OCIA_CATEGORY \
  --visual-anchor-mode ocia \
  --continual-update-mode domain_adapter \
  --adapter-routing osaf \
  --adapter-routing-scope all \
  --adapter-topk 2 \
  --adapter-semantic-weight 0.5 \
  --adapter-routing-temperature 0.1 \
  --adapter-debias-strength 1.0 \
  --adapter-fusion-weight 1.0 \
  --eval-descriptor raw \
  --batch-size 64 \
  --workers 8 \
  --testing "$CKPT_DIR" \
  --logs-dir ./RESULTS/eval_full
~~~

---

## 13. 建立未知域配置

当前 `config/cross_category_five_domains.json` 中：

~~~json
"test_domains": []
~~~

未知域配置示例：

~~~json
"test_domains": [
  {
    "name": "msmt17_unseen",
    "category": "person",
    "object_noun": "person",
    "manifest": "manifests/msmt17_unseen.csv",
    "root": "../data/MSMT17",
    "eval_protocol": "market1501"
  }
]
~~~

未知 manifest 只应包含 `query` 和 `gallery` 行：

~~~csv
path,pid,camid,split
query/0001_c1.jpg,p0001,0,query
gallery/0001_c2.jpg,p0001,1,gallery
~~~

不要为未知域增加 `prompt_checkpoint`，也不要把它加入 `train_domains`。

仓库已经提供 MSMT17 V2 未知域构建器。它只读取 `query` 与
`bounding_box_test`，不会读取 `bounding_box_train`：

~~~bash
python tools/build_msmt17_unseen_manifest.py
~~~

该命令会生成：

- `config/manifests/msmt17_unseen.csv`；
- `config/manifests/msmt17_unseen_audit.json`；
- `config/cross_category_five_domains_with_msmt17_unseen.json`。

在启动模型评测前可执行完整清单校验和图像解码冒烟测试：

~~~bash
python tools/validate_cross_category_config.py \
  --domain-config config/cross_category_five_domains_with_msmt17_unseen.json \
  --data-dir data

python tools/smoke_test_cross_category_loaders.py \
  --domain-config config/cross_category_five_domains_with_msmt17_unseen.json \
  --data-dir data \
  --workers 0
~~~

Stage 1 和 Stage 2 训练仍使用不含未知域的
`config/cross_category_five_domains.json`；仅最终 `--testing` 使用带
MSMT17 的评测配置。MSMT17 全量评测默认在距离矩阵超过
`100000000` 个元素时自动启用精确 Query 分块排名，可通过
`--eval-query-chunk-size` 调整块大小。

建议维护两类配置：

1. `cross_category_all5_with_external_unseen.json`：五域全部训练，增加外部未知域；
2. `cross_category_loo_<domain>.json`：五折 leave-one-domain-out，将一个现有域移到未知测试集。

---

## 14. 消融实验命令

### 14.1 公共 Stage 2 参数

Bash 可以先定义公共参数数组，减少重复：

~~~bash
S2_COMMON=(
  --data-dir data
  --domain-config config/cross_category_five_domains.json
  --prompt-checkpoint-source output-dir
  --stage1-prompts-out-dir ./_PROMPTS_OCIA_CATEGORY
  --attr-distill-mode none
  --classifier-scope current
  --eval-descriptor raw
  --stage2-base-lr 0.000005
  --classifier-lr-multiplier 10
  --global-loss-weight 1.0
  --anchor-temperature 0.07
  --batch-size 64
  --num-instances 4
  --workers 8
  --epochs0 80
  --epochs 60
  --milestones 30
  --eval-stage 0,1,2,3,4
  --seed 1234
)
~~~

### 14.2 OCIA 核心消融

OCIA 消融统一使用 `adapter-routing oracle`，先只比较已见域训练目标；
`visual-anchor-mode` 不是 `ocia` 时不能启用 OSAF。

#### A. 通用 object Prompt + 纯文本 Anchor

该实验需要单独生成通用 Prompt：

~~~bash
python train_stage1.py \
  --data-dir data \
  --domain-config config/cross_category_five_domains.json \
  --prompt-checkpoint-source output-dir \
  --stage1-prompts-out-dir ./_PROMPTS_GENERIC_OBJECT \
  --disable-category-prompt \
  --batch-size 64 \
  --num-instances 4 \
  --workers 8 \
  --prompt-epochs 120 \
  --seed 1234 \
  --logs-dir ./RESULTS/prompts_generic
~~~

对应 Stage 2：

~~~bash
python train_stage2.py \
  --data-dir data \
  --domain-config config/cross_category_five_domains.json \
  --prompt-checkpoint-source output-dir \
  --stage1-prompts-out-dir ./_PROMPTS_GENERIC_OBJECT \
  --disable-category-prompt \
  --visual-anchor-mode text \
  --global-loss-weight 1.0 \
  --attr-distill-mode none \
  --classifier-scope current \
  --eval-descriptor raw \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 6 \
  --adapter-bottleneck-dim 128 \
  --adapter-lr 0.0003 \
  --adapter-routing oracle \
  --stage2-base-lr 0.000005 \
  --classifier-lr-multiplier 10 \
  --batch-size 64 \
  --num-instances 4 \
  --workers 8 \
  --epochs0 80 \
  --epochs 60 \
  --milestones 30 \
  --eval-stage 0,1,2,3,4 \
  --seed 1234 \
  --logs-dir ./RESULTS/ablation_ocia_A_generic_text
~~~

#### B. 真实对象名词 + 纯文本 Anchor

~~~bash
python train_stage2.py "${S2_COMMON[@]}" \
  --visual-anchor-mode text \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 6 \
  --adapter-bottleneck-dim 128 \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/ablation_ocia_B_category_text
~~~

#### C. Prompt + 原始视觉身份原型

~~~bash
python train_stage2.py "${S2_COMMON[@]}" \
  --visual-anchor-mode prototype \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 6 \
  --adapter-bottleneck-dim 128 \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/ablation_ocia_C_prototype
~~~

#### D. Prompt + 类别中心化视觉残差

~~~bash
python train_stage2.py "${S2_COMMON[@]}" \
  --visual-anchor-mode centered \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 6 \
  --adapter-bottleneck-dim 128 \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/ablation_ocia_D_centered
~~~

#### E. 完整 OCIA

~~~bash
python train_stage2.py "${S2_COMMON[@]}" \
  --visual-anchor-mode ocia \
  --anchor-residual-weight 1.0 \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 6 \
  --adapter-bottleneck-dim 128 \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/ablation_ocia_E_full
~~~

#### F. OCIA 退化检查

`lambda_r=0` 理论上应退化为归一化文本 Anchor：

~~~bash
python train_stage2.py "${S2_COMMON[@]}" \
  --visual-anchor-mode ocia \
  --anchor-residual-weight 0.0 \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 6 \
  --adapter-bottleneck-dim 128 \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/ablation_ocia_F_lambda0
~~~

### 14.3 Adapter 抗遗忘消融

#### A0. 纯视觉顺序微调基线

该基线关闭全局 Anchor 损失，并在后续域继续更新共享骨干最后两块：

~~~bash
python train_stage2.py "${S2_COMMON[@]}" \
  --visual-anchor-mode text \
  --global-loss-weight 0.0 \
  --continual-update-mode partial_blocks \
  --continual-trainable-vision-blocks 2 \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/baseline_visual_sequential
~~~

#### A1. OCIA + 共享骨干后两块持续更新

~~~bash
python train_stage2.py "${S2_COMMON[@]}" \
  --visual-anchor-mode ocia \
  --anchor-residual-weight 1.0 \
  --continual-update-mode partial_blocks \
  --continual-trainable-vision-blocks 2 \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/ablation_adapter_A_partial_blocks
~~~

#### B. 轻量 Adapter：4 Block / 64

~~~bash
python train_stage2.py "${S2_COMMON[@]}" \
  --visual-anchor-mode ocia \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 4 \
  --adapter-bottleneck-dim 64 \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/ablation_adapter_B_4x64
~~~

#### C. 推荐 Adapter：6 Block / 128

~~~bash
python train_stage2.py "${S2_COMMON[@]}" \
  --visual-anchor-mode ocia \
  --continual-update-mode domain_adapter \
  --adapter-last-blocks 6 \
  --adapter-bottleneck-dim 128 \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/ablation_adapter_C_6x128
~~~

### 14.4 OSAF 核心消融

以下实验默认使用现有五域配置比较 `Seen-Oracle` 与 `Seen-Routed`。若要同时
评价未知域，只需把 `$EVAL_CONFIG` 换成含真实 `test_domains` 的配置；所有实验
都是评测期实验，可复用同一个完整 OCIA + Adapter checkpoint。

~~~bash
EVAL_CONFIG="config/cross_category_five_domains.json"
LATEST_RUN=$(find ./RESULTS/ocia_osaf_full/stage2 \
  -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | \
  sort -nr | head -n 1 | cut -d' ' -f2-)
CKPT_DIR="${LATEST_RUN}/_CKPTS"

EVAL_COMMON=(
  --data-dir data
  --domain-config "$EVAL_CONFIG"
  --prompt-checkpoint-source output-dir
  --stage1-prompts-out-dir ./_PROMPTS_OCIA_CATEGORY
  --visual-anchor-mode ocia
  --continual-update-mode domain_adapter
  --eval-descriptor raw
  --batch-size 64
  --workers 8
  --testing "$CKPT_DIR"
)
~~~

#### A. 未知域基础路径

~~~bash
python train_stage2.py "${EVAL_COMMON[@]}" \
  --adapter-routing oracle \
  --logs-dir ./RESULTS/osaf_A_base_unknown
~~~

#### B. Top-K 普通 Adapter 残差融合

~~~bash
python train_stage2.py "${EVAL_COMMON[@]}" \
  --adapter-routing osaf \
  --adapter-routing-scope all \
  --adapter-topk 2 \
  --adapter-semantic-weight 0.5 \
  --adapter-routing-temperature 0.1 \
  --adapter-debias-strength 0.0 \
  --adapter-fusion-weight 1.0 \
  --logs-dir ./RESULTS/osaf_B_topk_no_debias
~~~

#### C. 完整 OSAF

~~~bash
python train_stage2.py "${EVAL_COMMON[@]}" \
  --adapter-routing osaf \
  --adapter-routing-scope all \
  --adapter-topk 2 \
  --adapter-semantic-weight 0.5 \
  --adapter-routing-temperature 0.1 \
  --adapter-debias-strength 1.0 \
  --adapter-fusion-weight 1.0 \
  --logs-dir ./RESULTS/osaf_C_full
~~~

#### D. Top-K 数量

~~~bash
for k in 1 2 3 5; do
  python train_stage2.py "${EVAL_COMMON[@]}" \
    --adapter-routing osaf \
    --adapter-routing-scope all \
    --adapter-topk "$k" \
    --adapter-semantic-weight 0.5 \
    --adapter-routing-temperature 0.1 \
    --adapter-debias-strength 1.0 \
    --adapter-fusion-weight 1.0 \
    --logs-dir "./RESULTS/sweep_osaf_topk_${k}"
done
~~~

---

## 15. 参数实验命令

### 15.1 OCIA 视觉残差权重

这是训练期参数，每个值需要重新训练 Stage 2：

~~~bash
for value in 0.0 0.25 0.5 1.0 2.0; do
  python train_stage2.py "${S2_COMMON[@]}" \
    --visual-anchor-mode ocia \
    --anchor-residual-weight "$value" \
    --continual-update-mode domain_adapter \
    --adapter-last-blocks 6 \
    --adapter-bottleneck-dim 128 \
    --adapter-routing oracle \
    --logs-dir "./RESULTS/sweep_ocia_lambda_r_${value}"
done
~~~

### 15.2 OCIA Anchor 温度

~~~bash
for value in 0.03 0.05 0.07 0.1 0.2; do
  python train_stage2.py "${S2_COMMON[@]}" \
    --visual-anchor-mode ocia \
    --anchor-residual-weight 1.0 \
    --anchor-temperature "$value" \
    --continual-update-mode domain_adapter \
    --adapter-last-blocks 6 \
    --adapter-bottleneck-dim 128 \
    --adapter-routing oracle \
    --logs-dir "./RESULTS/sweep_anchor_temperature_${value}"
done
~~~

### 15.3 OCIA 损失权重

~~~bash
for value in 0.0 0.25 0.5 1.0 2.0; do
  python train_stage2.py "${S2_COMMON[@]}" \
    --visual-anchor-mode ocia \
    --global-loss-weight "$value" \
    --continual-update-mode domain_adapter \
    --adapter-last-blocks 6 \
    --adapter-bottleneck-dim 128 \
    --adapter-routing oracle \
    --logs-dir "./RESULTS/sweep_global_weight_${value}"
done
~~~

### 15.4 OSAF 语义/视觉路由平衡

以下均是评测期参数，可复用 checkpoint：

~~~bash
for value in 0.0 0.25 0.5 0.75 1.0; do
  python train_stage2.py "${EVAL_COMMON[@]}" \
    --adapter-routing osaf \
    --adapter-routing-scope all \
    --adapter-topk 2 \
    --adapter-semantic-weight "$value" \
    --adapter-routing-temperature 0.1 \
    --adapter-debias-strength 1.0 \
    --adapter-fusion-weight 1.0 \
    --logs-dir "./RESULTS/sweep_osaf_alpha_${value}"
done
~~~

### 15.5 OSAF 路由温度

~~~bash
for value in 0.03 0.05 0.1 0.2 0.5; do
  python train_stage2.py "${EVAL_COMMON[@]}" \
    --adapter-routing osaf \
    --adapter-routing-scope all \
    --adapter-topk 2 \
    --adapter-semantic-weight 0.5 \
    --adapter-routing-temperature "$value" \
    --adapter-debias-strength 1.0 \
    --adapter-fusion-weight 1.0 \
    --logs-dir "./RESULTS/sweep_osaf_temperature_${value}"
done
~~~

### 15.6 OSAF 去偏强度

~~~bash
for value in 0.0 0.25 0.5 0.75 1.0; do
  python train_stage2.py "${EVAL_COMMON[@]}" \
    --adapter-routing osaf \
    --adapter-routing-scope all \
    --adapter-topk 2 \
    --adapter-semantic-weight 0.5 \
    --adapter-routing-temperature 0.1 \
    --adapter-debias-strength "$value" \
    --adapter-fusion-weight 1.0 \
    --logs-dir "./RESULTS/sweep_osaf_rho_${value}"
done
~~~

### 15.7 OSAF 融合强度

~~~bash
for value in 0.0 0.25 0.5 1.0 1.5 2.0; do
  python train_stage2.py "${EVAL_COMMON[@]}" \
    --adapter-routing osaf \
    --adapter-routing-scope all \
    --adapter-topk 2 \
    --adapter-semantic-weight 0.5 \
    --adapter-routing-temperature 0.1 \
    --adapter-debias-strength 1.0 \
    --adapter-fusion-weight "$value" \
    --logs-dir "./RESULTS/sweep_osaf_lambda_f_${value}"
done
~~~

不建议直接运行所有参数的笛卡尔积。应先单因素扫描，再在验证域上对最有影响的
两个参数做小范围联合搜索。OSAF 超参数不能使用最终未知测试域选择。

---

## 16. 推荐实验协议

### 16.1 协议 A：五域顺序训练与已见域抗遗忘

目的：验证 OCIA 和私有 Adapter 在持续学习中的作用。每个阶段训练结束后，
在所有已见域上分别计算：

- mAP；
- Rank-1、Rank-5、Rank-10；
- Seen-Avg；
- 每域性能随阶段变化的矩阵。

建议补充平均遗忘：

\[
\mathrm{AF}
=
\frac{1}{T-1}
\sum_{j=1}^{T-1}
\left(
\max_{t\in\{j,...,T\}}R_{t,j}
-
R_{T,j}
\right).
\]

当前代码没有直接汇总 AF，需要从各阶段日志的性能矩阵计算。

### 16.2 协议 B：同类别未知域

目的：测试类别已见、数据域未知时的泛化。例如：

- 训练见过 person，测试 MSMT17、DukeMTMC-reID 或 CUHK03；
- 训练见过 vehicle，测试 UAV-VeID、VehicleID 或 VERI-Wild。

这类未知域与传统 LReID-Unseen 更接近，OSAF 更容易找到语义兼容的源 Adapter。

### 16.3 协议 C：未知类别

目的：测试训练中从未出现的对象类别。可参考：

- LeopardID2022；
- ELPephants；
- Wildlife71；
- PetFace；
- CUTE；
- University-1652。

必须分别报告每个数据集，避免聚合平均掩盖某些未知类别上的负迁移。

### 16.4 协议 D：五折 Leave-One-Category-Out

每次使用四个现有类别顺序训练，将第五个类别整体放入 `test_domains`：

1. hold out Market；
2. hold out iPanda；
3. hold out VeRi；
4. hold out ATRW；
5. hold out Boat。

该协议不需要额外下载数据，适合首先判断 OSAF 是否有基本跨类别迁移能力。
但它同时混合了“未知类别”和“未知数据集”两种偏移，论文中应明确这一点。

### 16.5 训练顺序与随机种子

至少使用：

- 当前顺序；
- 一个以动物域开始的顺序；
- 一个以车辆或船只域开始的顺序。

最终主结果建议使用 3 个随机种子，并报告均值和标准差。模块消融阶段可固定
同一 Stage 1 checkpoint 和种子以隔离变量；最终稳健性结果则应重跑完整
Stage 1 + Stage 2。

### 16.6 未知域调参原则

未知域测试集不能兼任验证集。可采用：

1. 从已见源域中轮流选一个“伪未知验证域”选择 OSAF 参数；
2. 固定参数后再测试真正外部未知域；
3. 或在 leave-one-out 外层测试、内层源域验证的嵌套协议中调参。

建议额外报告：

- OSAF 相对基础路径的 `Delta mAP` 和 `Delta R1`；
- 每个未知域的 Top-1 路由分布；
- Top-1 与 Top-2 得分间隔；
- 纯语义、纯视觉和双线索路由；
- 不同未知域上的负迁移比例；
- 推理时间和 Adapter 参数量。

---

## 17. 推荐实验表

### 17.1 核心模块表

| 行 | OCIA | 私有 Adapter | Top-K 融合 | 语义去偏 | 主要回答 |
|---|---:|---:|---:|---:|---|
| B0 | 否 | 否 | 否 | 否 | 顺序微调基线 |
| B1 | 是 | 否 | 否 | 否 | OCIA 是否改进身份目标 |
| B2 | 文本 | 是 | 否 | 否 | 参数隔离的抗遗忘收益 |
| B3 | 是 | 是 | 否 | 否 | OCIA + Adapter 未知域基础路径 |
| B4 | 是 | 是 | 是 | 否 | Top-K 路由和融合收益 |
| Full | 是 | 是 | 是 | 是 | 完整 OSAF |

### 17.2 OCIA 细分表

| 变体 | 对象名词 | 视觉成分 | 中心化 | 子空间去偏 |
|---|---|---|---:|---:|
| Generic Text | object | 无 | 否 | 否 |
| Category Text | 真实类别 | 无 | 否 | 否 |
| Prototype | 真实类别 | 原始身份原型 | 否 | 否 |
| Centered | 真实类别 | 域中心化残差 | 是 | 否 |
| OCIA | 真实类别 | 类别去偏残差 | 是 | 是 |

### 17.3 OSAF 细分表

| 变体 | 路由 | K | 去偏强度 | 融合 |
|---|---|---:|---:|---|
| Base | 无 | 0 | — | 无 |
| Top-1 | 双线索 | 1 | 1 | 单 Adapter |
| Top-2 no-debias | 双线索 | 2 | 0 | 加权 |
| Semantic-only | 纯语义 | 2 | 1 | 加权 |
| Visual-only | 纯视觉 | 2 | 1 | 加权 |
| Full OSAF | 双线索 | 2 | 1 | 加权 |

---

## 18. 五域训练后测试未知域的参考性

### 18.1 传统 LReID 先例

AKA 在 CVPR 2021 建立的 LReID benchmark 使用五个已见行人域顺序训练，并在
七个未参与训练的行人数据集上评测泛化。后续 LReID 工作及原始 VLADR 延续了
Seen-Avg 与 UnSeen-Avg 的评测思路：

- 五个已见域：CUHK03、Market1501、MSMT17 V2、DukeMTMC-reID、CUHK-SYSU；
- 七个未知域：VIPeR、PRID、GRID、i-LIDS、CUHK01、CUHK02、SenseReID。

- AKA：<https://openaccess.thecvf.com/content/CVPR2021/html/Pu_Lifelong_Person_Re-Identification_via_Adaptive_Knowledge_Accumulation_CVPR_2021_paper.html>
- VLADR：<https://arxiv.org/abs/2603.19678>

这一协议说明：终身 ReID 不只评价历史域遗忘，也可以评价最终表征对未知域的
泛化。

### 18.2 与当前课题更接近的先例

CVPR 2026 的 Object-Generalized Re-Identification 使用五个异构对象训练集
（Market1501、VeRi、VesselReID、iPanda50、ATRW），并在九个测试数据集和
大量未知类别上评测：

- 已见类别但新数据域：MSMT17、UAV-VeID；
- 未知类别或复合类别：LeopardID2022、ELPephants、University-1652、CUTE、
  GZGC、PetFace、Wildlife71。

<https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Object-Generalized_Re-Identification_A_Step_Towards_Universal_Instance_Perception_CVPR_2026_paper.pdf>

这表明“五个异构源类别训练、再做大范围未知类别评测”是已有实证先例的实验
范式，但不构成对当前顺序学习方法效果的证明。

### 18.3 为什么它们不是当前 OSAF 的性能保证

传统 LReID 的五个训练域和七个未知域全部是行人，人体结构、属性和 CLIP 语义
高度共享；当前任务要求跨人、车辆、动物和船只迁移，难度更高。

Object-Generalized ReID 虽然跨类别，但采用多源联合训练，不需要处理域顺序、
灾难性遗忘或私有 Adapter 选择。因此它支持当前数据规模和未知域协议，却不能
直接证明顺序学习下的 OSAF 会有效。

当前 OSAF 还没有低置信度回退。当未知类别与五个源类别都不相似时，强制融合
Top-K Adapter 可能低于基础 CLIP。合理预期是：

- 同类别未知域：最有希望获得稳定正迁移；
- 与某个源类别结构接近的未知类别：可能获得正迁移；
- 与全部源类别相距较远的未知类别：存在明显负迁移风险；
- 最终结论必须由 Base、no-debias 和 Full OSAF 的逐数据集对比支持。

---

## 19. 当前文档冲突与采用原则

### 19.1 当前采用

- 本文；
- `ocia_osaf_modules_manual_zh.md`；
- `object_aware_cross_modal_identity_anchoring_zh.md`；
- `domain_adapter_continual_learning_zh.md`；
- 当前代码和测试。

### 19.2 历史或可选

- `category_prompt_and_visual_residual_zh.md`：OCIA 的早期中心化版本；
- `semantic_compatibility_aware_selective_distillation_zh.md`：可选 SCSD；
- `method_ocia_tara.tex`、`method_ocia_tacee.tex`：早期 Adapter 方法稿；
- `VLADR_cross_category_analysis_zh.md`：从原 VLADR 改造时的早期审计。

### 19.3 尚未落到代码的论文设想

`ocia_osaf_methodology_zh.tex` 中以下内容尚未完整实现：

- OSAF 路由置信度门控和低置信度回退；
- 固定数量的视觉聚类原型；
- 去偏 Adapter 路径的单独训练监督；
- 训练后删除身份分类器；
- OCIA 对全部当前域身份而非 batch 唯一身份计算分母。

正式论文应选择其一：

1. 删除未实现内容，使公式与代码一致；
2. 在实验前补齐实现、测试和相应消融。

---

## 20. 建议执行顺序

1. 校验数据和 loader；
2. 重新生成五域 OCIA Stage 1 checkpoint；
3. 先完成小身份数 GPU smoke test；
4. 跑完整 OCIA + Adapter 五域训练，确认已见域性能矩阵；
5. 先做五折 leave-one-category-out，无需下载新数据即可初验 OSAF；
6. 加入同类别未知域，区分域泛化和类别泛化；
7. 再接入 Object-Generalized ReID 的外部未知类别测试集；
8. 先完成核心消融，再做单因素参数扫描；
9. 固定超参数后运行多个顺序和 3 个随机种子；
10. 最后根据真实结果决定是否实现置信度回退或压缩路由原型。
