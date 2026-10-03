# MSMT17 与 DogFaceNet 的训练后 Unseen 评估

新增入口：`tools/evaluate_unseen_category.py`。

仅加载已经完成全部训练阶段的模型并测试，不修改已有训练/评估代码，不修改原训练配置。Unseen 图片不用于训练、聚类、更新原型、选择超参数或创建 Adapter。

## 1. 两个数据集测量的能力不同

| 数据集 | 当前训练中是否见过类别 | 评估含义 |
|---|---|---|
| MSMT17 | person 已通过 Market1501 学习 | 未见数据集上的同类别泛化 |
| DogFaceNet | dog 不在现有五个类别中 | 未见数据集、未见类别的身份检索迁移 |

两者均使用图像驱动的 `prototype` 路由：每张 query/gallery 图片独立选择已有 Adapter，再提取归一化检索特征。身份标签只用于划分和计算指标，不进入模型。图库不按真实类别或预测类别筛选。

模型当前没有 dog Adapter，也没有未知类别拒识功能。DogFaceNet 会使用路由选中的已有专家；这不等同于预测出了 dog 类别，更不是训练了第六类。报告中的狗类别路由准确率为 `null`，记录的是其流向各已有专家的数量。

ECPM 在测试时提供冻结的路由原型；PGCA 的作用体现在已训练的 Adapter 参数中。不会在目标数据上再次执行 PGCA 的新类别初始化/训练，也不会利用目标图像重新估计 ECPM。

这里的 unseen 指未参与本项目的增量训练和原型构建，不声称知道 CLIP 预训练数据是否包含相关来源。检索任务仍要求 query 的正确身份在 gallery 中存在，不是未知身份拒识测试。

## 2. 本地结构与协议分析

检查只枚举目录、文件名和标签清单，没有遍历解码真实数据集图片。

### MSMT17

当前结构已转换为 Market 风格：

```text
data/MSMT17/
  query/                  11,659 张
  bounding_box_test/      82,161 张
  bounding_box_train/     不使用
  __MACOSX/               不使用
```

文件名形如 `0000_c14_0030.jpg`，可解析身份和摄像头。使用现有 query/gallery 划分，3,060 个测试身份、15 个摄像头；排名时排除 query 自身及同身份、同摄像头的 gallery 图片，保留同摄像头的其他身份作为负样本。

默认检查标准数量，每个 query 必须存在跨摄像头正样本。明确需要小规模调试子集时才加 `--allow-msmt-subset`；报告会标记数量不是标准规模。支持的是你当前的转换后结构，不自动解析其他版本的 `list_query.txt` 原始目录结构，也不推断图片属于 V1 还是 V2。

项目已有的 `tools/build_msmt17_unseen_manifest.py` 只被复用为元数据解析器；本入口不会调用它修改训练配置。

### DogFaceNet

当前是按狗身份组织的对齐人脸图片版本：

```text
data/DogFaceNet/
  0/0.0.jpg, 0.1.jpg, ...
  1/1.0.jpg, 1.1.jpg, ...
  ...
  classes_test.txt
  classes_train.txt
```

本地共 1,393 个身份、8,363 张图片。提供的 test 清单包含 139 个身份、697 张图片；train 清单为 1,254 个身份、7,666 张图片，二者身份无交集。文本中同一个 ID 会重复多次，脚本按身份去重，而不是把每一行当作新的身份。

**默认协议 `--dog-split test --dog-protocol one_query --seed 42`：**

1. 仅使用 `classes_test.txt` 列出的身份，不读取 train 身份的图片进行推理。
2. 每个身份使用固定种子选一张 query，其余所有图片放入 gallery。
3. 本地得到 139 张 query、558 张 gallery；二者路径不重叠。
4. 没有可靠摄像头元数据，使用 `exclude_self` 规则，不编造跨摄像头协议。
5. 如果某身份不足两张图片，明确记录并排除；本地默认 test 身份没有此问题。

身份名单来自数据集提供方，但 **query/gallery 划分是本项目为检索制定的协议**，不能直接称为 DogFaceNet 原论文的验证/识别协议，也不能直接把这里的 mAP/Rank-1 与其 verification accuracy 混合比较。

可选 `--dog-protocol leave_one_out`：全部 697 张测试图片同时作为 query 和 gallery，对每个 query 排除自身；使用全部图片评估，减少单次 query 抽样影响，但结果必须单独标记协议。共享图片只编码一次。

可选 `--dog-split all`：把全部身份用于外部检索评估；它仍不训练模型，但会包含数据集原 train 身份，必须标记为 all-identity 协议。默认不启用。

## 3. Ubuntu：完整测试命令

把新脚本上传至服务器的 `tools/`。使用原训练环境，保留现有项目源码，特别是 `tools/evaluate_category_progressive.py`、`tools/build_msmt17_unseen_manifest.py`、`reid/`、`lreid_dataset/`。无需增加绘图库或下载模型。

```bash
export REID_PROJECT="/home/haichao/ouyang/CVPR2026-VLADR-main"
export REID_MODEL="$REID_PROJECT/experiments/ecpm_pgca/full_seed42_only_Clustering_evolution_recurring_transfer"
export REID_UNSEEN="$REID_PROJECT/experiments/unseen/full_seed42_dog_test_q1_seed42"

cd "$REID_PROJECT"

python -u tools/evaluate_unseen_category.py \
  --run-dir "$REID_MODEL" \
  --output-dir "$REID_UNSEEN" \
  --datasets msmt17 dogfacenet \
  --msmt17-root "$REID_PROJECT/data/MSMT17" \
  --dogfacenet-root "$REID_PROJECT/data/DogFaceNet" \
  --device cuda \
  --batch-size 128 \
  --workers 0 \
  --encoder prototype \
  --gallery per_dataset \
  --dog-split test \
  --dog-protocol one_query \
  --seed 42
```

`REID_MODEL` 改成实际最终模型目录，须包含 `latest.pt` 和 `reference.pt`。必须完成该模型配置的全部阶段；支持原顺序或逆序训练，不假定最终阶段名称是 `t4`。模型保存的原 `stream_config` 路径、CSV 与 FINCH 环境应仍然可用，继续执行现有断点绑定校验。

无需把两个 unseen 数据集加到 `main.json`；添加它们反而会改变绑定的数据协议指纹。评估直接从新脚本的根目录参数读取。

`beta`、`routing_summary` 默认继承模型设置。如需显式覆盖可使用 `--beta 0.5 --summary ecpm`，报告会记录实际值。新输出目录必须不存在或为空。

MSMT17 默认需要读取 93,820 张测试图片做推理；这些图片按 batch 加载，不会全部解码到内存。CPU 保存特征，排名逐 query 计算，不分配完整的 11,659 × 82,161 距离矩阵。默认 512 维单路 float32 特征约 185 MiB，排名过程还会使用额外工作内存。`--encoder both` 会保存两路特征并增加计算量。

显存不足将 `--batch-size` 改成 32。也支持 `--device cpu`。排名使用 CPU，`--ranking-threads` 默认 1，避免逐 query 小矩阵运算被过多线程拖慢。控制台持续输出元数据审计、特征提取和排名进度。

## 4. 其他运行方式

### 只检查数据协议，不读取图片、不加载模型

```bash
python -u tools/evaluate_unseen_category.py \
  --audit-only \
  --stream-config "$REID_PROJECT/config/category_progressive_real/main.json" \
  --msmt17-root "$REID_PROJECT/data/MSMT17" \
  --dogfacenet-root "$REID_PROJECT/data/DogFaceNet" \
  --output-dir "$REID_PROJECT/experiments/unseen/protocol_audit"
```

`--stream-config` 在审计模式下可省略，此时只审计测试协议，不声称已检查训练数据。正式加载模型测试时，自动检查模型实际绑定的训练流，不需要这个参数。

本地已检查当前四阶段训练流的 55,853 条训练记录，没有发现这两个数据集的路径、来源名称或完整身份键参与训练。这是元数据检查，不是遍历图片做内容哈希去重，也不能识别所有改名复制的图像。

### 只测试 DogFaceNet

```bash
python -u tools/evaluate_unseen_category.py \
  --run-dir "$REID_MODEL" \
  --output-dir "$REID_PROJECT/experiments/unseen/dogfacenet_only_seed42" \
  --datasets dogfacenet \
  --dogfacenet-root "$REID_PROJECT/data/DogFaceNet" \
  --device cuda --batch-size 128 \
  --dog-split test --dog-protocol one_query --seed 42
```

### 可选对照

- `--encoder both`：同时报告 `prototype` 和 `reference`。后者是该 checkpoint 的冻结参考主干，不使用 Adapter，不等同于重新训练的 baseline。
- `--gallery both`：同时报告各数据集自身图库和两个选中 unseen 数据集图库的并集。混合图库仅包含 `--datasets` 选中的数据集，不包含原五个已见数据集，也不额外把 query 加入 gallery；DogFaceNet `leave_one_out` 协议本身允许 query/gallery 重叠，并在排名时排除自身。
- `--save-features`：额外保存 `features.pt`（两路或一路 CPU 特征、逐行元数据、预测路由与断点/协议指纹）。这是分析缓存，本入口尚未提供从缓存恢复评估的命令。

不同协议或不同模型请使用独立输出目录。对比模型时固定 DogFaceNet seed、类别名单和协议，并检查清单指纹一致；不要在 unseen 测试集上挑选表现最好的 seed 或路由参数。

## 5. 输出与论文报告方式

| 输出 | 内容 |
|---|---|
| `metrics.csv` | 各数据集、编码方式、图库范围的 query/gallery 数量、mAP、Rank1；数值为百分数 |
| `results.json` | 完整结果、路由分布、模型 SHA256、训练流指纹、运行环境与完成状态 |
| `protocol_audit.json` | 真实数量、划分规则、排除身份、协议清单 SHA256 |
| `msmt17_unseen_manifest.csv` | MSMT17 每张 query/gallery 的标签及路径 |
| `dogfacenet_unseen_manifest.csv` | DogFaceNet 确切的 query/gallery 划分及标签 |
| `features.pt` | 仅在指定 `--save-features` 时导出 |

先保存协议，排名每完成一个数据集即保存结果。中断后的 `results.json` 可能标记 `extracting` 或 `ranking`；只有 `status=complete` 才表示全部请求完成。脚本不会覆盖已有实验目录，也不自动续跑半完成结果。

建议将 MSMT17 和 DogFaceNet 分行报告，同时写明 `prototype`、per-dataset/mixed gallery 及 DogFaceNet 的 test 身份 + one-query/leave-one-out 协议。两者难度和协议不同，不建议只用一个平均数代替各自指标。MSMT17 的路由类别准确率是辅助诊断，DogFaceNet 不定义已有类别预测的“正确率”。

测试复用现有排名实现，因此 mAP、Rank1、稳定排序和摄像头过滤与项目已有评估一致；不会因为新类别不在训练登记表中而跳过该数据集。

## 6. 验证记录与来源

已在本地完成两个真实数据集的元数据审计，没有运行真实大模型的完整性能测试。另用小型模型验证：两数据集解析、固定划分、跨摄像头正样本、混合图库负样本、两种编码方式、图像独立路由，以及模型参数/ECPM 状态/断点文件均保持不变。

- [DogFaceNet 数据集提供方说明](https://zenodo.org/records/12578449)：对齐图片版本、身份文件夹与 train/test 类别清单。
- [DogFaceNet 官方项目](https://github.com/GuillaumeMougeot/DogFaceNet)：原任务为狗脸验证与识别。
- [Torchreid 的 MSMT17 实现](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/image/msmt17.py)：标准 query/gallery 数量与摄像头元数据。
