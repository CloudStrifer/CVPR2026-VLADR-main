# 指定模型的五类别 t-SNE 可视化

入口：`tools/visualize_category_tsne.py`。无需重新训练，无需手动整理图片目录。与检索图一样，指定包含 `latest.pt` 和 `reference.pt` 的实验目录，脚本读取模型绑定的数据配置，自动选取测试图片。

## 1. 两类图分别表达什么

| 文件 | 内容 | 默认抽样 |
|---|---|---|
| `tsne_categories.png` / `.pdf` | 五个类别放进同一次 t-SNE，按真实类别着色 | 每类最多 150 张，共最多 750 张 |
| `tsne_identities_<数据集名>.png` / `.pdf` | 每个数据集保存一张独立图片，展示类别内部的身份结构，按真实身份着色 | 每类随机最多 10 个身份，每个身份最多 10 张 |

类别为 Market1501（person）、VeRi（vehicle）、iPanda50（panda）、ATRW（tiger）、Boat（boat）。身份图分别保存为 `tsne_identities_market1501`、`tsne_identities_veri`、`tsne_identities_ipanda50`、`tsne_identities_atrw`、`tsne_identities_boat`，扩展名取决于 `--formats`；不再输出五列合并的 `tsne_identities.png`。

总览图没有顶部标题，底部图例只显示数据集名称，不显示 `(n=...)`。各身份图只保留顶部数据集名称及底部 `ID 1` 等身份图例，不显示身份/图片数量和底部说明句。每个数据集的身份图仍**独立进行 t-SNE**；图例编号是该图内的显示编号，真实身份映射保存在 `tsne_report.json`，不是原始 PID。

特征与之前 `mixed / prototype` 检索使用同一套 `RoutedEncoder`：图像 → 冻结参考编码器和 ECPM 原型路由 → 预测类别对应的 Adapter → L2 归一化检索描述子。模型只接收图像；真实类别和身份用于抽样、分组、着色，不用于选择 Adapter。路由预测错误的图片也保留，不按结果好坏挑样本。`beta` 和 `routing_summary` 默认继承模型保存的设置。

这里展示的是每张图片的特征，不是检索排名或 ECPM 原型散点。`mixed` 和 `per_dataset` 的图库范围差异并不意味着需要两套图片编码方式。

## 2. 模型和环境要求

使用原来能运行训练和检索的 Python 环境。额外绘图依赖可安装：

```bash
python -m pip install matplotlib scikit-learn
```

`--run-dir` 是实验目录，不是单独的 `.pt` 路径，更不是 CLIP 预训练文件。目录中必须同时保留：

```text
latest.pt
reference.pt
```

与检索工具相同：模型必须是已完成阶段提交的断点；默认绘制全部五类，应选择已学完五类的模型。断点中原始 `stream_config` 路径、CSV 清单及测试图片可访问，配置指纹和 FINCH 运行环境应与训练时一致。脚本复用现有断点校验逻辑。推荐使用结束训练的实验目录。

需把 `tools/visualize_category_tsne.py` 上传到服务器，同时保留此前的 `tools/visualize_category_retrieval.py` 与 `tools/evaluate_category_progressive.py`。本次只增加可视化脚本、说明和测试，不更改训练核心源码。

## 3. Ubuntu 完整命令

下面沿用此前检索示例的模型目录；更换模型时修改 `REID_MODEL`，并给 `REID_TSNE` 一个新的输出目录。

```bash
export REID_PROJECT="/home/haichao/ouyang/CVPR2026-VLADR-main"
export REID_MODEL="$REID_PROJECT/experiments/ecpm_pgca/full_seed42_only_Clustering_evolution_recurring_transfer"
export REID_TSNE="$REID_PROJECT/experiments/tsne/full_seed42_sample42"

cd "$REID_PROJECT"

python -u tools/visualize_category_tsne.py \
  --run-dir "$REID_MODEL" \
  --output-dir "$REID_TSNE" \
  --device cuda \
  --split test \
  --categories person vehicle panda tiger boat \
  --images-per-category 150 \
  --ids-per-category 10 \
  --images-per-id 10 \
  --batch-size 128 \
  --seed 42 \
  --perplexity 30 \
  --max-iter 1000 \
  --formats png pdf \
  --dpi 300
```

运行会依次输出加载模型、各类抽样数量、特征提取进度、各面板 t-SNE 进度和最终目录。特征提取使用 `--device`，t-SNE 在 CPU 上运行。显存不足可以把 `--batch-size` 改成 `32`；也支持 `--device cpu`。

输出目录必须不存在或为空，避免覆盖已有图和样本记录。如果一次运行提取完特征但绘图失败，可用下一节的缓存命令恢复到新目录。

## 4. 抽样、保存与复现

- 候选池为指定 split 的 query 与 gallery 合并，按图片路径去重；不混入训练图片。
- 总览图在每类内部轮流从不同身份取图片，避免照片多的身份占满样本。
- 身份子图随机选择至少有两张图片的身份，每个身份尽量先取不同摄像头，再补充同摄像头图片。摄像头多样性取决于数据实际提供的 camid。
- 图片或身份不足时使用实际可用数量，不重复补齐；实际数量保存在 JSON 中。少于 3 张图片或特征全部相同时，该面板说明原因而不生成虚假的散点分布。
- 两类图抽样集合可能不同，但共享图片只提取一次。默认合计最多提取 1,250 张唯一图片，通常更少，不会提取整个测试图库。
- 所有面板使用余弦距离、PCA 初始化、自动学习率。小样本面板的 perplexity 自动下调为 `min(指定值, max(1, (样本数 - 1) / 3))`；实际值写入报告。

除了图片，目录还会包含：

| 文件 | 用途 |
|---|---|
| `features.npz` | 归一化特征、预测路由和绑定的样本元数据；后续无需重新加载模型 |
| `sample_manifest.json` | 每张图片的路径、真实类别/身份、所属面板，以及模型 SHA256、阶段、抽样和路由设置 |
| `coordinates_00_overview.csv` | 总览图二维坐标与每个点的原始记录 |
| `coordinates_01_person.csv` 等 | 各身份子图二维坐标与原始记录 |
| `tsne_report.json` | 各面板实际 t-SNE 参数、优化结果、依赖版本、身份图例映射和图片路径 |

固定模型、数据清单、种子和运行环境可复现抽样及绘图。跨库版本、设备的结果不承诺逐位一致。已导出的坐标可直接用于后续排版，不需要重新降维。

## 5. 直接用缓存重新画图

如果只想改变 perplexity、t-SNE 随机种子或输出格式，运行：

```bash
python -u tools/visualize_category_tsne.py \
  --features-file "$REID_TSNE/features.npz" \
  --output-dir "${REID_TSNE}_perplexity20" \
  --seed 42 \
  --perplexity 20 \
  --max-iter 1000 \
  --formats png pdf svg \
  --dpi 300
```

缓存模式不加载模型、不访问原始图片、不初始化 GPU，也不需要 PyTorch；只需 NumPy、scikit-learn、matplotlib 和 Pillow。它保留缓存中的样本、类别顺序和路由设置，抽样参数、`--categories`、`--split`、`--beta`、`--summary` 不会重新应用；`--seed` 仅改变降维种子。需要换图片或换模型时应重新使用 `--run-dir`。

缓存重绘目录只记录原 `features.npz` 路径，不复制特征文件，请保留原缓存。

## 6. 论文中的解释边界

总览图辅助观察跨类别特征结构；身份子图辅助观察同类内部的身份局部结构。两者应结合 mixed-gallery mAP/Rank-1 和检索示例解释，不把“看起来分得更开”单独当作检索性能或模块有效性的证明。子图之间的绝对位置、距离、面积不能直接比较；总览中远距离簇的距离也不等于原始特征距离。

这是单个模型的可视化，不自动构成消融对比或持续学习遗忘分析。后续如对比其他模型，应复用完全相同的样本清单及处理参数，而不是挑选各自看起来最好的随机种子。
