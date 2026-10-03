# 混合图库 Top-10 检索与三色图片边框

入口：`tools/visualize_category_retrieval.py`。不需要重新训练。

Query 和每张 gallery 图片均独立执行 `prototype` 路由，模型只接收图片张量。所有 gallery 特征统一按余弦相似度降序排列，不按照真实类别或预测类别提前筛选候选。

| 候选图片外围的圆角边框 | 含义 |
|---|---|
| 绿色 | 同一真实身份，匹配正确 |
| 橙色 | 同类别、不同身份 |
| 红色 | 不同类别 |
| 灰色 | 缺少真实标签，无法判断匹配是否正确 |

图片保持原始宽高比，完整显示，不做裁剪或颜色增强。每行最左侧为中性灰框 query，其右侧依次排列 Top-1 至 Top-K。图上不显示 Query、Top、类别、PID、Route、cos、Match 等逐图文字，也不显示总标题；整张图只在底部显示一次颜色图例。预测类别、身份和余弦相似度仍完整保存在 JSON 中，余弦相似度不是概率。

## 1. 模型需要哪些文件

将新脚本上传到服务器对应的 `tools/` 目录。使用原来可以运行训练/评估的 Python 环境；绘图另外使用项目依赖中的 matplotlib 和 Pillow。新版脚本在加载模型、提取特征之前检查绘图库，缺少依赖时立即报错。

若报 `ModuleNotFoundError: No module named 'matplotlib'`，在当前已激活的训练环境安装并检查：

```bash
python -m pip install matplotlib
python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot; print('matplotlib:', matplotlib.__version__)"
```

旧脚本可能已经打印 `Gallery: 30633/30633` 和 `Query: 1/1`，随后才在绘图阶段退出。这表示特征提取已完成、图片未成功导出；旧脚本未持久保存特征，需要在补齐依赖后重新运行检索，无需重新训练模型。若原输出目录仍为空，可复用它；若非空，使用新的输出目录。

`--run-dir` 指向一个已经完成阶段提交的**实验目录**，例如：

```text
experiments/ecpm_pgca/full_seed42/
  latest.pt
  reference.pt
  ...
```

不能只提供 CLIP 预训练权重。`latest.pt` 包含已经训练的 Adapter 和 ECPM，`reference.pt` 包含对应冻结主干。脚本复用项目的已提交断点读取器，训练中的断点会被拒绝；请使用已结束训练的实验，或在阶段提交后保留的独立目录。

此外，断点中记录的原 `stream_config` 及其 CSV 清单必须能在原路径读取，数据配置指纹和 FINCH 依赖版本保持一致。即使使用外部 query/gallery，也保留这套模型绑定配置；本工具没有绕过断点校验或搬迁数据路径的功能。默认测试图库的图片也必须存在。

只新增 `tools/` 脚本，不修改训练入口和 `reid/`、`lreid_dataset/` 源码，不改变已有训练源码指纹。

## 2. 最简单：自动从每类测试 query 取一张

在 Ubuntu 服务器运行：

```bash
export REID_PROJECT="/home/haichao/ouyang/CVPR2026-VLADR-main"
export REID_MODEL="$REID_PROJECT/experiments/ecpm_pgca/full_seed42"

cd "$REID_PROJECT"

python -u tools/visualize_category_retrieval.py \
  --run-dir "$REID_MODEL" \
  --output-dir "$REID_PROJECT/experiments/retrieval/full_seed42_examples" \
  --device cuda \
  --split test \
  --queries-per-category 1 \
  --top-k 10 \
  --batch-size 128
```

完成五类训练后，这条命令通常生成一张五行的组合图，每类一张 query。选择规则是清单中每类最先出现的 N 个 query，不挑选表现最好的样例；默认行顺序跟随 evaluation 清单。需要指定各类编号和行顺序时，使用下一节的 `--query-select`。

图库自动合并该断点全部已见类别的测试 gallery，并按图片路径去重。中间阶段只使用当时已见类别。`beta` 和 `routing_summary` 默认继承该实验保存的设置，也可显式指定 `--beta 0.5 --summary ecpm`。

逆序模型只需把 `REID_MODEL` 改为 `.../experiments/ecpm_pgca/full_reverse_seed42`，并换一个输出目录。末阶段自动按照断点中的时间位置读取，不假定末阶段编号必须是 `t4`。

## 2A. 推荐：五类分别选一张，合并成五行图片

```bash
export REID_PROJECT="/home/haichao/ouyang/CVPR2026-VLADR-main"
export REID_MODEL="$REID_PROJECT/experiments/ecpm_pgca/full_seed42_only_Clustering_evolution_recurring_transfer"
cd "$REID_PROJECT"

python -u tools/visualize_category_retrieval.py \
  --run-dir "$REID_MODEL" \
  --query-select person:12 vehicle:1 panda:1 tiger:1 boat:1 \
  --output-dir "$REID_PROJECT/experiments/retrieval/five_categories_grid" \
  --device cuda --split test \
  --top-k 10 --batch-size 128 \
  --formats png pdf --dpi 300
```

每个参数是 `类别:该类别内的query图片序号`，从 1 开始。例如 `person:12` 表示行人的第 12 张 query，不是 PID=12。

以上命令从上到下依次显示 person、vehicle、panda、tiger、boat 五行，每行左侧一张 query、右侧十张检索候选。参数中类别的先后顺序就是最终行顺序；修改冒号后的编号即可换图，也可以同类别选择多张。五个 query 共用一次 gallery 特征提取。

底部仅显示一次 `Correct identity`、`Same category, wrong ID`、`Different category` 图例。若使用外部无标签图片，则额外显示灰色未知标签图例，不猜测正确性。

`--query-select` 与 `--query`、`--query-manifest`、`--query-index`、`--query-category` 不同时使用。它只负责选图，不给模型类别标签，也不限制图库。可以先用 `--list-queries --query-category person` 等命令查看编号。

## 3. 指定一张或多张 query，自动使用现有混合测试图库

```bash
export REID_QUERY="/绝对路径/某张测试query.jpg"

python -u tools/visualize_category_retrieval.py \
  --run-dir "$REID_MODEL" \
  --query "$REID_QUERY" \
  --output-dir "$REID_PROJECT/experiments/retrieval/my_query_top10" \
  --device cuda --split test --top-k 10 --batch-size 128
```

多张图片写在同一个 `--query` 后面：

```bash
--query "/绝对路径/query1.jpg" "/绝对路径/query2.jpg"
```

如果图片路径出现在当前已见类别的所选 test/validation 清单中，真实类别、身份和相机标签会自动匹配。建议直接传原 query 文件路径，不要复制、改名后再传入；工具不通过文件名猜身份，也不按图片外观猜真实类别。

外部图片或改名后的副本仍可检索，但没有标签时只显示灰色边框。要得到可靠三色标注，请使用下一节的 CSV。

## 3A. 按编号或“类别 + 编号”选 query（无需移动图片）

Query 直接读取原始测试清单中的图片，无需复制到新目录。可以先查看编号：

```bash
python tools/visualize_category_retrieval.py \
  --run-dir "$REID_MODEL" \
  --list-queries --query-category person
```

列出 `global_index`（全局序号）、`category_index`（类别内序号）、类别、身份 PID、相机及原始图片路径。默认打印前 50 行，加 `--list-limit 0` 可打印全部；加 `--output-dir "/新目录/query_catalog"` 会导出该筛选范围内的完整 `query_catalog.csv`，不受打印行数限制。查看编号只读取断点与清单元数据，不建立模型、不提取特征、不使用 GPU。

选择行人类别第 12 张 query：

```bash
python -u tools/visualize_category_retrieval.py \
  --run-dir "$REID_MODEL" \
  --query-category person --query-index 12 \
  --output-dir "$REID_PROJECT/experiments/retrieval/person_q12" \
  --device cuda --split test --top-k 10 --batch-size 128
```

类别名称对应关系：`person` = Market1501、`vehicle` = VeRi、`panda` = iPanda50、`tiger` = ATRW、`boat` = Boat。

一次选择该类别的多张图片：`--query-category person --query-index 1 12 30`。只给类别、不指定编号时，默认选该类别第一张；可追加 `--queries-per-category 3` 选前三张。

不指定类别时，编号代表整个 query 清单的全局序号：

```bash
python -u tools/visualize_category_retrieval.py \
  --run-dir "$REID_MODEL" \
  --query-index 100 \
  --output-dir "$REID_PROJECT/experiments/retrieval/global_q100" \
  --device cuda --top-k 10
```

**编号从 1 开始，是图片序号，不是身份 PID，也不是文件名里的数字。** 顺序由当前已见类别的 evaluation 配置顺序及各 query CSV 行顺序决定。同一配置、阶段和 split 下固定；换 test/validation、阶段或清单后应重新查看目录。

`--query-category` 仅帮助选图片，既不把真实类别交给模型，也不缩小 gallery。仍然是自动路由 + 全部已见类别混合检索。每次结果 JSON 的 `query_reference` 会记录选中图片的两种序号、PID 和原路径，便于复现。

## 4. 自定义有标签的 query 与 gallery

分别准备 `queries.csv` 和 `gallery.csv`。图片使用绝对路径，或相对于各自 CSV 目录的路径；也可以用 `--image-root` 统一指定相对路径根目录。

`queries.csv` 示例：

```csv
path,category,source_dataset,original_pid,camid,protocol
images/query_person.jpg,person,my_dataset,001,cam1,cross_camera
```

`gallery.csv` 示例：

```csv
path,category,source_dataset,original_pid,camid
images/person_001_cam2.jpg,person,my_dataset,001,cam2
images/person_002_cam2.jpg,person,my_dataset,002,cam2
images/car_001.jpg,vehicle,my_dataset,001,cam2
```

上面三张候选如果进入结果，分别是绿色、橙色、红色。这里仅解释标注含义，实际排名完全由模型分数决定。

运行：

```bash
python -u tools/visualize_category_retrieval.py \
  --run-dir "$REID_MODEL" \
  --query-manifest "/绝对路径/queries.csv" \
  --gallery-manifest "/绝对路径/gallery.csv" \
  --output-dir "$REID_PROJECT/experiments/retrieval/custom_top10" \
  --device cuda --top-k 10 --batch-size 128
```

也可以只提供 `--query-manifest`，继续使用原始混合测试图库。此时自定义 query 的 `category/source_dataset/original_pid` 必须与原图库的身份命名一致。不能把 `001` 随意写成 `1`；工具保留字符串 ID。已知清单路径的标签冲突会直接报错。原清单中的身份别名会自动保留；外部 CSV 的身份键按三列原样组合，不额外猜测或归并别名。

字段说明：

- `path`：必填图片路径。
- `category`、`source_dataset`、`original_pid`：三列一起填写，或一起留空。完整身份键是三者的组合，不仅仅是数值 PID。
- `camid`：相机编号；使用跨相机协议时，query 和同身份 gallery 均需要。
- `protocol`：query 的过滤协议。`cross_camera` 排除同一图片、以及同身份同相机候选；`exclude_self` 只排除同一图片。自定义 CSV 未填写时为 `exclude_self`，请按数据集的实际协议设置。已有测试清单自动使用其原协议。

这些标签只用于过滤和边框标注，不用于路由或相似度计算。若添加未见类别，系统仍只能路由到已见 Adapter，不具备未知类别拒识；这属于额外检索演示，不是原测试协议结果。

## 5. 只有图片文件夹，也可以检索

```bash
python -u tools/visualize_category_retrieval.py \
  --run-dir "$REID_MODEL" \
  --query "/绝对路径/query.jpg" \
  --gallery-dir "/绝对路径/gallery_images" \
  --output-dir "$REID_PROJECT/experiments/retrieval/folder_top10" \
  --device cuda --top-k 10
```

工具递归读取常见图片格式，按路径排序后提取特征。文件夹中的已知测试图片可以自动关联原始标签，其余图片没有标签时显示灰色。三种正确性颜色需要真实身份信息，模型的预测类别不能代替真实标签。

## 6. 输出与协议

```text
输出目录/
  retrieval_grid.png
  retrieval_grid.pdf
  retrieval_results.json
```

全部 query 合并为一张多行图片，一行一个 query，行顺序与输入选择顺序一致。每行左侧是 query，右侧按真实相似度顺序排列 Top-K；同分使用稳定排序。候选不足 K 时只显示有效候选，其余位置留白，不重复补图。额外指定 `--formats png pdf svg` 可导出 SVG。PNG 默认 300 dpi，可用 `--dpi 600` 提高导出分辨率，但不会增加原图片本身的信息。

JSON 保存模型断点哈希、原始路径、阶段、协议、query/候选真实标签、自动路由类别、相似度、颜色状态和有效图库大小，可以逐项追溯。真实标签不完整时会记录 `annotation_complete=false`。`known_positive_count=0` 表示未发现合法的已知同身份正样本，不应把相似图当成成功重识别。

输出目录必须是新目录或空目录，以免覆盖已有图。每次运行重新提取所用 gallery 特征，模型不更新，也不改动训练目录。大图库第一次运行可能耗时，终端会打印逐批特征提取进度；多个 query 一次传入时共用这次 gallery 提取。显存不足可减小 `--batch-size`。

## 7. 工程验证范围

```bash
python -m unittest discover -s tests -p test_retrieval_visualization.py -v
```

测试覆盖混合排序保留跨类别候选、稳定同分排序、同身份/相机过滤与原评估器一致、三色身份判断、未知标签、CSV 命名空间与重复路径、图像张量独立推理、PNG/PDF/SVG 导出，以及小型真实 Adapter 模型的训练后断点加载与完整检索流程。生成图片的测试仅验证工程链路和布局，不能代表真实数据集性能。
