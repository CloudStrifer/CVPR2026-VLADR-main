# 只用坐标 CSV 重画 t-SNE 图

两个全新、独立的脚本：

- `tools/plot_tsne_overview_csv.py`：读取总览 CSV，生成 `tsne_categories`。
- `tools/plot_tsne_identities_csv.py`：读取单个或多个身份 CSV，每个 CSV 生成对应数据集的独立身份图。

原来的 `tools/visualize_category_tsne.py` 不改动。新脚本不导入原脚本或其他项目模块，每个脚本都可以单独复制到其他目录或电脑运行。

只需要 CSV 文件及 NumPy、Matplotlib；不需要模型、原始图片、`features.npz`、JSON 报告、PyTorch、scikit-learn 或 GPU。

```bash
python -m pip install numpy matplotlib
```

## 1. 最简单的用法

在项目根目录运行，替换成实际 CSV 路径：

```bash
python tools/plot_tsne_overview_csv.py --csv "/path/to/coordinates_00_overview.csv"

python tools/plot_tsne_identities_csv.py --csv "/path/to/coordinates_01_person.csv"
```

只提供 `--csv` 即可。默认在 CSV 同级的 `csv_plots/` 中输出 PNG 和 PDF，分辨率 300 dpi。数据集名称由 CSV 中的 `category` 判断，不依赖文件名。

## 2. Ubuntu：重画总览及五张身份图

下面沿用之前的输出目录；将 `REID_TSNE` 改为实际存放坐标 CSV 的目录。

```bash
export REID_PROJECT="/home/haichao/ouyang/CVPR2026-VLADR-main"
export REID_TSNE="$REID_PROJECT/experiments/tsne/full_seed42_sample42"
export REID_REDRAW="$REID_TSNE/csv_plots"
cd "$REID_PROJECT"

python tools/plot_tsne_overview_csv.py \
  --csv "$REID_TSNE/coordinates_00_overview.csv" \
  --output-dir "$REID_REDRAW" \
  --formats png pdf \
  --dpi 300

python tools/plot_tsne_identities_csv.py \
  --csv \
    "$REID_TSNE/coordinates_01_person.csv" \
    "$REID_TSNE/coordinates_02_vehicle.csv" \
    "$REID_TSNE/coordinates_03_panda.csv" \
    "$REID_TSNE/coordinates_04_tiger.csv" \
    "$REID_TSNE/coordinates_05_boat.csv" \
  --output-dir "$REID_REDRAW" \
  --formats png pdf \
  --dpi 300
```

如果此前修改了类别顺序，CSV 前面的编号也可能不同，按实际文件名填写。

生成文件（同时生成同名 PDF）：

```text
tsne_categories.png
tsne_identities_market1501.png
tsne_identities_veri.png
tsne_identities_ipanda50.png
tsne_identities_atrw.png
tsne_identities_boat.png
```

支持 `--formats png pdf svg`。允许把总览和身份图输出到同一个目录；默认拒绝覆盖已经存在的同名文件，需要覆盖时显式加 `--overwrite`。

## 3. 与原图一致的内容

- 直接使用 CSV 原始 `x`、`y`；不重新降维、不旋转、不归一化、不改变坐标。
- 保持图尺寸、字体、散点大小、透明度、配色和图例布局。
- 总览无顶部标题，图例仅包含数据集名称，不显示 `(n=...)`。
- 身份图顶部仅显示数据集名称，底部保留 `ID 1` 等图例，无身份/图片数量及底部说明句。
- 总览类别顺序采用 CSV 首次出现顺序，与原始导出一致。
- 身份颜色和 ID 编号按完整 `identity_key` 排序，与原脚本一致。相同 PID、不同数据来源的身份不会被误合并；`001` 等前导零会保留。

同一份未改动的 CSV、相同 Matplotlib/字体环境与 DPI 下，PNG 可与原脚本逐像素一致；跨环境的字体或渲染版本差异可能影响像素效果，坐标和文字含义不受影响。PDF/SVG 文件的内部时间戳或对象编号可能不同。

不要把五个独立降维的身份 CSV 拼接成总览 CSV：它们属于不同坐标系。总览应读取原始 `coordinates_00_overview.csv`。

## 4. CSV 格式与异常处理

总览图至少需要 `category,x,y` 列。身份图至少需要 `category,identity_key,x,y` 列；每份身份 CSV 只能包含一个类别。原脚本导出的 CSV 可直接使用，其余列不影响绘图，`path` 对应的图片无需存在。

`identity_key` 保留导出时的 JSON 字符串数组，例如 `["person", "market1501", "001"]`。不要用 Excel 等工具把身份编号自动转成整数或删改身份键。

如果 CSV 为空、坐标缺失、包含 NaN/Inf、身份键损坏，脚本会明确报错，不跳过数据点或补造坐标。原脚本因样本不足或特征全部相同而跳过降维时，CSV 没有有效坐标，也无法凭 CSV 还原散点；CSV 本身不包含具体跳过原因。
