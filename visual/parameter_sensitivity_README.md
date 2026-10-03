# α、δ 参数敏感性折线图

两个独立 Python 脚本，使用提供的工作簿中 `mixed/prototype` 的“已见类别平均”数据：

- `plot_alpha_sensitivity.py`：Sheet1 的 C8:E12。
- `plot_delta_sensitivity.py`：Sheet1 的 C18:E22。

每个脚本的 `DATA` 列表已写入对应的五组真实结果，可直接运行，无需 Excel 文件或 openpyxl。更新实验结果时修改相应列表。未使用 F/G 列的“本阶段训练类别平均”，也未重新计算或合并两个参数实验的结果。输入 Excel 保持原样。

```bash
python -m pip install matplotlib
python visual/plot_alpha_sensitivity.py
python visual/plot_delta_sensitivity.py
```

默认在脚本旁边的 `parameter_sensitivity/` 文件夹生成：

```text
alpha_sensitivity.png / .pdf / .svg
delta_sensitivity.png / .pdf / .svg
alpha_source_data.csv
delta_source_data.csv
```

PNG 默认 600 dpi；PDF/SVG 为矢量输出，文字可编辑。脚本可以单独复制使用，默认输出位置随脚本移动。重复运行会覆盖对应同名输出。

可选参数：

```bash
python visual/plot_alpha_sensitivity.py --output-dir figures/alpha --formats png pdf --dpi 600
python visual/plot_delta_sensitivity.py --output-dir figures/delta
```

## 视觉与数据约定

这两张独立折线图用于展示已测参数取值下的性能变化，属于定量比较图。样式与提供的 Stage-wise Performance 参考图保持一致：mAP 使用蓝色 `#0072B2`、实心圆及实线；R1 使用橙色 `#D55E00`、实心方块及实线，图例显示为 `mAP` 和 `Rank-1`。这是参考图的冷暖配色设计，不声称期刊有统一强制色表。

图尺寸为 4.5 × 3.2 英寸，白底、浅灰横纵虚线网格，图例在绘图区右下角；黑色左/下边框及向内刻度。纵轴名称为 `Performance(%)`，横轴分别为 α、δ。两指标共用一个百分数纵轴，两张图均固定为 45–80%，不使用双轴或平滑曲线。直接连接实测点，不显示点旁的具体数值。`--no-values` 仅为兼容旧命令保留，现在默认就不显示数值。

表格未提供重复实验的标准差、置信区间或种子数量，因此不添加误差线或显著性标记。α=0.5 的 54.82/75.26 与 δ=0.5 的 54.88/75.22 分别按原表保留，不将两组记录强行统一。

已核对两个脚本中全部 30 个数值（各含 5 个参数值、5 个 mAP、5 个 R1）以及导出的 CSV，均与源单元格一致；样式更新后重新检查了 PNG 预览的坐标轴、图例及无点值标注。
