# 逐阶段性能输出与结果查看

## 1. 这次实现了什么

开启 `--evaluate` 后，每完成一个阶段的训练和评估，终端立即打印该阶段的类别、mAP、R1 和遗忘，并更新运行目录中的结果文件。不用等所有阶段结束才查看。

阶段数量从 `--stream-config` 指向的数据流配置读取，支持 T1–T5，也支持其他数量。本次没有改变数据划分、模型、损失或检索指标算法。

**现有真实配置仍是四阶段。** 五个数据集不等于五个训练阶段；当前配置如下：

| 阶段 | 本阶段训练类别 | 本阶段未训练、但仍评估的旧类别 |
|---|---|---|
| T1 | person, vehicle | 无 |
| T2 | person, panda, tiger | vehicle |
| T3 | vehicle, panda, person | tiger |
| T4 | tiger, boat, vehicle | person, panda |

若要更换为五阶段，需要先确定完整 T1–T5 类别安排，再按身份重新划分数据流。单纯修改 `--epochs` 不会改变阶段数。五阶段的交错流程已经用小型合成数据测试，但该测试没有覆盖真实配置。

## 2. 输出中的类别分别是什么意思

- **本阶段训练类别**：本阶段真正提供训练图片的类别。
- **首次出现**：以前从未训练过的类别。
- **再次训练**：以前训练过，本阶段提供新身份的类别。同一训练身份仍不跨阶段重复。
- **已见但本阶段未训练**：历史学过、本阶段没有训练图片的类别。
- **累计已见／本次评估类别**：截至当前阶段学过的全部类别；本次都会测试，用来观察旧类别是否遗忘。
- **本阶段来源数据集**：说明类别对应的数据集名称。例如 `vehicle` 是类别，`veri` 是来源数据集名称。

例如当前真实 T2 的训练类别是 person、panda、tiger，但评估类别是 person、vehicle、panda、tiger。vehicle 虽然没有参与 T2 训练，仍需测试。

训练开始时还会打印 `stage_training_start` JSON，包含类别及每类别训练身份数、图片数、来源数据集。这里的图片数是数据流样本数，不是采样器在多轮迭代中累计读取的次数。

## 3. 每阶段结果怎么看

终端在原有 `evaluation_progress` 进度之后打印可读表格，包含：

| 字段 | 含义 |
|---|---|
| 训练身份/图片 | 该类别在当前阶段的数据量；未参与训练的旧类别为 0/0 |
| mAP (%) | 当前模型的检索平均精度 |
| R1 (%) | 当前模型的 Rank-1 检索准确率 |
| mAP / R1 遗忘 (pp) | 对应指标相对历史最佳的非负下降量，单位为百分点 |
| 全部已见类别平均 | 截至当前阶段全部已见类别的等权平均 |
| 本阶段训练类别平均 | 只对当前参与训练的类别求等权平均 |
| 旧类别平均遗忘 | 只对本阶段开始前已经学过的类别求平均 |

“当前阶段性能”使用当前阶段结束后的模型，覆盖全部已见类别；另列当前训练类别平均供比较。新增类别会改变全部已见类别平均的组成，因此该平均升降本身不能直接代表遗忘。

结果按评估协议分开：

- `per_dataset / prototype`：各数据集原图库，自动类别路由，是主要结果。
- `mixed / prototype`：已见类别混合图库，自动类别路由。
- `oracle`：给定真实类别选择 Adapter 的诊断对照。

`--eval-gallery both` 会同时计算两种图库。终端阶段汇总默认显示 prototype；结果 Markdown、JSON、CSV 同时保留 prototype 和 oracle。类别路由准确率是另一项诊断指标，不等于检索 R1。

自动路由在已见类别中选择 Adapter，不代表可以识别从未训练过的类别。

## 4. 遗忘的准确计算方式

令 \(a_{s,c}\) 为第 \(s\) 阶段结束时类别 \(c\) 的 mAP 或 R1，分数采用 0–100 标度。对旧类别：

\[
b_{t,c}=\max_{s<t,\ c\text{ 已出现}}a_{s,c},\qquad
F_{t,c}=\max(0,b_{t,c}-a_{t,c}).
\]

例如历史最佳 mAP 为 80%，现在为 70%，遗忘是 **10 个百分点（10 pp）**。如果现在达到 85%，遗忘记为 0，而不是负数。

设 \(C_{<t}\) 为本阶段之前已经学过的类别集合：

\[
\bar F_t=\frac{1}{|C_{<t}|}\sum_{c\in C_{<t}}F_{t,c}.
\]

新类别没有历史性能可比较，因此显示 `--`，JSON 记为 `null`，CSV 留空；不会把新类别按零计入旧类别平均。首阶段没有旧类别，平均遗忘也为 `--`。

如果需要相对百分比，JSON/CSV 另存 `forgetting_relative_percent` 对应值：\(100F_{t,c}/b_{t,c}\)。上述例子为 12.5%，与 10 pp 是两种口径。历史最佳为零时相对百分比不定义，留空。

每个类别、每种图库、每种路由、每个指标分别维护历史最佳，不能交叉比较。这里测量的是固定留出身份测试集上的性能变化，不是直接重测旧训练身份。

原 `evaluation_summary.json` 中的 `lifelong` 计算方式保持原样。它的宏平均遗忘包含新类别的零值，因此可能与新版“旧类别平均遗忘”不同；比较实验时必须统一字段。

## 5. 新训练的使用方法

在项目根目录上传并覆盖补丁中的同路径文件，保留原有代码和数据。已有环境变量 `REID_STREAM`、`REID_CLIP`、`REID_RUN_ROOT` 的含义不变。以下使用新目录：

```bash
CUDA_VISIBLE_DEVICES=2 python -u train_category_progressive.py \
  --stream-config "$REID_STREAM" \
  --reference-checkpoint "$REID_CLIP" \
  --output-dir "$REID_RUN_ROOT/full_seed42_stage_reports" \
  --device cuda --amp \
  --epochs 10 --iterations-per-epoch 100 \
  --batch-size 32 --num-instances 4 \
  --prototype-batch-size 128 --eval-batch-size 128 \
  --workers 0 --seed 42 \
  --adapter-lr 0.0003 --head-lr 0.0003 --weight-decay 0.0001 \
  --lambda-tri 1.0 --triplet-margin 0.3 \
  --consistency drift --lambda-con 1.0 --gamma 1.0 \
  --init-mode similarity --alpha 0.5 --delta 0.5 \
  --control-summary ecpm --routing-summary ecpm --beta 0.5 \
  --evaluate --eval-split test --eval-gallery both \
  --checkpoint-every 1
```

每阶段训练 10 轮，阶段数取决于数据流配置。必须保留 `--evaluate` 才会逐阶段测试；没有开启该参数时只训练。

每次阶段评估结束后，运行目录更新：

| 文件 | 推荐用途 |
|---|---|
| `stage_results.md` | 首选阅读文件，按阶段展示类别、数据量、各项指标和平均值 |
| `stage_results.json` | 编程读取，含完整类别上下文、两类平均和旧类别遗忘 |
| `stage_metrics.csv` | 用 Excel 等打开；每行对应一个阶段、图库、路由、类别 |
| `evaluation_summary.json` | 原始详细报告与原有终身指标；阶段报告新增 `stage_context` |
| `evaluation.jsonl` | 已完成阶段评估的原始记录 |

三个新文件从已保存评估历史重新生成，不会在恢复时重复追加行。每个文件分别采用临时文件替换；若写出期间中断，下次用相同代码恢复时会重建派生报告。评估未完成时不会写入该阶段的最终分数。

**版本兼容：** 本次更改了主训练入口并新增源码文件，会改变精确恢复所校验的源码指纹。旧版本尚未完成的训练应继续使用其原版本完成；不要直接换成本次代码再对旧断点执行 `--resume`。旧的“首次评估提速迁移工具”不适用于本次报告功能。使用本次版本新建的实验，仍支持在相同代码、环境和数据下精确恢复。

## 6. 已经训练完的结果：无需重跑

只要有完整 `evaluation_summary.json`，可以直接转换为新版报告。下面的路径应指向实际完成训练的目录：

```bash
python tools/summarize_category_stages.py \
  --run-dir "$REID_RUN_ROOT/full_seed42"
```

仅在终端查看 T4，并附带 oracle：

```bash
python tools/summarize_category_stages.py \
  --run-dir "$REID_RUN_ROOT/full_seed42" \
  --stage t4 --include-oracle
```

`--stage` 只限制终端显示，导出文件仍包含全部已完成阶段。另存结果可增加 `--output-dir /path/to/report`。工具只读原始 JSON 并写三个派生文件，不加载模型、图片或执行 GPU 评估，不改原始 JSON 和训练断点。需在已安装项目依赖的 Python 环境执行。

旧结果没有 `stage_context` 时，用 ECPM 的类别最近更新阶段和累计身份数恢复当前训练类别与身份数量；旧报告没有记录的图片数量、来源数据集和总阶段数显示未知，不编造。若连这些 ECPM 记录也缺失，工具会明确报错，不能把所有已见类别误当成本阶段训练类别。

Windows 终端需要重定向中文输出时建议 `python -X utf8 tools/summarize_category_stages.py ...`；导出文件本身使用 UTF-8，CSV 带 BOM。

已用你提供的 seed 42 四阶段结果生成 [真实结果预览](stage_results_seed42_preview/stage_results.md)。T4 原图库自动路由：全部已见类别平均 mAP **54.07%**、R1 **73.61%**；本阶段训练类别平均 mAP **61.56%**、R1 **83.27%**；四个旧类别平均遗忘 mAP **0.17 pp**、R1 **1.46 pp**。这些值只是从已有结果整理得到，没有重新训练或评估。

## 7. 代码位置与验证

| 文件 | 本次职责 |
|---|---|
| `train_category_progressive.py` | 开始训练时说明类别；每阶段评估后保存上下文、打印表格、导出报告；恢复时重建派生文件 |
| `reid/evaluation/stage_reporting.py` | 提取类别上下文，计算旧类别遗忘与平均，生成 Markdown / JSON / CSV |
| `tools/summarize_category_stages.py` | 将历史结果转换为新版可读报告 |
| `tests/test_stage_reporting.py` | 类别语义、遗忘计算、未知值、输入校验、导出稳定性测试 |
| `tests/test_progressive_evaluation.py` | 阶段评估、恢复重建、五阶段交错训练逐次输出的集成测试 |

本地 40 项相关测试通过：6 项报告单元测试、19 项评估测试、14 项恢复测试、1 项新增五阶段集成测试。记录保存在 `stage_reporting_*_tests.txt`。五阶段测试使用小模型和合成图片，验证工程流程，不代表真实五阶段数据上的模型效果。

补丁包 `stage_reporting_upload.zip` 包含主入口、新报告模块、历史结果工具及本说明。它是覆盖现有项目的增量包，不含数据、模型权重，也不是完整训练项目。
