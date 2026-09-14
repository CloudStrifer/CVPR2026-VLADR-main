# 已生成的真实数据流：上传与使用

日期：2026-09-14。

**不需要手写 JSON，也不需要逐张图片填写 CSV。已经从仓库实际清单生成好：**

```text
config/category_progressive_real/main.json
```

这份配置使用 Market1501、VeRi、iPanda50、ATRW 和 boat 的真实图片条目，不是之前的占位示例。用户已确认四阶段方案、随机种子 42、约 10% 原训练身份作为验证集。

## 1. 已生成的阶段安排

下表中的数字是互不重叠的**训练身份数**，不是图片数：

| 阶段 | person | vehicle | panda | tiger | boat |
|---|---:|---:|---:|---:|---:|
| T1 | 225 | 173 | — | — | — |
| T2 | 225 | — | 16 | 48 | — |
| T3 | 225 | 173 | 15 | — | — |
| T4 | — | 172 | — | 48 | 72 |

例如 T1 的 225 个行人身份与 T2、T3 的行人身份完全不同；同一身份的所有训练图像一起分到一个阶段。

- 训练共 **1,392 个身份、55,853 张图像条目**。
- 验证共 **157 个身份**：person 76、vehicle 58、panda 4、tiger 11、boat 8。
- 验证数量按各数据集原训练身份数的 10% 向上取整。
- 除去验证身份后，各类别剩余训练身份尽量均匀分到它出现的阶段，余数优先分到更早阶段。
- 原仓库的测试 query/gallery 行、顺序、身份和摄像头标识保留；没有把测试身份挪回训练或验证。
- 最小阶段类别为 T3 的 panda，共 15 个身份，因此 `batch_size=32`、`num_instances=4`（P=8）满足身份数量要求。

这里是用户确认的首版实验划分，不表示阶段比例或随机种子已经被证明最优。保留了原始清单，后续可另存新版本研究其他划分。

## 2. 包里有什么

上传压缩包：`docs/category_progressive_real_upload.zip`。解压后保留仓库相对目录结构：

```text
config/
  category_stream_recipe.json
  category_progressive_real/
    main.json
    ablation_base.json
    manifests/                  # 31 个 CSV
    identity_assignments.csv    # 全部原训练身份最终分配到哪里
    preparation_report.json     # 原清单哈希、分配规则、统计
    audit_report.json           # 本机已完成的实际路径与身份审计
    README_server_zh.md
tools/
  build_category_progressive_stream.py
docs/
  real_data_stream_upload_zh.md
```

31 个 CSV 包括 11 份阶段训练清单，以及 5 类各自的 validation/test query/gallery，共 20 份评估清单。清单中的 `original_pid` 保留原始字符串，例如熊猫身份 `00` 不会改成 `0`；`source_dataset` 跨阶段保持不变。

**压缩包是数据配置补充包，不包含图片、CLIP 权重或完整训练代码。** 服务器还需要完整项目、实际数据和已配置的 Python/CUDA 环境。如果直接上传整个项目，本目录本身已经在项目中，无需重复解压配置包。

## 3. 服务器目录怎么放

默认 `main.json` 中写的是：

```json
"data_root": "../../data"
```

它从 `config/category_progressive_real/` 向上两级找到仓库根目录，再进入 `data/`。保持下面的目录结构，上传后就不需要修改任何图片路径：

```text
CVPR2026-VLADR-main/
  train_category_progressive.py
  reid/
  lreid_dataset/
  tools/
  config/
    category_progressive_real/
      main.json
      manifests/...
  data/
    market1501/
      bounding_box_train/...
      query/...
      bounding_box_test/...
    VeRi/
      image_train/...
      image_query/...
      image_test/...
    iPanda50/
      00_aibang/...
      ...
    atrw/
      train/...
      ...
    boat/
      ...
```

`VeRi`、`iPanda50` 等目录名区分大小写。各数据集内部的文件结构需与仓库现有数据一致，具体相对路径已写在 CSV 中。

如果服务器五个数据集位于另一个共同目录，例如 `/data/reid`，其下面仍有 `market1501`、`VeRi`、`iPanda50`、`atrw`、`boat`，则只需将 `main.json` 的 `data_root` 改为 `/data/reid`。不用逐行修改 CSV。若各数据集分散存储或目录名不同，则需按实际目录调整配置，不能直接假设路径一致。

## 4. 上传之后先检查

在服务器仓库根目录解压配置包；如果已经上传整个最新项目，跳过解压：

```bash
unzip /path/to/category_progressive_real_upload.zip
```

然后运行路径和身份审计：

```bash
python tools/validate_category_stream.py \
  --stream-config config/category_progressive_real/main.json \
  --check-images \
  --output logs/real_stream_server_audit.json
```

本机已经验证所有引用图片存在、无身份泄漏、全部类别的 validation/test 覆盖完整。服务器路径可能不同，因此在服务器上仍需要这一次审计。审计检查文件存在性，不解码全部图片；大清单可能需要一些时间。

`audit_report.json` 和 `preparation_report.json` 中部分绝对路径描述本机生成来源，仅供追溯；训练实际读取的是 `main.json` 和 CSV 中的可迁移路径。迁移后数据流指纹变化是正常现象，应从 Ubuntu 新建训练，不拿本机未完成断点跨环境精确续训。

## 5. 现在 REID_STREAM 可以这样写

在服务器仓库根目录：

```bash
export REID_STREAM="$PWD/config/category_progressive_real/main.json"
```

这次变量指向的是**已经生成的真实文件**，不再是手册里的假设路径 `/data/reid/streams/main.json`。

也可以不使用环境变量，在命令中直接写：

```bash
--stream-config config/category_progressive_real/main.json
```

## 6. 先短训练，再正式实验

在环境与权重已经准备好、服务器审计通过后，先保存两次更新验证启动和恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_category_progressive.py \
  --stream-config config/category_progressive_real/main.json \
  --output-dir logs/real_stream_check \
  --device cuda --amp --workers 0 --seed 42 \
  --epochs 2 --iterations-per-epoch 3 \
  --batch-size 32 --num-instances 4 \
  --prototype-batch-size 32 --eval-batch-size 32 \
  --evaluate --eval-split validation --eval-gallery both \
  --max-updates 2
```

默认读取 `~/.cache/clip/ViT-B-16.pt`。如果权重位于其他位置，在新建训练命令中加 `--reference-checkpoint /实际路径/ViT-B-16.pt`。

恢复该短训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_category_progressive.py \
  --output-dir logs/real_stream_check --resume
```

这是完整数据上的少量训练更新，不是少量图片：ECPM 仍会提取当前阶段全部训练身份原型，评估也会读取完整评估集。先用 validation 检查流程和选择超参数，正式 test 实验固定超参数后另建输出目录。

## 7. 已附带真实数据的消融配置

`config/category_progressive_real/ablation_base.json` 已指向这份真实 `main.json`，默认使用 validation、seed=42、训练 B=32/K=4 和完整方法默认参数。

在仓库根目录生成消融计划：

```bash
python tools/run_category_ablations.py \
  --base-config config/category_progressive_real/ablation_base.json \
  --output-dir logs/real_ablation_validation_s42
```

确认环境、数据和参数后执行；中断后也使用同一条命令恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/run_category_ablations.py \
  --base-config config/category_progressive_real/ablation_base.json \
  --output-dir logs/real_ablation_validation_s42 --execute
```

若 CLIP 权重不在默认缓存位置，在这个 JSON 中增加 `reference_checkpoint` 字段并填入实际路径，再开始新套件。正式 test 套件应另存配置，设置 `eval_split` 为 `test`，使用独立输出目录；不要修改已经开始运行的套件计划。

## 8. 验证集怎样生成

每类从满足协议的原训练身份中确定性抽取约 10%，选中的身份整体退出训练。每个验证身份选一张确定性的随机 query，其余保留图像进入 gallery。

- person、vehicle：跨摄像头。
- panda：跨原清单中的会话编号。
- boat：跨视角，validation 只保留原图；被选身份的 318 张已有增强图片也全部从训练移除，不放入验证。
- tiger：原训练摄像头未知，使用 `exclude_self`，不人为制造跨摄像头标签。

同身份同相机/会话/视角的 gallery 图像按跨摄像头协议排除，query 仍保证有另一个相机/会话/视角的正样本。测试集严格沿用仓库已有清单，不声称另行重建了各数据集的官方协议。

## 9. 可复现与检查证据

生成配方保存在 `config/category_stream_recipe.json`。需要研究新方案时，可以修改配方的副本，输出到一个新目录：

```bash
python tools/build_category_progressive_stream.py \
  --recipe config/category_stream_recipe.json \
  --output-dir config/category_progressive_another_run \
  --check-images
```

此命令依赖原来的五域配置与源 CSV；正常训练已有这份配置时无需重新生成。转换器拒绝覆盖已有输出目录，避免已经开始的实验身份分配被改变。

本次已完成：6 项转换器检查通过；真实数据全量文件存在性与身份审计通过；10 份原测试 query/gallery 在字段/相对路径转换后逐行一致；相对目录迁移检查通过。完整原训练身份分配见 `identity_assignments.csv`，源 CSV 哈希和输出 CSV 哈希见 `preparation_report.json`。

这里只准备数据协议，没有在该完整真实数据流上开始 GPU 训练。训练参数、恢复和更多消融说明见 `docs/ubuntu_training_commands_zh.md`。
