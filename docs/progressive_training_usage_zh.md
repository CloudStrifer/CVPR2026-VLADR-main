# 第 8 步：连续阶段训练、日志与精确恢复

更新日期：2026-09-14。

## 1. 这一步完成了什么

前 7 步已经能训练一个阶段。本步把它们串成一个连续任务：从 T1 自动训练到最后一个阶段，并在每次完整优化器更新之后保存可恢复状态。

入口为 `train_category_progressive.py`。默认启用 ECPM、PGCA 熟悉类别漂移蒸馏（`--consistency drift`）和陌生类别迁移初始化（`--init-mode similarity`）。原有单阶段入口继续保留。

一次“更新”是：每个当前类别各取一个 P×K 批次，分别计算类别内部 CE、Triplet 和适用的蒸馏损失，累加梯度，最后执行一次 AdamW 更新。类别的平均损失相加，不再除以类别数量。各类别的训练身份仍然只属于一个阶段。

这里的“精确恢复”指回到最后成功写入磁盘的完整更新边界，接着使用相同的采样顺序、增强随机数、教师、分类头和优化器状态。不保存反向传播中途的计算图；尚未落盘的更新会重做。

## 2. 阶段流程

```text
between_stages：上一阶段已提交（T1 前为空历史）
  → 仅读取当前阶段训练图像，生成 ECPM 候选
  → 固定旧类别教师、漂移权重和新类别迁移结果
  → 创建当前类别临时分类头和阶段优化器
  → 保存 training 状态，当前阶段更新数为 0
  → 每次完整更新后按保存频率写入断点
  → 最后一次更新落盘，ECPM 仍为旧状态
  → 结束训练、释放分类头和教师、提交 ECPM 候选
  → 原子保存 between_stages，最后一个阶段则保存 complete
```

“提交”表示把当前身份原型正式追加到历史记忆。未训练完不会提交。即使最后一次更新已完成、提交文件时失败，也会从已保存的候选恢复，再提交一次；历史身份不会因此重复追加。

恢复 `training` 状态时，直接读取保存的学生、历史教师和迁移记录，不重新提取当前身份原型，不重新执行迁移，也不把已更新的学生当作旧教师。恢复阶段准备之前的 `between_stages` 状态时，则需要重新执行当前阶段准备。

## 3. 启动和恢复

以下命令在仓库根目录执行。先按第 1 步准备真实的数据流 JSON、manifest 和图像；示例配置的占位图像不能直接用于正式训练。`P=batch_size/num_instances`，每个阶段的每个类别都必须至少有 P 个身份。

新建训练示例（把 JSON 路径换成自己的数据流配置）：

```powershell
python train_category_progressive.py --stream-config "D:\ReIDData\stream.json" --output-dir "logs\ecpm_pgca_run1" --device cuda --amp --epochs 10 --iterations-per-epoch 100 --batch-size 32 --num-instances 4 --workers 0 --checkpoint-every 1
```

首次默认读取本机 `~/.cache/clip/ViT-B-16.pt`，也可用 `--reference-checkpoint` 指定本地权重。不会自动下载。输出目录必须为空，避免覆盖已有实验。

从中断处继续，只需：

```powershell
python train_category_progressive.py --output-dir "logs\ecpm_pgca_run1" --resume
```

恢复时从断点读取训练配置、设备和数据流路径，无需再次输入超参数。如果显式提供不同的训练配置，会报错；不能在一次精确恢复中改变学习率、批大小、epoch 总数或随机种子。

主动运行 20 次更新后保存并退出：

```powershell
python train_category_progressive.py --output-dir "logs\ecpm_pgca_run1" --resume --max-updates 20
```

`--max-updates` 计算本次命令新增的更新数，可停在 epoch 内部。`--max-stages 1` 则在本次完成一个阶段提交后退出。二者均不自动预约后续运行；之后继续使用 `--resume`。

`--checkpoint-every 1` 默认每次更新保存。调大它可以减少写盘，但异常时会退回更早的断点，重做尚未保存的更新。最后一次阶段更新、阶段提交和主动暂停都会强制保存。保存频率与暂停控制可以在恢复命令中调整。

如果是首次加载权重或首次建立 `latest.pt` 之前就失败，此时尚无训练更新可恢复；应在另一个空输出目录重新启动。训练过程中异常退出后，应重新运行 `--resume`，不要继续使用发生异常的内存对象。

## 4. 保存内容与文件

| 文件 | 用途 |
|---|---|
| `reference.pt` | 一次性保存冻结参考主干、架构和参考配置。后续恢复不依赖原 CLIP 文件仍在缓存中 |
| `latest.pt` | 唯一有效的最新状态，绑定 `reference.pt` 的 SHA-256 |
| `run_config.json` | 便于查看的配置和环境记录；恢复以 `latest.pt` 为准 |
| `progress.jsonl` | 阶段准备、类别状态、提交、恢复、暂停、阶段耗时和存储统计 |
| `stage_0000.jsonl` 等 | 对应配置顺序的逐阶段训练日志，记录损失、漂移、蒸馏权重、迁移来源、采样批次和 epoch 汇总 |
| `*.recovered-*.jsonl` | 崩溃后超出有效断点的日志尾部归档，不应再次纳入正式统计 |

请保留整个输出目录。只有 `latest.pt` 而缺少参考文件或断点对应的日志前缀，不满足本入口的恢复要求。

阶段内 `latest.pt` 包含：

- 当前学生 Adapter bank 和临时身份分类头。
- AdamW 动量、参数组、AMP GradScaler；本步学习率恒定，没有调度器，状态显式记录 `scheduler=None`。
- 阶段开始时的历史教师 bank、固定蒸馏权重、漂移和已经执行的迁移元数据。
- 旧 ECPM 记忆以及尚未提交的当前候选。
- 阶段编号、epoch、epoch 内迭代数、总更新数、未完成 epoch 的统计累计量。
- 每类别的采样配置和下一批游标。
- Python、NumPy、PyTorch CPU 和 CUDA 随机数状态。
- 数据流指纹、参考文件哈希、运行环境、源代码哈希和日志字节位置。

保存流程为写临时文件、flush/fsync、原子替换 `latest.pt`。替换失败时保留上一份有效断点。日志先持久化，再提交对应断点；恢复前验证已有日志前缀，把多余尾部归档后截回断点位置，避免重复训练日志。

恢复采样时，只重新生成确定性的 P×K 索引，不读取跳过批次的图像，不消耗这些批次的增强随机数。采样轮次的批次数可能不同，因此游标按实际轮次逐段定位，不用固定批数相除。

## 5. 日志和资源统计的含义

每个 `train_step` 记录当前各类别 CE、Triplet、蒸馏及加权总损失、梯度范数、局部分类准确率和采样索引。这里的分类准确率是当前临时身份分类头的训练指标，不是检索 Rank-1。

阶段提交记录身份数、模式数、类别漂移、原型张量字节、Adapter 张量字节和参考文件大小。`compute_seconds.preparation` 包含本阶段原型提取和训练器准备；`optimizer_updates` 累加已保留更新的训练耗时，不包含停机等待、恢复重建和断点写盘，因此不是端到端总运行时间。

真实 CLIP 短训练验证中，`reference.pt` 约 344.8 MB，只写一次；T2 阶段内断点约 14.4 MB。该数字来自 3 类、12 身份的小实验。类别和身份增多后，Adapter、优化器和原型存储都会增长；频繁完整校验和保存也会产生开销。

## 6. 精确性的适用范围

当前入口限定单进程、单设备、`workers=0` 和内置随机增强。启用 PyTorch 确定性算法，固定 cuDNN、TF32 与 cuBLAS 相关设置；遇到不支持确定性执行的算子应直接报错。

恢复会核对软件版本、设备信息、CPU 线程配置、数据流元数据、参考文件及源代码哈希。当前代码指纹包括入口和 `reid/`、`lreid_dataset/` 下的 Python 文件；修改这些代码后，即使是后续步骤的新功能，也不能将旧断点视为同一环境下的精确恢复。

图像文件内容没有逐文件保存哈希，必须自行保持同一路径下的图像内容不变。跨设备、跨版本、修改代码或替换训练图像后，不承诺逐位一致。本步没有多卡 DDP、多进程预取或任意外部随机增强的恢复实现。

第 7 步的 `completed_stage.pt` 是单阶段格式，不能直接传给本入口作为阶段内断点；新入口的精确恢复从自己的 `latest.pt` 开始。第 9 步正式硬路由与终身检索评估尚未接入，也没有据此得出真实 ReID 性能结论。

## 7. 验证与后续入口

已有验证记录：

- `docs/step8_regression_tests.txt`：173 项回归检查通过，包含前序实现和最初 11 项恢复检查。
- `docs/step8_resume_tests.txt`：13 项第 8 步专项检查通过，另补充阶段提交写入失败、采样轮次批数变化两种边界情况。
- `docs/progressive_resume_smoke.json`：真实 CLIP、CUDA、AMP、随机增强及梯度裁剪，三阶段共 18 次更新，跨进程恢复验证通过。

真实 CLIP 对照在总更新数 8（T2 第一个 epoch 内部）退出，另一个进程恢复并完成第 9 次更新；与直接完成第 9 次更新的基准比较，教师、学生、分类头、优化器、scaler 和 RNG 均逐位一致。随后两个分支都从第 9 次更新继续到 T3 完成，最终 Adapter、原型张量、RNG 和各阶段训练日志均一致。CPU 小模型另验证了保持基准对象连续运行至最终阶段的对照。

复现专项检查：

```powershell
python -m unittest discover -s tests -p test_progressive_resume.py -v
python tools/smoke_test_progressive_resume.py --device cuda --output docs/progressive_resume_smoke.json
```

实现位置：`train_category_progressive.py` 管理阶段和提交；`reid/trainer_category_resumable.py` 管理阶段内训练与恢复；`reid/utils/progressive_checkpoint.py` 管理文件、日志和 RNG 事务。

下一步为总计划第 9 步：原型硬路由、终身检索评估和必要消融。继续开发前先阅读 `docs/ecpm_pgca_step8_progress.md`，再按总计划第 9 步推进。
