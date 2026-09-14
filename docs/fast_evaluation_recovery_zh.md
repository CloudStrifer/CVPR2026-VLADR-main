# 路径解析瓶颈修复与 T1 断点接续

## 已确认的问题

用户 Ctrl+C 的堆栈停在 `oracle_retrieval_metrics → _path_key → Path.resolve → os.lstat`。
这证实中断时正在 CPU 评估的路径解析中。旧实现对每一个 query-gallery 配对重复解析路径，
T1 的两种图库和两种路由合计约 8.47 亿次解析调用，造成长时间无结果。
运行中替换 Python 源文件可能使异常堆栈显示的行文本与已加载代码不一致；下一次启动才能完整采用新代码。

## 修复内容与当前进度

- 路径解析从内层配对循环移出，每次指标调用中每个 query/gallery 样本只解析一次。
- 身份键（类别、来源、身份）、相机和规范化路径在本次调用中编码，用张量完成过滤。
- 没有使用跨评估的全局路径缓存；保留原 cosine 分数、逐 query 计算、稳定排序及 FP64 AP 累加。
- 沿用上次加入的特征批次、排序 query 数、mAP/Rank1 和阶段完成输出。
- `tools/upgrade_evaluation_checkpoint.py` 提供一次受限的评估版本迁移，复制到新目录，不修改原实验。
- 已通过指标对照测试 3 项、评估与迁移恢复测试 17 项、损坏日志时迁移拒绝发布测试 1 项，合计 21 项。
- 随机特征与同分排序、同图排除、跨相机过滤、同 PID 不同类别/来源的对照，mAP 和 Rank1 与旧算法完全相等。
- 迁移测试确认原目录所有文件摘要不变；新断点 bank、ECPM、RNG、训练参数和更新数保留，T1 不重复训练，后续训练结果与连续运行一致。
- 本地 Windows 单线程 CPU 小规模对照（64 query × 1024 gallery，512 维合成特征）：旧版 27.98 秒，新版 0.259 秒，指标完全相同。这不是服务器全量评估耗时或真实 ReID 性能保证。
- 测试记录：`fast_evaluation_metric_tests.txt`、`fast_evaluation_resume_tests.txt`、`fast_evaluation_upgrade_failure_tests.txt`；速度对照：`fast_evaluation_benchmark.json`。

## 上传的文件

可将 `docs/fast_evaluation_fix_upload.zip` 解压到服务器项目根目录。包内只有下面的代码文件和本说明，
不包含或覆盖模型、数据、实验输出。文件级校验记录为 `docs/fast_evaluation_fix_upload_verification.json`。

服务器旧进程已经停止后，覆盖项目中的下面两个文件，并新增工具：

1. `reid/evaluation/category_oracle.py`
2. `reid/evaluation/category_progressive.py`
3. `tools/upgrade_evaluation_checkpoint.py`

保留服务器当前训练代码、Python 环境和数据。尤其不要覆盖 `reid/trainer_category_progressive.py`、
主训练入口或其他训练相关文件；此工具只处理这两个评估文件的差异。

## 两条命令保留 T1、继续训练

在原服务器、原虚拟环境、原项目目录运行，仍使用 GPU 2。
新目录 `full_seed42_evalfast` 必须不存在。原目录 `full_seed42` 无需删除，仍作为原始备份。

```bash
CUDA_VISIBLE_DEVICES=2 python -u tools/upgrade_evaluation_checkpoint.py \
  --run-dir "$REID_RUN_ROOT/full_seed42" \
  --output-dir "$REID_RUN_ROOT/full_seed42_evalfast"
```

成功出现 `evaluation_upgrade_complete` 后再运行：

```bash
CUDA_VISIBLE_DEVICES=2 python -u train_category_progressive.py \
  --output-dir "$REID_RUN_ROOT/full_seed42_evalfast" --resume
```

恢复顺序：读取已提交的 T1 模型和 ECPM → 重新计算尚未完成的 T1 评估 → T2 训练 → 后续阶段。
不重新训练已保存的 T1，不需要重新传入 epochs、学习率等训练参数。
旧进程中没有写出的评估中间特征不在断点里，因此需要重新提取。

新版本仍严格校验普通 `--resume`；这不是删除或关闭源码指纹检查。
迁移工具只接受：

- 第一阶段已提交、首次评估尚未完成；没有活动 trainer/candidate 或已有评估结果。
- 仅两个评估源文件的哈希不同；其余源码列表、训练代码、环境、设备和线程设置一致。
- 目标评估文件匹配工具内固定的已审阅版本（允许 CRLF/LF 换行差异）。
- 复制后通过正常恢复加载器的全部数据、参考权重、ECPM、进度和日志前缀校验。

如果报 `non-evaluation source changed` 或环境不同，保留错误信息并核对差异，不绕过检查。
若断点已进入 T2，或 T1 评估已经完成，工具会拒绝；本工具专门对应当前确认的 T1 待评估状态。
失败的复制目录带 `.upgrade-...` 后缀，留作诊断，不作为有效恢复目录。

迁移记录保存在新目录 `evaluation_upgrade.json`，包含原/新断点摘要、两版运行环境及文件差异。
原断点和训练历史保留；未完成的评估切换为新的等价实现，因此应记录这次版本迁移，
不把它描述为旧版评估程序的逐指令恢复。

## 性能查看

终端 `ranking_complete` 显示单数据集指标，`evaluation_complete` 显示正式路由的宏平均。
完整正式结果仍保存在新目录下的 `evaluation.jsonl`、`evaluation_summary.json`。
后续继续训练和独立评估时，都使用 `full_seed42_evalfast` 这个新目录。

未在用户 Ubuntu 服务器执行；本地验证不代表已经获得真实 ReID 性能。
