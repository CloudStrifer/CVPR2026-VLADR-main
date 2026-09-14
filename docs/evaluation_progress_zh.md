# 评估阶段进度输出

后续已完成路径解析提速，并提供保留 T1 的受限迁移工具。当前遇到该瓶颈时优先阅读
[评估提速与断点接续](fast_evaluation_recovery_zh.md)，不必因这次已知评估更新重新训练 T1。

## 本次修改

修改 `reid/evaluation/category_progressive.py` 和 `reid/evaluation/category_oracle.py`，
为连续训练中的阶段评估和 `tools/evaluate_category_progressive.py` 独立评估增加终端进度。
训练命令不需要增加参数，默认启用，所有输出立即 flush。

终端每行是一条 JSON，`event` 为 `evaluation_progress`，`stage_id` 标明当前阶段。

| phase | 含义 | 主要字段 |
|---|---|---|
| evaluation_start | 开始阶段评估 | split、gallery_scope |
| dataset_start | 开始处理一个数据集 | dataset、query_images、gallery_images |
| features_start | 开始提取 query 或 gallery 特征 | dataset、subset、total_batches、total_images |
| features_progress | 完成一批特征提取，含 oracle 与正式路由 | batches / total_batches、images / total_images、elapsed_seconds |
| retrieval_start | 开始一种检索协议；mixed 包含混合图库准备 | routing、gallery_scope |
| ranking_start | 开始一个数据集的检索指标计算 | dataset、routing、gallery_scope、total_queries、gallery_images |
| ranking_progress | 已完成一部分 query 的排序与指标计算 | queries / total_queries、elapsed_seconds |
| ranking_complete | 一个数据集在一种协议下计算完成 | mAP、Rank1、elapsed_seconds |
| routing_diagnostics | 正在汇总路由准确率和身份一致性 | stage_id |
| evaluation_complete | 阶段评估计算结束 | elapsed_seconds、prototype_macro |

特征提取在首批、每 10 批、最后一批输出；指标计算在首个 query、每 100 个 query、最后一个 query 输出。
两者还会在完成一个批次/query 后检查时间，距上次输出达到 10 秒时输出。
这是实际进度，**不是每 10 秒固定心跳**：若某个批次、query 或系统 I/O 本身阻塞，必须等它返回后才会有新计数。

`ranking_progress` 的计数持续增加，说明 CPU 排序与指标计算仍在推进。
每个 dataset / routing / gallery_scope 组合有独立的 query 计数，从头开始不代表重复训练。
进度输出不改变指标。后续提速补丁将路径解析移到配对循环外，并用张量执行过滤，保留评估协议、精度和指标公式。

下面仅为输出格式示意，不是实验结果：

```json
{"event":"evaluation_progress","stage_id":"t1","phase":"features_progress","dataset":"person_test","subset":"gallery","batches":10,"total_batches":125,"images":1280,"total_images":15913,"elapsed_seconds":12.5}
{"event":"evaluation_progress","stage_id":"t1","phase":"ranking_progress","dataset":"person_test","routing":"prototype","gallery_scope":"per_dataset","queries":100,"total_queries":3368,"gallery_images":15913,"elapsed_seconds":45.2}
```

## 上传服务器

在下一次启动前，将以下两个本地文件覆盖到服务器项目的对应位置：

1. `reid/evaluation/category_progressive.py`
2. `reid/evaluation/category_oracle.py`

正在运行的旧 Python 进程不会自动加载改动，可以继续等待它结束。
若需要暂停旧进程，按一次 Ctrl+C，保留输出目录；不要删除已有训练结果。

本次源码变化会改变严格恢复指纹。按用户本次选择，如要使用新版重新训练，
在原完整训练命令中只改为新的空输出目录，例如：

```bash
--output-dir "$REID_RUN_ROOT/full_seed42_evalprogress"
```

不要直接对旧版断点使用新版代码 `--resume`，也不要删除指纹检查来绕过。
新版产生的断点后续可正常用同一版代码续训。
已提交阶段的旧断点也可以用独立评估工具读取，查看该阶段性能，无须为单独评估重新训练；
独立评估不会把结果自动并回训练流程。应在旧进程停止后进行，避免训练继续覆盖 latest.pt。

## 保存终端输出

新增进度写入标准输出，不写入 `progress.jsonl` 或正式评估结果文件。
阶段结果仍写入 `evaluation.jsonl` 和 `evaluation_summary.json`。
`evaluation_complete` 表示计算已结束；主流程随后才写入结果和断点。

如需长期保存，先建立输出目录以外的日志目录：

```bash
mkdir -p "$REID_RUN_ROOT/console"
```

在原训练命令最后追加：

```bash
2>&1 | tee "$REID_RUN_ROOT/console/full_seed42_evalprogress.log"
```

不要把 tee 日志提前放进本次新实验目录，否则会触发“输出目录必须为空”的保护。

## 验证进度

- 已有评估回归测试 15 项全部通过，覆盖检索指标、评估与训练 RNG 隔离、评估中断恢复及独立断点评估。记录保存为 `docs/evaluation_progress_tests.txt`。
- 已检查测试运行实际输出的全部 10 类进度事件及批次/query 计数边界；核对结果保存在 `docs/evaluation_progress_verification.json`。
- 未在用户 Ubuntu 服务器执行，也没有重新启动其当前训练进程。
