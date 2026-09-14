# ECPM / PGCA 第 1 步执行记录

更新时间：2026-09-13

## 当前状态

已完成：第 1 步“类别交错、训练身份跨阶段不重复”的数据协议、审计、加载器和验证。
仅推进实施计划第 1 步，未修改模型、损失或现有训练入口；真实数据集的正式阶段划分尚未转换。

## 中断后如何继续

让 Codex 读取本文以及 `docs/ecpm_pgca_implementation_plan_zh.md` 第 1 步，然后检查工作区实际文件与测试结果，从第一项未完成工作继续。文件落盘是恢复依据；不依赖聊天上下文或后台定时恢复，也没有自动安排 5 小时后的任务。

## 分段清单

- [x] A. 检查现有数据接口、采样器及实施计划。
- [x] B. 实现独立协议解析、身份/图片泄漏审计和阶段视图，已验证。
- [x] C. 实现仅当前阶段的类别训练与确定性原型 loader，已验证。
- [x] D. 添加审计 CLI、三阶段示例配置/manifest、使用说明。
- [x] E. 添加并运行有效性、泄漏、loader 隔离与 CLI 测试（最终 41 项通过）。
- [x] F. 更新实施计划验收状态，写明验证结果与剩余限制。

## 已检查的现状

- 旧 `ManifestReID` 转换 tuple 后丢失原始身份信息，第四字段固定为 0。
- 旧 `get_data_loaders.py` 可在训练为空时回退到 query/gallery，新路径必须拒绝该行为。
- `RandomIdentityBatchSampler` 可复用 P×K 规则；原型 loader 应按每图一次的确定性顺序遍历，不能复用训练采样器。
- 审计应只读元数据，图像解码延迟到被明确请求的当前阶段 loader 迭代。
- 当前仓库已有用户改动，保持不动；新增文件与必要文档更新独立完成。

## 下一项工作

本次授权的第 1 步已完成，无中断遗留代码。用户要求继续时，从实施计划第 2 步“冻结参考编码器与持久类别 Adapter”开始。

后续更新：第 2 步现已完成，最新接续位置为第 3 步。请以 `docs/ecpm_pgca_step2_progress.md` 为准。

如用户想先准备实际训练流，按 `docs/category_stream_usage_zh.md` 转换已有 manifest，确定每类别到达阶段/身份比例和验证划分后另存新配置；不要覆盖现有五数据集文件。

## 已确定接口

- `load_category_stream(config_path, check_images=False)`：全量元数据审计，不解码图片。
- `stream.stage(stage_id)`：只返回该阶段的训练视图。
- `build_stage_loaders(stage, ...)`：按类别返回 train/prototype loader，默认 P×K 训练、原型顺序遍历。
- 配置显式区分 validation/test；评估协议必须选择 cross_camera 或 exclude_self。
- 可选 identity_aliases 映射多个数据源中的同一真实身份，参与所有泄漏审计。

## 验证记录

2026-09-13 第一轮：

```powershell
python -m unittest discover -s tests -p "test_category_stream*.py" -v
```

结果：36 tests，全部通过；包含 Windows workers=1 子进程加载和默认 torchvision 变换的实际运行。

```powershell
python tools/validate_category_stream.py --stream-config config/category_progressive_example.json --output docs/category_stream_example_audit.json
```

结果：3 阶段、12 个训练身份、24 个训练图像条目、4 个评估集合，审计通过。报告已保存。示例图像路径是占位路径，因此此次仅为元数据审计，没有验证实际训练数据集图像。

最终验证：同一 unittest 命令，**41 tests / OK**，运行时 5.575 秒（不含解释器导入开销）。完整输出保存为 `docs/category_stream_verification.txt`。运行环境：Python 3.13.7，torch 2.8.0+cu126，torchvision 0.23.0+cu126；测试使用 CPU。环境出现 pynvml 弃用提示，不影响测试通过。

## 已保存的交付物

- `lreid_dataset/category_stream.py`：纯元数据解析、审计、身份映射、阶段及评估视图。
- `lreid_dataset/category_stream_loaders.py`：类别 P×K 采样、确定性原型遍历和独立评估 loader。
- `tools/validate_category_stream.py`：CLI 和 JSON 审计报告。
- `config/category_progressive_example.json` 和 `config/manifests/category_progressive_example/`：三阶段教程数据结构。
- `tests/test_category_stream.py`、`tests/test_category_stream_loaders.py`：41 项验证。
- `docs/category_stream_usage_zh.md`：真实数据转换规则、命令和 Python 接口说明。
- `docs/category_stream_example_audit.json`：示例审计结果。
- `docs/category_stream_verification.txt`：最终测试完整输出。

所有内容已写入工作区文件，未创建 Git commit，未安排自动定时继续。现有用户改动保持不动。第 8 步的训练断点恢复尚未实现，本记录解决的是本次代码开发中断后的接续。
