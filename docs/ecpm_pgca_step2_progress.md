# ECPM / PGCA 第 2 步执行记录

日期：2026-09-13

## 当前状态

已完成：冻结参考视觉编码器、持久类别 Adapter、临时分类头、快照与恢复接口。
范围限于模型与参数管理；尚未实现第 3 步 Trainer、ECPM 或 PGCA 损失。

## 中断恢复

恢复时先读取本文和 `docs/ecpm_pgca_implementation_plan_zh.md` 第 2 步，检查工作区文件，从首项未完成工作继续。每段代码和测试结果落盘，不依赖对话记忆或后台进程；不安排自动定时恢复。

## 分段清单

- [x] A. 阅读现有模型、Adapter 实现、数据接口及第 2 步计划。
- [x] B. 实现类别模型：参考/类别特征、持久注册、训练集合和冻结状态，已验证。
- [x] C. 实现 Adapter 导出/复制/载入、模型状态恢复、临时分类头与构建入口，已验证。
- [x] D. 添加针对冻结、梯度、参数独立性、前向一致性和恢复的测试，最终 24 项通过。
- [x] E. 运行测试及现有 Adapter/第 1 步回归，保存日志；合计 70 项测试通过，完整 CLIP CUDA 冒烟通过。
- [x] F. 编写使用文档，更新总计划和本记录。

## 已确认的工程细节

- 复用现有 Transformer 内 `DomainAdapter`，持久键改为稳定类别映射，不改旧模型的行为。
- 新接口统一使用投影 CLS 原始特征；不走旧 classifier、BN、prompt 或普通拼接描述符路径。
- 现有 `VisionTransformer.forward` 按 12 层组织，因此模型测试将使用宽度很小但真实 12 层的视觉 Transformer。
- 模型从创建起冻结参考权重和共享状态；学生前向仍保留经过 Adapter 的梯度。
- 用户已有 `.gitignore`、`attribute_pooling.py` 等改动保持不动。

## 下一项工作

第 2 步已完成，无未收尾代码。用户要求继续时，从第 3 步“类别内部 CE + Triplet 基础训练”开始。

后续更新：第 3 步现已完成，最新接续位置为第 4 步。请以 `docs/ecpm_pgca_step3_progress.md` 为准。

接入时先读 `docs/category_adapter_bank_usage_zh.md`，使用 `wrapper.make_category_model`，先注册当前类别和临时分类头，再创建 optimizer。不要沿用旧 train_stage2.py 的第一阶段解冻逻辑，也不要把普通模型 state_dict 当作完整类别模型恢复格式。

## 已落盘接口

- `ReferenceConfig`：冻结输入尺寸、resize、归一化和权重来源。
- `CategoryAdapterBank.encode_reference / encode_category`：统一输出原始投影 CLS。
- `add_category / set_trainable_categories / adapter_parameters`：类别持久注册与更新集合。
- `export_adapter / load_adapter / copy_category`：独立 CPU 快照和无参数共享的复制。
- `export_bank / load_bank / export_checkpoint / from_checkpoint`：类别注册、参考标识和权重恢复。
- `create_temporary_head / classify / discard_temporary_heads`：类别内临时线性分类头，无 BN。
- `wrapper.make_category_model`：从本地 CLIP ViT-B/16 权重构建新路径，保留旧入口。

本机检测到已有 `C:\Users\Cloud\.cache\clip\ViT-B-16.pt`，约 351 MB；CUDA 可用。

## 验证记录

第一轮：`python -m unittest discover -s tests -p "test_category_adapter_bank.py" -v`，22 tests / OK。完整输出已保存 `docs/category_adapter_bank_tests.txt`。

旧 Adapter 回归：5 tests / OK；第 1 步数据层回归：41 tests / OK，输出分别保存为 `docs/category_adapter_bank_legacy_regression.txt` 和 `docs/category_adapter_bank_data_regression.txt`。

首轮完整 CLIP ViT-B/16 CUDA + AMP 冒烟通过（224×224，输出 2×512）；参考权重 86,192,640 参数，每类别 Adapter 396,544 参数。person 更新后，参考特征和 vehicle 特征逐元素不变，恢复输出最大差异 0。详见 `docs/category_adapter_bank_smoke.json`。随机输入仅验证实现，不是 ReID 实验。

最终补充验证后：新模型 **24 tests / OK**，已覆盖外层 autocast 中的参考输出稳定性，以及整个 bank 在恢复前完整校验。完整 CLIP CUDA 冒烟再次通过，输出报告已覆盖更新。

## 保存的文件

- 新增 `reid/models/category_adapter_bank.py`。
- `reid/models/wrapper.py` 仅新增 `make_category_model` 构建入口，旧入口不变。
- 新增 `tests/test_category_adapter_bank.py`、`tools/smoke_test_category_adapter_bank.py`。
- 新增 `docs/category_adapter_bank_usage_zh.md`。
- 日志：`docs/category_adapter_bank_tests.txt`、`docs/category_adapter_bank_legacy_regression.txt`、`docs/category_adapter_bank_data_regression.txt`、`docs/category_adapter_bank_smoke.txt`。
- 结构化实模型验证：`docs/category_adapter_bank_smoke.json`。

第 2 步支持模型权重恢复；完整训练阶段/优化器/RNG 恢复仍在第 8 步。没有开始正式数据训练，没有更改真实数据划分，没有创建 Git commit 或后台定时任务。所有开发进度均已落盘。
