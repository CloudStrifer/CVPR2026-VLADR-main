# ECPM / PGCA 第 4 步执行记录

日期：2026-09-13

## 当前状态与恢复位置

已完成：身份原型提取、跨阶段累积、全阶段提交与文件原子保存恢复。第 5 步聚类/模式/类别中心暂不实现。

恢复时先读取本文和总计划第 4 步，核对文件与验证结果，从首个未完成项接续。所有开发内容分段落盘，不安排自动定时任务。

## 分段清单

- [x] A. 检查参考特征接口、当前阶段 loader 与身份协议。
- [x] B. 实现独立原型 loader、FP32 身份聚合与覆盖检查。
- [x] C. 实现阶段候选更新、全阶段提交、保存恢复和重复阶段保护。
- [x] D. 添加可运行提取工具及单元/集成测试。
- [x] E. 运行测试、必要回归和真实 CLIP 冒烟，保存报告。
- [x] F. 更新使用文档、总计划和当前记录。

## 已确定设计

- 原型为 Norm(mean(F0(x)))，不先对单张图归一化。
- 提取前后检查冻结参考权重，自动使用模型固定预处理。
- 一个类别只有一个身份或一张图也能提取原型，不受训练 P×K 限制，因此新增独立原型 loader。
- 每个阶段先完成全部当前类别的候选原型，再整体提交；失败不污染历史记忆。
- 保存完整身份键、FP32 向量、图像数和首次阶段，不保存历史路径、图片或逐图特征。
- 通过参考签名、协议指纹和阶段顺序防止错误恢复与重复累积。

## 下一项

本步骤没有未完成实现。后续更新：第 5 步也已完成，见 `docs/ecpm_pgca_step5_progress.md`。下一步从第 6 步 PGCA 熟悉类别分支开始，使用完整 ECPM 候选的旧快照和漂移。

代码和说明：`reid/memory/ecpm.py`、`lreid_dataset/category_stream_loaders.py`、`tools/extract_identity_prototypes.py`、`tools/smoke_test_identity_prototype_memory.py`、`tests/test_identity_prototype_memory.py`、`docs/identity_prototype_memory_usage_zh.md`。没有修改真实划分、旧训练入口，也没有创建 Git commit 或自动任务。

## 验证结果

- `docs/identity_prototype_memory_tests.txt`：22 项测试通过。
- `docs/identity_prototype_memory_regression.txt`：前 3 步 86 项回归通过。
- `docs/identity_prototype_memory_smoke.json` / `.txt`：真实本地 CLIP、CUDA、外层 autocast 两阶段合成图像验证通过；8 个累计身份、16 张累计图像、512 维、向量载荷 16384 字节。
- 历史向量逐元素保持，参考权重保持，独立均值核对最大差 `1.1920928955078125e-07`；每阶段保存恢复，删除旧图片后继续下一阶段。

仅完成工程正确性验证，未运行正式 ReID 效果实验。当前支持阶段之间恢复；提取途中中断时从上一完整记忆重提当前阶段，不支持逐图游标。文件原子保存和内存提交分开，保存失败时重试保存即可，不应重复提交。
