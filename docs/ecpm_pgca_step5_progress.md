# ECPM / PGCA 第 5 步执行记录

日期：2026-09-13。

## 当前状态与恢复位置

已完成：FINCH 第一层余弦聚类、模式等权类别原型、旧快照与漂移、完整 ECPM 状态保存恢复。后续 PGCA 蒸馏/迁移不在本步实现。

恢复时读取本文、总计划第 5 步和相关测试输出，从首个未完成项继续；开发内容分段落盘。

- [x] A. 核对论文公式、计划和第 4 步接口。
- [x] B. 核对 FINCH 官方实现/版本，确定可复现的第一层调用和边界策略。
- [x] C. 实现模式计算、ECPM 候选/快照/整体提交与状态验证。
- [x] D. 提供运行入口、测试和合成诊断。
- [x] E. 运行针对性测试、回归和真实 CLIP 冒烟，保存结果。
- [x] F. 更新使用文档、总计划、恢复位置。

## 已确定原则

模式为 Norm(mean(identity prototypes))；类别为 Norm(mean(mode prototypes))，模式等权。仅重聚类当前类别的累计身份，缺席类别不动；recurring 漂移为 1-cos(old,new)，范围 [0,2]；新类别漂移为 None。保存阶段开始时所有历史类别的独立快照，供后续 PGCA 使用。

第 4 步已有接口保持可用。FINCH 的层选择、距离、依赖版本、簇大小、耗时和向量内存需可检查。当前不修改真实数据划分和旧训练入口。

## 已保存实现与当前验证

`reid/memory/finch_modes.py`、`reid/memory/ecpm_modes.py`、`tools/update_ecpm_memory.py`、`tests/test_ecpm_modes.py`。第 4 步仅提取提交前验证为共用方法，保留原接口。

使用官方 `finch-clust==0.2.3` 的 `clust_rank/get_clust` 第一层路径，分块精确余弦 1-NN，不计算无用高层，不启用 ANN。随机向量、重复向量、两点输入及不同分块大小已对照官方完整 FINCH 第一层一致。依赖缺失明确失败；单身份显式单簇；均值近零明确失败。

21 项测试通过（`docs/ecpm_modes_tests.txt`），前 4 步 108 项回归通过（`docs/ecpm_modes_regression.txt`）。真实 CLIP CUDA 合成三阶段通过（`docs/ecpm_modes_smoke.json` / `.txt`），身份累计 4→8→12；旧身份、缺席类别统计保持，逐阶段保存恢复且删除旧图片后可继续。对称新增模式使模式数 1→3、中心漂移为 0 的诊断已保存，作为信号局限。

实际环境新增 FINCH、scipy、scikit-learn 等依赖，安装日志 `docs/ecpm_finch_install.txt`；实际版本 numpy 2.3.3、scipy 1.18.1、scikit-learn 1.9.1、FINCH 0.2.3，与仓库部分既有版本约束不同；未改动既有约束，仅新增 FINCH 固定版本。完整状态记录实际依赖版本与官方源文件哈希，恢复严格核对。

## 下一步与恢复注意

本步已完成，没有未收尾代码。后续更新：第 6 步也已完成，见 `docs/ecpm_pgca_step6_progress.md`；下一步从第 7 步陌生类别迁移初始化开始。

训练前 prepare 得到 `old_categories`（所有历史类别）、`category_updates`（当前类别）、`drifts`（新类别 None）。这些是原型快照；第 6 步仍须单独冻结历史 Adapter 教师。暂未把 ECPM 自动接入基础 Trainer，第 8 步才串联完整阶段流程。

阶段内提取/聚类中断后，从上一完整检查点重做当前阶段；没有逐批次游标。内存提交与磁盘保存分开，保存失败后重试 save。CLI 原型升级可重建历史，仅用已保存向量，不读历史图片。

使用说明 `docs/ecpm_modes_usage_zh.md`、总计划均已更新。未运行真实 ReID 正式实验，未修改真实数据划分和旧训练入口，未创建 Git commit 或自动任务。
