# ECPM / PGCA 第 7 步执行记录

日期：2026-09-13。

## 当前状态与恢复位置

已完成：陌生类别的单向模式覆盖匹配、综合相似度、阈值选源和独立 Adapter 初始化。第 8 步连续阶段调度/精确恢复和第 9 步路由评估不在本步实现。

- [x] A. 核对论文、计划、历史快照及 Trainer 注册时序。
- [x] B. 实现相似度/配置、阶段开始历史 Adapter 快照、选源与复制接口。
- [x] C. 接入 Trainer 和单阶段入口，记录全部候选分数、阈值及初始化来源。
- [x] D. 添加方向/阈值/顺序/存储独立性/训练与恢复测试。
- [x] E. 运行前 6 步回归及真实 CLIP 验证并保存结果。
- [x] F. 更新使用说明、总计划和恢复位置。

## 已确定原则

S_mode(c,j)=mean_k max_l cos(g_new[k],g_old[l])，方向为新类别到历史类别；score=alpha*cos(m_new,m_old)+(1-alpha)*S_mode。仅从 seen_before 中选源，允许缺席历史类别，不允许当前新类别互借。分数 >= delta 才复制全部 Adapter，否则默认零残差；空历史直接默认。并列按类别键排序。

所有来源参数在阶段开始时复制保存，不使用已更新的当前阶段源参数。只初始化新类别，不重置 recurring；新类别仍不增加源教师一致性。保留基础与第 6 步默认行为，新增显式迁移配置。开发内容分段落盘，中断后从首个未完成项继续。

## 当前实现与验证

`reid/adaptation/pgca.py` 新增配置、单向模式相似度、稳定选源与阶段初始化器；Trainer 在教师冻结后、优化器构造前应用。注册/新分类头维持原基线按类别排序的交替顺序，避免空历史回退导致随机初始化次序变化。

新入口 `tools/train_pgca_stage.py` 默认 drift＋similarity，复用第 6 步单阶段实现；原第 6 步工具仍默认 default 初始化。完整检查点 kind 为 pgca_stage，可接续第 6 步或 baseline＋ECPM。记录全部候选分数、最佳源、阈值回退原因、源/目标哈希和 CPU 快照内存。

最终 14 项测试通过（`docs/pgca_transfer_tests.txt`），包括方向、阈值等号、负分/并列、外层 CPU autocast 不改变 FP32 阈值判断、空历史、旧中心/缺席源、源被后续修改时仍复制旧快照、独立存储、多个新类别及反转输入顺序、无源蒸馏与 CLI 接续。

前 6 步 147 项回归通过（`docs/pgca_transfer_regression.txt`）。真实 CLIP CUDA＋AMP 三阶段通过（`docs/pgca_transfer_smoke.json` / `.txt` 及 `_t1.jsonl`、`_t2.jsonl`、`_t3.jsonl`）。T2 panda 对两个历史来源分数并列约 0.994191，按规则选择 T1 person；源/目标初始化哈希相同、存储不同，接受迁移 1 次，后续训练和恢复通过。T2 临时 CPU 源副本载荷 3172352 字节，T3 无新类别不分配源副本。

## 下一步与恢复注意

本步没有未完成实现。下一次用户要求继续时，从第 8 步完整阶段流程、日志及精确断点恢复开始。先读 `docs/pgca_transfer_usage_zh.md`、`tools/train_pgca_stage.py`（包装入口）、`tools/train_pgca_recurring_stage.py`（共用单阶段实现）和总计划第 8 步。

第 8 步需保持同一阶段开始的来源快照/旧教师/ECPM 候选，恢复未完成阶段不能重新迁移已在训练的新 Adapter。当前组合 checkpoint 仅在阶段成功后保存；没有优化器/RNG/采样器/教师的 epoch 内完整状态。失败时从上一完整阶段在新输出目录重跑。

使用说明、总计划、上一执行记录均已更新。阈值不能保证消除负迁移，尚未运行正式 ReID 效果实验；未修改真实划分与旧训练入口，未创建 Git commit、自动任务或子代理。
