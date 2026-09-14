# ECPM / PGCA 第 6 步执行记录

日期：2026-09-13。

## 当前状态与恢复位置

已完成：PGCA 熟悉类别的阶段固定漂移权重、冻结历史教师、当前图像特征一致性及 Trainer 三种模式。第 7 步新类别迁移及第 8 步完整阶段调度/精确断点暂不实现。

- [x] A. 核对论文/计划、Trainer、Adapter 和 ECPM 接口。
- [x] B. 实现一致性损失、权重配置和独立冻结教师。
- [x] C. 接入 Trainer，记录漂移/原始损失/权重/加权损失，保留 off 基线。
- [x] D. 添加可运行示例与有意义的公式/状态/集成测试。
- [x] E. 运行回归与真实 CLIP 短训练并保存结果。
- [x] F. 更新使用文档、总计划和恢复位置。

## 已确定原则

只对 recurring 类别蒸馏，教师为阶段开始前同类别 Adapter。教师独立于学生，不在活跃计算图上覆盖参数；同一增强后的当前图像张量同时给师生。Lcon=mean(1-cos(student,stopgrad(teacher)))；lambda=lambda_con*exp(-gamma*drift)，每阶段固定。off / fixed / drift 三种模式，零权重退化为基础训练，不计算多余教师前向。新类别、缺席类别不加该损失。

开发内容分段保存。恢复时读取本文及总计划第 6 步，从首个未完成项继续。

## 当前落盘内容与验证

新增 `reid/loss/pgca.py`、`reid/adaptation/pgca.py`、`tools/train_pgca_recurring_stage.py`、`tests/test_pgca_recurring.py`；Trainer 增加可选一致性上下文，ECPM 增加不提交的候选验证方法。新单阶段 CLI 在成功后把模型和对应 ECPM 放入同一个原子检查点，仍不支持 epoch 内恢复；新类别只用基础损失。

18 项测试通过（`docs/pgca_recurring_tests.txt`）：公式/停止梯度、独立教师/同类别路由/相同输入、学生一致性梯度、零权重与默认基础 Trainer 参数逐元素一致、gamma=0 与 fixed 一致、旧记忆/缺席 Adapter 保持、错误候选和教师损坏保护、错误教师特征在当前优化步骤前终止、CLI 两阶段接续和 baseline＋ECPM 导入。

初次测试发现自定义模型 eval 未递归设置空 ModuleDict 状态；教师初始化已改为显式递归 eval，全部测试通过。

前 5 步 129 项回归通过（`docs/pgca_recurring_regression.txt`）。真实 CLIP CUDA＋AMP 三阶段、每阶段 2 epoch×2 updates 短训练通过（`docs/pgca_recurring_smoke.json` / `.txt` 及 `_t1.jsonl`、`_t2.jsonl`、`_t3.jsonl`）。教师为各自阶段开始版本、参考/教师特征保持、缺席 Adapter 保持、当前 Adapter 更新；每阶段保存恢复后删除旧图片并继续。T2 教师张量载荷 347942912 字节，T3 为 349529088 字节，报告显式记录。

## 下一步与中断恢复

本步没有未完成实现。后续更新：第 7 步也已完成，见 `docs/ecpm_pgca_step7_progress.md`；下一步从第 8 步完整阶段流程与精确恢复开始。

第 6 步报告本身仅验证熟悉分支；后续第 7 步新增显式 similarity 初始化和默认双分支入口，源比较/参数复制使用同一阶段开始历史状态，陌生类别不加源教师一致性。原第 6 步工具仍默认 default 初始化。

开发内容、运行说明和总计划均已保存。运行中断时加载前一完整 `completed_stage.pt`，新输出目录重跑当前阶段；现有日志只用于审计，不是 epoch 内恢复状态。完整调度与精确恢复待第 8 步。

未运行正式 ReID 效果实验，未修改真实划分及旧训练入口；原基础 Trainer 默认 off。未创建 Git commit、自动任务或子代理。
