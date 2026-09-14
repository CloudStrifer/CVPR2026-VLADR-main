# ECPM / PGCA 第 9 步执行记录

日期：2026-09-14。已完成：硬路由、终身检索评估与可执行消融。

- [x] A. 核对论文实施计划、现有 oracle 检索协议和第 8 步恢复接口。
- [x] B. 纯图像硬路由、分组 Adapter 前向、ECPM/身份均值路由对照。
- [x] C. 原协议与混合 gallery 检索、路由诊断、性能/遗忘矩阵和资源统计。
- [x] D. 阶段提交后评估、评估事务恢复、独立检查点评估入口。
- [x] E. 固定蒸馏、初始化来源与 ECPM 控制/路由分离的可执行消融；验证集机制诊断。
- [x] F. 公式/协议/恢复检查、前序回归、真实 CLIP 三阶段评估短实验。
- [x] G. 使用说明、总计划和完成记录。

保持：训练身份不跨阶段重复；正式路由函数只接收图像/参考向量，不接收真实类别或路径；候选为所有已见类别。oracle 仅诊断。主协议保持各评估集 gallery，混合 gallery 单独标记。缺少已见类别评估集时不得默默报告完整宏平均。

评估不改变训练 RNG 或模型训练策略。阶段提交与评估分成可恢复事务，评估失败从该已提交阶段重评，不重训、不重复提交身份原型。正式数据结果未提供，验收使用生成图像和小模型/真实 CLIP 短实验，不宣称实际精度或创新成立。

开发分段落盘，中断时从首个未完成项继续。

## 开发检查点 1

已实现纯图像路由、两种 gallery 协议、oracle/prototype 对照、性能与遗忘矩阵、阶段评估事务、独立检查点评估、9 个消融配方及同阶段验证集来源/权重枚举工具。正在运行专项测试，尚未做最终回归和真实 CLIP 评估验收。

## 最终验收

- `docs/step9_regression_tests.txt`：190 项检查全部通过（72.023 秒），包含 15 项本步测试。`docs/step9_evaluation_tests.txt` 是增加混合 gallery 手算检查之前的 14 项初轮通过记录。
- 9 个消融配方均在小模型三阶段流执行并恢复通过；来源诊断和权重诊断从同一阶段边界起步，使用 validation，执行和结果续接通过。
- `docs/progressive_evaluation_smoke.json`：真实 CLIP ViT-B/16、CUDA＋AMP、随机增强，三阶段共 6 次更新。基准连续运行到完成；对照在 T1 边界及 T2 内部中断，用独立 Python 进程恢复。最终 Adapter、RNG 和训练日志逐位一致，路由/检索报告一致；重载后的路由分数、预测与描述符逐位一致。
- `docs/progressive_evaluation_smoke_metrics.json`、`docs/progressive_evaluation_smoke.jsonl`：保留三阶段的 oracle/prototype、原 gallery/混合 gallery 指标、路由诊断、性能/遗忘矩阵与资源记录。生成图像不用于证明真实 ReID 效果。
- `docs/ablation_example/ablation_plan.json`：由示例基础配置生成的 9 实验计划；没有对占位图像启动正式训练。

核心实现：

1. `reid/evaluation/prototype_router.py`：纯张量路由和分组 Adapter 前向。
2. `reid/evaluation/category_progressive.py`：协议评估、路由诊断、终身矩阵和遗忘。
3. `reid/memory/prototype_views.py`：身份均值路由及控制视图，不修改 ECPM。
4. `train_category_progressive.py`：可选阶段评估、pending_evaluation 事务和报告恢复。
5. `reid/adaptation/pgca.py`、`reid/loss/pgca.py`、`reid/trainer_category_resumable.py`：随机/指定历史源、控制摘要选择和对应恢复校验。
6. `tools/evaluate_category_progressive.py`：完成阶段检查点独立评估。
7. `tools/run_category_ablations.py`：计划、执行及恢复 9 个消融。
8. `tools/diagnose_pgca_validation.py`：共同阶段前状态下的验证集来源/权重对照，每个完成变体落盘。
9. `tests/test_progressive_evaluation.py`、`tools/smoke_test_progressive_evaluation.py`：验收。

## 下次继续的位置

九步基础实现已经完成，本步没有未完成代码或验收。后续应根据用户要求准备真实数据流、完善各类别 validation/test manifests，选择验证集超参数和多个随机种子，然后执行正式实验。不要把生成图像的短实验数值作为论文结果，不要用 test 选权重/来源/阈值。

入口说明为 `docs/progressive_evaluation_usage_zh.md`，基础配置样例为 `config/category_ablation_example.json`。真实训练需显式 `--evaluate` 才会生成逐阶段评估。代码指纹已因本步实现变化；旧版未完成训练不能跳过第 8 步的代码校验宣称精确续训，旧版已完成阶段可用独立评估工具读取。

工程限制已写入使用说明：单设备 workers=0；无未知类别拒识；身份均值对照仍保留 FINCH 计算/存储；资源不声称 CPU/native 聚类峰值；诊断按变体边界保存而非变体内部每步保存。
