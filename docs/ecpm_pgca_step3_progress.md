# ECPM / PGCA 第 3 步执行记录

日期：2026-09-13

## 当前状态

已完成：类别内部 CE + Triplet 基础 Trainer、单阶段运行工具和 oracle 诊断。只接入前两步的数据流和类别模型；未添加 ECPM、蒸馏、迁移选源或自动类别路由。

## 中断恢复

恢复时先读取本文及总计划第 3 步，核对实际文件，从第一项未完成内容继续。每段代码和测试结果及时落盘；不依赖后台进程，不自动安排额度恢复后的任务。

## 分段清单

- [x] A. 检查阶段数据、类别模型接口和第 3 步要求。
- [x] B. 实现类别内归一化特征的 batch-hard Triplet 与基础 Trainer，已验证。
- [x] C. 添加单阶段基线运行工具和 oracle 类别诊断评估，已验证。
- [x] D. 添加损失、类别隔离、阶段切换、梯度累积和小样本学习测试，最终 21 项通过。
- [x] E. 运行测试、必要回归及真实 CLIP 短训练验证，保存日志；21 + 24 + 41 = 86 项通过，CUDA + AMP 两阶段短训练通过。
- [x] F. 完成使用文档并更新总计划。

## 实施约定

- 临时分类头只包含当前阶段当前类别身份；原始投影 CLS 做 CE，L2 归一化特征做欧氏距离 batch-hard Triplet。
- 每个训练迭代从每个当前类别取一个 batch，类别内求平均损失，各类别梯度相加后统一更新一次。
- 较小类别只循环当前 loader；只捕获 StopIteration，不吞掉图片错误或退回历史/测试数据。
- 单阶段运行工具用于验证基础训练，不替代第 8 步完整终身流程与训练断点恢复。
- 原有用户改动及 OCIA/OSAF 训练入口保持不动。

## 下一项工作

本步骤已完成，无未收尾代码。后续更新：第 4 步“身份原型记忆”也已完成，使用 `encode_reference` 和独立顺序原型 loader。下一步为第 5 步；具体见 `docs/ecpm_pgca_step4_progress.md` 和总计划。

先读 `docs/category_training_usage_zh.md` 理解现有阶段生命周期。真正的 ECPM/PGCA 整合与完整训练断点恢复仍属后续步骤。

## 验证结果

第一轮：`python -m unittest discover -s tests -p "test_category_training.py" -v`，18 tests / OK，完整输出保存 `docs/category_training_tests.txt`。测试包括梯度累加等价性、阶段切换、损失手算、数据保护、单阶段工具接续和小样本学习（CE 下降，最后一轮分类准确率 >=95%）。

最终新增不同身份数分类头、不等长 loader 平衡与非有限损失原子更新检查后：**21 tests / OK**，运行时 3.333 秒（不含导入）。日志已更新。

模型层回归：24 tests / OK；数据层回归：41 tests / OK。

完整本地 CLIP ViT-B/16 CUDA + AMP 冒烟通过，T1(person/vehicle)、T2(person/panda) 各 2 epochs × 2 iterations，当前类别均更新 4 次；参考输出不变，T2 中 vehicle 输出不变。仅使用临时生成图片，不是真实数据实验。

## 已保存文件

- `reid/loss/category_triplet.py`
- `reid/trainer_category_progressive.py`
- `reid/evaluation/category_oracle.py`
- `tools/train_category_baseline_stage.py`
- `tools/smoke_test_category_training.py`
- `tests/test_category_training.py`
- `docs/category_training_usage_zh.md`
- `docs/category_training_tests.txt`
- `docs/category_training_model_regression.txt`
- `docs/category_training_data_regression.txt`
- `docs/category_training_smoke.json`、`docs/category_training_smoke.txt`
- `docs/category_training_smoke_t1.jsonl`、`docs/category_training_smoke_t2.jsonl`

未运行真实 ReID 正式实验，未改动现有真实数据划分，未修改旧训练入口。单阶段工具支持完成阶段之间接续，阶段内精确 resume 尚未实现。开发文件均已保存，未创建 Git commit 或自动定时任务。
