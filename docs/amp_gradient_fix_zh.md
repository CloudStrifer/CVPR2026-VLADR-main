# AMP 非有限梯度修复与服务器重跑

## 问题与原因

用户使用 `--amp` 后，在 `clip_grad_norm_(..., error_if_nonfinite=True)` 处遇到非有限梯度错误。
代码会先检查损失是否有限，因此这次堆栈说明反向传播后的梯度或其总范数出现 NaN/Inf，并不等于损失已经 NaN。
AMP 的初始损失缩放可能使 FP16 反向传播溢出；旧代码在 GradScaler 有机会降低倍率前终止了训练。
仅凭远程堆栈不能确定所有数值问题都来自缩放，持续溢出仍需检查。

## 已修改的功能

- `reid/trainer_category_progressive.py`：把反向传播与更新分开。AMP 梯度非有限时，由 GradScaler 跳过 AdamW 更新并降低倍率。
- 重试同一批已增强图像，恢复本次前向传播前的 RNG；重新计算所有当前类别的梯度，不重复采样和增强。
- 最多重试 20 次。失败尝试不增加有效更新数、采样游标或 epoch 统计。持续失败仍报错。
- 非 AMP 梯度错误、非有限损失、有限梯度导致的总范数溢出仍停止；不把 `error_if_nonfinite` 改为 False。
- 写入 `amp_overflow_retry` 事件，包含阶段、目标更新步、失败尝试编号及缩放前后倍率。
- 成功后原有断点会保存调整后的 scaler；重试期间中断则按原有日志事务恢复最后已保存的更新。
- 当前 ViT、Adapter 和线性分类头没有训练时更新的 BatchNorm 统计；若将来加入有状态前向层，需要扩展重试的状态回滚。

参考：[PyTorch AMP 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。

## 上传及启动

服务器运行所需更新仅一份文件：把本地 `reid/trainer_category_progressive.py` 覆盖到
`/workspace/GuangjinOuyang/code/reid/trainer_category_progressive.py`。
测试文件和本文可一并保存，但不是运行依赖。

保留原训练命令，包括 `--amp`，只把输出目录换成新的目录，例如：

```bash
--output-dir "$REID_RUN_ROOT/full_seed42_ampfix"
```

代码更新会改变精确恢复的源码指纹；本次应新建实验，不对旧版本断点使用 `--resume`。
不需要再次删除旧目录。修复版本产生的新断点，之后仍可正常续训：

```bash
CUDA_VISIBLE_DEVICES=2 python -u train_category_progressive.py \
  --output-dir "$REID_RUN_ROOT/full_seed42_ampfix" --resume
```

查看溢出重试事件：

```bash
grep 'amp_overflow_retry' "$REID_RUN_ROOT/full_seed42_ampfix"/stage_*.jsonl
```

如果暂时不上传修复，或修复后仍持续溢出，可从原始完整命令中去掉 `--amp`，
并换一个新的输出目录（例如 `full_seed42_fp32`）启动 FP32 对照诊断。
FP32 通常需要更多显存；它是排查方式，不能保证所有数值问题都消失。

前面的 Cython、TF32 和 pynndescent 警告不是这次堆栈的直接终止原因。

## 验证进度

- 已通过基础训练测试 24 项，含 CUDA 单次 Inf 后恢复、持续 NaN 有界失败、非 AMP 仍拒绝非法梯度。
- 同一批次重试结果与直接使用降低后倍率的更新，参数、损失和 CUDA RNG 逐项相等。
- 已通过恢复测试 14 项，其中 CUDA 测试在注入一次溢出后保存断点，恢复的 scaler、参数、优化器、计数、日志和 RNG 与连续训练相等。
- 已通过 PGCA 蒸馏与迁移测试 32 项；本次相关测试合计 70 项通过。记录分别为 `amp_gradient_fix_tests.txt`、`amp_gradient_fix_resume_tests.txt`、`amp_gradient_fix_pgca_tests.txt`。
- 真实 CLIP ViT-B/16 + 本地真实 T1 图像短测通过：person 225 身份、vehicle 173 身份，B=32、K=4、seed=42、AMP，完成 2 次 CE+Triplet 更新。
- 该短测第 2 次更新实际触发一次溢出，倍率由 65536 降到 32768，重试成功。这为远程报错的缩放溢出解释提供了本地复现证据。
- 真实短测环境 PyTorch 2.8.0+cu126、RTX 4060 Ti；使用 baseline trainer 检查 T1 的真实前向/反向路径，没有执行完整 ECPM 准备、后续阶段或检索评估。
- 真实短测结果保存在 `amp_real_t1_smoke.json`，逐步日志在 `amp_real_t1_smoke.jsonl`，终端输出在 `amp_real_t1_smoke.txt`。
- 未在用户 Ubuntu 服务器执行，未进行完整真实数据训练。
