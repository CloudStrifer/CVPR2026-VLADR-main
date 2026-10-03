# 类别渐进式 LwF 基线

入口：`train_category_progressive_lwf.py`。这是独立实验，不会修改 `train_category_progressive.py`、旧版 `train_stage2_lwf.py` 或 ECPM/PGCA 的训练结果。它读取同一份四阶段 JSON 流，沿用每类别一个 P×K 批次、每阶段相同优化步数、交叉熵与 Triplet 损失，以及相同的 query/gallery 检索协议。

## 方法对应关系

旧版 `train_stage2_lwf.py` 按数据域依次扩展全局身份分类器，不能直接处理一个阶段内多类别、类别再次出现且身份不重复的当前数据流。本入口改为**按类别保留历史阶段的身份分类头**。进入新阶段时冻结阶段开始前的教师模型和旧分类头；对于本阶段再次出现的类别，仅在该类别本阶段的新训练图像上，分别计算教师与学生的旧身份 logits，并对每个历史头计算温度为 $T$ 的 KL 蒸馏，最后取平均。学生侧的旧分类头和该类别 Adapter 一同更新，符合 LwF 联合优化旧任务输出的原则；教师侧始终冻结。教师不使用历史图像。首次出现的类别没有旧身份头，只计算本阶段交叉熵与 Triplet 损失。

每类别损失为 `CE + lambda_tri × Triplet + lwf_weight × mean_old_heads(KL_T)`。默认 `lwf_weight=1.0`，`lwf_temperature=2.0`，与旧版 LwF 入口的默认值一致。历史分类头仅用于蒸馏，不参与最终 ReID 特征提取或检索。

由于当前主干冻结、类别 Adapter 相互独立，旧类别缺席阶段不会被更新，也不会产生该类别的 LwF 项。LwF 项作用于**再次出现的类别**：它约束当前 Adapter 和旧身份分类头更新时对该类别旧身份的响应。这是原始分类 LwF 在本项目架构下的适配，不能称为作者原始网络结构的逐项复现。为了保持与主实验相同的训练步数，本入口不额外加入原论文针对新分类头的 warm-up 阶段。

ECPM 聚类、原型演化和 PGCA 两个训练分支均关闭。评估仍需要在未知输入类别标签时选择 Adapter，因此仅用冻结参考编码器提取的简单类别中心做路由；该中心不参与 LwF 训练损失。`--eval-gallery both` 与主实验一致，生成 `per_dataset` 和 `mixed` 两套指标，其中论文中无类别标签的混合检索应读取 `mixed / prototype`。输出目录必须与主实验分开。

## Ubuntu 完整训练

```bash
export REID_PROJECT="/home/haichao/ouyang/CVPR2026-VLADR-main"
export REID_STREAM="$REID_PROJECT/config/category_progressive_real/main.json"
export REID_CLIP="$REID_PROJECT/data/pretrained/ViT-B-16.pt"
export REID_LWF_ROOT="$REID_PROJECT/experiments/lwf"
cd "$REID_PROJECT"

python -u train_category_progressive_lwf.py \
  --stream-config "$REID_STREAM" \
  --reference-checkpoint "$REID_CLIP" \
  --output-dir "$REID_LWF_ROOT/full_seed42" \
  --device cuda --amp \
  --epochs 10 --iterations-per-epoch 100 \
  --batch-size 32 --num-instances 4 \
  --prototype-batch-size 128 --eval-batch-size 128 \
  --workers 0 --seed 42 \
  --adapter-lr 0.0003 --head-lr 0.0003 --weight-decay 0.0001 \
  --lambda-tri 1.0 --triplet-margin 0.3 \
  --lwf-weight 1.0 --lwf-temperature 2.0 \
  --routing-summary ecpm --beta 0.5 \
  --evaluate --eval-split test --eval-gallery both \
  --checkpoint-every 1
```

断点恢复：

```bash
python -u train_category_progressive_lwf.py \
  --output-dir "$REID_LWF_ROOT/full_seed42" \
  --resume --checkpoint-every 1
```

训练输出包含 `latest.pt`、`evaluation_summary.json`、逐阶段 JSONL 日志及与主实验同格式的 stage results。`latest.pt` 内额外保存历史分类头和阶段开始时冻结的教师，因此只应使用本入口恢复 LwF 实验。结果与 ECPM/PGCA 主实验比较时，应保持数据流、训练步数、种子和评估设置一致。
