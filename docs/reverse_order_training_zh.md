# ECPM / PGCA 完整逆序实验

新增配置：`config/category_progressive_real/main_reverse.json`。

该配置将 `main.json` 的完整阶段对象逆序排列，保持各阶段身份划分、训练清单、验证集、测试集及数据根目录不变。保留原始 `stage_id`，执行顺序由 JSON 的 `stages` 列表位置决定。

| 实际时间位置 | 原阶段编号 | 当前训练类别 | 首次出现类别 |
|---|---|---|---|
| 第 1 阶段 | t4 | tiger、boat、vehicle | tiger、boat、vehicle |
| 第 2 阶段 | t3 | vehicle、panda、person | panda、person |
| 第 3 阶段 | t2 | person、panda、tiger | 无 |
| 第 4 阶段 | t1 | person、vehicle | 无 |

日志中的 `stage_id` 因此依次为 `t4, t3, t2, t1`；`stage_index` 依次为 `0, 1, 2, 3`。最后完成的是原始编号 `t1`，它对应逆序实验的第四个时间阶段。整理论文时应按实际时间位置标注，并保留原数据块映射。

## Ubuntu 完整训练命令

先把 `main_reverse.json` 上传到服务器对应的 `config/category_progressive_real/` 目录，和原来的 `main.json` 放在一起，继续使用该目录已有的 `manifests/`。以下命令与原完整训练保持相同参数，使用独立输出目录 `full_reverse_seed42`。

```bash
export REID_PROJECT="/home/haichao/ouyang/CVPR2026-VLADR-main"
export REID_STREAM="$REID_PROJECT/config/category_progressive_real/main_reverse.json"
export REID_CLIP="$REID_PROJECT/data/pretrained/ViT-B-16.pt"
export REID_RUN_ROOT="$REID_PROJECT/experiments/ecpm_pgca"

cd "$REID_PROJECT"

python -u train_category_progressive.py \
  --stream-config "$REID_STREAM" \
  --reference-checkpoint "$REID_CLIP" \
  --output-dir "$REID_RUN_ROOT/full_reverse_seed42" \
  --device cuda --amp \
  --epochs 10 --iterations-per-epoch 100 \
  --batch-size 32 --num-instances 4 \
  --prototype-batch-size 128 --eval-batch-size 128 \
  --workers 0 --seed 42 \
  --adapter-lr 0.0003 --head-lr 0.0003 --weight-decay 0.0001 \
  --lambda-tri 1.0 --triplet-margin 0.3 \
  --consistency drift --lambda-con 1.0 --gamma 1.0 \
  --init-mode similarity --alpha 0.5 --delta 0.5 \
  --control-summary ecpm --routing-summary ecpm --beta 0.5 \
  --evaluate --eval-split test --eval-gallery both \
  --checkpoint-every 1
```

这是从头开始的新实验，不使用原顺序实验的断点。输出目录应不存在或为空；训练程序会创建目录。

如需先检查服务器上的图片路径，在训练前运行：

```bash
python tools/validate_category_stream.py \
  --stream-config "$REID_STREAM" \
  --check-images
```

元数据审计应显示 4 个阶段、1,392 个训练身份、55,853 张训练图片、10 个评估集。`--check-images` 只检查文件存在性，不解码图片。

## 中断后恢复

```bash
cd /home/haichao/ouyang/CVPR2026-VLADR-main
python -u train_category_progressive.py \
  --output-dir /home/haichao/ouyang/CVPR2026-VLADR-main/experiments/ecpm_pgca/full_reverse_seed42 \
  --resume
```

恢复时使用逆序实验自身的输出目录，保持同一份代码、数据配置和运行环境。

## 结果读取

查看 `full_reverse_seed42/stage_results.md`、`stage_results.json` 和 `evaluation_summary.json`，按实际时间顺序读取最后一个阶段的五类指标。原顺序结果仍位于 `full_seed42`。

跨顺序比较时固定 `prototype` 路由及同一 gallery 协议，优先比较最终五类宏平均 mAP、Rank1 和逐类性能。原顺序末阶段 Boat 刚出现，其旧类别平均遗忘仅包含四类；逆序末阶段五类均有历史评估。比较平均遗忘时说明类别集合，或另外报告共同四类（person、vehicle、panda、tiger）的平均值。
