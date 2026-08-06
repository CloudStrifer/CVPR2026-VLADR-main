# order 1
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
--setting 1 \
--testing ./_PRETRAINED/order1 \
--data-dir /data/dataset/xukunlun/PRID \
--stage1-prompts-out-dir ./_STAGE1_PROMPTS_WEIGHT \
--logs-dir ./_RESULTS \
--batch-size 64 \
--other-details global_clipreid_eval_order1

# order 2
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
--setting 2 \
--testing ./_PRETRAINED/order2 \
--data-dir /data/dataset/xukunlun/PRID \
--stage1-prompts-out-dir ./_STAGE1_PROMPTS_WEIGHT \
--logs-dir ./_RESULTS \
--batch-size 64 \
--other-details global_clipreid_eval_order2
