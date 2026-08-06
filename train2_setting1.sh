CUDA_VISIBLE_DEVICES=0,1 python train_stage2.py \
--setting 1 \
--data-dir /data/dataset/xukunlun/PRID \
--logs-dir ./_RESULTS \
--stage1-prompts-out-dir ./_STAGE1_PROMPTS_WEIGHT \
--batch-size 64 \
--other-details global_clipreid_setting1 \
--eval-stage 0,1,2,3,4 \
--epochs 60
