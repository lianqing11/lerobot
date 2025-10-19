HF_USER=lianqing11
rm -rf outputs/train/act_so101_test_1019
lerobot-train \
  --dataset.repo_id=${HF_USER}/record-train_1019v2_middle \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test_1019 \
  --job_name=act_so101_test_1019 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_so101_test_1019 