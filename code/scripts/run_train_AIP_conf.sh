CUDA_VISIBLE_DEVICES=3 python train_AIP_conf.py \
  --exp_suffix model_AIP_conf \
  --category_types StorageFurniture,Window,Faucet \
  --train_data_dir ../data/gt_data-train_fixed_cam_train_data-pushing \
  --val_data_dir ../data/gt_data-train_fixed_cam_test_data-pushing \
  --num_interaction_data_offline 128 \
  --epochs 80 \
  --AAP_dir ../logs/model_AAP_conf \
  --AAP_epoch 20 \
  --overwrite