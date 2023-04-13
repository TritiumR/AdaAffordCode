CUDA_VISIBLE_DEVICES=3 python train_AIP_conf.py \
  --exp_suffix model_AIP_conf_world \
  --category_types StorageFurniture,Window,Faucet \
  --train_data_dir ../data/gt_data-train_fixed_cam_train_data-pushing \
  --val_data_dir ../data/gt_data-train_fixed_cam_test_data-pushing \
  --num_interaction_data_offline 128 \
  --epochs 40 \
  --AAP_dir ../logs/model_AAP_conf_world \
  --AAP_epoch 20 \
