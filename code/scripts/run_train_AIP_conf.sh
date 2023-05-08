CUDA_VISIBLE_DEVICES=0 python train_AIP_conf.py \
  --exp_suffix model_AIP_conf_StorageFurniture,Switch,Refrigerator_pulling \
  --primact_type pulling \
  --category_types StorageFurniture,Switch,Refrigerator \
  --train_data_dir ../data/gt_data-train_fixed_cam_train_data-pulling \
  --val_data_dir ../data/gt_data-train_fixed_cam_test_data-pulling \
  --num_interaction_data_offline 128 \
  --epochs 15 \
  --AAP_dir ../logs/model_AAP_conf_world_pulling_StorageFurniture,Switch,Refrigerator \
  --AAP_epoch 14 \
