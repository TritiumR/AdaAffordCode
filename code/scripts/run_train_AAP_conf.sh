CUDA_VISIBLE_DEVICES=1 python train_AAP_conf.py \
  --exp_suffix model_AAP_conf_world_pulling_StorageFurniture,Switch,Refrigerator \
  --primact_type pulling \
  --category_types StorageFurniture,Switch,Refrigerator \
  --num_interaction_data_offline 96 \
  --batch_size 32 \
  --epochs 15 \
  --train_data_dir ../data/gt_data-train_fixed_cam_train_data-pulling \
  --val_data_dir ../data/gt_data-train_fixed_cam_test_data-pulling \
  --overwrite
