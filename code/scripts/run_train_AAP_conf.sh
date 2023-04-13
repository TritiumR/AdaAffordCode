CUDA_VISIBLE_DEVICES=1 python train_AAP_conf.py \
  --exp_suffix model_AAP_conf_test_1 \
  --category_types StorageFurniture,Window,Faucet \
  --num_interaction_data_offline 3 \
  --epochs 80 \
  --train_data_dir ../data/gt_data-train_fixed_cam_train_data-pushing \
  --val_data_dir ../data/gt_data-train_fixed_cam_test_data-pushing \
  --visual_shape 46180,40147,1028,1011,100982,102798 \
  --overwrite
