CUDA_VISIBLE_DEVICES=3 xvfb-run -a python iter_train_conf.py \
  --AAP_dir ../logs/model_AAP_conf_world_pulling \
  --AAP_epoch 20-network.pth \
  --AIP_dir ../logs/model_AIP_conf_world_pulling \
  --AIP_epoch 20-network.pth  \
  --primact_type pulling \
  --category_types TrashCan,Refrigerator,WashingMachine,KitchenPot,Box,Microwave,Door,Table,Kettle,Bucket \
  --exp_name final_model_conf_new_100_pulling \
  --num_interaction_data_offline 6 \
  --step 4 \
  --no_gui \
  --hidden_dim 128 \
  --train_data_dir ../data/gt_data-train_fixed_cam_new_new_ruler_data-pulling \
  --visual_shape 101311,103521,100015,100431,8930,7265,12483,100367,10036,47645,22367