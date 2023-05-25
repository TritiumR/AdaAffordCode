#categories="Switch Bucket Table Door TrashCan Refrigerator WashingMachine Microwave Window"
categories="Faucet"
for category in $categories
do
  CUDA_VISIBLE_DEVICES=3 xvfb-run -a python iter_train_conf.py \
    --AAP_dir ../logs/model_AAP_conf_world_pulling_Table,Switch,Window \
    --AAP_epoch 12-network.pth \
    --AIP_dir ../logs/model_AIP_conf_Table,Switch,Window_pulling \
    --AIP_epoch 9-network.pth  \
    --primact_type pulling \
    --category_types Refrigerator \
    --exp_name final_model_conf_Bucket_Table,Switch,Window_100_pulling \
    --save_interval 10 \
    --epoch 10 \
    --num_interaction_data_offline 6 \
    --step 4 \
    --no_gui \
    --hidden_dim 128 \
    --train_data_dir ../data/gt_data-train_fixed_cam_new_new_ruler_data-pulling \
    --visual_shape 101311
done