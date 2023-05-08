#categories="Switch Bucket Table Door TrashCan Refrigerator WashingMachine Microwave"
categories="Door TrashCan WashingMachine Faucet"
for category in $categories
do
  CUDA_VISIBLE_DEVICES=1 xvfb-run -a python iter_train_conf.py \
    --AAP_dir ../logs/model_AAP_conf_world_pushing_StorageFurniture,Switch,Refrigerator \
    --AAP_epoch 19-network.pth \
    --AIP_dir ../logs/model_AIP_conf_StorageFurniture,Switch,Refrigerator_pushing \
    --AIP_epoch 13-network.pth  \
    --primact_type pushing \
    --category_types "$category" \
    --exp_name final_model_conf_"$category"_StorageFurniture,Switch,Refrigerator_100 \
    --save_interval 10 \
    --epoch 10 \
    --num_interaction_data_offline 6 \
    --step 4 \
    --no_gui \
    --hidden_dim 128 \
    --train_data_dir ../data/gt_data-train_fixed_cam_new_new_ruler_data-pushing \
    --visual_shape 101311
done