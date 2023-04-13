CUDA_VISIBLE_DEVICES=2 python test_3d_critic.py \
    --model_version model_AAP \
    --primact_type pushing \
    --category_types TrashCan,Refrigerator,WashingMachine,KitchenPot,Box,Microwave,Door,Table,Kettle,Bucket \
    --val_data_dir ../data/gt_data-train_fixed_cam_new_new_test_data-pushing \
    --val_data_fn data_tuple_list.txt \
    --buffer_max_num 100000 \
    --no_true_false_equal \
    --load_model model_AAP_conf_world \
    --num_interaction_data_offline 32 \
    --batch_size 32 \
    --start_epoch 20 \
    --end_epoch 20

