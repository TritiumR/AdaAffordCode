CUDA_VISIBLE_DEVICES=3 python test_3d_critic.py \
    --model_version model_AAP \
    --primact_type pushing \
    --category_types Door \
    --val_data_dir ../data/gt_data-train_fixed_cam_new_new_test_data-pushing \
    --val_data_fn data_tuple_list.txt \
    --buffer_max_num 100000 \
    --no_true_false_equal \
    --load_model final_model_conf_Door_StorageFurniture,Switch,Refrigerator_100 \
    --num_interaction_data_offline 48 \
    --batch_size 16 \
    --start_epoch 0 \
    --end_epoch 10

