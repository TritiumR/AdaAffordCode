"""
    SAPIENVisionDataset
        Joint data loader for six primacts
        for panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
"""

import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from progressbar import ProgressBar
from pyquaternion import Quaternion
from camera import Camera


class SAPIENVisionDataset(data.Dataset):

    def __init__(self, primact_types, category_types, data_features, buffer_max_num, \
            abs_thres=0.01, rel_thres=0.5, dp_thres=0.5, img_size=224, \
            only_true_data=False, world_coordinate=False, batch_zise=32):
        self.primact_types = primact_types
        self.category_types = category_types

        self.buffer_max_num = buffer_max_num
        self.img_size = img_size
        self.abs_thres = abs_thres
        self.rel_thres = rel_thres
        self.dp_thres = dp_thres
        self.only_true_data = only_true_data
        self.world_coordinate = world_coordinate
        self.batch_size = batch_zise

        # data buffer
        self.data_buffer = []  # (gripper_direction_world, gripper_action_dis, gt_motion)
        self.seq = []

        # data features
        self.data_features = data_features
        
    def load_data(self, data_list):
        bar = ProgressBar()
        for i in bar(range(len(data_list))):
            cur_dir = data_list[i]
            cur_shape_id, cur_category, cur_cnt_id, cur_primact_type, cur_trial_id = cur_dir.split('/')[-1].split('_')
            # print(cur_category)

            if cur_primact_type not in self.primact_types:
                continue

            if cur_category not in self.category_types:
                continue

            with open(os.path.join(cur_dir, 'result.json'), 'r') as fin:
                result_data = json.load(fin)

                camera_mat44 = np.array(result_data['camera_metadata']['mat44'], dtype=np.float32)
                state = result_data['object_state']
                camera_direction = result_data['direction_camera']

                # print('camera_mat44: ', camera_mat44)

                if self.world_coordinate:
                    gripper_direction = np.array(result_data['gripper_direction_world'], dtype=np.float32)
                    gripper_forward_direction = np.array(result_data['gripper_forward_direction_world'], dtype=np.float32)
                else:
                    gripper_direction = np.array(result_data['gripper_direction_camera'], dtype=np.float32)
                    gripper_forward_direction = np.array(result_data['gripper_forward_direction_camera'], dtype=np.float32)

                ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)
                pixel_ids = np.round(np.array(result_data['pixel_locs'], dtype=np.float32) / 448 * self.img_size).astype(np.int32)
                ctpt = result_data['position_world']
                target_object_part_actor_id = ['target_object_part_actor_id']
                ctpt_2d = result_data['pixel_locs']
                joint_angles = result_data['joint_angles']

                success = self.check_success(result_data, cur_primact_type)

                start_pos = result_data['start_target_part_qpos']
                if result_data['result'] != 'VALID':
                    end_pos = start_pos
                else:
                    end_pos = result_data['final_target_part_qpos']

                # load original data
                if success:
                    cur_data = (cur_dir, cur_shape_id, state, cur_category, cur_cnt_id, cur_trial_id, ctpt, ctpt_2d, camera_direction, camera_mat44,
                            ori_pixel_ids, pixel_ids, gripper_direction, gripper_forward_direction, start_pos, end_pos, target_object_part_actor_id, joint_angles, True, True)
                else:
                    if not self.only_true_data:
                        cur_data = (cur_dir, cur_shape_id, state, cur_category, cur_cnt_id, cur_trial_id, ctpt, ctpt_2d, camera_direction, camera_mat44,
                                ori_pixel_ids, pixel_ids, gripper_direction, gripper_forward_direction, start_pos, end_pos, target_object_part_actor_id, joint_angles, True, False)

                self.data_buffer.append(cur_data)

    def check_success(self, result_data, primact_type):
        if result_data['result'] != 'VALID':
            return False
        
        abs_motion = abs(result_data['final_target_part_qpos'] - result_data['start_target_part_qpos'])
        j = result_data['target_object_part_joint_id']
        tot_motion = result_data['joint_angles_upper'][j] - result_data['joint_angles_lower'][j] + 1e-8
        success = (abs_motion > self.abs_thres) or (abs_motion / tot_motion > self.rel_thres)
        if not success:
            return False

        if primact_type == 'pushing':
            pass

        elif primact_type == 'pulling':
            mov_dir = np.array(result_data['touch_position_world_xyz_end'], dtype=np.float32) - \
                    np.array(result_data['touch_position_world_xyz_start'], dtype=np.float32)
            mov_dir /= np.linalg.norm(mov_dir)
            intended_dir = -np.array(result_data['gripper_direction_world'], dtype=np.float32)
            success = (intended_dir @ mov_dir > self.dp_thres)
       
        elif primact_type == 'pushing-left' or primact_type == 'pulling-left':
            mov_dir = np.array(result_data['touch_position_world_xyz_end'], dtype=np.float32) - \
                    np.array(result_data['touch_position_world_xyz_start'], dtype=np.float32)
            mov_dir /= np.linalg.norm(mov_dir)
            intended_dir = np.array(result_data['gripper_forward_direction_world'], dtype=np.float32)
            success = (intended_dir @ mov_dir > self.dp_thres)

        elif primact_type == 'pushing-up' or primact_type == 'pulling-up':
            mov_dir = np.array(result_data['touch_position_world_xyz_end'], dtype=np.float32) - \
                    np.array(result_data['touch_position_world_xyz_start'], dtype=np.float32)
            mov_dir /= np.linalg.norm(mov_dir)
            up_dir = np.array(result_data['gripper_direction_world'], dtype=np.float32)
            forward_dir = np.array(result_data['gripper_forward_direction_world'], dtype=np.float32)
            intended_dir = np.cross(up_dir, forward_dir)
            success = (intended_dir @ mov_dir > self.dp_thres)

        else:
            raise ValueError('ERROR: primact_type %s not supported in check_success!' % primact_type)
        
        return success

    def get_seq(self):
        length = len(self.data_buffer) // self.batch_size
        self.seq = np.arange(length)
        np.random.shuffle(self.seq)
        
    def __str__(self):
        strout = '[SAPIENVisionDataset %d] primact_types: %s, abs_thres: %f, rel_thres: %f, dp_thres: %f, img_size: %d\n' % \
                (len(self), ','.join(self.primact_types), self.abs_thres, self.rel_thres, self.dp_thres, self.img_size)
        for primact_type in self.primact_types:
            strout += '\t<%s> len %d\n' % (primact_type, len(self.data_buffer))
        return strout

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, index):
        ind = index // self.batch_size
        index = self.seq[ind] * self.batch_size + (index % self.batch_size)
        cur_dir, shape_id, state, category, cnt_id, trial_id, ctpt, ctpt_2d, camera_direction, camera_mat44, ori_pixel_ids, pixel_ids, \
            gripper_direction, gripper_forward_direction, start_pos, end_pos, target_object_part_actor_id, joint_angles, is_original, result = self.data_buffer[index]

        # grids
        grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
        grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448

        data_feats = ()
        out2 = None
        out3 = None
        for feat in self.data_features:
            if feat == 'img':
                with Image.open(os.path.join(cur_dir, 'rgb.png')) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'pcs':
                x, y = ori_pixel_ids[0], ori_pixel_ids[1]
                with h5py.File(os.path.join(cur_dir, 'cam_XYZA.h5'), 'r') as fin:
                    cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                    cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                    cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
                with Image.open(os.path.join(cur_dir, 'interaction_mask.png')) as fimg:
                    out3 = (np.array(fimg, dtype=np.float32) > 127)
                pt = out[x, y, :3]
                ptid = np.array([x, y], dtype=np.int32)
                mask = (out[:, :, 3] > 0.5)
                mask[x, y] = False
                pc = out[mask, :3]
                pcids = grid_xy[:, mask].T
                out3 = out3[mask]
                idx = np.arange(pc.shape[0])
                np.random.shuffle(idx)
                while len(idx) < 30000:
                    idx = np.concatenate([idx, idx])
                idx = idx[:30000-1]
                pc = pc[idx, :]
                pcids = pcids[idx, :]
                out3 = out3[idx]
                pc = np.vstack([pt, pc])
                pcids = np.vstack([ptid, pcids])
                out3 = np.append(True, out3)
                # normalize to zero-centered
                pc[:, 0] -= 5

                if self.world_coordinate:
                    pc = (camera_mat44[:3, :3] @ pc.T).T

                out = torch.from_numpy(pc).unsqueeze(0)
                out2 = torch.from_numpy(pcids).unsqueeze(0)
                out3 = torch.from_numpy(out3).unsqueeze(0).float()
                data_feats = data_feats + (out,)

            elif feat == 'pc_pxids':
                data_feats = data_feats + (out2,)
             
            elif feat == 'pc_movables':
                data_feats = data_feats + (out3,)
             
            elif feat == 'interaction_mask_small':
                with Image.open(os.path.join(cur_dir, 'interaction_mask.png')) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = (torch.from_numpy(out) > 0.5).float().unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'interaction_mask':
                with Image.open(os.path.join(cur_dir, 'interaction_mask.png')) as fimg:
                    out = np.array(fimg, dtype=np.float32) / 255
                data_feats = data_feats + (out,)
             
            elif feat == 'gripper_img_target':
                if is_original:
                    with Image.open(os.path.join(cur_dir, 'viz_target_pose.png')) as fimg:
                        out = np.array(fimg, dtype=np.float32) / 255
                    out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                else:
                    out = torch.ones(1, 3, 448, 448).float()
                data_feats = data_feats + (out,)

            elif feat == 'is_original':
                data_feats = data_feats + (is_original,)
            
            elif feat == 'pixel_id':
                out = torch.from_numpy(pixel_ids).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'gripper_direction':
                out = torch.from_numpy(gripper_direction).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'gripper_forward_direction':
                out = torch.from_numpy(gripper_forward_direction).unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'result':
                out = result
                data_feats = data_feats + (out,)

            elif feat == 'cur_dir':
                data_feats = data_feats + (cur_dir,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)
            
            elif feat == 'category':
                data_feats = data_feats + (category,)

            elif feat == 'cnt_id':
                data_feats = data_feats + (cnt_id,)

            elif feat == 'trial_id':
                data_feats = data_feats + (trial_id,)

            elif feat == 'obj_state':
                data_feats = data_feats + (state,)

            elif feat == 'camera_direction':
                data_feats = data_feats + (camera_direction,)

            elif feat == 'start_pos':
                data_feats = data_feats + (start_pos,)

            elif feat == 'end_pos':
                data_feats = data_feats + (end_pos,)

            elif feat == 'joint_info':
                data_feats = data_feats + (np.array([0, 0, 0, 0]),)

            elif feat == 'ctpt':
                data_feats = data_feats + (ctpt,)

            elif feat == 'target_object_part_actor_id':
                data_feats = data_feats + (target_object_part_actor_id,)

            elif feat == 'ctpt_2d':
                data_feats = data_feats + (ctpt_2d,)

            elif feat == 'joint_angles':
                data_feats = data_feats + (joint_angles,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

