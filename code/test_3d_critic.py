"""
    Test the Action Scoring Module only
"""
import os
import sys
import random
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
from data_conf import SAPIENVisionDataset
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))


def test(conf, val_data_list):
    # create training and validation datasets and data loaders
    data_features = ['gripper_direction', 'pcs', 'gripper_forward_direction', 'result', 'start_pos', 'end_pos',
                     'joint_info', 'category', 'shape_id', 'ctpt', 'pc_movables']

    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.network(input_dim=17, pnpp_feat_dim=conf.feat_dim, hidden_feat_dim=conf.feat_dim,
                                feat_dim=conf.feat_dim, hidden_dim=conf.hidden_dim)

    # send parameters to device
    network.to(conf.device)

    # set models to evaluation mode
    network.eval()

    # load test data
    val_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num, \
                                      abs_thres=conf.abs_thres, rel_thres=conf.rel_thres, dp_thres=conf.dp_thres,
                                      img_size=conf.img_size, world_coordinate=True, batch_zise=conf.batch_size)
    val_dataset.load_data(val_data_list)
    val_dataset.get_seq()
    utils.printout(None, str(val_dataset))

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, \
                                                 num_workers=0, drop_last=True, collate_fn=utils.collate_feats,
                                                 worker_init_fn=utils.worker_init_fn)

    # load model
    for model_epoch in range(conf.start_epoch, conf.end_epoch + 1):
        data_to_restore = torch.load(os.path.join(conf.load_model, 'ckpts', '%d_secondAAP-network.pth' % model_epoch))
        network.load_state_dict(data_to_restore)
        print('loading model from: ', os.path.join(conf.load_model, 'ckpts', '%d_secondAAP-network.pth' % model_epoch))

        total_pred_and_gt = 0
        total_pred_sum = 1
        total_gt_sum = 1
        correct_sum = 0
        all_sum = 0

        val_batches = enumerate(val_dataloader, 0)

        ### test for every batch
        for val_batch_ind, batch in val_batches:
            with torch.no_grad():
                # forward pass (including logging)
                pred_and_gt, pred_sum, gt_sum, correct = forward(batch=batch, data_features=data_features, network=network, conf=conf, batch_id=val_batch_ind)
            total_pred_and_gt += pred_and_gt
            total_pred_sum += pred_sum
            total_gt_sum += gt_sum
            correct_sum += correct
            all_sum += conf.batch_size

        precise = total_pred_and_gt / total_pred_sum
        recall = total_pred_and_gt / total_gt_sum
        F1 = 2 * (precise * recall) / (precise + recall)

        precise = round(precise, 3)
        recall = round(recall, 3)
        F1 = round(F1, 3)
        print('score from: ', conf.load_model + f'-{model_epoch}')
        print('Score: ', precise, recall, F1)

        with open(os.path.join(conf.load_model, 'F1_score.txt'), 'a') as fout:
            if model_epoch == conf.start_epoch:
                fout.write(f'{conf.val_data_dir} Score: \n')
            fout.write(f'{model_epoch} {precise} {recall} {F1}\n')


def forward(batch, data_features, network, conf, batch_id):
    shape_id = batch[data_features.index('shape_id')]
    if batch_id % 10 == 0:
        # categories = batch[data_features.index('category')]
        print(shape_id)
    # prepare input
    input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(conf.device)  # B x 3N x 3
    input_movables = torch.cat(batch[data_features.index('pc_movables')], dim=0).to(conf.device)  # B x 3N
    batch_size = input_pcs.shape[0]

    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(input_pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
    input_movables = input_movables[input_pcid1, input_pcid2].reshape(batch_size, conf.num_point_per_shape)

    input_dirs1 = torch.cat(batch[data_features.index('gripper_direction')], dim=0).to(conf.device)  # B x 3
    input_dirs2 = torch.cat(batch[data_features.index('gripper_forward_direction')], dim=0).to(conf.device)  # B x 3
    joint_info = batch[data_features.index('joint_info')]
    ctpt = batch[data_features.index('ctpt')]

    ctpt = torch.FloatTensor(np.array(ctpt)).view(batch_size, -1).to(conf.device)
    joint_info = torch.FloatTensor(joint_info).view(batch_size, -1).to(conf.device)

    final_dist = 0.1
    if conf.primact_type == 'pushing-left' or conf.primact_type == 'pushing-up':
        final_dist = 0.11

    start_pos = np.array(batch[data_features.index('start_pos')])
    end_pos = np.array(batch[data_features.index('end_pos')])
    push_dis = abs(start_pos - end_pos)

    dis = torch.Tensor(np.ones((batch_size, 1)) * final_dist).to(conf.device)
    push_dis = torch.FloatTensor(push_dis).view(batch_size, -1).to(conf.device)
    start_pos = torch.FloatTensor(start_pos).view(batch_size, -1).to(conf.device)
    end_pos = torch.FloatTensor(end_pos).view(batch_size, -1).to(conf.device)

    # forward through the network
    pred_result_logits = network.inference(input_dirs1, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, input_dirs2)  # B x 2, B x F x N

    # prepare gt
    gt_result = torch.Tensor(batch[data_features.index('result')]).long().numpy()  # B

    # compute correctness
    pred_results = pred_result_logits.detach().cpu().numpy() > 0
    # print('pred_results: ', pred_results)
    pred_and_gt = np.logical_and(pred_results, gt_result)

    correct = (pred_results == gt_result)

    # precise = pred_and_gt.sum() / pred_results.sum()
    # recall = pred_and_gt.sum() / gt_result.sum()
    # F1_score = (precise + recall) / 2
    # print('batch ', batch_id, 'Score: ', precise, recall, F1_score)

    return pred_and_gt.sum(), pred_results.sum(), gt_result.sum(), correct.sum()


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--primact_type', type=str, help='the primact type')
    parser.add_argument('--category_types', type=str,
                        help='list all categories [Default: None, meaning all 10 categories]', default=None)
    parser.add_argument('--val_data_dir', type=str, help='data directory')
    parser.add_argument('--val_data_fn', type=str, help='data directory', default='data_tuple_list_val_subset.txt')
    parser.add_argument('--buffer_max_num', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_interaction_data_offline', type=int, default=150)

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    # parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_point_per_shape', type=int, default=10000)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--abs_thres', type=float, default=0.01, help='abs thres')
    parser.add_argument('--rel_thres', type=float, default=0.5, help='rel thres')
    parser.add_argument('--dp_thres', type=float, default=0.5, help='dp thres')
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--load_model', type=str, default=None, help='what model to load')
    parser.add_argument('--start_epoch', type=int, default=None, help='what model to load')
    parser.add_argument('--end_epoch', type=int, default=None, help='what model to load')

    # parse args
    conf = parser.parse_args()

    ### prepare before training
    conf.load_model = os.path.join('../logs', conf.load_model)

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # set training device
    device = torch.device(conf.device)
    conf.device = device

    if conf.category_types is None:
        conf.category_types = ['Box', 'Door', 'Faucet', 'Kettle', 'Microwave', 'Refrigerator', 'StorageFurniture',
                               'Switch', 'TrashCan', 'Window']
    else:
        conf.category_types = conf.category_types.split(',')
    print('category_types: %s' % str(conf.category_types))

    with open(os.path.join(conf.val_data_dir, 'data_tuple_list.txt'), 'r') as fin:
        all_val_data_list = [os.path.join(conf.val_data_dir, l.rstrip()) for l in fin.readlines()]

    val_shape_ids = []
    for item in all_val_data_list:
        if item.split('_')[-5] not in val_shape_ids:
            val_shape_ids.append(item.split('_')[-5])

    val_data_list = []
    for id in val_shape_ids:
        for item in all_val_data_list:
            if int(item.split('_')[-1]) < conf.num_interaction_data_offline and item.split('_')[-5] == id:
                val_data_list.append(item)

    ### start training
    test(conf, val_data_list)


