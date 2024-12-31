import copy
import pickle
import numpy as np

from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from .once_toolkits import Octopus
import multiprocessing
from tqdm import tqdm
import os
from functools import partial
import math
from collections import defaultdict

class ONCEDatasetPatchTeacher(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = dataset_cfg.DATA_SPLIT['train'] if training else dataset_cfg.DATA_SPLIT['test']
        assert self.split in ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.cam_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
        self.cam_tags = ['top', 'top2', 'left_back', 'left_front', 'right_front', 'right_back', 'back']

        self.ceph_path = self.dataset_cfg.get('CEPH_PATH', None)
        if self.ceph_path:
            from ..ceph import PetrelBackend
            self.petrel_client = PetrelBackend()
            self.ceph_path_save = self.ceph_path.replace('cluster3', 'cluster7')

        self.toolkits = Octopus(self.root_path, self.ceph_path, self.petrel_client)

        self.once_infos = []
        self.include_once_data(self.split)

        self.point_cloud_range = self.dataset_cfg.POINT_CLOUD_RANGE
        self.partial_patch = self.dataset_cfg.PARTIAL_PATCH

    def include_once_data(self, split):
        if self.logger is not None:
            self.logger.info('Loading ONCE dataset')
        once_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[split]:
            if self.ceph_path is None:
                info_path = self.root_path / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    once_infos.extend(infos)
            else:
                info_path = os.path.join(self.ceph_path, info_path)
                infos = self.petrel_client.load_pkl(info_path, 1)[0]
                once_infos.extend(infos)

        def check_annos(info):
            return 'annos' in info

        if 'raw' not in self.split:
            once_infos = list(filter(check_annos,once_infos))

        self.once_infos.extend(once_infos)

        if self.logger is not None:
            self.logger.info('Total samples for ONCE dataset: %d' % (len(once_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, sequence_id, frame_id):
        return self.toolkits.load_point_cloud(sequence_id, frame_id)

    def get_image(self, sequence_id, frame_id, cam_name):
        return self.toolkits.load_image(sequence_id, frame_id, cam_name)

    def project_lidar_to_image(self, sequence_id, frame_id):
        return self.toolkits.project_lidar_to_image(sequence_id, frame_id)

    def point_painting(self, points, info):
        semseg_dir = './' # add your own seg directory
        used_classes = [0,1,2,3,4,5]
        num_classes = len(used_classes)
        frame_id = str(info['frame_id'])
        seq_id = str(info['sequence_id'])
        painted = np.zeros((points.shape[0], num_classes)) # classes + bg
        for cam_name in self.cam_names:
            img_path = Path(semseg_dir) / Path(seq_id) / Path(cam_name) / Path(frame_id+'_label.png')
            calib_info = info['calib'][cam_name]
            cam_2_velo = calib_info['cam_to_velo']
            cam_intri = np.hstack([calib_info['cam_intrinsic'], np.zeros((3, 1), dtype=np.float32)])
            point_xyz = points[:, :3]
            points_homo = np.hstack(
                [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_lidar = np.dot(points_homo, np.linalg.inv(cam_2_velo).T)
            mask = points_lidar[:, 2] > 0
            points_lidar = points_lidar[mask]
            points_img = np.dot(points_lidar, cam_intri.T)
            points_img = points_img / points_img[:, [2]]
            uv = points_img[:, [0,1]]
            #depth = points_img[:, [2]]
            seg_map = np.array(Image.open(img_path)) # (H, W)
            H, W = seg_map.shape
            seg_feats = np.zeros((H*W, num_classes))
            seg_map = seg_map.reshape(-1)
            for cls_i in used_classes:
                seg_feats[seg_map==cls_i, cls_i] = 1
            seg_feats = seg_feats.reshape(H, W, num_classes).transpose(2, 0, 1)
            uv[:, 0] = (uv[:, 0] - W / 2) / (W / 2)
            uv[:, 1] = (uv[:, 1] - H / 2) / (H / 2)
            uv_tensor = torch.from_numpy(uv).unsqueeze(0).unsqueeze(0)  # [1,1,N,2]
            seg_feats = torch.from_numpy(seg_feats).unsqueeze(0) # [1,C,H,W]
            proj_scores = F.grid_sample(seg_feats, uv_tensor, mode='bilinear', padding_mode='zeros')  # [1, C, 1, N]
            proj_scores = proj_scores.squeeze(0).squeeze(1).transpose(0, 1).contiguous() # [N, C]
            painted[mask] = proj_scores.numpy()
        return np.concatenate([points, painted], axis=1)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.once_infos) * self.total_epochs

        return len(self.once_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.once_infos)

        info = copy.deepcopy(self.once_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id)

        if self.dataset_cfg.get('POINT_PAINTING', False):
            points = self.point_painting(points, info)

        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })


        data_dict_list = self.prepare_data(data_dict=input_dict)
        for data_dict in data_dict_list:
            data_dict.pop('num_points_in_gt', None)

        if self.training:
            data_dict_list_valid = []
            for data_dict_list_ in data_dict_list:
                if (len(data_dict_list_['points'])>10) and (len(data_dict_list_['gt_boxes'])>1):
                    data_dict_list_valid.append(data_dict_list_)
            if self.partial_patch > len(data_dict_list_valid):
                data_dict_list_valid.extend(data_dict_list[:self.partial_patch-len(data_dict_list_valid)])
            np.random.shuffle(data_dict_list_valid)
            return data_dict_list_valid[:self.partial_patch]
        else:
            return data_dict_list
        
    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            
            if 'calib' in data_dict:
                calib = data_dict['calib']
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if 'calib' in data_dict:
                data_dict['calib'] = calib
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes']

        data_dict_list = []
        for patch_i in range(16):
            data_dict_patch = {}
            ##### rotation
            if patch_i // 4 == 0:
                points_patch = copy.deepcopy(points[(points[:, 0] > self.point_cloud_range[0]) & (points[:, 1] > self.point_cloud_range[1])])
                gt_boxes_patch = copy.deepcopy(gt_boxes[(gt_boxes[:, 0] > self.point_cloud_range[0]) & (gt_boxes[:, 1] > self.point_cloud_range[1])])
                rotation = 0

            if patch_i // 4 == 1:
                points_patch = copy.deepcopy(points[(points[:, 0] > self.point_cloud_range[0]) & (points[:, 1] < -self.point_cloud_range[1])])
                gt_boxes_patch = copy.deepcopy(gt_boxes[(gt_boxes[:, 0] > self.point_cloud_range[0]) & (gt_boxes[:, 1] < -self.point_cloud_range[1])])
                rotation = np.pi / 2

            if patch_i // 4 == 2:
                points_patch = copy.deepcopy(points[(points[:, 0] < -self.point_cloud_range[0]) & (points[:, 1] < -self.point_cloud_range[1])])
                gt_boxes_patch = copy.deepcopy(gt_boxes[(gt_boxes[:, 0] < -self.point_cloud_range[0]) & (gt_boxes[:, 1] < -self.point_cloud_range[1])])
                rotation = np.pi

            if patch_i // 4 == 3:
                points_patch = copy.deepcopy(points[(points[:, 0] < -self.point_cloud_range[0]) & (points[:, 1] > self.point_cloud_range[1])])
                gt_boxes_patch = copy.deepcopy(gt_boxes[(gt_boxes[:, 0] < -self.point_cloud_range[0]) & (gt_boxes[:, 1] > self.point_cloud_range[1])])
                rotation = 3 * np.pi / 2

            data_dict_patch['gt_nums'] = len(gt_boxes)
            points_patch = common_utils.rotate_points_along_z(points_patch[np.newaxis, :, :], np.array([rotation]))[0]
            gt_boxes_patch[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes_patch[np.newaxis, :, 0:3], np.array([rotation]))[0]
            gt_boxes_patch[:, 6] +=rotation

            ##### shift
            if patch_i % 4 == 0:
                points_patch = points_patch[(points_patch[:, 0] >= self.point_cloud_range[3] + 2 * self.point_cloud_range[0]) & (points_patch[:, 1] >=self.point_cloud_range[4] + 2 * self.point_cloud_range[1])]
                gt_boxes_patch = gt_boxes_patch[(gt_boxes_patch[:, 0] >= self.point_cloud_range[3] + 2 * self.point_cloud_range[0]) & (gt_boxes_patch[:, 1] >=self.point_cloud_range[4] + 2 * self.point_cloud_range[1])]
                shift_x, shift_y = self.point_cloud_range[3] + self.point_cloud_range[0], self.point_cloud_range[4] + self.point_cloud_range[1]

            if patch_i % 4 == 1:
                points_patch = points_patch[(points_patch[:, 0] >= self.point_cloud_range[3] + 2 * self.point_cloud_range[0]) & (points_patch[:, 1] <self.point_cloud_range[4])]
                gt_boxes_patch = gt_boxes_patch[(gt_boxes_patch[:, 0] >= self.point_cloud_range[3] + 2 * self.point_cloud_range[0]) & (gt_boxes_patch[:, 1] <self.point_cloud_range[4])]
                shift_x, shift_y = self.point_cloud_range[3] + self.point_cloud_range[0], 0.0

            if patch_i % 4 == 2:
                points_patch = points_patch[(points_patch[:, 0] < self.point_cloud_range[3]) & (points_patch[:, 1] <self.point_cloud_range[4])]
                gt_boxes_patch = gt_boxes_patch[(gt_boxes_patch[:, 0] < self.point_cloud_range[3]) & (gt_boxes_patch[:, 1] <self.point_cloud_range[4])]
                shift_x, shift_y = 0.0, 0.0

            if patch_i % 4 == 3:
                points_patch = points_patch[(points_patch[:, 0] < self.point_cloud_range[3]) & (points_patch[:, 1] >=self.point_cloud_range[4] + 2 * self.point_cloud_range[1])]
                gt_boxes_patch = gt_boxes_patch[(gt_boxes_patch[:, 0] < self.point_cloud_range[3]) & (gt_boxes_patch[:, 1] >=self.point_cloud_range[4] + 2 * self.point_cloud_range[1])]
                shift_x, shift_y = 0.0, self.point_cloud_range[4] + self.point_cloud_range[1]

            points_patch[:, 0] -= shift_x
            points_patch[:, 1] -= shift_y
            gt_boxes_patch[:, 0] -= shift_x
            gt_boxes_patch[:, 1] -= shift_y

            if patch_i == 0:
                data_dict_patch['gt_boxes_origin'] = data_dict['gt_boxes']
                data_dict_patch['frame_id'] = data_dict['frame_id']
            data_dict_patch['points'] = points_patch
            data_dict_patch['gt_boxes'] = gt_boxes_patch
            data_dict_patch['use_lead_xyz'] = data_dict['use_lead_xyz']
            data_dict_list.append(data_dict_patch)

        for data_dict_ in data_dict_list:
            data_dict_ = self.data_processor.forward(
                data_dict=data_dict_
            )
            data_dict.pop('gt_names', None)

        return data_dict_list

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        if isinstance(batch_list[0], list):
            for cur_sample in batch_list:
                for cur_cur_sample in cur_sample:
                    for key, val in cur_cur_sample.items():
                        data_dict[key].append(val)
            batch_size = len(batch_list) * len(batch_list[0])
        else:
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    data_dict[key].append(val)
            batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points', 'voxels_dense', 'voxel_num_points_dense']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'points_dense', 'voxel_coords', 'voxel_coords_dense']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes_origin']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((len(val), max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(len(val)):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len-len(_points)), (0,0))
                        points_pad = np.pad(_points,
                                pad_width=pad_width,
                                mode='constant',
                                constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_3d': np.zeros((num_samples, 7))
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_3d'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                raise NotImplementedError
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        from .once_eval.evaluation import get_evaluation_results

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.once_infos]
        ap_result_str, ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict
