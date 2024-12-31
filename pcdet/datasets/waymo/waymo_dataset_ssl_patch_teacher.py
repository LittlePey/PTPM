# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from functools import partial

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from collections import defaultdict
from ..augmentor.pillarmix import pillarmix

class WaymoDatasetSSLPatchTeacher(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.data_path_unlabel = Path(dataset_cfg.DATA_PATH_UNLABEL)  / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        split_dir_unlabel = self.root_path / 'ImageSets' / ('unlabel.txt')
        self.sample_sequence_list_unlabel = [x.strip() for x in open(split_dir_unlabel).readlines()]

        self.ceph_path = self.dataset_cfg.get('CEPH_PATH', None)
        self.ceph_path_unlabel = self.dataset_cfg.get('CEPH_PATH_UNLABEL', None)
        if self.ceph_path:
            from ..ceph import PetrelBackend
            self.petrel_client = PetrelBackend()
            self.data_path_ceph = os.path.join(self.ceph_path, self.dataset_cfg.PROCESSED_DATA_TAG)
            self.data_path_ceph_unlabel = os.path.join(self.ceph_path_unlabel, self.dataset_cfg.PROCESSED_DATA_TAG)

        self.infos = []
        self.seq_name_to_infos = self.include_waymo_data(self.mode)
        self.unlabel_infos = []
        self.seq_name_to_infos_unlabel = self.include_waymo_data_unlabel(self.mode)
        self.total_infos = copy.deepcopy(self.infos) + copy.deepcopy(self.unlabel_infos)

        self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and self.training
        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
            self.load_data_to_shared_memory()

        if self.dataset_cfg.get('USE_PREDBOX', False):
            self.pred_boxes_dict = self.load_pred_boxes_to_dict(
                pred_boxes_path=self.dataset_cfg.ROI_BOXES_PATH[self.mode]
            )
        else:
            self.pred_boxes_dict = {}
        self.train_batch_size = self.dataset_cfg.TRAIN_BATCH_SIZE
        self.unlabel_times = self.dataset_cfg.get('UNLABEL_TIMES', 4)
        self.pseudo_box_score_thr = self.dataset_cfg.PSEUDO_BOX_SCORE_THR
        self.point_cloud_range = self.dataset_cfg.POINT_CLOUD_RANGE
        self.partial_patch = self.dataset_cfg.PARTIAL_PATCH

        self.mix_type = self.dataset_cfg.MIX_TYPE
        self.mix_prob = self.dataset_cfg.MIX_PROB
        self.mix_ratio = self.dataset_cfg.MIX_RATIO
        self.mix_frames = self.dataset_cfg.MIX_FRAMES
        self.mix_aug = self.dataset_cfg.MIX_AUG
        self.bin_size = self.dataset_cfg.BIN_SIZE
        self.strategy = self.dataset_cfg.STRATEGY

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.seq_name_to_infos = self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        seq_name_to_infos = {}

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]

            if self.ceph_path is None:
                info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
                info_path = self.check_sequence_name_with_all_version(info_path)
                if not info_path.exists():
                    num_skipped_infos += 1
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    waymo_infos.extend(infos)
            else:
                info_path = os.path.join(self.data_path_ceph, sequence_name, ('%s.pkl' % sequence_name))
                infos = self.petrel_client.load_pkl(str(info_path), 1)[0]
                waymo_infos.extend(infos)

            seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))
            
        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        if not use_sequence_data:
            seq_name_to_infos = None 
        return seq_name_to_infos

    def include_waymo_data_unlabel(self, mode):
        self.logger.info('Loading Waymo Unlabel dataset')
        waymo_infos = []
        seq_name_to_infos = {}

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list_unlabel)):
            sequence_name = os.path.splitext(self.sample_sequence_list_unlabel[k])[0]

            if self.ceph_path is None:
                info_path = self.data_path_unlabel / sequence_name / ('%s.pkl' % sequence_name)
                info_path = self.check_sequence_name_with_all_version(info_path)
                if not info_path.exists():
                    num_skipped_infos += 1
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    waymo_infos.extend(infos)
            else:
                info_path = os.path.join(self.data_path_ceph_unlabel, sequence_name, ('%s.pkl' % sequence_name))
                infos = self.petrel_client.load_pkl(str(info_path), 1)[0]
                waymo_infos.extend(infos)

            seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

        self.unlabel_infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.unlabel_infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.unlabel_infos[k])
            self.unlabel_infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.unlabel_infos))

        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        if not use_sequence_data:
            seq_name_to_infos = None 
        return seq_name_to_infos

    def load_pred_boxes_to_dict(self, pred_boxes_path):
        self.logger.info(f'Loading and reorganizing pred_boxes to dict from path: {pred_boxes_path}')
        with open(pred_boxes_path, 'rb') as f:
            pred_dicts = pickle.load(f)

        pred_boxes_dict = {}
        for index, box_dict in enumerate(pred_dicts):
            seq_name = box_dict['frame_id'][:-4].replace('training_', '').replace('validation_', '')
            sample_idx = int(box_dict['frame_id'][-3:])

            if seq_name not in pred_boxes_dict:
                pred_boxes_dict[seq_name] = {}

            pred_labels = np.array([self.class_names.index(box_dict['name'][k]) + 1 for k in range(box_dict['name'].shape[0])])
            pred_boxes = np.concatenate((box_dict['boxes_lidar'], box_dict['score'][:, np.newaxis], pred_labels[:, np.newaxis]), axis=-1)
            pred_boxes_dict[seq_name][sample_idx] = pred_boxes

        self.logger.info(f'Predicted boxes has been loaded, total sequences: {len(pred_boxes_dict)}')
        return pred_boxes_dict

    def load_data_to_shared_memory(self):
        self.logger.info(f'Loading training data to shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            points = self.get_lidar(sequence_name, sample_idx)
            common_utils.sa_create(f"shm://{sa_key}", points)

        dist.barrier()
        self.logger.info('Training data has been saved to shared memory')

    def clean_shared_memory(self):
        self.logger.info(f'Clean training data from shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if not os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            SharedArray.delete(f"shm://{sa_key}")

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('Training data has been deleted from shared memory')

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1, update_info_only=False):
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label, update_info_only=update_info_only
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        # process_single_sequence(sample_sequence_file_list[0])
        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        if self.ceph_path is None:
            lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
            point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]
        else:
            lidar_file = os.path.join(self.data_path_ceph, sequence_name, ('%04d.npy' % sample_idx))
            point_features = self.petrel_client.load_np(str(lidar_file))

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not self.dataset_cfg.get('DISABLE_NLZ_FLAG_ON_POINTS', False):
            points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    @staticmethod
    def transform_prebox_to_current(pred_boxes3d, pose_pre, pose_cur):
        """

        Args:
            pred_boxes3d (N, 9 or 11): [x, y, z, dx, dy, dz, raw, <vx, vy,> score, label]
            pose_pre (4, 4):
            pose_cur (4, 4):
        Returns:

        """
        assert pred_boxes3d.shape[-1] in [9, 11]
        pred_boxes3d = pred_boxes3d.copy()
        expand_bboxes = np.concatenate([pred_boxes3d[:, :3], np.ones((pred_boxes3d.shape[0], 1))], axis=-1)

        bboxes_global = np.dot(expand_bboxes, pose_pre.T)[:, :3]
        expand_bboxes_global = np.concatenate([bboxes_global[:, :3],np.ones((bboxes_global.shape[0], 1))], axis=-1)
        bboxes_pre2cur = np.dot(expand_bboxes_global, np.linalg.inv(pose_cur.T))[:, :3]
        pred_boxes3d[:, 0:3] = bboxes_pre2cur

        if pred_boxes3d.shape[-1] == 11:
            expand_vels = np.concatenate([pred_boxes3d[:, 7:9], np.zeros((pred_boxes3d.shape[0], 1))], axis=-1)
            vels_global = np.dot(expand_vels, pose_pre[:3, :3].T)
            vels_pre2cur = np.dot(vels_global, np.linalg.inv(pose_cur[:3, :3].T))[:,:2]
            pred_boxes3d[:, 7:9] = vels_pre2cur

        pred_boxes3d[:, 6]  = pred_boxes3d[..., 6] + np.arctan2(pose_pre[..., 1, 0], pose_pre[..., 0, 0])
        pred_boxes3d[:, 6]  = pred_boxes3d[..., 6] - np.arctan2(pose_cur[..., 1, 0], pose_cur[..., 0, 0])
        return pred_boxes3d

    @staticmethod
    def reorder_rois_for_refining(pred_bboxes):
        num_max_rois = max([len(bbox) for bbox in pred_bboxes])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        ordered_bboxes = np.zeros([len(pred_bboxes), num_max_rois, pred_bboxes[0].shape[-1]], dtype=np.float32)

        for bs_idx in range(ordered_bboxes.shape[0]):
            ordered_bboxes[bs_idx, :len(pred_bboxes[bs_idx])] = pred_bboxes[bs_idx]
        return ordered_bboxes

    def get_sequence_data(self, info, points, sequence_name, sample_idx, sequence_cfg, load_pred_boxes=False):
        """
        Args:
            info:
            points:
            sequence_name:
            sample_idx:
            sequence_cfg:
        Returns:
        """

        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        def load_pred_boxes_from_dict(sequence_name, sample_idx):
            """
            boxes: (N, 11)  [x, y, z, dx, dy, dn, raw, vx, vy, score, label]
            """
            sequence_name = sequence_name.replace('training_', '').replace('validation_', '')
            load_boxes = self.pred_boxes_dict[sequence_name][sample_idx]
            assert load_boxes.shape[-1] == 11
            load_boxes[:, 7:9] = -0.1 * load_boxes[:, 7:9]  # transfer speed to negtive motion from t to t-1
            return load_boxes

        pose_cur = info['pose'].reshape((4, 4))
        num_pts_cur = points.shape[0]
        sample_idx_pre_list = np.clip(sample_idx + np.arange(sequence_cfg.SAMPLE_OFFSET[0], sequence_cfg.SAMPLE_OFFSET[1]), 0, 0x7FFFFFFF)
        sample_idx_pre_list = sample_idx_pre_list[::-1]

        if sequence_cfg.get('ONEHOT_TIMESTAMP', False):
            onehot_cur = np.zeros((points.shape[0], len(sample_idx_pre_list) + 1)).astype(points.dtype)
            onehot_cur[:, 0] = 1
            points = np.hstack([points, onehot_cur])
        else:
            points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])
        points_pre_all = []
        num_points_pre = []

        pose_all = [pose_cur]
        pred_boxes_all = []
        if load_pred_boxes:
            pred_boxes = load_pred_boxes_from_dict(sequence_name, sample_idx)
            pred_boxes_all.append(pred_boxes)

        sequence_info = self.seq_name_to_infos[sequence_name]

        for idx, sample_idx_pre in enumerate(sample_idx_pre_list):

            points_pre = self.get_lidar(sequence_name, sample_idx_pre)
            pose_pre = sequence_info[sample_idx_pre]['pose'].reshape((4, 4))
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1)
            points_pre_global = np.dot(expand_points_pre, pose_pre.T)[:, :3]
            expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
            points_pre2cur = np.dot(expand_points_pre_global, np.linalg.inv(pose_cur.T))[:, :3]
            points_pre = np.concatenate([points_pre2cur, points_pre[:, 3:]], axis=-1)
            if sequence_cfg.get('ONEHOT_TIMESTAMP', False):
                onehot_vector = np.zeros((points_pre.shape[0], len(sample_idx_pre_list) + 1))
                onehot_vector[:, idx + 1] = 1
                points_pre = np.hstack([points_pre, onehot_vector])
            else:
                # add timestamp
                points_pre = np.hstack([points_pre, 0.1 * (sample_idx - sample_idx_pre) * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame 0.1s
            points_pre = remove_ego_points(points_pre, 1.0)
            points_pre_all.append(points_pre)
            num_points_pre.append(points_pre.shape[0])
            pose_all.append(pose_pre)

            if load_pred_boxes:
                pose_pre = sequence_info[sample_idx_pre]['pose'].reshape((4, 4))
                pred_boxes = load_pred_boxes_from_dict(sequence_name, sample_idx_pre)
                pred_boxes = self.transform_prebox_to_current(pred_boxes, pose_pre, pose_cur)
                pred_boxes_all.append(pred_boxes)

        points = np.concatenate([points] + points_pre_all, axis=0).astype(np.float32)
        num_points_all = np.array([num_pts_cur] + num_points_pre).astype(np.int32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)

        if load_pred_boxes:
            temp_pred_boxes = self.reorder_rois_for_refining(pred_boxes_all)
            pred_boxes = temp_pred_boxes[:, :, 0:9]
            pred_scores = temp_pred_boxes[:, :, 9]
            pred_labels = temp_pred_boxes[:, :, 10]
        else:
            pred_boxes = pred_scores = pred_labels = None

        return points, num_points_all, sample_idx_pre_list, poses, pred_boxes, pred_scores, pred_labels

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = copy.deepcopy(self.infos[index])
        if self.training:
            data_dict_labeled_list = self.get_item_single(info)[:self.partial_patch]
        else:
            data_dict_labeled_list = self.get_item_single(info)

        if self.training:
            all_list = []
            all_list.extend(data_dict_labeled_list)

            if self.train_batch_size > 1:
                indexes = np.random.choice(range(len(self.infos)), self.train_batch_size - 1, False)
                for index in indexes:
                    info = copy.deepcopy(self.infos[index])
                    all_list.extend(self.get_item_single(info)[:self.partial_patch])

            unlabel_indexes = np.random.choice(range(len(self.unlabel_infos)), self.unlabel_times * self.train_batch_size, False)
            for unlabel_index in unlabel_indexes:
                unlabel_info = copy.deepcopy(self.unlabel_infos[unlabel_index])
                all_list.extend(self.get_item_single(unlabel_info, unlabel_frame=True)[:self.partial_patch])
            return all_list
        else:
            return data_dict_labeled_list

    def get_item_single(self, info, unlabel_frame=False):
        input_dict = self.get_item_single_(info, unlabel_frame)

        enable_mix = np.random.choice([False, True], replace=False, p=[1-self.mix_prob, self.mix_prob])
        if self.training and unlabel_frame and enable_mix:

            input_dict_queue = []
            for _ in range(self.mix_frames):
                index2 = np.random.choice(len(self.total_infos))
                info2 = copy.deepcopy(self.total_infos[index2])
                input_dict2 = self.get_item_single_(info2, index2 >= len(self.infos))
                for aug in self.mix_aug:
                    if aug == 'flip_x' and np.random.choice([False, True], replace=False, p=[0.5, 0.5]):
                        input_dict2['gt_boxes'][:, 1] = -input_dict2['gt_boxes'][:, 1]
                        input_dict2['gt_boxes'][:, 6] = -input_dict2['gt_boxes'][:, 6]
                        input_dict2['points'][:, 1] = -input_dict2['points'][:, 1]
                    if aug == 'flip_y' and np.random.choice([False, True], replace=False, p=[0.5, 0.5]):
                        input_dict2['gt_boxes'][:, 0] = -input_dict2['gt_boxes'][:, 0]
                        input_dict2['gt_boxes'][:, 6] = -(input_dict2['gt_boxes'][:, 6] + np.pi)
                        input_dict2['points'][:, 0] = -input_dict2['points'][:, 0]
                    if aug == 'rotation':
                        noise_rotation = np.random.uniform(-0.78539816, 0.78539816)
                        input_dict2['points'] = common_utils.rotate_points_along_z(input_dict2['points'][np.newaxis, :, :], np.array([noise_rotation]))[0]
                        input_dict2['gt_boxes'][:, 0:3] = common_utils.rotate_points_along_z(input_dict2['gt_boxes'][np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
                        input_dict2['gt_boxes'][:, 6] += noise_rotation

                input_dict_queue.append(input_dict2)

            if self.mix_type == 'pillarmix':
                input_dict = pillarmix(input_dict, input_dict_queue, self.mix_ratio, self.bin_size, self.strategy)

        data_dict_list = self.prepare_data(data_dict=input_dict, unlabel_frame=unlabel_frame)
        for data_dict in data_dict_list:
            data_dict['metadata'] = info.get('metadata', info['frame_id'])
            data_dict.pop('num_points_in_gt', None)

        if self.training:
            data_dict_list_valid = []
            for data_dict_list_ in data_dict_list:
                if (len(data_dict_list_['points'])>10) and (len(data_dict_list_['gt_boxes'])>1):
                    data_dict_list_valid.append(data_dict_list_)
            if self.partial_patch > len(data_dict_list_valid):
                data_dict_list_valid.extend(data_dict_list[:self.partial_patch-len(data_dict_list_valid)])
            np.random.shuffle(data_dict_list_valid)
            return data_dict_list_valid
        else:
            return data_dict_list

    def get_item_single_(self, info, unlabel_frame=False):
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        input_dict = {
            'sample_idx': sample_idx
        }
        points = self.get_lidar(sequence_name, sample_idx)

        input_dict.update({
            'points': points,
            'frame_id': info['frame_id'],
        })

        if 'annos' in info:
            annos = info['annos']
            annos.pop('speed_global', None)
            annos.pop('accel_global', None)
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.dataset_cfg.get('TRAIN_WITH_SPEED', False):
                assert gt_boxes_lidar.shape[-1] == 9
            else:
                gt_boxes_lidar = gt_boxes_lidar[:, 0:7]

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False) and 'num_points_in_gt' in annos.keys():
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            if 'score' in annos.keys():
                if isinstance(self.pseudo_box_score_thr,list):
                    filter_mask = np.array([(annos['name'][k] in self.class_names) and (annos['score'][k] > self.pseudo_box_score_thr[self.class_names.index(annos['name'][k])]) for k in range(len(annos['score']))], dtype=np.bool)
                else:
                    filter_mask = annos['score'] > self.pseudo_box_score_thr
                # filter_mask = annos['score'] > self.pseudo_box_score_thr
                annos['name'] = annos['name'][filter_mask]
                gt_boxes_lidar = gt_boxes_lidar[filter_mask]
                annos['score'] = annos['score'][filter_mask]
                gt_score = annos['score']
                # print('pseudo boxes %d ==> %d' % (len(filter_mask), filter_mask.sum()))
            else:
                gt_score = np.ones(len(gt_boxes_lidar)).astype(np.float32)

            input_dict.update({
                'gt_names': annos['name'],
                'gt_score': gt_score,
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })
        return input_dict

    def prepare_data(self, data_dict, unlabel_frame=False):
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
                },
                unlabel_frame=unlabel_frame
            )
            if 'calib' in data_dict:
                data_dict['calib'] = calib
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            data_dict['gt_score'] = data_dict['gt_score'][selected]
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
                elif key in ['gt_score']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__()] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['noise_local_offset_x', 'noise_local_offset_y', 'noise_local_rot', 'noise_local_scale']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_augments = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_augments[k, :val[k].__len__()] = val[k]
                    ret[key] = batch_gt_augments
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
    
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
