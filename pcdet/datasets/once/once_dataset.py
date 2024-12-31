import copy
import pickle
import numpy as np

from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils
from .once_toolkits import Octopus
import multiprocessing
from tqdm import tqdm
import os
from functools import partial
import math

class ONCEDataset(DatasetTemplate):
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

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    def get_infos(self, num_workers=4, sample_seq_list=None):
        import concurrent.futures as futures
        import json
        root_path = self.root_path
        cam_names = self.cam_names

        """
        # dataset json format
        {
            'meta_info': 
            'calib': {
                'cam01': {
                    'cam_to_velo': list
                    'cam_intrinsic': list
                    'distortion': list
                }
                ...
            }
            'frames': [
                {
                    'frame_id': timestamp,
                    'annos': {
                        'names': list
                        'boxes_3d': list of list
                        'boxes_2d': {
                            'cam01': list of list
                            ...
                        }
                    }
                    'pose': list
                },
                ...
            ]
        }
        # open pcdet format
        {
            'meta_info':
            'sequence_id': seq_idx
            'frame_id': timestamp
            'timestamp': timestamp
            'lidar': path
            'cam01': path
            ...
            'calib': {
                'cam01': {
                    'cam_to_velo': np.array
                    'cam_intrinsic': np.array
                    'distortion': np.array
                }
                ...
            }
            'pose': np.array
            'annos': {
                'name': np.array
                'boxes_3d': np.array
                'boxes_2d': {
                    'cam01': np.array
                    ....
                }
            }          
        }
        """
        def process_single_sequence(seq_idx):
            print('%s seq_idx: %s' % (self.split, seq_idx))
            seq_infos = []
            seq_path = Path(root_path) / 'data' / seq_idx
            json_path = seq_path / ('%s.json' % seq_idx)
            with open(json_path, 'r') as f:
                info_this_seq = json.load(f)
            meta_info = info_this_seq['meta_info']
            calib = info_this_seq['calib']
            for f_idx, frame in enumerate(info_this_seq['frames']):
                frame_id = frame['frame_id']
                if f_idx == 0:
                    prev_id = None
                else:
                    prev_id = info_this_seq['frames'][f_idx-1]['frame_id']
                if f_idx == len(info_this_seq['frames'])-1:
                    next_id = None
                else:
                    next_id = info_this_seq['frames'][f_idx+1]['frame_id']
                pc_path = str(seq_path / 'lidar_roof' / ('%s.bin' % frame_id))
                pose = np.array(frame['pose'])
                frame_dict = {
                    'sequence_id': seq_idx,
                    'frame_id': frame_id,
                    'timestamp': int(frame_id),
                    'prev_id': prev_id,
                    'next_id': next_id,
                    'meta_info': meta_info,
                    'lidar': pc_path,
                    'pose': pose,
                    'sample_idx': f_idx
                }
                calib_dict = {}
                for cam_name in cam_names:
                    cam_path = str(seq_path / cam_name / ('%s.jpg' % frame_id))
                    frame_dict.update({cam_name: cam_path})
                    calib_dict[cam_name] = {}
                    calib_dict[cam_name]['cam_to_velo'] = np.array(calib[cam_name]['cam_to_velo'])
                    calib_dict[cam_name]['cam_intrinsic'] = np.array(calib[cam_name]['cam_intrinsic'])
                    calib_dict[cam_name]['distortion'] = np.array(calib[cam_name]['distortion'])
                frame_dict.update({'calib': calib_dict})

                if 'annos' in frame:
                    annos = frame['annos']
                    boxes_3d = np.array(annos['boxes_3d'])
                    if boxes_3d.shape[0] == 0:
                        print(frame_id)
                        continue
                    boxes_2d_dict = {}
                    for cam_name in cam_names:
                        boxes_2d_dict[cam_name] = np.array(annos['boxes_2d'][cam_name])
                    annos_dict = {
                        'name': np.array(annos['names']),
                        'boxes_3d': boxes_3d,
                        'boxes_2d': boxes_2d_dict
                    }

                    points = self.get_lidar(seq_idx, frame_id)
                    corners_lidar = box_utils.boxes_to_corners_3d(np.array(annos['boxes_3d']))
                    num_gt = boxes_3d.shape[0]
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_gt):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annos_dict['num_points_in_gt'] = num_points_in_gt

                    frame_dict.update({'annos': annos_dict})
                seq_infos.append(frame_dict)

            pkl_path = seq_path / ('%s.pkl' % seq_idx)
            with open(pkl_path, 'wb') as f:
                pickle.dump(seq_infos, f)
            return seq_infos

        sample_seq_list = sample_seq_list if sample_seq_list is not None else self.sample_seq_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_sequence, sample_seq_list)
        all_infos = []
        for info in infos:
            all_infos.extend(info)
        return all_infos

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('once_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            if 'annos' not in infos[k]:
                continue
            print('gt_database sample: %d' % (k + 1))
            info = infos[k]
            frame_id = info['frame_id']
            seq_id = info['sequence_id']
            points = self.get_lidar(seq_id, frame_id)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['boxes_3d']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (frame_id, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

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

    def create_pseudo_database(self, infos=None, used_classes=None, split='raw_small', workers=16, use_ceph=False, save_path=''):
        import torch
        if use_ceph:
            from ..ceph import PetrelBackend
            self.petrel_client = PetrelBackend()

        database_save_path = Path(save_path) / ('pseudo_database' if split == 'train' else ('pseudo_database_%s' % split))
        db_info_save_path = Path(save_path) / ('once_pseudo_dbinfos_%s.pkl' % split)

        if use_ceph is False:
            database_save_path.mkdir(parents=True, exist_ok=True)

        print(f'Number workers: {workers}')
        create_pseudo_database_of_single_scene = partial(
            self.create_pseudo_database_of_single_scene,
            save_path=save_path, 
            database_save_path=database_save_path, 
            used_classes=used_classes, 
            total_samples=len(infos), 
            use_ceph=use_ceph,
        )
        with multiprocessing.Pool(workers) as p:
            all_db_infos_list = list(p.map(create_pseudo_database_of_single_scene, zip(infos, np.arange(len(infos)))))

        all_db_infos = {}

        for cur_db_infos in all_db_infos_list:
            for key, val in cur_db_infos.items():
                if key not in all_db_infos:
                    all_db_infos[key] = val
                else:
                    all_db_infos[key].extend(val)

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        if use_ceph:
            db_info_save_path_list = str(db_info_save_path).split('/')
            db_info_save_path_ceph = os.path.join(os.path.join(*self.ceph_path_save.split('/')[:-1]), db_info_save_path_list[-2], db_info_save_path_list[-1]).replace('s3:/', 's3://')
            self.petrel_client.save_pkl(db_info_save_path_ceph, [all_db_infos])
        else:
            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)

    def create_pseudo_database_of_single_scene(self, info_with_idx, save_path, database_save_path,
                used_classes=None, total_samples=0, use_ceph=False):
        info, info_idx = info_with_idx
        print('gt_database sample: %d/%d' % (info_idx, total_samples))

        all_db_infos = {}
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id)
        annos = info['annos']
        names = annos['name']
        gt_boxes = annos['boxes_3d']
        score = annos['score']
        
        # filter
        if info_idx % 4 != 0 and len(names) > 0:
            mask = (names == 'Car')
            names = names[~mask]
            gt_boxes = gt_boxes[~mask]
            score = score[~mask]

        if info_idx % 2 != 0 and len(names) > 0:
            mask = (names == 'Bicycle')
            names = names[~mask]
            gt_boxes = gt_boxes[~mask]
            score = score[~mask]

        # filter
        filter_mask = score >= 0.5
        names = names[filter_mask]
        gt_boxes = gt_boxes[filter_mask]
        score  = score[filter_mask]

        num_obj = gt_boxes.shape[0]
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()  # (nboxes, npoints)

        for i in range(num_obj):
            filename = '%s_%s_%d.bin' % (frame_id, names[i], i)
            filepath = database_save_path / filename
            gt_points = points[point_indices[i] > 0]

            gt_points[:, :3] -= gt_boxes[i, :3]
            if use_ceph:
                filepath_list = str(filepath).split('/')
                filepath_ceph = os.path.join(os.path.join(*self.ceph_path_save.split('/')[:-1]), filepath_list[-3], filepath_list[-2], filepath_list[-1]).replace('s3:/', 's3://')
                self.petrel_client.save_bin(filepath_ceph, gt_points)
            else:
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

            db_path = str(filepath.relative_to(save_path))  # gt_database/xxxxx.bin
            db_info = {'name': names[i], 'path': db_path, 'gt_idx': i, 'score': score[i],
                        'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
            if names[i] in all_db_infos:
                all_db_infos[names[i]].append(db_info)
            else:
                all_db_infos[names[i]] = [db_info]
        return all_db_infos

    def get_pseudo_infos(self, info_path, result_path='', sampled_interval=1, use_ceph=False):
        print('---------------The ONCE sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_seq_list)))
        
        info_path_list = str(info_path).split('/')
        info_path_ceph = os.path.join(os.path.join(*self.ceph_path_save.split('/')[:-1]), *info_path_list[-2:]).replace('s3:/', 's3://')
        if use_ceph and self.petrel_client.isfile(info_path_ceph):
            infos = self.petrel_client.load_pkl(info_path_ceph, 1)[0]
            return infos

        total_info = []
        infer_result = pickle.load(open(result_path, 'rb'))
        for idx in tqdm(range(0, len(self.once_infos), sampled_interval)):
            self.once_infos[idx]['annos'] = dict()
            self.once_infos[idx]['annos']['name'] = infer_result[idx]['name']
            self.once_infos[idx]['annos']['score'] = infer_result[idx]['score']
            self.once_infos[idx]['annos']['boxes_3d'] = infer_result[idx]['boxes_3d'][:, :7]
            assert self.once_infos[idx]['frame_id'] == infer_result[idx]['frame_id']
            total_info.append(self.once_infos[idx])
        
        if use_ceph:
            info_path_list = str(info_path).split('/')
            info_path_ceph = os.path.join(os.path.join(*self.ceph_path_save.split('/')[:-1]), *info_path_list[-2:]).replace('s3:/', 's3://')
            self.petrel_client.save_pkl(info_path_ceph, [total_info])
        else:
            with open(info_path, 'wb') as f:
                pickle.dump(total_info, f)

        return total_info

def create_once_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = ONCEDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    splits = ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']
    ignore = ['test']

    print('---------------Start to generate data infos---------------')
    for split in splits:
        if split in ignore:
            continue

        filename = 'once_infos_%s.pkl' % split
        filename = save_path / Path(filename)
        dataset.set_split(split)
        once_infos = dataset.get_infos(num_workers=workers)
        with open(filename, 'wb') as f:
            pickle.dump(once_infos, f)
        print('ONCE info %s file is saved to %s' % (split, filename))

    train_filename = save_path / 'once_infos_train.pkl'
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split('train')
    dataset.create_groundtruth_database(train_filename, split='train')
    print('---------------Data preparation Done---------------')

def create_once_infos_upart(dataset_cfg, class_names, data_path, save_path, use_ceph=False, split='raw_small', result_path='', 
                            workers=min(16, multiprocessing.cpu_count())):
    dataset = ONCEDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    print('---------------Start to load result---------------')
    dataset.set_split(split)
    dataset.once_infos = []
    dataset.include_once_data(split)
    info_path = save_path / ('once_infos_%s.pkl' % split)
    infos = dataset.get_pseudo_infos(
        info_path=info_path,
        result_path=result_path,
        sampled_interval=1 if split!='raw_large' else 2,
        use_ceph=use_ceph
    )

    # print('---------------Start create groundtruth database for data augmentation---------------')
    # dataset.create_pseudo_database(infos, split=split, workers=workers, use_ceph=use_ceph, save_path=save_path)
    # print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--runs_on', type=str, default='server', help='')
    parser.add_argument('--use_ceph', action='store_true', default=False)
    parser.add_argument('--split', type=str, default='raw_small', help='')
    parser.add_argument('--result_path', type=str, default='')
    parser.add_argument('--post_fix', type=str, default='')
    args = parser.parse_args()

    if args.func == 'create_once_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)

        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        once_data_path = ROOT_DIR / 'data' / 'once'
        once_save_path = ROOT_DIR / 'data' / 'once'

        if args.runs_on == 'cloud':
            once_data_path = Path('/cache/once/')
            once_save_path = Path('/cache/once/')
            dataset_cfg.DATA_PATH = dataset_cfg.CLOUD_DATA_PATH

        create_once_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Bus', 'Truck', 'Pedestrian', 'Bicycle'],
            data_path=once_data_path,
            save_path=once_save_path
        )
    elif args.func == 'create_once_infos_upart':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)

        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        once_data_path = ROOT_DIR / 'data' / 'once'
        if args.post_fix == '':
            once_save_path = ROOT_DIR / 'data' / ('once_%s' % (args.split))
        else:
            once_save_path = ROOT_DIR / 'data' / ('once_%s_%s' % (args.split, args.post_fix))

        if args.runs_on == 'cloud':
            once_data_path = Path('/cache/once/')
            once_save_path = Path('/cache/once/')
            dataset_cfg.DATA_PATH = dataset_cfg.CLOUD_DATA_PATH

        create_once_infos_upart(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Bus', 'Truck', 'Pedestrian', 'Bicycle'],
            data_path=once_data_path,
            save_path=once_save_path,
            use_ceph=args.use_ceph,
            split=args.split,
            result_path=args.result_path,
        )