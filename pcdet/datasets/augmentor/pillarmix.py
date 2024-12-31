import numpy as np
import pdb

def pillarmix(input_dict, input_dict_queue, mix_ratio=0.5, bin_size=5.0, strategy='random'):
    side = int((2 * 80) // bin_size)
    partition =side ** 2
    partition_shuffle = np.random.permutation(partition)

    if len(input_dict_queue) == 1:

        ### points
        points = input_dict['points']
        points2 = input_dict_queue[0]['points']

        gt_boxes = input_dict['gt_boxes']
        gt_boxes2 = input_dict_queue[0]['gt_boxes']

        mix_ratio = 1.0 / (1 + len(input_dict_queue))

        ### coords
        coords = (points[:, :2] + 80) // bin_size
        coords2 = (points2[:, :2] + 80) // bin_size

        centers = (gt_boxes[:, :2] + 80) // bin_size
        centers2 = (gt_boxes2[:, :2] + 80) // bin_size

        ### mask
        mask = np.zeros(len(points), dtype=np.bool)
        mask2 = np.zeros(len(points2), dtype=np.bool)

        mask_box = np.zeros(len(gt_boxes), dtype=np.bool)
        mask_box2 = np.zeros(len(gt_boxes2), dtype=np.bool)

        for i in range(partition):
            coord_x = partition_shuffle[i] // side
            coord_y = partition_shuffle[i] % side
            if strategy == 'random':
                if i < partition * mix_ratio:
                    mask = mask | ((coords[:, 0] == coord_x ) & (coords[:, 1] == coord_y ))
                    mask_box = mask_box | ((centers[:, 0] == coord_x ) & (centers[:, 1] == coord_y ))
                elif i < partition * mix_ratio * 2:
                    mask2 = mask2 | ((coords2[:, 0] == coord_x ) & (coords2[:, 1] == coord_y ))
                    mask_box2 = mask_box2 | ((centers2[:, 0] == coord_x ) & (centers2[:, 1] == coord_y ))
            elif strategy == 'cross':
                if ((coord_x % 2 ==0) & (coord_y % 2 ==0)) | ((coord_x % 2 !=0) & (coord_y % 2 !=0)):
                    mask = mask | ((coords[:, 0] == coord_x ) & (coords[:, 1] == coord_y ))
                    mask_box = mask_box | ((centers[:, 0] == coord_x ) & (centers[:, 1] == coord_y ))
                if ((coord_x % 2 !=0) & (coord_y % 2 ==0)) | ((coord_x % 2 ==0) & (coord_y % 2 !=0)):
                    mask2 = mask2 | ((coords2[:, 0] == coord_x ) & (coords2[:, 1] == coord_y ))
                    mask_box2 = mask_box2 | ((centers2[:, 0] == coord_x ) & (centers2[:, 1] == coord_y ))

        input_dict['points'] = np.concatenate([points[mask], points2[mask2]], axis=0)
        input_dict['gt_boxes'] = np.concatenate([input_dict['gt_boxes'][mask_box], input_dict_queue[0]['gt_boxes'][mask_box2]], axis=0)
        input_dict['gt_names'] = np.concatenate([input_dict['gt_names'][mask_box], input_dict_queue[0]['gt_names'][mask_box2]], axis=0)
        input_dict['gt_score'] = np.concatenate([input_dict['gt_score'][mask_box], input_dict_queue[0]['gt_score'][mask_box2]], axis=0)
        input_dict['num_points_in_gt'] = None

        return input_dict