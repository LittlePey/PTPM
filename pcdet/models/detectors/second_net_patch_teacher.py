from .detector3d_template import Detector3DTemplate
from ...utils import box_utils, common_utils
from ..model_utils import model_nms_utils
import numpy as np
import torch
import pdb

class SECONDNetPatchTeacher(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.patch_num = 16
        self.point_cloud_range = model_cfg.POINT_CLOUD_RANGE

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            batch_pred_boxes = []
            batch_pred_scores = []
            batch_pred_labels = []
            batch_points = []

            batch_dict_new = {'gt_boxes' : batch_dict['gt_boxes_origin']} if 'gt_boxes_origin' in batch_dict.keys() else {}
            recall_dict_new = {}
            pred_dicts_new = []

            for i, pred_dict in enumerate(pred_dicts):
                pred_dict['pred_boxes'] = pred_dict['pred_boxes'].cpu().numpy()
                batch_points_ = batch_dict['points'][batch_dict['points'][:,0] == i].cpu().numpy()[:, 1:]

                box_mask = (pred_dict['pred_boxes'][:, 0] >=0) & (pred_dict['pred_boxes'][:, 1] >=0) & \
                    (pred_dict['pred_boxes'][:, 0] <=self.point_cloud_range[3] + self.point_cloud_range[0]) & \
                    (pred_dict['pred_boxes'][:, 1] <=self.point_cloud_range[4] + self.point_cloud_range[1])
                point_mask =  (batch_points_[:, 0] >=0) & (batch_points_[:, 1] >=0) & \
                    (batch_points_[:, 0] <=self.point_cloud_range[3] + self.point_cloud_range[0]) & \
                    (batch_points_[:, 1] <=self.point_cloud_range[4] + self.point_cloud_range[1])

                rotation = ((i % self.patch_num) // 4) * np.pi / 2
                if i % 4 == 0:
                    shift_x, shift_y = self.point_cloud_range[3] + self.point_cloud_range[0], self.point_cloud_range[4] + self.point_cloud_range[1]
                if i % 4 == 1:
                    shift_x, shift_y = self.point_cloud_range[3] + self.point_cloud_range[0], 0.0
                if i % 4 == 2:
                    shift_x, shift_y = 0.0, 0.0
                if i % 4 == 3:
                    shift_x, shift_y = 0.0, self.point_cloud_range[4] + self.point_cloud_range[1]

                pred_dict['pred_boxes'][:, 0] += shift_x
                pred_dict['pred_boxes'][:, 1] += shift_y
                pred_dict['pred_boxes'] = common_utils.rotate_points_along_z(pred_dict['pred_boxes'][np.newaxis, :, :], np.array([-rotation]))[0]
                pred_dict['pred_boxes'][:, 6] -=rotation
                batch_points_[:, 0] += shift_x
                batch_points_[:, 1] += shift_y
                batch_points_ = common_utils.rotate_points_along_z(batch_points_[np.newaxis, :, :], np.array([-rotation]))[0]

                pred_dict['pred_boxes'] = torch.from_numpy(pred_dict['pred_boxes']).to(pred_dict['pred_scores'])
                batch_pred_boxes.append(pred_dict['pred_boxes'][box_mask])
                batch_pred_scores.append(pred_dict['pred_scores'][box_mask])
                batch_pred_labels.append(pred_dict['pred_labels'][box_mask])
                batch_points.append(batch_points_[point_mask])

                if (i + 1) % self.patch_num == 0:
                    batch_pred_boxes = torch.cat(batch_pred_boxes, dim=0)
                    batch_pred_scores = torch.cat(batch_pred_scores, dim=0)
                    batch_pred_labels = torch.cat(batch_pred_labels, dim=0)
                    batch_points = np.concatenate(batch_points, axis=0)

                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=batch_pred_scores, box_preds=batch_pred_boxes,
                        nms_config=self.model_cfg.POST_PROCESSING.NMS_CONFIG,
                        score_thresh=self.model_cfg.POST_PROCESSING.SCORE_THRESH
                    )

                    recall_dict_new = self.generate_recall_record(
                        box_preds=batch_pred_boxes if 'rois' not in batch_dict_new else src_box_preds,
                        recall_dict=recall_dict_new, batch_index=i // self.patch_num, data_dict=batch_dict_new,
                        thresh_list=self.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST
                    )

                    pred_dict_new = {
                        'pred_boxes': batch_pred_boxes[selected],
                        'pred_scores': selected_scores,
                        'pred_labels': batch_pred_labels[selected],
                        'batch_points': batch_points,
                    }
                    if 'gt_boxes' in batch_dict_new.keys():
                        pred_dict_new['batch_gt_boxes'] = batch_dict_new['gt_boxes'][i // self.patch_num],

                    pred_dicts_new.append(pred_dict_new)

                    batch_pred_boxes = []
                    batch_pred_scores = []
                    batch_pred_labels = []
                    batch_points = []

            return pred_dicts_new, recall_dict_new

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
