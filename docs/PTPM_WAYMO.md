```
cd PTPM/tools/
```
### Stage 1: PatchTeacher Semi-Supervised Learning
#### 1.Train PatchTeacher with Labeled Data
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh ./scripts/dist_train.sh 4 \
--cfg_file cfgs/ptpm/waymo_0.05/patchteacher/second_0.05_tea.yaml

Comment Line 11 and uncomment Line 12 of the config before running the following scripts:
CUDA_VISIBLE_DEVICES=0,1,2,3 sh ./scripts/dist_test.sh 4 \
--cfg_file cfgs/ptpm/waymo_0.05/patchteacher/second_0.05_tea.yaml --batch_size 8 \
--ckpt ../output/ptpm/waymo_0.05/patchteacher/second_0.05_tea/default/ckpt/checkpoint_epoch_120.pth

python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos_upart \
--cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml  --ratio 0.05  --pseudo_label  --post_fix round1 \
--result_path output/ptpm/waymo_0.05/patchteacher/second_0.05_tea/default/eval/epoch_120/unlabel/default/result.pkl
```

#### 2.Train PatchTeacher with Labeled Data and Pseudo Label Generated by itself
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh ./scripts/dist_train.sh 4 \
--cfg_file cfgs/ptpm/waymo_0.05/patchteacher/second_0.05_ssl_tea.yaml \
--pretrained_model ../output/ptpm/waymo_0.05/patchteacher/second_0.05_tea/default/ckpt/checkpoint_epoch_120.pth

Comment Line 12 and uncomment Line 13 of the config before running the following scripts:
CUDA_VISIBLE_DEVICES=0,1,2,3 sh ./scripts/dist_test.sh 4 \
--cfg_file cfgs/ptpm/waymo_0.05/patchteacher/second_0.05_ssl_tea.yaml --batch_size 16 \
--ckpt ../output/ptpm/waymo_0.05/patchteacher/second_0.05_ssl_tea/default/ckpt/checkpoint_epoch_30.pth

python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos_upart \
--cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml  --ratio 0.05  --pseudo_label  --post_fix round2 \
--result_path output/ptpm/waymo_0.05/patchteacher/second_0.05_ssl_tea/default/eval/epoch_30/unlabel/default/result.pkl
```

### Stage 2: Student Semi-Supervised Learning
#### 1.Train Student with Labeled Data
```
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos_lpart \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml  --ratio 0.05

CUDA_VISIBLE_DEVICES=0,1,2,3 sh ./scripts/dist_train.sh 4 \
--cfg_file cfgs/ptpm/waymo_0.05/second_0.05.yaml

Comment Line 8 and uncomment Line 9 of the config before running the following scripts:
CUDA_VISIBLE_DEVICES=0,1,2,3 sh ./scripts/dist_test.sh 4 \
--cfg_file cfgs/ptpm/waymo_0.05/second_0.05.yaml --batch_size 32 \
--ckpt ../output/ptpm/waymo_0.05/second_0.05/default/ckpt/checkpoint_epoch_35.pth

python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos_upart \
--cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml  --ratio 0.05  --pseudo_label \
--result_path output/ptpm/waymo_0.05/second_0.05/default/eval/epoch_35/unlabel/default/result.pkl
```

#### 2.Train Student with Labeled Data and Pseudo Label Generated by itself
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh ./scripts/dist_train.sh 4 \
--cfg_file cfgs/ptpm/waymo_0.05/pillarmix/second_0.05_ssl_stu_stu.yaml \
--pretrained_model ../output/ptpm/waymo_0.05/second_0.05/default/ckpt/checkpoint_epoch_35.pth
```

#### 3.Train Student with Labeled Data and Pseudo Label Generated by PatchTeacher
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh ./scripts/dist_train.sh 4 \
--cfg_file cfgs/ptpm/waymo_0.05/pillarmix/second_0.05_ssl_stu_tea.yaml \
--pretrained_model ../output/ptpm/waymo_0.05/pillarmix/second_0.05_ssl_stu_stu/default/ckpt/checkpoint_epoch_30.pth
```