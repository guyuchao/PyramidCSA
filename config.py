import argparse
parser = argparse.ArgumentParser()
#optimizer
parser.add_argument('--lr_mode', type=str, default="poly")
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--finetune_lr', type=float, default=1e-6)

#train schedule
parser.add_argument('--pretrain_epoches', type=int, default=15)
parser.add_argument('--finetune_epoches', type=int, default=15)
parser.add_argument('--log_inteval', type=int, default=50)


##data
parser.add_argument('--data_statistics', type=str, default="Data/['DUTS-TR']_statistics.pth")
parser.add_argument('--img_dataset_list', type=str, default=["DUTS-TR"])
parser.add_argument('--video_dataset_list', type=str, default=["DAVIS","DAVSOD"])
parser.add_argument('--img_dataset_root', type=str,default="/media/data/guyuchao/dataset/fastsaliency")
parser.add_argument('--video_dataset_root', type=str,default="/media/data/guyuchao/dataset/saliency/trainDataset")
parser.add_argument('--size', type=tuple,default=(256,448))
parser.add_argument('--pretrain_batchsize', type=int, default=24)
parser.add_argument('--video_batchsize', type=int, default=12)
parser.add_argument('--video_time_clips', type=int, default=5)
parser.add_argument('--video_testset_root', type=str,default="/media/data/guyuchao/dataset/saliency/testDataset")
parser.add_argument('--parallel', type=bool, default=True)
parser.add_argument('--device_idxs', type=list, default=[0,1,2,3])
parser.add_argument('--local_rank', type=int,default=0)

#pretrain
parser.add_argument('--pretrain_state_dict', type=str, default="/media/data/guyuchao/project/released/videofastsal/checkpoints/tensorboard/pretrain_baseline/epoch_15_batch_0/checkpoint.pth")
parser.add_argument('--backbone_imagenet_pretrain', type=str, default="Models/statedict/mobilenetv3-large.pth")

config = parser.parse_args()
