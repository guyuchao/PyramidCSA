import os
from torch.utils.data import Dataset,DataLoader
from Data.preprocess import *
from PIL import Image
import torch
from config import config
import glob
##pretrain_dataset
class Pretrain(Dataset):
    def __init__(self,img_dataset_list,video_dataset_list,transform):
        self.file_list=[]
        for dataset in img_dataset_list:
            listfile=os.path.join(config.img_dataset_root,dataset+'.lst')
            with open(listfile,"r") as f:
                file_list=f.readlines()
            file_list=[filename.strip() for filename in file_list]
            self.file_list.extend([(os.path.join(config.img_dataset_root,filename.split(' ')[0][1:]),os.path.join(config.img_dataset_root,filename.split(' ')[1][1:])) for filename in file_list])

        for dataset in video_dataset_list:
            path=os.path.join(config.video_dataset_root,dataset)
            video_list=glob.glob(path+"/**/*.jpg",recursive=True)
            self.file_list.extend([(filepath,filepath.replace("Imgs","ground-truth").replace("jpg","png")) for filepath in video_list])

        self.img_label_transform = transform

    def __getitem__(self, idx):
        img_path, label_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        img, label = self._process(img, label)
        return img,label

    def _process(self, img, label):
        img, label = self.img_label_transform(img, label)
        return img, label

    def __len__(self):
        return len(self.file_list)

def get_pretrain_loader():
    statistics = torch.load(config.data_statistics)
    trsf_main=Compose_imglabel([
        Resize(config.size[0],config.size[1]),
        Random_crop_Resize(15),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"],statistics["std"])
    ])
    trsf_scale1 = Compose_imglabel([
        Resize(int(config.size[0] * 1.5), int(config.size[1] * 1.5)),
        # ColorAug(),
        Random_crop_Resize(50),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])
    trsf_scale2 = Compose_imglabel([
        Resize(int(config.size[0] * 1.25), int(config.size[1] * 1.25)),
        Random_crop_Resize(25),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])
    trsf_scale3 = Compose_imglabel([
        Resize(int(config.size[0] * 0.75), int(config.size[1] * 0.75)),
        Random_crop_Resize(15),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])

    trsf_scale4 = Compose_imglabel([
        Resize(int(config.size[0] * 0.5), int(config.size[1] * 0.5)),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])
    train_loader=DataLoader(Pretrain(config.img_dataset_list,config.video_dataset_list,transform=trsf_main),batch_size=config.pretrain_batchsize,shuffle=True,drop_last=False,num_workers=8)
    multicale_loader=[
        DataLoader(Pretrain(config.img_dataset_list, config.video_dataset_list, transform=trsf_scale1),
                   batch_size=config.pretrain_batchsize, shuffle=True, drop_last=False,num_workers=8),
        DataLoader(Pretrain(config.img_dataset_list, config.video_dataset_list, transform=trsf_scale2),
                   batch_size=config.pretrain_batchsize, shuffle=True, drop_last=False,num_workers=8),
        DataLoader(Pretrain(config.img_dataset_list, config.video_dataset_list, transform=trsf_scale3),
                   batch_size=config.pretrain_batchsize, shuffle=True, drop_last=False,num_workers=8),
        DataLoader(Pretrain(config.img_dataset_list, config.video_dataset_list, transform=trsf_scale4),
                   batch_size=config.pretrain_batchsize, shuffle=True, drop_last=False,num_workers=8)
    ]
    return train_loader,multicale_loader,statistics

def get_pretrain_dataset():
    statistics = torch.load(config.data_statistics)
    trsf_main=Compose_imglabel([
        Resize(config.size[0],config.size[1]),
        Random_crop_Resize(15),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"],statistics["std"])
    ])
    trsf_scale1 = Compose_imglabel([
        Resize(int(config.size[0] * 1.5), int(config.size[1] * 1.5)),
        # ColorAug(),
        Random_crop_Resize(50),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])
    trsf_scale2 = Compose_imglabel([
        Resize(int(config.size[0] * 1.25), int(config.size[1] * 1.25)),
        Random_crop_Resize(25),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])
    trsf_scale3 = Compose_imglabel([
        Resize(int(config.size[0] * 0.75), int(config.size[1] * 0.75)),
        Random_crop_Resize(15),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])

    trsf_scale4 = Compose_imglabel([
        Resize(int(config.size[0] * 0.5), int(config.size[1] * 0.5)),
        Random_horizontal_flip(0.5),
        toTensor(),
        Normalize(statistics["mean"], statistics["std"])
    ])
    train_loader=Pretrain(config.img_dataset_list,config.video_dataset_list,transform=trsf_main)
    multicale_loader=[
        Pretrain(config.img_dataset_list, config.video_dataset_list, transform=trsf_scale1),
        Pretrain(config.img_dataset_list, config.video_dataset_list, transform=trsf_scale2),
        Pretrain(config.img_dataset_list, config.video_dataset_list, transform=trsf_scale3),
        Pretrain(config.img_dataset_list, config.video_dataset_list, transform=trsf_scale4)
    ]
    return train_loader,multicale_loader,statistics

class VideoDataset(Dataset):
    def __init__(self,video_dataset_list,transform=None,time_interval=1):
        super(VideoDataset, self).__init__()
        self.video_filelist=video_dataset_list
        self.time_clips=config.video_time_clips
        self.video_train_list = []

        for video_name in video_dataset_list:
            video_root=os.path.join(config.video_dataset_root,video_name)
            cls_list=os.listdir(video_root)
            self.video_filelist={}
            for cls in cls_list:
                self.video_filelist[cls]=[]
                cls_path=os.path.join(video_root,cls)
                cls_img_path=os.path.join(cls_path,"Imgs")
                cls_label_path=os.path.join(cls_path,"ground-truth")
                tmp_list=os.listdir(cls_img_path)
                tmp_list.sort()
                for filename in tmp_list:
                    self.video_filelist[cls].append((
                        os.path.join(cls_img_path,filename),
                        os.path.join(cls_label_path,filename.replace(".jpg",".png"))
                    ))
            #emsemble
            for cls in cls_list:
                li=self.video_filelist[cls]
                for begin in range(len(li)-(self.time_clips-1)*time_interval):
                    batch_clips=[]
                    for t in range(self.time_clips):
                        batch_clips.append(li[begin+time_interval*t])
                    self.video_train_list.append(batch_clips)
            self.img_label_transform=transform

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li=[]
        label_li=[]
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img_li.append(img)
            label_li.append(label)
        img_li,label_li=self.img_label_transform(img_li,label_li)
        for idx,(img,label) in enumerate(zip(img_li,label_li)):
            if IMG is not None:
                IMG[idx,:,:,:]=img
                LABEL[idx,:,:,:]=label
            else:
                IMG=torch.zeros(len(img_li),*(img.shape))
                LABEL=torch.zeros(len(img_li),*(label.shape))
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
        return IMG, LABEL

    def __len__(self):
        return len(self.video_train_list)

def get_video_loader():
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])

    trsf_scale1 = Compose_imglabel([
        Resize_video(int(config.size[0]), int(config.size[1])),
        Random_crop_Resize_Video(22),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])

    train_loader = DataLoader(VideoDataset(config.video_dataset_list, transform=trsf_main, time_interval=1),
                              batch_size=config.video_batchsize, shuffle=True, num_workers=8)
    multiscale_loader = [
        DataLoader(VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=1),
                   batch_size=config.video_batchsize,
                   shuffle=True, num_workers=8),
        DataLoader(VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=2),
                   batch_size=config.video_batchsize,
                   shuffle=True, num_workers=8),
        DataLoader(VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=3),
                   batch_size=config.video_batchsize,
                   shuffle=True, num_workers=8),
        DataLoader(VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=4),
                   batch_size=config.video_batchsize,
                   shuffle=True, num_workers=8),
        DataLoader(VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=5),
                   batch_size=config.video_batchsize,
                   shuffle=True, num_workers=8),
        DataLoader(VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=6),
                   batch_size=config.video_batchsize,
                   shuffle=True, num_workers=8)]
    return train_loader,multiscale_loader,statistics

def get_video_dataset():
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])

    trsf_scale1 = Compose_imglabel([
        Resize_video(int(config.size[0]), int(config.size[1])),
        Random_crop_Resize_Video(22),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])

    train_loader = VideoDataset(config.video_dataset_list, transform=trsf_main, time_interval=1)
    multiscale_loader = [
        VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=1),
        VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=2),
        VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=3),
        VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=4),
        VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=5),
        VideoDataset(config.video_dataset_list, transform=trsf_scale1, time_interval=6)]
    return train_loader,multiscale_loader,statistics

if __name__=="__main__":
   pass