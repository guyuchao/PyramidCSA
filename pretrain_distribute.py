from torch.optim import Adam
import torch
from utils.Distribute.engine import Engine
from Data.dataloader import get_pretrain_dataset
import torch.nn.functional as F
from utils.utils import get_Logger_and_SummaryWriter
import os
from config import config
from Models.mobilenetv3_pretrain import Fastnet
import torch.nn as nn
from utils.SalEval import SalEval
from torchnet.meter import AverageValueMeter
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, *inputs):
        pred,target = tuple(inputs)
        total_loss = F.binary_cross_entropy(pred, target.float())
        return total_loss

class TrainSchedule(object):
    def __init__(self,batches_per_epoch):
        self.cur_epoch = 0
        self.total_epoch=config.pretrain_epoches
        self.cur_batches = 0
        self.batches_per_epoch=batches_per_epoch

    def update(self):
        self.cur_batches+=1
        if not self.cur_batches<self.batches_per_epoch:
            self.cur_batches=0
            self.cur_epoch+=1

    def state_dict(self):
        state_dict={"cur_batches":self.cur_batches,
                    "cur_epoch":self.cur_epoch,
                    "total_epoch":self.total_epoch,
                    "batches_per_epoch":self.batches_per_epoch}
        return state_dict

    def load_state_dict(self,state_dict):
        self.cur_batches=state_dict["cur_batches"]
        self.cur_epoch=state_dict["cur_epoch"]
        self.total_epoch=state_dict["total_epoch"]
        self.batches_per_epoch=state_dict["batches_per_epoch"]

class Train(object):

    def __init__(self):
        self.logger,self.writer,self.tag_dir=get_Logger_and_SummaryWriter()
        self.engine=Engine(self.logger)
        self.device = torch.device("cuda")
        self.network=Fastnet().cuda()
        self.network = self.engine.data_parallel(self.network)

        self.criterion=CrossEntropyLoss().to(self.device)
        self.optim=Adam(self.network.parameters(),lr=config.base_lr)
        self.train_dataset,self.train_multiscale_dataset,statistics=get_pretrain_dataset()
        self.train_multiscale_loader=[]
        self.train_multiscale_smapler=[]
        for dst in self.train_multiscale_dataset:
            ld,sp=self.engine.get_train_loader(dst,config.pretrain_batchsize)
            self.train_multiscale_loader.append(ld)
            self.train_multiscale_smapler.append(sp)
        self.train_loader,self.train_sampler=self.engine.get_train_loader(self.train_dataset,config.pretrain_batchsize)

        batches_per_epoch=0
        batches_per_epoch+=len(self.train_loader)

        for loader in self.train_multiscale_loader:
            batches_per_epoch+=len(loader)

        self.sche=TrainSchedule(batches_per_epoch)
        if self.engine.local_rank==0:
            self.logger.info(self.network)
            total_paramters = sum([np.prod(p.size()) for p in self.network.parameters()])
            self.logger.info('Total network parameters: ' + str(total_paramters))
            self.logger.info(config)

    def save_checkpoint(self):
        os.makedirs(os.path.join(self.tag_dir, "epoch_%d_batch_%d" % (
        self.sche.cur_epoch,self.sche.cur_batches)), exist_ok=True)
        save_root = os.path.join(self.tag_dir, "epoch_%d_batch_%d" % (
        self.sche.cur_epoch,self.sche.cur_batches))
        torch.save(self.state_dict(), os.path.join(save_root, "checkpoint.pth"))

    def adjust_learning_rate(self):
        if config.lr_mode == 'poly':
            cur_iter = self.sche.batches_per_epoch * self.sche.cur_epoch + self.sche.cur_batches
            max_iter = self.sche.batches_per_epoch * self.sche.total_epoch
            base_lr = config.base_lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9

        for param_group in self.optim.param_groups:
            param_group['lr'] = base_lr

        return base_lr

    def train_per_loader(self,trainloader):
        self.network.train()
        loss_meter=AverageValueMeter()
        for idx,(img,label) in enumerate(trainloader):
            baselr=self.adjust_learning_rate()
            img=img.cuda(non_blocking=True)
            label=label.cuda(non_blocking=True)
            if len(label.shape)==5:
                label=label.view(-1,*(label.shape[2:]))
            output=self.network(img)
            loss=self.criterion(output,label)
            reduce_loss = self.engine.all_reduce_tensor(loss)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            loss_meter.add(float(reduce_loss))

            if self.engine.local_rank == 0:
                if self.sche.cur_batches % config.log_inteval == 0:
                    self.logger.info("%s-epoch:%d/%d batch:%d/%d loss:%.4f base_lr:%e" % (
                    self.tag_dir.split("/")[-1], self.sche.cur_epoch, self.sche.total_epoch, self.sche.cur_batches,
                    self.sche.batches_per_epoch,loss_meter.value()[0],baselr))
            self.sche.update()
        return loss_meter.value()[0]

    def train_per_epoch(self):
        for loader in self.train_multiscale_loader:
            self.train_per_loader(loader)
        loss_train=self.train_per_loader(self.train_loader)
        if self.engine.local_rank == 0:
            self.logger.info("train_img_loss:%.4f"%(loss_train))

    def train(self):
        while self.sche.cur_epoch < self.sche.total_epoch:

            self.train_sampler.set_epoch(self.sche.cur_epoch)
            for sp in self.train_multiscale_smapler:
                sp.set_epoch(self.sche.cur_epoch)
            self.train_per_epoch()

            if self.engine.local_rank == 0:
                self.save_checkpoint()

    def state_dict(self):
        if config.parallel is True:
            state_dict={"net":self.network.module.state_dict(),
                        'optimizer':self.optim.state_dict(),
                        'sche':self.sche.state_dict()}
        else:
            state_dict = {"net": self.network.state_dict(),
                          'optimizer': self.optim.state_dict(),
                          'sche': self.sche.state_dict()}
        return state_dict

    def load_state_dict(self,state_dict):
        self.sche.load_state_dict(state_dict["sche"])
        self.optim.load_state_dict(state_dict["optimizer"])
        self.network.load_state_dict(state_dict["net"])

if __name__=="__main__":
    torch.backends.cudnn.benchmark=True
    setup_seed(0)
    trainer=Train()
    trainer.train()

