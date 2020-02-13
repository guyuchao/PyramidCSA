import os
from config import config
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler,DataLoader
import torch.nn as nn

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)
    return tensor

class Engine(object):

    ## distribute init
    def __init__(self, logger=None):
        self.distributed = False
        self.logger=logger
        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1
        else:
            raise NotImplementedError

        if self.distributed:
            self.local_rank = config.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
        else:
            raise NotImplementedError

    ## convert model

    def data_parallel(self, model):
        if self.distributed:
            #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model,device_ids=[self.local_rank],output_device=self.local_rank)
        else:
            raise NotImplementedError
        return model

    def get_train_loader(self, train_dataset,batchsize):
        if self.distributed:
            train_sampler = DistributedSampler(
                train_dataset)
            local_bs = batchsize // self.world_size
            is_shuffle = False
            train_loader = DataLoader(train_dataset,
                   batch_size=local_bs,
                   num_workers=2,
                   drop_last=False,
                   shuffle=is_shuffle,
                   pin_memory=False,
                   sampler=train_sampler)

        else:
            raise NotImplementedError

        return train_loader, train_sampler

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            raise NotImplementedError


    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            self.logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False
