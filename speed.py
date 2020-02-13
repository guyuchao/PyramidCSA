import numpy as np
import torch
import time

def computeTime(model, device='cuda'):
    inputs = torch.randn(1,1, 3, 256, 448)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Avg execution time (ms): %.4f, FPS:%d'%(np.mean(time_spent),1*1//np.mean(time_spent)))
    return 1*1//np.mean(time_spent)

if __name__=="__main__":

    torch.backends.cudnn.benchmark = True

    from Models import mobilenetv3temporal_PCSA as net
    model = net.Fastnet()

    computeTime(model)
