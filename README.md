# PyramidCSA

Code for "Pyramid Constrained Self-Attention Network for Fast Video Salient Object Detection" (AAAI 2020)

## Build

```bash
conda create -n PCSA python=3.6
conda activate PCSA
conda install pytorch=1.1.0 torchvision -c pytorch
pip install tensorboardX tqdm Pillow==6.2.2
pip install git+https://github.com/pytorch/tnt.git@master
cd Models/PCSA
python setup.py build develop
```

## Training

### pretrain phase
```bash
bash pretrain.sh
```
### finetune phase
```bash
bash finetune.sh
```

## Results
The result saliency map can be downloaded [here](https://pan.baidu.com/s/1bktiBwBUprIpfstK9fDehg) (password t781).

## Evaluation
For VSOD, we use the evaluation code provided by [DAVSOD](https://github.com/DengPingFan/DAVSOD).

For UVOS, we use the evaluation code provided by [Davis16](https://github.com/fperazzi/davis).

## Speed Evaluation
```python3
python speed.py
```

## Cite
If you think this work is helpful, please cite
```latex
@inproceedings{gu2020PCSA,
 title={Pyramid Constrained Self-Attention Network for Fast Video Salient Object Detection},
 author={Gu, Yuchao and Wang, Lijuan and Wang, Ziqin and Liu, Yun and Cheng, Ming-Ming and Lu, Shao-Ping},
 booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
 year={2020},
}
```

## Related Project
The feature extraction backbone is borrowed from [d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)

## Concat
Any questions and suggestions, please email [ycgu@mail.nankai.edu.cn](ycgu@mail.nankai.edu.cn).
