conda create -n PCSA python=3.6
conda activate PCSA
conda install pytorch=1.1.0 torchvision -c pytorch
pip install tensorboardX tqdm Pillow==6.2.2
pip install git+https://github.com/pytorch/tnt.git@master
cd Models/PCSA
python setup.py build develop
