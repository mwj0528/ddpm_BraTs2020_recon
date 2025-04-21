# DDPM BraTs2020 T1 Reconstruction

## Installation
Our implementation is conducted on Python 3.8. To install the environment, please run the following.
```
conda create -n ddpm python=3.10
conda activate ddpm
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy
pip install tqdm
pip install tensorboard
pip install matplotlib
```
## Run

We use a single NVIDIA TITAN RTX GPU for our experiments.
Train
```
python main.py --mode train
```
Sample
```
python main.py --mode sample
```
