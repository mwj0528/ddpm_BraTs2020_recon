# DDPM BraTs2020 T1 Reconstruction

## Installation
Implementation is conducted on Python 3.10. To install the environment, please run the following.
```
conda create -n ddpm python=3.10
conda activate ddpm
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy
pip install tqdm
pip install tensorboard
pip install h5py
pip install matplotlib
```
## Run

I use a single NVIDIA TITAN RTX GPU for our experiments.

### Train
```
python main.py --mode train
```
### Sample
```
python main.py --mode sample
```

## Example Result

![sample_epoch100](https://github.com/user-attachments/assets/a6ddbf65-40e0-4630-aea4-7811635c96dd)
![generated_0](https://github.com/user-attachments/assets/16548e1a-ae93-410b-a434-18ebc6465420)
