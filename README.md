
#  🎇 LGFIN 🎇

LGFIN: A Dual-Branch Local-Global Feature Integration Network
for Medical Image Classification



![LGFIN整体架构](https://github.com/user-attachments/assets/9668152d-71b1-41df-a522-ec176ef7ef96)


# 📌Installation📌
* `pip install packaging`
* `pip install timm==0.4.12`
* `pip install pytest chardet yacs termcolor`
* `pip install submitit tensorboardX`
* `pip install triton==2.0.0`
* `pip install addict==2.4.0`
* `pip install dataclasses`
* `pip install pyyaml`
* `pip install albumentations`
* `pip install tensorboardX`
* `pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs`
* `pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117`


## 📜Other requirements📜
* Linux System
* NVIDIA GPU
* CUDA 12.0+


# 📊Datasets📊
The dataset format is as follows：
```
├── datasets
│   ├── Brain
│     ├── class1
│           ├── 1.png
|           ├── ...
|     ├── class2
│           ├── 1.png
|           ├── ...
│   ├── Brain MRI
│     ├── class1
│           ├── 1.png
|           ├── ...
|     ├── class2
│           ├── 1.png
|           ├── ...
│     ├── class3
│           ├── 1.png
|           ├── ...
|     ├── class4
│           ├── 1.png
|           ├── ...
│   ├──...
```



## Train
To train the model, we used the PyTorch deep learning framework and selected the Adam optimizer to optimize the model parameters. Specifically, we used normalization for data preprocessing,and used CrossEntropyLoss to calculate the loss function. During training, the batch size was set to 16, and iterative training was performed in a training cycle of 100 epochs.
```
python train.py
```




## Test
The following metrics are used to evaluate the classification performance of the model: Accuracy, Precision, Recall, Specificity, and F1-Score

```
python test.py
```




