# bag-of-tricks-for-classification-pytorch
bag of tricks for image classification tutorials using pytorch. Based on ["Bag of Tricks for Image Classification with Convolutional Neural Networks", 2019 CVPR Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf), implement classification codebase using custom dataset.

- author: hoya012  
- last update: 2020.06.22
- [supplementary materials (blog post written in Korean)](https://hoya012.github.io/blog/Bag-of-Tricks-for-Image-Classification-with-Convolutional-Neural-Networks-Review/)

## 0. Experimental Setup

### 0-1. Prepare Library

```python
pip install -r requirements.txt
```

### 0-2. Download dataset (Kaggle Intel Image Classification)

- [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification/)

This Data contains around 25k images of size 150x150 distributed under 6 categories.
{'buildings' -> 0,
'forest' -> 1,
'glacier' -> 2,
'mountain' -> 3,
'sea' -> 4,
'street' -> 5 }

### 1. Baseline Training Setting
- ImageNet Pratrained ResNet-50 from torchvision.models
- 1080 Ti 1 GPU / Batch Size 64 / Epochs 120
- Training Augmentation: Resize((256, 256)), RandomCrop(224, 224), RandomHorizontalFlip(), RandomVerticalFlip(), Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
- SGD + Momentum(0.9) + learning rate step decay (x0.1 at 30, 60, 90 epoch)

### 2. Bag of Tricks from Original Papers
#### 2-1. Learning Rate Warmup 
- first 5 epochs to warmup

#### 2-2. Zero gamma in Batch Normalization
- zero-initialize the last BN in each residual branch

#### 2-3. Cosine Learning Rate Annealing
![](assets/cosine_warmup.PNG)

#### 2-4. Label Smoothing

#### 2-5. MixUp Augmentation

### 3. Additional Tricks from hoya012's survey note
#### 3-1. CutMix Augmentation

#### 3-2. Adam Optimizer, LARS Optimizer

#### 3-3. RandAugment

#### 3-4. EvoNorm

#### 3-5. Other Architecture (EfficientNet, RegNet)

## Code Reference
- GradualWarmupScheduler: https://github.com/ildoonet/pytorch-gradual-warmup-lr
- Label Smoothing: https://github.com/pytorch/pytorch/issues/7455
- MixUp Augmentation:
- CutMix Augmentation:https://github.com/clovaai/CutMix-PyTorch
- LARS Optimizer: https://github.com/kakaobrain/torchlars
- RandAugment: https://github.com/ildoonet/pytorch-randaugment
- EvoNorm: https://github.com/digantamisra98/EvoNorm
- ImageNet-Pretrained EfficientNet, RegNet: https://github.com/facebookresearch/pycls
