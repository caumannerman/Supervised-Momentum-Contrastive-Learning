## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">

  <img src="https://user-images.githubusercontent.com/28627878/146727551-e659df13-e596-440f-bc4d-90b4fb9c13b8.png" width = "300">
</p>
This is a PyTorch implementation of the 

[MoCo paper](https://arxiv.org/abs/1911.05722) + [SupCon paper](https://arxiv.org/abs/2004.11362):

```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```

```
@Article{Khosla SupCon,
  author  = {Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan},
  title   = {Supervised Contrastive Learning},
  journal = {arXiv preprint arXiv:2004.11362v5},
  year    = {2020},
}
```


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on that code. Check the modifications by:
```
diff main_moco.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
# #작성 논문 


<img width="673" alt="스크린샷 2022-02-17 오후 4 10 20" src="https://user-images.githubusercontent.com/75043852/154423949-5ecacd0a-fb98-4be3-a562-d034ab07f36b.png">

<img width="674" alt="스크린샷 2022-02-17 오후 4 10 35" src="https://user-images.githubusercontent.com/75043852/154423965-bf6665c6-7ee3-4631-9411-4f4f9b3ee7c5.png">

<img width="674" alt="스크린샷 2022-02-17 오후 4 10 49" src="https://user-images.githubusercontent.com/75043852/154423977-4c7f16b0-c680-46bd-be6e-fcf3a0e98675.png">

<img width="675" alt="스크린샷 2022-02-17 오후 4 11 01" src="https://user-images.githubusercontent.com/75043852/154423983-c25c04a0-b701-4037-bc1e-e6e91092ad86.png">
