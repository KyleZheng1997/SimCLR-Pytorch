# SimCLR-Pytorch
Pytorch Implementation for SimCLR (Verified on Imagenet)

We found that using the SimCLR augmentation directly will sometimes cause the model to collapse, this may be because the SimCLR augmentation is too strong. Hence, we adopt the augmentation from MoCoV2 (same with SimCLR, but a little bit weaker) during the warmup stage and then change to SimCLR augmentation after warmup.

This code has been verified on Imagenet, and get higher performance than the result reported in the original paper. With apex(fp16), we train the model on 32 V100 GPU, it roughly takes 7 to 8 mins per epoch.

To run the code, please change the Dataset setting (dataset/Imagenet.py), and Pytorch DDP setting (util/dist_init.py) for your own server enviroments.

Results on Imagenet:
|          |Arch | BatchSize | Epochs | Accuracy (our reproduce)| Accuracy (paper) |
|----------|:----:|:---:|:---:|:---:|:---:|
|  SimCRL | ResNet50 | 4096 | 100  |  65.6 % | 64.5 % |
|  SimCRL | ResNet50 | 4096 | 200  |  67.8 % | 66.8 % |




