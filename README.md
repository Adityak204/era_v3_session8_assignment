# CIFAR10 Image Classification
This project is a simple image classification project using the CIFAR10 dataset. The CIFAR10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

# CNN Model Architecture
- The model used in this project is a Convolutional Neural Network (CNN) model. 
- The model consists of 4 convolutional blocks, followed by Global Average Pooling and finally fully collected layer.
- The model consists following type of layers in different convolutional blocks:
    - Normal Conv2D layers
    - Conv2D layers with dilation
    - Depthwise Separable Conv2D layers
- The model uses ReLU activation function and Batch Normalization after each convolutional layer.
- Total number of parameters in the model is 111,674.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
       BatchNorm2d-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
            Conv2d-4           [-1, 32, 32, 32]           4,608
       BatchNorm2d-5           [-1, 32, 32, 32]              64
              ReLU-6           [-1, 32, 32, 32]               0
            Conv2d-7           [-1, 64, 32, 32]          18,432
       BatchNorm2d-8           [-1, 64, 32, 32]             128
              ReLU-9           [-1, 64, 32, 32]               0
           Conv2d-10           [-1, 16, 32, 32]           1,024
           Conv2d-11           [-1, 32, 32, 32]           4,608
      BatchNorm2d-12           [-1, 32, 32, 32]              64
             ReLU-13           [-1, 32, 32, 32]               0
           Conv2d-14           [-1, 32, 32, 32]           9,216
      BatchNorm2d-15           [-1, 32, 32, 32]              64
             ReLU-16           [-1, 32, 32, 32]               0
           Conv2d-17           [-1, 64, 30, 30]          18,432
      BatchNorm2d-18           [-1, 64, 30, 30]             128
             ReLU-19           [-1, 64, 30, 30]               0
           Conv2d-20           [-1, 16, 30, 30]           1,024
           Conv2d-21           [-1, 32, 30, 30]           4,608
      BatchNorm2d-22           [-1, 32, 30, 30]              64
             ReLU-23           [-1, 32, 30, 30]               0
           Conv2d-24           [-1, 32, 30, 30]           9,216
      BatchNorm2d-25           [-1, 32, 30, 30]              64
             ReLU-26           [-1, 32, 30, 30]               0
           Conv2d-27           [-1, 64, 26, 26]          18,432
      BatchNorm2d-28           [-1, 64, 26, 26]             128
             ReLU-29           [-1, 64, 26, 26]               0
           Conv2d-30           [-1, 16, 26, 26]           1,024
           Conv2d-31           [-1, 32, 26, 26]           4,608
      BatchNorm2d-32           [-1, 32, 26, 26]              64
             ReLU-33           [-1, 32, 26, 26]               0
           Conv2d-34           [-1, 32, 26, 26]           9,216
      BatchNorm2d-35           [-1, 32, 26, 26]              64
             ReLU-36           [-1, 32, 26, 26]               0
           Conv2d-37           [-1, 32, 26, 26]             288
           Conv2d-38          [-1, 128, 26, 26]           4,096
DepthwiseSeparableConv-39          [-1, 128, 26, 26]               0
      BatchNorm2d-40          [-1, 128, 26, 26]             256
             ReLU-41          [-1, 128, 26, 26]               0
AdaptiveAvgPool2d-42            [-1, 128, 1, 1]               0
           Linear-43                   [-1, 10]           1,290
================================================================
Total params: 111,674
Trainable params: 111,674
Non-trainable params: 0
----------------------------------------------------------------
```

# Training Highlights
- The model was trained for 60 epochs with a batch size of 64.
- The model was trained using the Adam optimizer with a learning rate scheduler.
- Albumentation library has been used for image augmentation. Augmentation techniques used are:
    - Horizontal Flip
    - ShiftScaleRotate
    - CoarseDropout
    - Normalize
- The model achieved a max accuracy of 85.23% on the test dataset.

## Training Logs
```
********* Epoch = 1 *********
loss=1.4951 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.30it/s]

Epoch 1: Train set: Average loss: 0.0279, Accuracy: 15447/50000 (30.89%)



Test set: Average loss: 0.0243, Accuracy: 4183/10000 (41.83%)

LR =  [0.01]
********* Epoch = 2 *********
loss=1.2240 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 36.79it/s]

Epoch 2: Train set: Average loss: 0.0228, Accuracy: 22944/50000 (45.89%)



Test set: Average loss: 0.0227, Accuracy: 4729/10000 (47.29%)

LR =  [0.01]
********* Epoch = 3 *********
loss=1.1569 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.01it/s]

Epoch 3: Train set: Average loss: 0.0197, Accuracy: 26891/50000 (53.78%)



Test set: Average loss: 0.0195, Accuracy: 5472/10000 (54.72%)

LR =  [0.01]
********* Epoch = 4 *********
loss=1.1949 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.42it/s]

Epoch 4: Train set: Average loss: 0.0177, Accuracy: 29626/50000 (59.25%)



Test set: Average loss: 0.0177, Accuracy: 5985/10000 (59.85%)

LR =  [0.01]
********* Epoch = 5 *********
loss=1.2434 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.24it/s]

Epoch 5: Train set: Average loss: 0.0162, Accuracy: 31497/50000 (62.99%)



Test set: Average loss: 0.0171, Accuracy: 6131/10000 (61.31%)

LR =  [0.01]
********* Epoch = 6 *********
loss=0.8792 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.13it/s]

Epoch 6: Train set: Average loss: 0.0150, Accuracy: 33005/50000 (66.01%)



Test set: Average loss: 0.0166, Accuracy: 6308/10000 (63.08%)

LR =  [0.01]
********* Epoch = 7 *********
loss=0.8106 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.20it/s]

Epoch 7: Train set: Average loss: 0.0141, Accuracy: 34143/50000 (68.29%)



Test set: Average loss: 0.0153, Accuracy: 6589/10000 (65.89%)

LR =  [0.01]
********* Epoch = 8 *********
loss=1.0132 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.36it/s]

Epoch 8: Train set: Average loss: 0.0133, Accuracy: 35189/50000 (70.38%)



Test set: Average loss: 0.0145, Accuracy: 6781/10000 (67.81%)

LR =  [0.01]
********* Epoch = 9 *********
loss=1.0848 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.19it/s]

Epoch 9: Train set: Average loss: 0.0126, Accuracy: 35826/50000 (71.65%)



Test set: Average loss: 0.0135, Accuracy: 6959/10000 (69.59%)

LR =  [0.01]
********* Epoch = 10 *********
loss=1.1055 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.17it/s]

Epoch 10: Train set: Average loss: 0.0121, Accuracy: 36444/50000 (72.89%)



Test set: Average loss: 0.0136, Accuracy: 6995/10000 (69.95%)

LR =  [0.01]
********* Epoch = 11 *********
loss=0.7769 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.23it/s]

Epoch 11: Train set: Average loss: 0.0116, Accuracy: 37064/50000 (74.13%)



Test set: Average loss: 0.0122, Accuracy: 7389/10000 (73.89%)

LR =  [0.01]
********* Epoch = 12 *********
loss=0.1727 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.25it/s]

Epoch 12: Train set: Average loss: 0.0111, Accuracy: 37733/50000 (75.47%)



Test set: Average loss: 0.0123, Accuracy: 7365/10000 (73.65%)

LR =  [0.01]
********* Epoch = 13 *********
loss=0.4531 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.26it/s]

Epoch 13: Train set: Average loss: 0.0106, Accuracy: 38168/50000 (76.34%)



Test set: Average loss: 0.0125, Accuracy: 7295/10000 (72.95%)

LR =  [0.01]
********* Epoch = 14 *********
loss=0.8959 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.30it/s]

Epoch 14: Train set: Average loss: 0.0103, Accuracy: 38620/50000 (77.24%)



Test set: Average loss: 0.0123, Accuracy: 7271/10000 (72.71%)

LR =  [0.01]
********* Epoch = 15 *********
loss=0.3257 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.39it/s]

Epoch 15: Train set: Average loss: 0.0100, Accuracy: 38975/50000 (77.95%)



Test set: Average loss: 0.0113, Accuracy: 7478/10000 (74.78%)

LR =  [0.01]
********* Epoch = 16 *********
loss=0.6993 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.25it/s]

Epoch 16: Train set: Average loss: 0.0096, Accuracy: 39390/50000 (78.78%)



Test set: Average loss: 0.0110, Accuracy: 7670/10000 (76.70%)

LR =  [0.01]
********* Epoch = 17 *********
loss=0.5470 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.25it/s]

Epoch 17: Train set: Average loss: 0.0094, Accuracy: 39705/50000 (79.41%)



Test set: Average loss: 0.0118, Accuracy: 7461/10000 (74.61%)

LR =  [0.01]
********* Epoch = 18 *********
loss=0.3618 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.28it/s]

Epoch 18: Train set: Average loss: 0.0091, Accuracy: 39986/50000 (79.97%)



Test set: Average loss: 0.0104, Accuracy: 7712/10000 (77.12%)

LR =  [0.01]
********* Epoch = 19 *********
loss=0.7373 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.24it/s]

Epoch 19: Train set: Average loss: 0.0089, Accuracy: 40169/50000 (80.34%)



Test set: Average loss: 0.0102, Accuracy: 7775/10000 (77.75%)

LR =  [0.01]
********* Epoch = 20 *********
loss=0.5694 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.23it/s]

Epoch 20: Train set: Average loss: 0.0087, Accuracy: 40513/50000 (81.03%)



Test set: Average loss: 0.0111, Accuracy: 7595/10000 (75.95%)

LR =  [0.01]
********* Epoch = 21 *********
loss=0.7081 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.32it/s]

Epoch 21: Train set: Average loss: 0.0085, Accuracy: 40660/50000 (81.32%)



Test set: Average loss: 0.0108, Accuracy: 7594/10000 (75.94%)

LR =  [0.01]
********* Epoch = 22 *********
loss=0.5615 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.27it/s]

Epoch 22: Train set: Average loss: 0.0083, Accuracy: 40848/50000 (81.70%)



Test set: Average loss: 0.0106, Accuracy: 7775/10000 (77.75%)

LR =  [0.01]
********* Epoch = 23 *********
loss=0.4829 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.27it/s]

Epoch 23: Train set: Average loss: 0.0081, Accuracy: 41092/50000 (82.18%)



Test set: Average loss: 0.0094, Accuracy: 7955/10000 (79.55%)

LR =  [0.01]
********* Epoch = 24 *********
loss=1.4345 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.17it/s]

Epoch 24: Train set: Average loss: 0.0080, Accuracy: 41238/50000 (82.48%)



Test set: Average loss: 0.0094, Accuracy: 7928/10000 (79.28%)

LR =  [0.01]
********* Epoch = 25 *********
loss=0.7686 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.26it/s]

Epoch 25: Train set: Average loss: 0.0078, Accuracy: 41356/50000 (82.71%)



Test set: Average loss: 0.0105, Accuracy: 7832/10000 (78.32%)

LR =  [0.01]
********* Epoch = 26 *********
loss=1.2402 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.24it/s]

Epoch 26: Train set: Average loss: 0.0077, Accuracy: 41555/50000 (83.11%)



Test set: Average loss: 0.0093, Accuracy: 7988/10000 (79.88%)

LR =  [0.01]
********* Epoch = 27 *********
loss=0.4272 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.24it/s]

Epoch 27: Train set: Average loss: 0.0075, Accuracy: 41729/50000 (83.46%)



Test set: Average loss: 0.0095, Accuracy: 7940/10000 (79.40%)

LR =  [0.01]
********* Epoch = 28 *********
loss=0.6921 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.31it/s]

Epoch 28: Train set: Average loss: 0.0075, Accuracy: 41821/50000 (83.64%)



Test set: Average loss: 0.0101, Accuracy: 7854/10000 (78.54%)

LR =  [0.01]
********* Epoch = 29 *********
loss=0.4317 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.20it/s]

Epoch 29: Train set: Average loss: 0.0073, Accuracy: 41942/50000 (83.88%)



Test set: Average loss: 0.0092, Accuracy: 8018/10000 (80.18%)

LR =  [0.01]
********* Epoch = 30 *********
loss=0.5680 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.19it/s]

Epoch 30: Train set: Average loss: 0.0071, Accuracy: 42199/50000 (84.40%)



Test set: Average loss: 0.0097, Accuracy: 7907/10000 (79.07%)

LR =  [0.01]
********* Epoch = 31 *********
loss=0.3377 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.17it/s]

Epoch 31: Train set: Average loss: 0.0070, Accuracy: 42346/50000 (84.69%)



Test set: Average loss: 0.0093, Accuracy: 8021/10000 (80.21%)

LR =  [0.01]
********* Epoch = 32 *********
loss=0.4431 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.20it/s]

Epoch 32: Train set: Average loss: 0.0069, Accuracy: 42375/50000 (84.75%)



Test set: Average loss: 0.0110, Accuracy: 7808/10000 (78.08%)

LR =  [0.01]
********* Epoch = 33 *********
loss=0.8008 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.12it/s]

Epoch 33: Train set: Average loss: 0.0067, Accuracy: 42606/50000 (85.21%)



Test set: Average loss: 0.0088, Accuracy: 8080/10000 (80.80%)

LR =  [0.01]
********* Epoch = 34 *********
loss=0.5024 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.17it/s]

Epoch 34: Train set: Average loss: 0.0067, Accuracy: 42607/50000 (85.21%)



Test set: Average loss: 0.0087, Accuracy: 8090/10000 (80.90%)

LR =  [0.01]
********* Epoch = 35 *********
loss=0.5538 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.18it/s]

Epoch 35: Train set: Average loss: 0.0066, Accuracy: 42715/50000 (85.43%)



Test set: Average loss: 0.0085, Accuracy: 8153/10000 (81.53%)

LR =  [0.01]
********* Epoch = 36 *********
loss=0.4132 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.17it/s]

Epoch 36: Train set: Average loss: 0.0065, Accuracy: 42783/50000 (85.57%)



Test set: Average loss: 0.0096, Accuracy: 7956/10000 (79.56%)

LR =  [0.01]
********* Epoch = 37 *********
loss=0.5231 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.16it/s]

Epoch 37: Train set: Average loss: 0.0064, Accuracy: 42959/50000 (85.92%)



Test set: Average loss: 0.0090, Accuracy: 8080/10000 (80.80%)

LR =  [0.01]
********* Epoch = 38 *********
loss=0.8874 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.26it/s]

Epoch 38: Train set: Average loss: 0.0064, Accuracy: 42943/50000 (85.89%)



Test set: Average loss: 0.0083, Accuracy: 8227/10000 (82.27%)

LR =  [0.01]
********* Epoch = 39 *********
loss=0.3228 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.22it/s]

Epoch 39: Train set: Average loss: 0.0063, Accuracy: 43124/50000 (86.25%)



Test set: Average loss: 0.0089, Accuracy: 8146/10000 (81.46%)

LR =  [0.01]
********* Epoch = 40 *********
loss=0.5558 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.18it/s]

Epoch 40: Train set: Average loss: 0.0062, Accuracy: 43122/50000 (86.24%)



Test set: Average loss: 0.0084, Accuracy: 8263/10000 (82.63%)

LR =  [0.01]
********* Epoch = 41 *********
loss=0.6084 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.23it/s]

Epoch 41: Train set: Average loss: 0.0060, Accuracy: 43345/50000 (86.69%)



Test set: Average loss: 0.0086, Accuracy: 8183/10000 (81.83%)

LR =  [0.01]
********* Epoch = 42 *********
loss=0.4483 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.20it/s]

Epoch 42: Train set: Average loss: 0.0060, Accuracy: 43297/50000 (86.59%)



Test set: Average loss: 0.0083, Accuracy: 8258/10000 (82.58%)

LR =  [0.01]
********* Epoch = 43 *********
loss=1.5113 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.14it/s]

Epoch 43: Train set: Average loss: 0.0059, Accuracy: 43408/50000 (86.82%)



Test set: Average loss: 0.0085, Accuracy: 8195/10000 (81.95%)

LR =  [0.01]
********* Epoch = 44 *********
loss=0.3994 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.09it/s]

Epoch 44: Train set: Average loss: 0.0057, Accuracy: 43649/50000 (87.30%)



Test set: Average loss: 0.0087, Accuracy: 8210/10000 (82.10%)

LR =  [0.001]
********* Epoch = 45 *********
loss=0.4177 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.21it/s]

Epoch 45: Train set: Average loss: 0.0046, Accuracy: 44936/50000 (89.87%)



Test set: Average loss: 0.0071, Accuracy: 8495/10000 (84.95%)

LR =  [0.001]
********* Epoch = 46 *********
loss=0.2643 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.19it/s]

Epoch 46: Train set: Average loss: 0.0043, Accuracy: 45328/50000 (90.66%)



Test set: Average loss: 0.0072, Accuracy: 8471/10000 (84.71%)

LR =  [0.001]
********* Epoch = 47 *********
loss=0.2188 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.18it/s]

Epoch 47: Train set: Average loss: 0.0042, Accuracy: 45382/50000 (90.76%)



Test set: Average loss: 0.0071, Accuracy: 8496/10000 (84.96%)

LR =  [0.001]
********* Epoch = 48 *********
loss=0.9536 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.19it/s]

Epoch 48: Train set: Average loss: 0.0041, Accuracy: 45597/50000 (91.19%)



Test set: Average loss: 0.0071, Accuracy: 8501/10000 (85.01%)

LR =  [0.001]
********* Epoch = 49 *********
loss=0.5204 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.28it/s]

Epoch 49: Train set: Average loss: 0.0041, Accuracy: 45555/50000 (91.11%)



Test set: Average loss: 0.0071, Accuracy: 8502/10000 (85.02%)

LR =  [0.001]
********* Epoch = 50 *********
loss=0.0479 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.13it/s]

Epoch 50: Train set: Average loss: 0.0039, Accuracy: 45705/50000 (91.41%)



Test set: Average loss: 0.0073, Accuracy: 8519/10000 (85.19%)

LR =  [0.001]
********* Epoch = 51 *********
loss=0.2950 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.22it/s]

Epoch 51: Train set: Average loss: 0.0040, Accuracy: 45634/50000 (91.27%)



Test set: Average loss: 0.0072, Accuracy: 8502/10000 (85.02%)

LR =  [0.001]
********* Epoch = 52 *********
loss=0.4774 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.25it/s]

Epoch 52: Train set: Average loss: 0.0039, Accuracy: 45741/50000 (91.48%)



Test set: Average loss: 0.0073, Accuracy: 8475/10000 (84.75%)

LR =  [0.001]
********* Epoch = 53 *********
loss=0.1270 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.14it/s]

Epoch 53: Train set: Average loss: 0.0039, Accuracy: 45775/50000 (91.55%)



Test set: Average loss: 0.0072, Accuracy: 8498/10000 (84.98%)

LR =  [0.001]
********* Epoch = 54 *********
loss=0.6109 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.16it/s]

Epoch 54: Train set: Average loss: 0.0039, Accuracy: 45738/50000 (91.48%)



Test set: Average loss: 0.0073, Accuracy: 8487/10000 (84.87%)

LR =  [0.0001]
********* Epoch = 55 *********
loss=0.1203 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.20it/s]

Epoch 55: Train set: Average loss: 0.0038, Accuracy: 45944/50000 (91.89%)



Test set: Average loss: 0.0073, Accuracy: 8505/10000 (85.05%)

LR =  [0.0001]
********* Epoch = 56 *********
loss=0.6471 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.14it/s]

Epoch 56: Train set: Average loss: 0.0037, Accuracy: 45960/50000 (91.92%)



Test set: Average loss: 0.0073, Accuracy: 8497/10000 (84.97%)

LR =  [0.0001]
********* Epoch = 57 *********
loss=0.2732 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.21it/s]

Epoch 57: Train set: Average loss: 0.0037, Accuracy: 46024/50000 (92.05%)



Test set: Average loss: 0.0073, Accuracy: 8523/10000 (85.23%)

LR =  [0.0001]
********* Epoch = 58 *********
loss=0.2903 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.22it/s]

Epoch 58: Train set: Average loss: 0.0037, Accuracy: 46069/50000 (92.14%)



Test set: Average loss: 0.0073, Accuracy: 8510/10000 (85.10%)

LR =  [0.0001]
********* Epoch = 59 *********
loss=0.2102 batch_id=781: 100%|██████████| 782/782 [00:20<00:00, 37.30it/s]

Epoch 59: Train set: Average loss: 0.0037, Accuracy: 46065/50000 (92.13%)



Test set: Average loss: 0.0073, Accuracy: 8501/10000 (85.01%)

LR =  [0.0001]
********* Epoch = 60 *********
loss=0.2171 batch_id=781: 100%|██████████| 782/782 [00:21<00:00, 37.23it/s]

Epoch 60: Train set: Average loss: 0.0037, Accuracy: 46033/50000 (92.07%)



Test set: Average loss: 0.0073, Accuracy: 8503/10000 (85.03%)

LR =  [0.0001]
```