"""
Note:

My CNN integrates several features to act as a good architecural model. The network layout uses the sequential feature-channel
doubling(64 -> 512) inspired by VGG NET (1) along with identity skip connections from the ResNet framework (2) to help prevent
vanishing gradients during deeper feature extarctions. I've also used a GLobal Average Pooling layer (3) before the final classification
to help with stronger regularaisation against overfitting. I have also added Kaiming Normal initialisation (4) to help prevent early learning stalls 
across the ReLU functions.



[1] K.Simonyan and A.Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014. [Online]. Available: https://arxiv.org/abs/1409.1556
[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. [Online]. Available: https://ieeexplore.ieee.org/document/7780459/
[3] M. Lin, Q. Chen, and S. Yan, "Network in network," arXiv preprint arXiv:1312.4400, 2013. [Online]. Available: https://arxiv.org/abs/1312.4400
[4] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1026-1034. [Online]. Available: https://arxiv.org/abs/1502.01852
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    """
    Residual mapping block inspired by the Resnet framework.
    Passes the original input arount the convolution networks and adds it to the final features.
    This keeps the network form losing informatin and stops problems with vanishing gradients
    """
    def __init__(self, channels):
        super (ResidualBlock, self).__init__()

        #sequential sub-block: main feature extraction path
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),        #first convolutional layer 
            nn.BatchNorm2d(channels),       #normalises activations 
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),        #second convolutional layer
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU() 
    
    def forward(self, x):
        #skip connection: adds raw input "x" directly ot the output then applies final relu

        return self.relu(x + self.block(x))

class CNN(nn.Module):
    """
    Custom CNN with residual blocks, batch normalisation, and adaptive average pooling for multi-class classification

    """
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        #LAYER 1 (64 channels)
        #input: (Batch, 3, 224, 224)
        #output: (Batch, 64, 112, 112)
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.resb1 = ResidualBlock(64)

        #LAYER 2 (128 channels)
        #input: (Batch, 64, 112, 112)
        #output: (Batch, 128, 56, 56)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.resb2 = ResidualBlock(128)

        #LAYER 3 (256 channels)
        #input: (Batch, 128, 56, 56)
        #output: (Batch, 256, 28, 28)
        self.conv5 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        self.resb3 = ResidualBlock(256)

        #LAYER 4 (512 channels)
        #input: (Batch, 256, 28, 28)
        #output: (Batch, 512, 14, 14)
        self.conv7 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2)
        self.resb4 = ResidualBlock(512)

        self.dropout = nn.Dropout(0.25)     #regularisation, randomly zeroes 25% of neurons to help with overfitting
        self.gap = nn.AdaptiveAvgPool2d(1)      #global average pooling 
        self.fc1 = nn.Linear(512, 256)      #fully connected layers
        self.fc2 = nn.Linear(256, num_classes)      #final output vector mapping to class


        self._initialise_weights()      #initialise weights

    def _initialise_weights(self):
        """
        Custom weight initialisation using the Kaiming Normal method.
        Optimised for networks using the ReLU activation functions to ensure stable gradient flow 
        """
        for layer in self.modules():

            #initialise 2D Convolutional layers
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode = "fan_out", nonlinearity = "relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            #initialise Batch Normalisation layers (weights to 1, biases to 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

            #initisalise Fully Connected layers
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):

        #forward pass: Group 1 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.resb1(x)

        #forward pass: Group 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.resb2(x)

        #forward pass: Group 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.resb3(x)

        #forward pass: Group 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.resb4(x)    

        x = self.gap(x)     #dim changes from (Batch, 512, 14, 14) to (Batch, 512, 1, 1)
        x = torch.flatten(x, 1)     #flattens tensor to 1D feature vector (Batch, 512)
        x = F.relu(self.fc1(x))     #fully connected layer with ReLU activation
        x = self.dropout(x)     #regularisation, dropout
        x = self.fc2(x) 

        return x