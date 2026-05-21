ARCHITECTURE & DESIGN RATIONALE:

My CNN integrates several features to act as a good architecural model. The network layout uses the sequential feature-channel
doubling(64 -> 512) inspired by VGG NET (1) along with identity skip connections from the ResNet framework (2) to help prevent
vanishing gradients during deeper feature extarctions. I've also used a GLobal Average Pooling layer (3) before the final classification
to help with stronger regularaisation against overfitting. I have also added Kaiming Normal initialisation (4) to help prevent early learning stalls 
across the ReLU functions.


Note on validation strategy and final training:

For the majority of the coursework, I split the pets dataset into trainval and test and then split the trainval into train and val.
I used the val accuracy as a measure of how well the model was doing with unseen data and adjusted my code with that and the training
accuracy as a reference. Once i was happy with the model's performance, I then decided to train the model on the whole trainval dataset 
for its final run to get the maximum predictive performance possible. I have left the validation code intact within the codebase as 
evidence of this.


RESULTS:

Best Train Accuracy: 59.29%
Final Test Accuracy: 45.95%


REFERENCES: 

[1] K.Simonyan and A.Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014. [Online]. Available: https://arxiv.org/abs/1409.1556
[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. [Online]. Available: https://ieeexplore.ieee.org/document/7780459/
[3] M. Lin, Q. Chen, and S. Yan, "Network in network," arXiv preprint arXiv:1312.4400, 2013. [Online]. Available: https://arxiv.org/abs/1312.4400
[4] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1026-1034. [Online]. Available: https://arxiv.org/abs/1502.01852