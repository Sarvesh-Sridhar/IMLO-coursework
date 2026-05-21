Note on validation strategy and final training:

For the majority of the coursework, I split the pets dataset into trainval and test and then split the trainval into train and val.
I used the val accuracy as a measure of how well the model was doing with unseen data and adjusted my code with that and the training
accuracy as a reference. Once i was happy with the model's performance, I then decided to train the model on the whole trainval dataset 
for its final run to get the maximum predictive performance possible. I have left the validation code intact within the codebase as 
evidence of this.