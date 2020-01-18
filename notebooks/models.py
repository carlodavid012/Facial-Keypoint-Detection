## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # sees 224
        # output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        # 32 x 221 x 221
        self.conv1 = nn.Conv2d(1, 32, 4)
        
        # output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # 64 x 108 x 108 
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # output size = (W-F)/S +1 = (54-2)/1 +1 = 53
        # 128 x 53 x 53
        self.conv3 = nn.Conv2d(64, 128, 2)
        
        # output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # 256 x 25 x 25
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)


        self.fc1 = nn.Linear(256*13*13, 512)
        
        self.fc2 = nn.Linear(512, 256)
        
        self.fc3 = nn.Linear(256, 136)

        
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        #print('conv1', x.shape)
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        #print('conv2', x.shape)
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
        #print('conv3', x.shape)
        x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        #print('conv4', x.shape)
        
        x = x.view(-1, 256*13*13)
        #print(x.shape)
        
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout1(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
  
        # a modified x, having gone through all the layers of your model, should be returned
        return x
