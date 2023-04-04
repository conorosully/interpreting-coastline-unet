# Trianing unet for coastline detection
# Conor O'Sullivan 
# 07 Feb 2023

#Imports
import numpy as np
import pandas as pd
import sys
import random
import glob

import cv2 as cv
from PIL import Image
from osgeo import gdal

import torch 
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Variables
train_path = "../../data/SWED/train/images/" #UPDATE
save_path = "../../models/{}.pth" #UPDATE
model_name = sys.argv[1]
try: 
    sample = sys.argv[2].lower() == 'true'
    scale = sys.argv[3].lower() == 'true'
    clean_data = int(sys.argv[4]) # 0 = no cleaning, 1 = remove missing, 2 = remove incorrect, 3 = remove both, 4 = remove all
except:
    sample = False
    clean_data = 0
    scale = True
batch_size = 32 

# Paths for data cleaning 
missing_data = open("missing_data.txt", "r").read().splitlines() #images with no data
incorrect = open("incorrect.txt", "r").read().splitlines() #images of water incorrectly labelled as land
remove = open("remove.txt", "r").read().splitlines() #images to remove for other reasons

missing_data = [train_path + i for i in missing_data]
incorrect = [train_path + i for i in incorrect]
remove = [train_path + i for i in remove]

if clean_data == 0:
    missing_data = []
    incorrect = []
    remove = []
elif clean_data == 1:
    incorrect = []
    remove = []
elif clean_data == 2:
    missing_data = []
    remove = []
elif clean_data == 3:
    remove = []


print("Missing: {}, Incorrect: {}, Remove: {}".format(len(missing_data),len(incorrect),len(remove)))
print("Sample: {}, Clean: {}, Scale: {}".format(sample,clean_data,scale))

# Classes
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transforms, incorrect):

        self.paths = paths
        self.transforms = transforms
        self.incorrect = incorrect #images of water incorrectly labelled as land

    def __getitem__(self, idx):
        """Get image and binary mask for a given index"""

        path = self.paths[idx]
            
        if 'train' in path: 
            image, target = self.load_train(path)
        elif 'test' in path: 
            image, target = self.load_test(path)
        
        return image, target

    def load_train(self,path):
        """Return image and mask for training images"""

        # Get image
        image = np.load(path)
        image = self.transform_img(image)
        # Get target
        mask_path = path.replace("images","labels").replace("image","chip")
        mask = np.load(mask_path)
        mask = self.transform_mask(mask,path)

        return image, mask

    def load_test(self, path):
       
        """Return image and mask for test set"""
    
        # Get image 
        image = gdal.Open(path).ReadAsArray()
        image = np.stack(image, axis=-1)
        image = self.transform_img(image)

        # Get mask
        mask_path = path.replace("images","labels").replace("image","label")
        mask= gdal.Open(mask_path).ReadAsArray()
        mask = self.transform_mask(mask,path)
        
        return image, mask
        
    
    def transform_img(self,image):
        """Scale all image bands to 0-1"""

        image = np.array(image)
        image = image.astype(np.float32)

        # Scale to 0-1
        if scale:
            image = image/10000
            image = np.clip(image,0,1)
        

        # Convert to tensor
        image = image.transpose(2,0,1)
        image = torch.tensor(image)

        return image
    
    def transform_mask(self,mask,path):

        """ 1. Convert mask to tensor and 
            2. replace -1 with 0
            3. relabel images incorrectly labelled as land"""
        
        # Land = 0, Water = 1
        mask_1 = np.array(mask).astype(np.int8)
        mask_1[np.where(mask_1 == -1)] = 0

        # relabel as water 
        if path in self.incorrect:
            mask_1 = np.ones(mask_1.shape)
         

        # Land = 1, Water = 0
        mask_0 = 1-mask_1

        mask = np.array([mask_0,mask_1])
        mask = torch.Tensor(mask).squeeze()

        return mask

    def __len__(self):
        return len(self.paths)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.elu = nn.ELU()
    

    def forward(self, inputs):
        #Layer 1
        x = self.conv1(inputs) #convolution
        x = self.elu(x) #activation
        x = self.bn1(x) #normalisation
        
        #Layer 2
        x = self.conv2(x)
        x = self.elu(x)
        x = self.bn2(x)
        
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c) 
        self.pool = nn.MaxPool2d((2, 2)) 
    def forward(self, inputs):
        x = self.conv(inputs) #convolutional block
        p = self.pool(x) #max pooling
        return x, p

class decoder_block(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0) 
        self.conv = conv_block(out_c+out_c, out_c) 

    def forward(self, inputs, skip):
        x = self.up(inputs) #upsampling
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x) #convolutional block
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(12, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)
        self.e4 = encoder_block(128, 256)
        """ Bottleneck """
        self.b = conv_block(256, 512)
        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)
        """ Classifier """
        self.outputs = nn.Conv2d(32, 2, kernel_size=1, padding=0)
        self.sm = nn.Softmax(dim=1)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        outputs = self.sm(outputs) 

        return outputs

# Functions
def load_data():
    """Load data from disk"""

    TRANSFORMS = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    paths = glob.glob(train_path + "*")
    print("Paths before cleaning: {}".format(len(paths)))
    
    if sample: 
        paths = paths[:1000]

    # Remove paths with no data and other errors
    paths = [path for path in paths if path not in missing_data]
    paths = [path for path in paths if path not in remove]

    print("Paths after cleaning: {}".format(len(paths)))

    # Shuffle the paths
    random.shuffle(paths)

    # Create a datasets for training and validation
    split = int(0.9 * len(paths))
    train_data = TrainDataset(paths[:split], TRANSFORMS,incorrect)
    valid_data = TrainDataset(paths[split:], TRANSFORMS, incorrect)

    # Prepare data for Pytorch model
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    print("Training images: {}".format(train_data.__len__()))
    print("Validation images: {}".format(valid_data.__len__()))

    return train_loader, valid_loader

def train_model(train_loader, valid_loader,ephocs=50):
    
    # define the model
    model = build_unet()
    
    # move tensors to GPU if available
    device = torch.device('mps')  #UPDATE
    print("Using device: {}\n".format(device))
    model.to(device)
    
    # specify loss function (binary cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train the model
    min_loss = np.inf

    for epoch in range(ephocs):
        print("Epoch {} |".format(epoch+1),end=" ")
        count = 1
        model = model.train()

        for images, target in iter(train_loader):

            print(count,end=" ")
            count += 1

            images = images.to(device)
            target = target.to(device)

            # Zero gradients of parameters
            optimizer.zero_grad()  

            # Execute model to get outputs
            output = model(images)
         
            # Calculate loss
            loss = criterion(output, target)

            # Run backpropogation to accumulate gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

        # Calculate validation loss
        model = model.eval()

        valid_loss = 0
        for images, target in iter(valid_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)

            loss = criterion(output, target)
            
            valid_loss += loss.item()
    
        valid_loss /= len(valid_loader)
        print("| Validation Loss: {}".format(round(valid_loss,5)))
        
        if valid_loss < min_loss:
            print("Saving model")
            torch.save(model, save_path.format(model_name))

            min_loss = valid_loss

if __name__ == "__main__":
    print("Training model: {}".format(model_name))
    print("Clean: {}".format(clean_data))

    # Test incorrect
    if len(incorrect) > 0:
        test_path = incorrect[100]
        image, mask = TrainDataset([], None, []).load_train(test_path)
        print("Before: ", mask[1].min(),mask[1].max())
        image, mask = TrainDataset([], None, incorrect).load_train(test_path)
        print("After: ", mask[1].min(),mask[1].max())

    # Load data
    train_loader, valid_loader = load_data()
   
    # Train the model
    train_model(train_loader, valid_loader)


    
