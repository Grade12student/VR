import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import cv2

# Model with increased capacity
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        # Adjusted Convolutional Layers
        self.enc1 = nn.Conv3d(1, 64, 3, stride=1, padding=1)  # Adjust kernel size, stride, and padding
        self.enc2 = nn.Conv3d(64, 128, 3, stride=1, padding=1)  # Adjust kernel size, stride, and padding
        self.enc3 = nn.Conv3d(128, 256, 3, stride=1, padding=1)  # Adjust kernel size, stride, and padding

        self.dec1 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # Adjust kernel size, stride, and padding
        self.dec2 = nn.Conv3d(128, 64, 3, stride=1, padding=1)  # Adjust kernel size, stride, and padding
        self.dec3 = nn.ConvTranspose3d(64, 3, 3, stride=1, padding=1)  # Adjust kernel size, stride, and padding


    def forward(self, x):
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        x = F.leaky_relu(self.enc3(x))
        x = F.leaky_relu(self.dec1(x))
        x = F.leaky_relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x

# Load data
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, start_frame, end_frame):
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame

    def __len__(self):
        return self.end_frame - self.start_frame + 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame + idx)
        ret, frame = cap.read()
        if ret:
            small_frame = cv2.resize(frame, (256, 256))
            if small_frame.shape[-1] == 1:  # Check if the frame is grayscale
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB)  # Convert to RGB
            cap.release()
            return small_frame
        cap.release()
        return np.zeros((256, 256, 3), dtype=np.uint8)


# Create model
model = Autoencoder()

# Train for more epochs
num_epochs = 100

# Use perceptual loss 
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Define any necessary components or initialization here

    def forward(self, output, target):
        # Define the computation for the perceptual loss between the output and target
        perceptual_loss = torch.mean(torch.abs(output - target))  # Example loss computation, replace with appropriate calculation
        # Return the computed loss
        return perceptual_loss


# Data augmentation
class RandomHorizontalFlip(nn.Module):
    def __init__(self):
        super(RandomHorizontalFlip, self).__init__()

    def forward(self, clip):
        # Apply random horizontal flip to the clip
        return clip

class RandomCrop(nn.Module):
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, clip):
        # Apply random crop to the clip
        return clip

transforms = nn.Sequential(
    RandomHorizontalFlip(),
    RandomCrop((128, 128)) 
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust the learning rate if necessary
video_path = '05300535.mp4'  # Replace with your video file path
start_frame = 30
end_frame = 40
dataset = VideoDataset(video_path, start_frame, end_frame)


for epoch in range(num_epochs):
    for idx in range(len(dataset)):
        data = dataset[idx]
        data = transforms(data)
        data = Variable(torch.tensor(data, dtype=torch.float).permute(2, 0, 1).unsqueeze(0) / 255.0)
        data = data.repeat(1, 3, 1, 1)
        optimizer.zero_grad()
        output = model(data)
        criterion = PerceptualLoss()  # Define the actual perceptual loss function
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
