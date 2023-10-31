import os
import numpy as np
import torch
from torch import nn
from lib import ParseGRU, Visualizer
from torch.autograd import Variable
from torchvision import transforms  # Add this line to import transforms
from torchvision.utils import save_image
from network import ThreeD_conv
import cv2
import glob
from extract import extract_frames

parse = ParseGRU()
opt = parse.args
autoencoder = ThreeD_conv(opt)
autoencoder.train()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

files = glob.glob(opt.dataset + '/*')

videos = [extract_frames(file, opt.T) for file in files]
n_videos = len(videos)



trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(1),
    transforms.Resize((opt.image_size, opt.image_size)),
    transforms.ToTensor(),
])

def transform(frames):
    trans_frames = torch.empty(opt.n_channels, opt.T, opt.image_size, opt.image_size)
    for i in range(opt.T):
        if i < len(frames):  # Check if the index is within the bounds of the list
            img = frames[i]
            img = trans(img).reshape(opt.n_channels, opt.image_size, opt.image_size)
            trans_frames[:, i] = img
        else:
            # Handle the case when the index is out of range
            # You can choose to break the loop or take other appropriate action
            break
    return trans_frames


def trim(video):
    start = np.random.randint(0, video.shape[1] - (opt.T + 1))
    end = start + opt.T
    return video[:, start:end, :, :]


def random_choice():
    X = []
    for _ in range(opt.batch_size):
        video_frames = videos[np.random.randint(0, n_videos)]
        trans_video = transform(video_frames)
        X.append(trans_video)
    X = torch.stack(X)
    return X


losses = np.zeros(opt.n_itrs)
visual = Visualizer(opt)

for itr in range(opt.n_itrs):
    real_videos = random_choice()
    x = real_videos

    x = Variable(x)

    xhat = autoencoder(x)

    loss = mse_loss(xhat, x)
    losses[itr] = losses[itr] * (itr / (itr + 1.)) + loss.data * (1. / (itr + 1.))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('itr [{}/{}], loss: {:.4f}'.format(
        itr + 1,
        opt.n_itrs,
        loss))
    visual.losses = losses
    visual.plot_loss()

    if itr % opt.check_point == 0:
        tests = x[:opt.n_test].reshape(-1, opt.T, opt.n_channels, opt.image_size, opt.image_size)
        recon = xhat[:opt.n_test].reshape(-1, opt.T, opt.n_channels, opt.image_size, opt.image_size)

        for i in range(opt.n_test):
            save_image((tests[i] / 2 + 0.5), os.path.join(opt.log_folder + '/generated_videos/3dconv', f"real_itr{itr}_no{i}.png"))
            save_image((recon[i] / 2 + 0.5), os.path.join(opt.log_folder + '/generated_videos/3dconv', f"recon_itr{itr}_no{i}.png"))
