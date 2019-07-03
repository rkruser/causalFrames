# Outline of structure of video processor

# Videodataset abstraction that loads video frames, a pytorch dataset
# A convnet abstraction

import torch
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms.functional as tfunc
import torchvision.transforms as transforms
import torchvision
#import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time



datapath = '/scratch1/datasets/beamng/crash_vids'
vidnames = {
    'crash_1.mp4':1,
    'no_crash_2.mp4':0,
    'crash_3.mp4':1,
    'crash_4.mp4':1,
    'no_crash_5.mp4':0,
    'no_crash_6.mp4':0,
    'crash_7.mp4':1,
    'crash_8.mp4':1,
    'crash_9.mp4':1,
    'crash_10.mp4':1,
    'no_crash_11.mp4':0,
    'crash_12.mp4':1,
    'no_crash_13.mp4':0,
    'crash_14.mp4':1
}

class VideoLoader(Dataset):
    def __init__(self, path=datapath, fnames=vidnames, keepEvery=5, cropsize=512):
        super(VideoLoader, self).__init__()
        self.label0frames = []
        self.label1frames = []
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor()
        ])
        for vid in fnames:
            label = fnames[vid]
            frames = []
            fullpath = os.path.join(path, vid)
            videoFile = cv2.VideoCapture(fullpath)
            if not videoFile.isOpened():
                print("Error reading {}".format(vid))
                continue
            else:
                print("Reading {}".format(vid))
            counter = 0
            while videoFile.isOpened():
                ret, frame = videoFile.read()
                if ret:
                    if (counter%keepEvery == 0):
                        frames.append(trans(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                else:
                    print("Breaking after {}".format(counter))
                    break
                counter += 1
            videoFile.release()
            if label == 0:
                self.label0frames += frames
            else:
                self.label1frames += frames
                
    def __len__(self):
        return len(self.label0frames)+len(self.label1frames)
    
    def __getitem__(self, index):
        if index < len(self.label0frames):
            return self.label0frames[index], 0
        else:
            return self.label1frames[index-len(self.label0frames)], 1
    
def main():
    loader = VideoLoader()  
    print(len(loader))
    #frame, label = loader[0]
    #print(frame)
#    print(frame.numpy().shape)
#    plt.imshow(frame.numpy().transpose(1,2,0))
#    plt.show()
    
    allframes = []
    for i in range(min(len(loader),500)):
        frame, label = loader[i]
        allframes.append(frame)
        
    #allframes = torch.Tensor(allframes)
    grid = torchvision.utils.make_grid(allframes, nrow=8)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.show()
       
    
    
if __name__=='__main__':
    main()
                    
            

