# Outline of structure of video processor

# Videodataset abstraction that loads video frames, a pytorch dataset
# A convnet abstraction

import torch
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms.functional as tfunc
#import torchvision.transforms as transforms
import torchvision
#import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


#import numpy as np
#import torch
#import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
#from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from array2gif import write_gif
#import matplotlib.pyplot as plt
import imageio



datapath = '/scratch1/datasets/beamng/crash_vids'
vidnames = {
    'crash_1.mp4':1,
    'no_crash_2.mp4':0,
  #  'crash_3.mp4':1,
    'crash_4.mp4':1,
  #  'no_crash_5.mp4':0,
    'no_crash_6.mp4':0,
    'crash_7.mp4':1,
  #  'crash_8.mp4':1,
    'crash_9.mp4':1,
  #  'crash_10.mp4':1,
    'no_crash_11.mp4':0,
    'crash_12.mp4':1,
    'no_crash_13.mp4':0,
  #  'crash_14.mp4':1
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
    
    
########################################33    
class Simple_Video_Net(nn.Module):
    def __init__(self, windowsize, numclasses):
        super(Simple_Video_Net, self).__init__()
        self.main = nn.Sequential(
            # N x 512 x 512
            nn.Conv2d(3*windowsize, 64, kernel_size=7, stride=4, padding=3),
            # N x 128 x 128
            nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3),
            # 64 x 64 x 64
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=, stride=),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            # 64 x 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3, stride=2, padding=1),
            #64 x 16 x 16
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*16*16, numclasses)
        )
        
    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x).squeeze(1)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        init.normal(m.weight, std=1e-2)
        # init.orthogonal(m.weight)
        #init.xavier_uniform(m.weight, gain=1.4)
        if m.bias is not None:
            init.constant(m.bias, 0.0)

class AverageMeter(object):
    def __init__(self):
        self.count = 0
        self.value = 0
    def update(self, value, count=1):
        self.value += value
        self.count += count
    def average(self):
        return self.value / self.count
    def reset(self):
        self.count = 0
        self.value = 0
            
def train(nepochs, outname='beamng_model.pth', windowsize=1, cont=False, noise=0.1, mode='normal'):
    device='cuda:0'
    dset = VideoLoader()
    loader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True, num_workers=4)
    
    #model = MNIST_Net(6,1)
    model = Simple_Video_Net(windowsize,1)
    model.apply(weights_init)
    if cont:
        print("Continuing")
        model.load_state_dict(torch.load(outname))
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5,0.999))
    lossfunc = nn.BCEWithLogitsLoss()
    
    accuracyMeter = AverageMeter()
    lossMeter = AverageMeter()
    model.train()
    for epoch in range(nepochs):
        print("Epoch", epoch)
        accuracyMeter.reset()
        lossMeter.reset()
        for i, data in enumerate(loader):
            # Extract training batches
            ims, labels = data
            device_labels = labels.float().to(device)
            device_ims = ims.to(device)

            # Train network
            model.zero_grad()
            predictions = model(device_ims)
            loss = lossfunc(predictions, device_labels)
            loss.backward()
            optimizer.step()

            # Compute statistics
            rounded = torch.sigmoid(predictions).round().long()
            rounded = rounded.to('cpu')
            correct = torch.sum(rounded == labels).item()
            batch_size = predictions.size(0)
            accuracyMeter.update(correct, batch_size)
            lossMeter.update(loss.item())

            if i%10 == 0:
               # print("labels", labels)
               # print("predictions", rounded)
                accuracy = correct/batch_size
                print("batch {1}, batch_loss={2}, batch_accuracy={3}".format(
                    epoch,i,loss.item(),accuracy))
    
    torch.save(model.state_dict(), outname)   
    
#### Needs fixing up
def test(modelname, windowsize=6, noise=0.1, mode='normal'):
    testmodel = Simple_MNIST_Net(windowsize,1)
    testmodel.load_state_dict(torch.load(modelname))
    testmodel.eval()
    testdata = MNIST_Video(train=False, windowsize=windowsize, noise=0.1, mode=mode)
    
    preds = []
    for i in range(460):
        if i%100 == 0:
            print(i)
        pt, y, choicept = testdata[i]
        #print(y)
        #allpts.append(pt)
        preds.append(torch.sigmoid(testmodel(pt.unsqueeze(0))).item())
    preds = np.array(preds).reshape((10,testdata.netFrames()))
    #pred_indices = np.where(np.logical_or((preds>0.999), (preds < 0.001)))
    pred_indices = firstPerRow(np.logical_or(preds>0.999,preds<0.001))
    print(pred_indices)
    for i in range(10): #range(testdata.numGifs()):
        testdata.write_gif(i,'testgif{}.gif'.format(i),predicted_choice=pred_indices[i])
        ind = i*testdata.netFrames()
        pt, y, choicept = testdata[ind]
        print("Y=",y, "Choicept=", choicept, "Predicted=",pred_indices[i])
        plt.plot(preds[i,:])
        plt.show()
        input("Press enter")
#########################################################3    
    
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
    #main()
    train(20)
                    
            

