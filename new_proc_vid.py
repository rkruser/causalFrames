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
#import time
import pickle


#import numpy as np
#import torch
#import torchvision.datasets as datasets
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.init as init
#from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
#from array2gif import write_gif
#import matplotlib.pyplot as plt
#import imageio



datapath = '/scratch1/datasets/beamng/extracted_vids'
vidnames = [
    'v1_1.mp4',
    'v1_2.mp4'
]


class VideoDataset(Dataset):
  def __init__(self, path=datapath, 
        names=vidnames, 
        labelfile='times.txt',
        keepEvery=5,
        cropsize=768,
        resize=224, 
        windowsize=1, 
        chopoff=6, 
        load_from_tensorfile=False,
        tensorfile_prefix='allframes', 
        save_to_tensorfile=True):
    super(VideoDataset, self).__init__()
    self.chopoff=chopoff # How many seconds to chop off front of each
    self.windowsize=windowsize # How 
    self.cropsize=cropsize
    self.resize=resize
    self.keepEvery=keepEvery
    self.names=names
    self.path=path
    self.labelfile = labelfile
    self.tensorfile_prefix=tensorfile_prefix # save tensor and metadata
    self.save_to_tensorfile = save_to_tensorfile
    self.load_from_tensorfile = load_from_tensorfile
    
    self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(self.cropsize),
            transforms.Resize(self.resize,interpolation=Image.LANCZOS),
            transforms.ToTensor()
        ])

    self.metadata = []
    self.framesPerVid = []
    self.allframes = []
    self.alllabels = []

# Useful: np.searchsorted(sortedarr, val), # get first index of elt >= val
#    vidobj.get(cv2.CAP_PROP_FRAME_COUNT), # get nframes
#  vidobj.get(cv2.CAP_PROP_FPS)  # get fps

    if self.load_from_tensorfile:
      self.metadata = pickle.load(open(os.path.join(self.path,
             self.tensorfile_prefix+'_metadata.pkl'),'rb'))
      self.allframes, self.alllabels = torch.load(os.path.join(self.path,
            self.tensorfile_prefix+'.th'))
      self.framesPerVid = [m[1] for m in self.metadata]

    else:
      # load all videos
      # load labels
          # ...
      allLabels = open(os.path.join(self.path,self.labelfile),'r').readlines()
      nameMap = {}
      for s in allLabels:
        s = s.split()
        nameMap[s[0]] = float(s[1])

      for ind, vid in enumerate(self.names):
        fullpath = os.path.join(self.path,vid)
        video = cv2.VideoCapture(fullpath)
        if not video.isOpened():
          print("Error reading {}".format(vid))
          continue
        else:
          print("Reading {}".format(vid))

        nframes = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        toChop = round(self.chopoff*fps)

        if toChop >= nframes:
            print("Skipping {} because too short".format(vid))
            continue
        
        #vidinfo = allLabels[ind].split() # Problems here
        #assert(vidinfo[0] == os.path.splitext(vid)[0])
        #crashtime = float(vidinfo[1])
        
        vidPrefix = os.path.splitext(vid)[0]
        crashtime = nameMap[vidPrefix] # Map file name to crash cutoff time
        crashframe = round(crashtime*fps)

        frames = []
        frameLabels = []

        counter = 0
        while video.isOpened():
          ret, frame = video.read()
          if ret:
            if (counter >= toChop and counter%self.keepEvery == 0):
              frames.append(self.transform(cv2.cvtColor(frame, 
                      cv2.COLOR_BGR2RGB)))
              if counter >= crashframe:
                frameLabels.append(1)
              else:
                frameLabels.append(0)
          else:
            print("Breaking after {}".format(counter))
            break
          counter += 1

        video.release()

        # vidname framesused  startframe  crashframe  totalvidframes  vidfps
        self.metadata.append((vidPrefix, len(frames),
              toChop+(self.keepEvery-toChop%self.keepEvery)%self.keepEvery,
              crashframe, nframes, fps))
        self.framesPerVid.append(len(frames))
        self.allframes.append(torch.stack(frames))
        self.alllabels.append(torch.Tensor(frameLabels))
      
      self.allframes = torch.cat(self.allframes)
      self.alllabels = torch.cat(self.alllabels)
      self.alllabels = self.alllabels.type(torch.ByteTensor)
      
      if self.save_to_tensorfile:
        print("Saving tensor")
        torch.save((self.allframes,self.alllabels), 
            os.path.join(self.path,self.tensorfile_prefix+'.th'))
        pickle.dump(self.metadata, open(os.path.join(self.path,
                self.tensorfile_prefix+'_metadata.pkl'),'wb'))


    ##### index math

    self.effectiveFramesPerVid = np.array([0]+[v-self.windowsize+1 for v in self.framesPerVid])
    self.framesPerVid = np.array([0]+self.framesPerVid)
    self.cumulFrames = np.cumsum(self.framesPerVid)
    self.cumulEffectiveFrames = np.cumsum(self.effectiveFramesPerVid)



  

  def __len__(self):
    return self.cumulEffectiveFrames[-1]


  def __getitem__(self, index):
    place = np.searchsorted(self.cumulEffectiveFrames, index)
    if index < self.cumulEffectiveFrames[place]:
        place = place-1
    leftover = index-self.cumulEffectiveFrames[place]
    actual_index = self.cumulFrames[place]+leftover

#    returnFrames = self.allframes[actual_index:actual_index+self.windowsize]
#    returnFrames = torch.cat([self.allframes[j] for j in range(actual_index,
 #       actual_index+self.windowsize)])
    returnFrames = torch.cat(list(self.allframes[actual_index:actual_index+self.windowsize]))
    returnLabel = self.alllabels[actual_index:actual_index+self.windowsize].any().item()

    return returnFrames, returnLabel

    def numVideos(self):
        return len(self.cumulFrames)-1

    def getVideo(self, number):
        startindex = self.cumulFrames[number]
        endindex = self.cumulFrames[number+1]
        return self.allframes[startindex:endindex]



    # Use place to jump back to framesPerVid


if __name__ == '__main__':
    d = VideoDataset(load_from_tensorfile=False, windowsize=1, 
        save_to_tensorfile=False)
    print(len(d))
    out0, out1 = d[96]
    out0 = out0.numpy().transpose(1,2,0)

    for j in range(0,len(d),10):
        imj, labj = d[j]
        print(labj)
        plt.imshow(imj.numpy().transpose(1,2,0))
        plt.show()
        input("Press enter")


# cv2.CAP_PROP_FRAME_WIDTH
# cv2.CAP_PROP_FRAME_HEIGHT
      




















    

