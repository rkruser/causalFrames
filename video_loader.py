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
# vidnames = [
#     'v1_1.mp4',
#     'v1_2.mp4',
#     'v1_3.mp4',
#     'v16_1.mp4'
# ]

trainvids = [
    'v1_1.mp4',
    'v1_2.mp4',
    'v1_3.mp4',
    'v1_4.mp4',
    'v1_5.mp4',
    'v1_6.mp4',
    'v1_7.mp4',
    'v1_8.mp4',
    'v1_9.mp4',
    'v1_10.mp4',
    'v1_11.mp4',
    'v1_12.mp4',
    'v1_13.mp4',
    'v1_14.mp4',
    'v2_2.mp4',
    'v2_4.mp4',
    'v2_7.mp4',
    'v2_9.mp4',
    'v2_11.mp4',
    'v3_1.mp4',
    'v3_2.mp4',
    'v3_3.mp4',
    'v3_4.mp4',
    'v3_5.mp4',
    'v3_7.mp4',
    'v4_1.mp4',
    'v4_3.mp4',
    'v4_4.mp4',
    'v4_5.mp4',
    'v4_6.mp4',
    'v4_8.mp4',
    'v4_9.mp4',
    'v4_10.mp4',
    'v4_13.mp4',
    'v4_14.mp4',
    'v4_15.mp4',
    'v4_17.mp4',
    'v4_18.mp4',
    'v5_1.mp4',
    'v5_2.mp4',
    'v5_3.mp4',
    'v5_4.mp4',
    'v5_5.mp4',
    'v5_6.mp4',
    'v5_7.mp4',
    'v5_8.mp4',
    'v6_1.mp4',
    'v6_2.mp4',
    'v6_3.mp4',
    'v6_4.mp4',
    'v6_5.mp4',
    'v7_1.mp4',
    'v7_2.mp4',
    'v7_3.mp4',
    'v7_4.mp4',
    'v7_5.mp4',
    'v7_6.mp4',
    'v7_8.mp4',
    'v7_9.mp4',
    'v7_11.mp4',
    'v7_14.mp4',
    'v7_15.mp4',
    'v7_17.mp4',
    'v7_18.mp4',
    'v8_1.mp4',
    'v8_2.mp4',
    'v8_3.mp4',
    'v8_4.mp4',
    'v8_5.mp4',
    'v8_6.mp4',
    'v9_1.mp4',
    'v9_3.mp4',
    'v9_4.mp4',
    'v10_2.mp4',
    'v10_3.mp4',
    'v10_4.mp4',
    'v11_1.mp4',
    'v11_2.mp4',
    'v11_3.mp4',
    'v11_4.mp4',
    'v11_5.mp4',
    'v11_6.mp4',
    'v12_1.mp4',
    'v13_1.mp4',
    'v13_2.mp4',
    'v13_3.mp4',
    'v13_4.mp4',
    'v13_5.mp4',
    'v13_6.mp4',
    'v13_7.mp4',
    'v14_1.mp4',
    'v14_2.mp4',
    'v14_3.mp4',
    'v14_4.mp4',
    'v14_6.mp4',
    'v14_7.mp4',
    'v15_1.mp4',
    'v15_2.mp4',
    'v15_3.mp4',
    'v15_4.mp4',
    'v15_5.mp4',
    'v15_7.mp4',
    'v15_8.mp4',
    'v15_9.mp4',
    'v15_11.mp4',
    'v15_12.mp4',
    'v15_13.mp4',
    'v16_1.mp4',
    'v16_2.mp4',
    'v16_3.mp4',
    'v16_4.mp4',
    'v16_6.mp4',
    'v16_7.mp4',
    'v16_9.mp4',
    'v16_10.mp4'
]

testvids = [
    'v12_2.mp4',
    'v4_12.mp4',
    'v15_10.mp4',
    'v7_16.mp4',
    'v10_1.mp4',
    'v4_19.mp4',
    'v4_7.mp4',
    'v2_6.mp4',
    'v13_8.mp4',
    'v3_6.mp4',
    'v4_16.mp4',
    'v7_13.mp4',
    'v7_10.mp4',
    'v5_9.mp4',
    'v4_11.mp4',
    'v7_7.mp4',
    'v9_2.mp4',
    'v6_6.mp4',
    'v2_5.mp4',
    'v16_5.mp4',
    'v7_12.mp4',
    'v4_2.mp4',
    'v2_3.mp4',
    'v14_5.mp4',
    'v2_1.mp4',
    'v16_8.mp4',
    'v15_6.mp4',
    'v2_8.mp4',
    'v2_10.mp4'
]


class VideoDataset(Dataset):
  def __init__(self, path=datapath, 
        names=trainvids, 
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

  def videoPrefix(self, number):
    return self.metadata[number][0]

  def getVideo(self, number):
    startindex = self.cumulFrames[number]
    endindex = self.cumulFrames[number+1]
    return self.allframes[startindex:endindex], self.alllabels[startindex:endindex]



    # Use place to jump back to framesPerVid


if __name__ == '__main__':
    traindata = VideoDataset(load_from_tensorfile=True, 
        windowsize=2, 
        save_to_tensorfile=True, 
        tensorfile_prefix='allframes_224_train',
        names=trainvids)
    print(len(traindata))
    testdata = VideoDataset(load_from_tensorfile=True,
        windowsize=2,
        save_to_tensorfile=True,
        tensorfile_prefix='allframes_224_test',
        names=testvids
        )
    print(len(testdata))
    # print(d.numVideos())
    # vid2 = d.getVideo(2)
    # print(vid2.size())
    # print(d.metadata)
    # print(len(d))
    # print(len(d))
    # out0, out1 = d[96]
    # out0 = out0.numpy().transpose(1,2,0)

    # for j in range(0,len(d),10):
    #     imj, labj = d[j]
    #     print(labj)
    #     plt.imshow(imj.numpy().transpose(1,2,0))
    #     plt.show()
    #     input("Press enter")

    # for j in range(0,len(vid2),10):
    #     imj, labj = d[j]
    #     print(labj)
    #     plt.imshow(imj.numpy().transpose(1,2,0))
    #     plt.show()
    #     input("Enter:")


# cv2.CAP_PROP_FRAME_WIDTH
# cv2.CAP_PROP_FRAME_HEIGHT
      




















    

