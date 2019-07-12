from video_loader import VideoDataset


import torchvision.models as models


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

#print(len(testdata))


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
            
def train(nepochs, outname='beamng_model_resnet.pth', windowsize=1, cont=False):
		 #keepEvery=5, nhidden=64):
    model = models.resnet18(pretrained=False, num_classes=1)
   # model = models.alexnet(pretrained=False, )
    traindata = VideoDataset(load_from_tensorfile=True, 
	    windowsize=windowsize, 
	    save_to_tensorfile=False, 
	    tensorfile_prefix='allframes_224_train')

	#print(len(traindata))
	# testdata = VideoDataset(load_from_tensorfile=True,
	#     windowsize=windowsize,
	#     save_to_tensorfile=False,
	#     tensorfile_prefix='allframes_224_test'
	#     )

    device='cuda:0'
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True, 
    	num_workers=4)
    
    #model = MNIST_Net(6,1)
    #model.apply(weights_init)
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
        for i, data in enumerate(trainloader):
            # Extract training batches
            ims, labels = data
            device_labels = labels.float().to(device)
            device_ims = ims.to(device)

            # Train network
            model.zero_grad()
            predictions = model(device_ims).squeeze(1)
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


def runHuge(model, data, maxbatchsize=128, device='cuda:0'):
    nchunks = (len(data)//maxbatchsize) + 1
    chunks = torch.chunk(data, nchunks)
    outputs = []
    for i in range(nchunks):
        devChunk = chunks[i].to(device)
        out = model(devChunk)
        outputs.append(out.to('cpu'))
    return torch.cat(outputs).detach().sigmoid().numpy()
  
    
def write_gif(vid, fname, special_frame=None):
    vid = vid.numpy()
#     if predicted_choice is not None:
#         pts[predicted_choice:predicted_choice+6,1:3,:,:] = 0
    if special_frame is not None:
	    end_idx = min(len(vid),special_frame+5)
	    vid[special_frame:end_idx,1:3,:,:].fill(0)

    vid = np.transpose((vid*255).astype('uint8'),(0,2,3,1))
    #write_gif(pts*255, fname, fps=6)
    imageio.mimsave(fname,vid,fps=5)



def test(modelname='beamng_model_resnet.pth', windowsize=1):
    testmodel = models.resnet18(pretrained=False, num_classes=1)  #Simple_Video_Net(windowsize,1,chan=nhidden)
    testmodel.load_state_dict(torch.load(modelname))
    testmodel = testmodel.to('cuda:0')
    testmodel.eval()
    testdata = VideoDataset(load_from_tensorfile=True, 
	    windowsize=windowsize, 
	    save_to_tensorfile=False, 
	    tensorfile_prefix='allframes_224_test')

    numVideos = testdata.numVideos()
    for j in range(numVideos):
    	vid, labels = testdata.getVideo(j)
    	labels = labels.numpy()
    	vidPrefix = testdata.videoPrefix(j)
    	outputs = runHuge(testmodel, vid)
    	nonzero_inds = np.nonzero(labels)[0]
    	if len(nonzero_inds) > 0:
    		special_frame = np.nonzero(labels)[0][0] # First index of crash
    	else:
    		special_frame = None
    	if (j>24):
	    	write_gif(vid, vidPrefix+'_render_special.gif', special_frame=special_frame)
	    	plt.plot(outputs)
	    	plt.plot(labels)
	    	plt.title(vidPrefix)
	    	plt.savefig(vidPrefix+'_plot.png')
	    	plt.show()
	    	input("Press enter")

    #estdata = VideoLoader(fnames=vidnames, windowsize=windowsize)
    
    # nsafe = testdata.labelLength(0, count_videos=True)
    # ncrashes = testdata.labelLength(1, count_videos=True)
    # print("Crash test vid predictions")
    # for j in range(ncrashes):
    #     vid = testdata.getVideo(1, j)
    #     vidname = os.path.splitext(testdata.getVideoName(1,j))[0]
    #     print(vidname)
    #     result = runHuge(testmodel, vid)
    #     plt.plot(result)
    #     plt.show()
    #     input("Press enter")
    #   #  write_gif(vid, vidname+'.gif')
    # print("No crash test vid predictions")
    # for j in range(nsafe):
    #     vid = testdata.getVideo(0, j)
    #     vidname = os.path.splitext(testdata.getVideoName(0,j))[0]
    #     print(vidname)
    #     result = runHuge(testmodel, vid)
    #     plt.plot(result)
    #     plt.show()
    #     input("Press enter")


if __name__ == '__main__':
	#train(10)
	test()