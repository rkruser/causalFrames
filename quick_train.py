from video_loader import datapath, trainvids, testvids 
from memory import VideoDataset, BeamNG_FileTracker, random_proportion_true, frame_transform_1, frame_transform_2, label_func_1, label_func_2 #need to change name of memory module

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from flexible_resnet import resnet18_flexible, resnet50_flexible

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')

parser.add_argument('--train_data', action='store_true', help='Use full training data')
parser.add_argument('--partial_data', action='store_true', help='Use one video for an example')
parser.add_argument('--test_data', action='store_true', help='Use full test data')
parser.add_argument('--explicit_data', type=str, default=None, help='Load an explicit video')

parser.add_argument('--device', type=str, default='cuda:0', help='Device to run model on')
parser.add_argument('--loadname', type=str, default=None, help='Name of model to load')

parser.add_argument('--even_crash_truncate', action='store_true', help='Truncate every other vid')
parser.add_argument('--random_crash_truncate', action='store_true', help='Truncate random vids')
parser.add_argument('--no_crash_truncate', action='store_true', help='Do not truncate vids')

parser.add_argument('--sample_every', type=int, default=1, help='Frequency of frame sampling')
parser.add_argument('--checkpoint_every', type=int, default=5, help='How often to checkpoint')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--nepochs', type=int, default=100, help="Training epochs")
parser.add_argument('--frames_per_datapoint', type=int, default=2, help="Num frames per datapoint")
parser.add_argument('--overlap_datapoints', type=bool, default=True, help="Overlap datapoints or not")
parser.add_argument('--gamma', type=float, default=0.95, help='RL discount')
parser.add_argument('--model_name', type=str, default='model', help='Name of the model')

opt = parser.parse_args()
print(opt)


############## Data Selection ##############

if opt.train_data:
    vid_list = trainvids
    opt.even_crash_truncate=True
elif opt.partial_data:
    vid_list = ['v1_1.mp4']
    opt.no_crash_truncate=True
elif opt.test_data:
    vid_list = testvids
    opt.no_crash_truncate=True
elif opt.explicit_data is not None:
    vid_list = [opt.explicit_data]
    opt.no_crash_truncate=True
else:
    print("One of opt.train_data, opt.partial_data, opt.test_data must be set")
    exit()
############################################


crash_truncate_list = np.zeros(len(vid_list)).astype('bool')

######## Custom truncate if desired ########
if opt.even_crash_truncate:
    print("Even crash truncate")
    crash_truncate_list[np.arange(len(vid_list))%2 == 0] = True
elif opt.random_crash_truncate:
    print("Random crash truncate")
    crash_truncate_list = random_proportion_true(len(vid_list))  
elif opt.no_crash_truncate:
    print("No crash truncate")
    #crash_truncate_list = np.zeros(len(vid_list)).astype('bool')
############################################


ftracker = BeamNG_FileTracker(datapath, basename_list=vid_list, crash_truncate_list=crash_truncate_list)
dataset = VideoDataset(vidfiles=ftracker.file_list(), 
                       videoinfo=ftracker.file_info(),
                       label_func = label_func_1, #testing -1 now
                       frame_transform=frame_transform_1, #color is transform 2
                       return_transitions=True,
                       frames_per_datapoint = opt.frames_per_datapoint, #Former values: 10, 1, 2, 3
                       overlap_datapoints=opt.overlap_datapoints,
                       sample_every=opt.sample_every,
                       verbose=False,
                       is_color=False)




#m = models.resnet50(num_classes=1) # Need input and output sizes
#model_name = 'resnet50_gray_weighted_3frame_every5th_take2'
#model_name = 'resnet50_10frame_everyframe_gamma_95'
#model_name = 'another_test'
#m = resnet18_flexible(num_classes=1, data_channels=3)
m = resnet50_flexible(num_classes=1, data_channels=opt.frames_per_datapoint)

#m.conv1 = nn.Conv2d(1, m.inplanes, kernel_size=7, stride=2, padding=3, bias=False) #Allow it to take black-and-white ims
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

#m = m.to('cuda:1')


# *****
# This just collapses to 1, but the loss is too low for that to be the case. What's the deal?
# Check loss function for right shape at every step
# Check batches with truncated videos with label 0

if opt.train:
    print("Did you truncate the crashes from some points?")
    m = m.to(opt.device)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    # Output size should be: (32, 2, 224, 224), (32, 2)

    optimizer = optim.Adam(m.parameters())
    
    for epoch in range(opt.nepochs):
        for i, batch in enumerate(dataloader):
            xtran, ytran = batch
            xtran = xtran.to(opt.device)
            ytran = ytran.to(opt.device)

            if opt.frames_per_datapoint > 1:
                x_current = xtran[:,0,:,:,:]
                x_future = xtran[:,1,:,:,:]
            else:
                x_current = xtran[:,0,:,:].unsqueeze(1)
                x_future = xtran[:,1,:,:].unsqueeze(1)

            y_current = ytran[:,0]
            y_future = ytran[:,1]
    
    
            # Check size issues! That was the issue with quick_data!
            weights = torch.ones(len(y_current)).to(opt.device)
            with torch.no_grad():
                q_future = opt.gamma*m(x_future).squeeze()  # What happens at a terminal state?
                inds = torch.abs(y_future+2)<0.0001
                q_future[inds] = 0 #If terminal, only use current reward
                weights[inds] = 64 #Terminal states highly valued
                q_future = q_future+y_current
    
            q_current = m(x_current).squeeze()

            diff = q_future-q_current
            diff = diff**2
            diff = weights*diff
            loss = diff.mean()
    
            #loss = (weights*(q_future-q_current)**2).mean()
            print("epoch", epoch, "iter", i, "root loss", np.sqrt(loss.item()))
    
            m.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)%opt.checkpoint_every == 0:
            torch.save(m.state_dict(), './models/{0}_epoch_{1}.pth'.format(opt.model_name,epoch))

#    torch.save(m.state_dict(), '.models/model_2.pth')



if opt.eval:
    print("Did you make sure to not truncate crashes?")

    m.load_state_dict(torch.load('./models/{}.pth'.format(opt.loadname), map_location=opt.device))
    m=m.to(opt.device)
    m.eval()

    # Graphing video results
    for i in range(min(dataset.num_videos(),100)):
        q_vals = []
        x_vals = []
        vid = dataset.get_video(i)
        print(vid.labelinfo)
        print(vid.playback_info)
        frame_duration = vid.playback_info['frame_duration']
        for j in range(len(vid)):
            fm, lbl = dataset.access_video(i,j)
            fm = fm.to(opt.device)
            fm = fm.unsqueeze(0) #give a batch dimension
            with torch.no_grad():
                if opt.frames_per_datapoint > 1:
                    q_pred = m(fm[:,0,:,:,:]).item()
                else:
                    q_pred = m(fm[:,0,:,:].unsqueeze(1)).item()

                q_vals.append(q_pred)
                x_vals.append(j*opt.sample_every*frame_duration+vid.labelinfo.starttime)

        q_vals = np.array(q_vals)
        x_vals = np.array(x_vals)
        x_vals = x_vals + (opt.frames_per_datapoint-1)*frame_duration #Shift time to end of prediction

        #fig = plt.figure()
        #plt.ylim(0,1)
        plt.plot(x_vals, q_vals)
        plt.vlines(vid.labelinfo.crashtime,0,1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Q-value')
        plt.title('Predicted crash potential')

        ########### Saving #########
        folder = './results/{}'.format(opt.loadname)
        fname = '{}.jpg'.format(vid.labelinfo.name)
        if opt.test_data:
            folder = os.path.join(folder,'test')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        savepath = os.path.join(folder,fname)
        plt.savefig(savepath)
        ############################

        plt.clf()
         

