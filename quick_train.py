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

#trainvids = ['v5_1.mp4', 'v2_1.mp4', 'v2_2.mp4', 'v2_3.mp4', 'v2_4.mp4'] # for testing the code
#trainvids = ['v1_1.mp4', 'v1_2.mp4', 'v1_3.mp4', 'v1_4.mp4', 'v1_5.mp4']
#trainvids = ['v1_1.mp4']
trainvids=testvids

#crash_truncate_list = random_proportion_true(len(trainvids)) 
#crash_truncate_list = np.zeros(len(trainvids)).astype('bool')
#crash_truncate_list[np.arange(len(trainvids))%2 == 0] = True #Truncate even videos

crash_truncate_list = np.zeros(len(trainvids)).astype('bool')  ## !!!! Was this my problem?!?!

sample_every=1
checkpoint_every = 5

ftracker = BeamNG_FileTracker(datapath, basename_list=trainvids, crash_truncate_list=crash_truncate_list)
dataset = VideoDataset(vidfiles=ftracker.file_list(), 
                       videoinfo=ftracker.file_info(),
                       label_func = label_func_1, #testing -1 now
                       frame_transform=frame_transform_1, #color is transform 2
                       return_transitions=True,
                       frames_per_datapoint = 10, #Windows of 3 frames
                       overlap_datapoints=True,
                       sample_every=sample_every,
                       verbose=False,
                       is_color=False)




#m = models.resnet50(num_classes=1) # Need input and output sizes
#model_name = 'resnet50_gray_weighted_3frame_every5th_take2'
model_name = 'resnet50_10frame_everyframe_gamma_95'
#model_name = 'another_test'
#m = resnet18_flexible(num_classes=1, data_channels=3)
m = resnet50_flexible(num_classes=1, data_channels=10)

#m.conv1 = nn.Conv2d(1, m.inplanes, kernel_size=7, stride=2, padding=3, bias=False) #Allow it to take black-and-white ims
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

#m = m.to('cuda:1')


# *****
# This just collapses to 1, but the loss is too low for that to be the case. What's the deal?
# Check loss function for right shape at every step
# Check batches with truncated videos with label 0

if False:
    m = m.to('cuda:1')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    # Output size should be: (32, 2, 224, 224), (32, 2)

    optimizer = optim.Adam(m.parameters())
    
    n_epochs = 100
    gamma = 0.95 #Chosen so that 5 seconds in the past it's at 0.5, with sampling every 5th frame
#    gamma = 0.995 # 0.977
    
    for epoch in range(n_epochs):
        for i, batch in enumerate(dataloader):
            xtran, ytran = batch
#            xtran = xtran.cuda()
#            ytran = ytran.cuda()
            xtran = xtran.to('cuda:1')
            ytran = ytran.to('cuda:1')
            
            x_current = xtran[:,0,:,:,:]#.unsqueeze(1)
#            x_current = xtran[:,0,:,:].unsqueeze(1)
            y_current = ytran[:,0]
            x_future = xtran[:,1,:,:,:]#.unsqueeze(1)
#            x_future = xtran[:,1,:,:].unsqueeze(1)

            y_future = ytran[:,1]
    
    
            # Check size issues! That was the issue with quick_data!
            weights = torch.ones(len(y_current)).to('cuda:1')
            with torch.no_grad():
                q_future = gamma*m(x_future).squeeze()  # What happens at a terminal state?
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

        if (epoch+1)%checkpoint_every == 0:
            torch.save(m.state_dict(), './models/{0}_epoch_{1}.pth'.format(model_name,epoch))

#    torch.save(m.state_dict(), '.models/model_2.pth')



else:


    #load_name = 'resnet18_every_1_neg_epoch_4'
#load_name = 'resnet50_gray_weighted_3frame_every5th_take2_epoch_99'
    load_name = 'resnet50_10frame_everyframe_epoch_99'
    m.load_state_dict(torch.load('./models/{}.pth'.format(load_name)))
    m = m.to('cuda:1')
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
            #fm = fm.cuda()
            fm = fm.to('cuda:1')
            fm = fm.unsqueeze(0) #give a batch dimension
            with torch.no_grad():
                q_pred = m(fm[:,0,:,:,:]).item()
#                q_pred = m(fm[:,0,:,:].unsqueeze(1)).item()

                q_vals.append(q_pred)
                x_vals.append(j*sample_every*frame_duration+vid.labelinfo.starttime)

        #fig = plt.figure()
        #plt.ylim(0,1)
        plt.plot(x_vals, q_vals)
        plt.vlines(vid.labelinfo.crashtime,0,1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Q-value')
        plt.title('Predicted crash potential')
        plt.savefig('./results/{0}/test/{1}.jpg'.format(load_name, vid.labelinfo.name))
        plt.clf()
         
#        plt.show()
#        vid.play()
#        input()

