from video_loader import datapath, trainvids, testvids 
from memory import VideoDataset, BeamNG_FileTracker, random_proportion_true, frame_transform_2 #need to change name of memory module

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt

#trainvids = ['v1_1.mp4', 'v1_2.mp4', 'v1_3.mp4', 'v1_4.mp4'] # for testing the code
sample_every=5

crash_truncate_list = random_proportion_true(len(trainvids))
ftracker = BeamNG_FileTracker(datapath, basename_list=trainvids, crash_truncate_list=crash_truncate_list)
dataset = VideoDataset(vidfiles=ftracker.file_list(), 
                       videoinfo=ftracker.file_info(),
                       frame_transform=frame_transform_2, #color
                       return_transitions=True,
                       frames_per_datapoint = 1,
                       overlap_datapoints=True,
                       sample_every=sample_every,
                       verbose=False)




m = models.resnet50(num_classes=1) # Need input and output sizes
#m.conv1 = nn.Conv2d(1, m.inplanes, kernel_size=7, stride=2, padding=3, bias=False) #Allow it to take black-and-white ims
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
m = m.cuda()

if True:
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    # Output size should be: (32, 2, 224, 224), (32, 2)

    optimizer = optim.Adam(m.parameters())
    
    n_epochs = 100
    gamma = 0.977 #Chosen so that 5 seconds in the past it's at 0.5, with sampling every 5th frame
#    gamma = 0.995
    
    for epoch in range(n_epochs):
        for i, batch in enumerate(dataloader):
            xtran, ytran = batch
            xtran = xtran.cuda()
            ytran = ytran.cuda()
            
            x_current = xtran[:,0,:,:,:]#.unsqueeze(1)
            y_current = ytran[:,0]
            x_future = xtran[:,1,:,:,:]#.unsqueeze(1)
            y_future = ytran[:,1]
    
    
            with torch.no_grad():
                q_future = gamma*m(x_future)
                q_future[y_future==-2] = 0 #If terminal, only use current reward
                q_future = q_future+y_current
    
            q_current = m(x_current)
    
            loss = ((q_future-q_current)**2).sum()
            print("epoch", epoch, "iter", i, "loss", loss.item())
    
            m.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)%10 == 0:
            torch.save(m.state_dict(), './models/model_fulldata_epoch_{}.pth'.format(epoch))

#    torch.save(m.state_dict(), '.models/model_2.pth')
else:
    m.load_state_dict(torch.load('./models/model.pth'))
    m.eval()

    # Graphing video results
    for i in range(min(dataset.num_videos(),10)):
        q_vals = []
        x_vals = []
        vid = dataset.get_video(i)
        print(vid.labelinfo)
        print(vid.playback_info)
        frame_duration = vid.playback_info['frame_duration']
        for j in range(len(vid)):
            fm, lbl = dataset.access_video(i,j)
            fm = fm.cuda()
            fm = fm.unsqueeze(0) #give a batch dimension
            with torch.no_grad():
                q_pred = m(fm[:,0,:,:,:]).item()
                q_vals.append(q_pred)
                x_vals.append(j*sample_every*frame_duration+vid.labelinfo.starttime)


        plt.ylim(0,1)
        plt.plot(x_vals, q_vals)
        plt.vlines(vid.labelinfo.crashtime,0,1)
        plt.show()
        vid.play()
#        input()

