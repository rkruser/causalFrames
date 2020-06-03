from video_loader import datapath, trainvids, testvids 
from memory import VideoDataset, BeamNG_FileTracker, random_proportion_true, frame_transform_1, frame_transform_2, label_func_1, label_func_2, label_func_prediction_only #need to change name of memory module

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle

import matplotlib.pyplot as plt

from flexible_resnet import resnet18_flexible, resnet50_flexible

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resultsgen', action='store_true')
parser.add_argument('--nograph', action='store_true')
parser.add_argument('--picklefile', type=str, default='results.pkl', help='Name of the file to dump the qval info')

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

# Arguments for metrics and plotting

# Scratch that, just load the data once, load a bunch of models sequentially from a list,
#  and then save their outputs on each video into a pickle file that can be loaded by results.py




opt = parser.parse_args()
print(opt)


############## Data Selection ##############

if opt.train_data:
    vid_list = trainvids
    if not opt.no_crash_truncate:
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
                       label_func = label_func_prediction_only, #testing -1 now
                       frame_transform=frame_transform_1, #color is transform 2
                       return_transitions=False, # **** One main difference between this and quick_train.py
                       frames_per_datapoint = opt.frames_per_datapoint, #Former values: 10, 1, 2, 3
                       overlap_datapoints=opt.overlap_datapoints,
                       sample_every=opt.sample_every,
                       verbose=False,
                       is_color=False,
                       label_postprocess=False
                       )




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
            xcurrent, ycurrent = batch
            xcurrent = xcurrent.to(opt.device)
            ycurrent = ycurrent.to(opt.device) #Need to change what is returned by y_current

            p_current = torch.sigmoid(m(xcurrent)).squeeze()
            loss = torch.nn.functional.binary_cross_entropy(p_current, ycurrent)

            print("epoch", epoch, "iter", i, "root loss", loss.item()/len(ycurrent))
    
            m.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)%opt.checkpoint_every == 0:
            torch.save(m.state_dict(), './models/{0}_epoch_{1}.pth'.format(opt.model_name,epoch))

    torch.save(m.state_dict(), './models/{0}_epoch_{1}.pth'.format(opt.model_name,opt.nepochs))



# Upgraded eval needs to save x_vals and q_vals into a file, mapped to the names of the videos
# Do this for each model, for train and test
# Make a list of model names to load during the same run, so that the data need only be loaded once
# Load both train and test (memory can handle it)
# Uhhh... is it possible to change the return parameters of video data objects on the fly?
#   Not yet... need to load every frame of the data, then construct various videodata objects out of that depending on model
if opt.eval:
    print("Did you make sure to not truncate crashes?")

    m.load_state_dict(torch.load('./models/{}.pth'.format(opt.loadname), map_location=opt.device))
    m=m.to(opt.device)
    m.eval()

    results = {}

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
                    q_pred = torch.sigmoid(m(fm)).item()
                else:
                    q_pred = torch.sigmoid(m(fm.unsqueeze(1))).item() # ???????

                q_vals.append(q_pred)
                x_vals.append(j*opt.sample_every*frame_duration+vid.labelinfo.starttime)

        q_vals = np.array(q_vals)
        x_vals = np.array(x_vals)
        x_vals = x_vals + (opt.frames_per_datapoint-1)*frame_duration #Shift time to end of prediction

        results[vid.labelinfo.name] = (x_vals, q_vals, vid.labelinfo, vid.playback_info)

        #fig = plt.figure()
        #plt.ylim(0,1)
        if not opt.nograph:
            plt.plot(x_vals, q_vals)
            plt.vlines(vid.labelinfo.crashtime,0,1)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Probability-value')
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


    pickle.dump(results, open(opt.picklefile, 'wb'))
         
if opt.resultsgen:
    pass
































