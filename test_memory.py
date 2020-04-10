from memory import *

#vid = VideoArray(videofile='/home/ryen/Downloads/stock_video_2.mp4', end_frame=10)


crash_truncate_list = random_proportion_true(len(basename_list), proportion=0.5)
ftracker = BeamNG_FileTracker(datapath, basename_list=basename_list, crash_truncate_list=crash_truncate_list)


videoDataset = VideoDataset(vidfiles=ftracker.file_list(),
                            videoinfo=ftracker.file_info(),
                            return_transitions=True,
                            frames_per_datapoint=3,
                            overlap_datapoints=True,
                            sample_every=5,
                            verbose=True)


print(videoDataset)
print(len(videoDataset))
print(videoDataset.num_videos())
print(videoDataset[-1][0].size())
print(videoDataset[-1][1])
for i in range(videoDataset.num_videos()):
    vid = videoDataset.get_video(i)
    print(vid[-1][0].shape)
    print(vid[-1][1])
    videoDataset.play_video(i)

