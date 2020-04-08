# memory.py
# Maintain a "replay" memory for Q-learning
# Input: A list of sequence lengths in frames
#  [ vid1length, vid2length, vid3length, ..., vidnlength]


# videofile: mp4 file name
# videolabel: a named tuple consisting of attributes of the video, like crash time, etc.
# 


# Somewhere I need a loader function to truncate vids, etc.
import numpy as np
import cv2
import matplotlib.pyplot as plt

T = 14

class Zipindices:
    # length_list : a list of integers specifying lengths of other objects
    def __init__(self, length_list):
        self.llist = np.array(length_list)
        self.clist = np.cumsum(length_list)-1
        
    def __len__(self):
        if len(self.clist)>0:
            return self.clist[-1]+1
        else:
            return 0
        
    def __getitem__(self, key):
        if key < 0:
            key = self.__len__()+key
        if key < 0 or key >= self.__len__():
            raise KeyError('Zipindices key out of bounds')
        place = np.searchsorted(self.clist, key)
        if place==0:
            return place, key
        else:
            return place, key-self.clist[place-1]-1

class Zipindexibles:
    # indexibles is an iterable of objects with len and getitem defined
    def __init__(self, indexibles, slice_is_list=True):
        self.items = indexibles
        self.slice_is_list = slice_is_list
        self._buildindex()
        
    def _buildindex(self):
        llist = []
        for l in self.items:
            llist.append(len(l))
        self.zipindex = Zipindices(llist)       
        
    def __len__(self):
        return len(self.zipindex)
    
    def __getitem__(self, key):
        if isinstance(key,int) or isinstance(key,np.int64):
            return self._access(key)
        elif isinstance(key, slice):
            return self._slice(key)
        elif isinstance(key, np.ndarray):
            if len(key.shape)==1:
                if key.dtype=='bool':
                    return self._bool_array_index(key)
                elif 'int' in str(key.dtype):
                    return self._int_array_index(key)
                else:
                    raise KeyError('Zipindexibles key array has invalid type')
            else:
                raise(KeyError('Zipindexibles key array has wrong number of dimensions'))
        else:
            try:
                return self._access(key)
            except KeyError as e:
                raise KeyError('Zipindexibles key error') from e
            
    def get_indexible(self, key):
        return self.items[key]
    
    def num_indexibles(self):
        return len(self.items)
    
    def __str__(self):
        return self.__repr__()+', contents = '+str(self.items)
        
    def __repr__(self):
        return 'Zipindexibles object, indexibles {0}, items {1}'.format(
            self.num_indexibles(), self.__len__())
            
    def _access(self, key):
        place, ind = self.zipindex[key]
        return self.items[place][ind]
        
    # Not maximally efficient, but whatever
    def _slice(self, key):
        start = key.start
        stop = key.stop
        step = key.step
        if start is None:
            start = 0
        if step is None:
            step = 1
        if stop is None:
            stop = self.__len__()
            
        if self.slice_is_list:
            return self._int_array_index(np.arange(start,stop,step))
        
        if step < 0:
            print("Warning: negative step size produces undefined behavior when slicing a Zipindexibles object")
        
        place_list = []
        place_inds = {}
        for i in range(start, stop, step):
            place, ind = self.zipindex[i]
            if place not in place_inds:
                place_inds[place] = [ind, ind]
                place_list.append(place)
            else:
                place_inds[place][1] = ind
            
        new_items = []
        for j in place_list:
            sl = place_inds[j]
            new_items.append(self.items[j][sl[0]:sl[1]+step:step])
                             
        return Zipindexibles(new_items)
            
    def _int_array_index(self, key):
        all_items = []
        for i in key:
            all_items.append(self._access(i))
        return all_items
        
    def _bool_array_index(self, key):
        if len(key) != self.__len__():
            raise KeyError('Zipindexibles numpy boolean key has wrong length')
        all_items = []
        for i in range(len(key)):
            if key[i]:
                all_items.append(self._access(i))
        return all_items
    



'''
video_object is a sliceable object with the interface of the Video class.
videoinfo is a named tuple with external attributes of the video
'''
def vid_filter(video_object, videoinfo):
    pass


def ftransform(frame):
    pass


# Returns a generator that takes a valid videocapture object
#  and iterates through its frames, returning frame number and frame
#def video_frame_generator(videocapture_object):
#    pass


# Instead, have video file wrapper that provides an interface for
#  loading frames. Can swap between cv2 and ffmpeg more easily that way
#  Video class below uses this wrapper
class VideoWrapper:
    class VideoLoadError(Exception):
        def __init__(self, message='Video did not load'):
            super().__init__(message)
    
    # Need absolute path of file apparently
    def __init__(self, filename):
        self.filename = filename
        self.video_object = cv2.VideoCapture(self.filename)
        if not self.video_object.isOpened():
            raise VideoWrapper.VideoLoadError()
        self.frame_pos = 0
        
        self.num_frames = int(self.video_object.get(cv2.CAP_PROP_FRAME_COUNT))
        self.framerate = self.video_object.get(cv2.CAP_PROP_FPS)
        self.width = self.video_object.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_object.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_duration = 1.0/self.framerate       
        self.duration = self.num_frames*self.frame_duration
        
    def __len__(self):
        return self.num_frames
    
    # This getitem is slow and is not recommended for use in a performance loop
    # Instead, use the iterator
    def __getitem__(self, n):
        if n < 0:
            n = self.__len__()+n
        if n < 0 or n >= self.__len__():
            raise KeyError('VideoWrapper key out of bounds')
        ret, frame = self._nthframe(n)
        if not ret:
            raise KeyError('Could not return frame {}'.format(n))
        return frame
    
    def _nthframe(self, n):
        self._setframe(n)
        return self._read_cur_frame()
    
    def _setframe(self, n):
        self.video_object.set(cv2.CAP_PROP_POS_FRAMES, n)
        self.frame_pos = n
        
    def _read_cur_frame(self):
        ret, frame = self.video_object.read()
        self.frame_pos += 1
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV is BGR by default
        return ret, frame
    
    # Time in seconds
    def framenum_at_time(self, time):
        return int(time*self.framerate)
    
    def reset_video(self):
        self._setframe(0)
    
    def __iter__(self):
        self.reset_video()
        return self
    
    def __next__(self):
        ret, frame = self._read_cur_frame()
        if ret:
            return frame
        else:
            if self.frame_pos < self.num_frames:
                print("Warning: VideoWrapper iterator exited before end of video")
            raise StopIteration
            
    def play(self, scale=0.25, playback_speed=1.0, window_name=None):
        if window_name is None:
            window_name = self.filename
        play_rgb(self, frame_duration=self.frame_duration, scale=scale, playback_speed=playback_speed,
                 window_name=window_name)
            
    def _cleanup(self):
        self.video_object.release()
            
#    def __del__(self):
#        self._cleanup()
        
    def __enter__(self):
        return self
        
    def __exit__(self):
        self._cleanup()
        

        
def play_rgb(frames, frame_duration=40, fps=None, scale=0.25, playback_speed=1.0, window_name='video'):
    print("Press q to stop playback")

    playback_multiplier = 1.0/playback_speed
    
    if fps is not None:
        frame_duration = 1.0/fps

    #If duration is too short then video won't work
    frame_duration_ms = int(frame_duration*1000*playback_multiplier) 

    continue_playing = True
    while continue_playing:
        for frame in frames:
            frame = cv2.resize(frame, (int(np.size(frame,1)*scale), int(np.size(frame,0)*scale)))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #undo RGB for playing
            cv2.imshow(window_name, frame)
            if cv2.waitKey(frame_duration_ms) & 0xFF == ord('q'):
                continue_playing = False
                break
    cv2.destroyAllWindows()       
        
        
# Treat a numpy array as a video file
# Or possibly just an object that can be indexed like a numpy array
# Possibly just merge this with vidLoader instead of separating them
# Have a base class
class Video:
    '''
    videofile is a string with a path to the file (must be absolute)
    videolabel is a named tuple with external attributes of the video
    '''
    def __init__(self, 
                 videofile, 
                 videoinfo, 
                 frames_per_datapoint=1, 
                 overlap_datapoints=True, #only relevant if more than 1 frame per datapoint
                 frame_subtraction=False,
                 return_transitions=False,
                 frame_transform=None, #Transformation to apply per-frame
                 sample_every=1, #Rate to keep frames
                 start_frame=0,
                 end_frame=-1
                ):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            pass
        else:
            pass
        

    
    @staticmethod
    def _load_video(videofile):
        pass
    
    @property
    def num_frames(self):
        pass
    
    @property
    def num_bundles(self):
        pass
    
    @property
    def num_frame_transitions(self):
        pass
    
    @property
    def num_bundle_transitions(self):
        pass
    
    @property
    def frame_rate(self):
        pass
    
    @property
    def frame_duration(self):
        pass
        
    def getframe(self, key):
        pass
    
    def getbundle(self, key):
        pass
    
    def getframetransition(self, key):
        pass
    
    def getbundletransition(self, key):
        pass
    
    

    
    

class VideoDataset:
    '''
    Vidfiles is a list of full video file strings
    videoinfo is a dict associating file basenames (no extension) with named tuples of video info.
    vid_filter takes video objects and processes them based on videoinfo
    '''
    def __init__(self, vidfiles, videoinfo, vid_filter, frames_per_datapoint=1, frame_subtraction=False,
                 return_transitions=True):
        pass
    
    @staticmethod
    def _apply_filter(vidfile, vidinfo, vid_filter):
        pass
    
    def __len__(self):
        pass
    
    def __getitem(self, i):
        pass

    @property
    def num_videos(self):
        pass
    
    def get_video(self,i):
        pass
    
    