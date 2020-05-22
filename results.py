# Provides code for an object that takes one time-series / label at a time
#   and computes online accuracy/time difference of prediction to crash
#   and also AUROC

# Maybe also, later on, sweep hyperparams in an automated way and plot their effects on outcome

import numpy as np
import os
import pickle
import argparse



# Get the first array index whose array value exceeds the threshold
# series is a numpy array
def first_past_value(series, threshold):
    markers = (series >= threshold)
    candidates = markers.nonzero()[0]
    if len(candidates) == 0:
        return -1
    else:
        return candidates[0]
    

# Fits a 0-1 step function to the series
# Problem: this presumes you know the future; can't really be used online 
def best_step_fit(series):
    l = len(series)
    scores = np.zeros(l+1)
    for i in range(l+1):
        before = series[:i]
        after = 1-series[i:]
        scores[i] = np.sum(before*before) + np.sum(after*after)

    argmin = np.argmin(scores)
    if argmin >= l:
        return -1
    else:
        return argmin


def max_discrete_gradient(series):
    kernel = np.array([1,-1])
    result = np.convolve(series,kernel,'valid')
#    print(result)
    position = np.argmax(result)+np.floor((len(kernel)/2.0))
    return position


def calculate_stats(picklefile, seriesEval, sampleRate=10, overlap=True):
    stats = pickle.load(open(picklefile,'rb'))
    differentials = []
# (xvals, qvals, labelinfo, playbackinfo)
    for vidname in stats:
        xvals, qvals, labelinfo, playbackinfo = stats[vidname]
        predicted_position = seriesEval(qvals)
        if predicted_position < 0:
            print("No crash predicted for", vidname)
            continue
        if overlap:
            predicted_time = (predicted_position+sampleRate-1)/playbackinfo['fps']+labelinfo.starttime
        else:
            pass

        difference = predicted_time-labelinfo.crashtime
        print(vidname, difference)
        differentials.append(difference)

    differentials = np.array(differentials)
    mean = differentials.mean()
    std = differentials.std()

    newFileName = os.path.splitext(picklefile)
    newFileName = newFileName[0]+'_processed'+newFileName[1]

    print("Mean time difference", mean, "Std time difference", std)
#    pickle.dump({'mean':mean, 'std':std, 'differentials':differentials, 'keys':list(stats.keys())},
#                open(newFileName, 'wb'))






def test():
    a = np.zeros(10)
    a[7:] = 1
    print(a)

    print("First past", first_past_value(a,0.5))
    print("Best fit", best_step_fit(a))
    print("Max gradient", max_discrete_gradient(a))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--picklefile', type=str, default='pickles/results.pkl', help='Pickle file to load')
    opt = parser.parse_args()

#    calculate_stats(opt.picklefile, lambda x: first_past_value(x,0.9), sampleRate=10, overlap=True)
    calculate_stats(opt.picklefile, best_step_fit, sampleRate=10, overlap=True)



