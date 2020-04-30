import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



END_MARKER = 5

# Define simple NN class

class SimpleAutoregressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.net = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size,hidden_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_size,output_size),
#                        nn.Tanh()
                        )

    def forward(self, x):
        return self.net(x)



def get_values(start, winsize, switch, granularity): #, stdev):
    t = torch.zeros(winsize)
    if switch >= start:
        if switch < start+winsize:
            pre_switch_size = switch-start
            post_switch_size = winsize-pre_switch_size
            t[pre_switch_size:] = granularity*torch.arange(post_switch_size)
    else:
        t = granularity*torch.arange(start-switch, start-switch+winsize)


    return t


# Define batch loader (For moving MNIST and single-data-point)

def get_series_batch(batch_size, window_size, event_probability=0.5, 
                     terminal_probability=0.1, sequence_length=10, granularity = 0.1, stdev=0.05):
    # Terminal probability: prob that a given data point will have the terminal reward value
    # event_probability: prob that the given data point will be part of a sequence with terminal reward 1


        #t = t + stdev*torch.randn(winsize)

    # Start from 0 and count up to sequence length
    # Function = piecewise {0+noise  up to random x, then x+noise the rest of the way}
    #  if random x > sequence length, label is 0, otherwise label is 1
    max_points = int(sequence_length/granularity)
    max_startpoint = max_points-window_size
    
    batch = torch.zeros(batch_size, 2, window_size)
    labels = torch.zeros(batch_size,2)
    keypoints = torch.zeros(batch_size,2)
    for i in range(batch_size):
        event_happens = np.random.rand()
        is_terminal = np.random.rand()

        if event_happens < event_probability:
            terminal_label = 1.0
            switch_point = np.random.randint(max_startpoint)
        else:
            terminal_label = 0.0
            switch_point = max_points+10

        if is_terminal<terminal_probability:
            labels[i,0] = terminal_label
            labels[i,1] = END_MARKER #End marker, not a reward
            startpoint = max_startpoint
        else:
            labels[i,0] = 0.0
            startpoint = np.random.randint(max_startpoint)
            if startpoint+1 == max_startpoint:
                labels[i,1] = terminal_label
            else:
                labels[i,1] = 0.0

        keypoints[i,0] = startpoint
        keypoints[i,1] = switch_point

        noise = stdev*torch.randn(window_size+1) #Noise here for consistency

        batch[i,0,:] = get_values(startpoint, window_size, switch_point, granularity) + noise[:window_size]

        # Get transition
        if startpoint < max_startpoint:
            batch[i,1,:] = get_values(startpoint+1, window_size, switch_point, granularity) + noise[1:]

    return batch, labels, keypoints


# Get a batch of a single run in sequence
def get_sequential_batch(sequence_length, winsize, switch_val, granularity=0.1, stdev=0.05):
    max_points = int(sequence_length/granularity)
    max_startpoint = max_points-winsize
    switch = int(switch_val/granularity)

    batch = torch.zeros(max_startpoint, winsize)
    noise = stdev*torch.randn(max_points)
    for i in range(max_startpoint):
        batch[i,:] = get_values(i,winsize,switch,granularity)+noise[i:i+winsize]

    xran = granularity*torch.arange(max_startpoint)

    return xran, batch
    

# Define Training loop and output graph printing after each epoch


def train():
    winsize=10
    batch_size = 32
    sequence_length = 50
    standard_split_point = 25
    granularity = 0.1
    stdev=10
    event_probability = 0.5
    terminal_probability = 0.01

    n_epochs = 10
    batches_per_epoch=1000
    gamma = 0.9999
    
    M = SimpleAutoregressor(winsize,64,1) #!!
    optimizer = optim.Adam(M.parameters())
#optimizer = optim.RMSprop(M.parameters())
    
    standard_xran, standard_test_batch = get_sequential_batch(sequence_length,winsize,standard_split_point,granularity=granularity,stdev=stdev)
    plt.figure(0)
    plt.plot(standard_xran, standard_test_batch[:,0])
    plt.title('Standard test batch')
    plt.show()
    input()

    for epoch in range(n_epochs):
        for i in range(batches_per_epoch):
            batch, labels, _ = get_series_batch(batch_size,winsize,
                            event_probability=event_probability,
                            terminal_probability=terminal_probability,
                            sequence_length=sequence_length,
                            granularity=granularity,
                            stdev=stdev)
            x_current = batch[:,0,:]
            x_future = batch[:,1,:]
            y_current = labels[:,0]
            y_future = labels[:,1]
    
            with torch.no_grad():
                q_future = gamma*M(x_future).squeeze()
                q_future[torch.abs(y_future-END_MARKER)<0.0001] = 0
                q_future = q_future+y_current
    
            q_current = M(x_current).squeeze()
            loss = ((q_future-q_current)**2).mean() # Loss is mean squared error
    
            if i%100 == 0:
#                print("y_current", y_current, "y_future", y_future, "q_future", q_future)
                print('Epoch {0} batch {1}: root loss={2}'.format(epoch, i, np.sqrt(loss.item())))
#                input()
    
            M.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            test_predictions = M(standard_test_batch).squeeze()
            plt.figure(epoch+1)
            plt.plot(standard_xran, test_predictions)
#            plt.ylim(0,1)
            plt.title('Epoch {} q-values'.format(epoch))
            plt.show()
            input()
            





def test1():
    batch_size = 10
    winsize = 5
    granularity = 0.01
    batch, labels, keypoints = get_series_batch(batch_size,winsize,
                        event_probability=0.5,
                        terminal_probability=0.1,
                        sequence_length=1,
                        granularity=granularity,
                        stdev=0.05)
    print(labels)
    print(keypoints)

    for i in range(batch_size):
        plt.scatter(granularity*torch.arange(keypoints[i,0],keypoints[i,0]+winsize), batch[i,0])
        if labels[i,1] != END_MARKER:
            plt.scatter(granularity*torch.arange(keypoints[i,0]+1,keypoints[i,0]+winsize+1), batch[i,1])

        plt.show()
        input()
        plt.clf()
#    plt.show()




if __name__ == '__main__':
#    test1()
    train()
