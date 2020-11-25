"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream, local_clock
import numpy as np


def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    data  = []
    stamp = local_clock()
    timeOut = 0.5 #segundos
    while (True):
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample(timeout=timeOut)
        data.append(np.hstack((timestamp, sample)))
        if(local_clock()-stamp >= timeOut):
            print("Error")
            break
        stamp = local_clock()
    
    data = np.asarray(data)
    print(data.shape)
    data[:,0] = data[:,0]-data[0,0]
    print(data)
    print(data.shape)

    info = inlet.info()
    ch = info.desc().child("channels").child("channel")
    for k in range(info.channel_count()):
        print("  " + ch.child_value("label"))
        ch = ch.next_sibling()
    
    print(data[-1,:])

if __name__ == '__main__':
    main()