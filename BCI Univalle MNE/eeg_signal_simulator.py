"""Example program to demonstrate how to send a multi-channel time-series
with proper meta-data to LSL."""

"""Example program to demonstrate how to send a multi-channel time-series
with proper meta-data to LSL."""

import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock


def main():
    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover).
    info = StreamInfo('BioSemi', 'EEG', 14, 128, 'float32', 'myuid2424')

    # append some meta-data
    info.desc().append_child_value("manufacturer", "BioSemi")
    channels = info.desc().append_child("channels")
    for c in ["C3", "C4", "Cz", "FPz", "POz", "CPz", "O1", "O2"]:
        channels.append_child("channel") \
            .append_child_value("label", c) \
            .append_child_value("unit", "microvolts") \
            .append_child_value("type", "EEG")

    # next make an outlet; we set the transmission chunk size to 32 samples and
    # the outgoing buffer size to 1000 seconds (max.)
    outlet = StreamOutlet(info, 32, 1000)

    stamp = local_clock()
    tInicial = stamp
    while True:
        if(local_clock()-stamp >= (1./128)):
            stamp = local_clock()
            # make a new random 8-channel sample; this is converted into a
            # pylsl.vectorf (the data type that is expected by push_sample)
            mysample = [rand(), rand(), rand(), rand(), rand(), rand(), 
            rand(), rand(), rand(), rand(), rand(), rand(), 
            rand(), rand()]
            # get a time stamp in seconds (we pretend that our samples are actually
            # 125ms old, e.g., as if coming from some external hardware)
            # now send it and wait for a bit
            outlet.push_sample(mysample, stamp-tInicial)


if __name__ == '__main__':
    main()

