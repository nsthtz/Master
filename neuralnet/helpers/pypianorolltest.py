import pypianoroll
import os

import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt

# Create a piano-roll matrix, where the first and second axes represent time
# and pitch, respectively, and assign a C major chord to the piano-roll
""""
for filename in os.listdir("C:\DeepLearning\Master\\neuralnet\datasets\\training"):
    print(filename)
    pianoroll = pypianoroll.parse("C:\DeepLearning\Master\\neuralnet\datasets\\training\\" + filename, beat_resolution=24, name='unknown')
    del pianoroll.tracks[4:]
    print(pianoroll)
    pypianoroll.write(pianoroll, "C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly\\" + filename)
"""

# Create a `pypianoroll.Track` instance


pianoroll = pypianoroll.parse("C:\DeepLearning\Master\\neuralnet\datasets\\training\\AC-11-Wann.mid", beat_resolution=24, name='unknown')
print(pianoroll)
del pianoroll.tracks[4:]
print(pianoroll)
pypianoroll.write(pianoroll, "C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly\\AC-11-Wann.mid")


# Plot the piano-roll
#fig, ax = pianoroll.plot()
#plt.show()

