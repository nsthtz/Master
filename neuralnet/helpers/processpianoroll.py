
import numpy as np
import pypianoroll
import os
import sys
import copy
import datapreparation as datp
from pypianoroll import Multitrack, Track

#np.set_printoptions(threshold=sys.maxsize)


""" Step 1: Extract only SATB voices from midi file"""
files = [f for f in os.listdir('C:\DeepLearning\Master\\neuralnet\datasets\\training') if os.path.isfile(os.path.join('C:\DeepLearning\Master\\neuralnet\datasets\\training', f))]
print("Extracting SATB voices from midi...")
for filename in files:

    pianoroll = pypianoroll.parse("C:\DeepLearning\Master\\neuralnet\datasets\\training\\" + filename, beat_resolution=24, name='unknown')
    del pianoroll.tracks[4:]
    pypianoroll.write(pianoroll, "C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly\\" + filename)

    # Not used, unnecessarily slow
    """
    for i in range(1, 7):
        transposed = copy.deepcopy(pianoroll)
        transposed.transpose(i)
        pypianoroll.write(transposed, "C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly\\" + filename[:-4] + str(i) + ".mid")
        transposed = copy.deepcopy(pianoroll)
        transposed.transpose(-i)
        pypianoroll.write(transposed, "C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly\\" + filename[:-4] + str(-i) + ".mid")
    """

""" Step 2: Create .csv piano roll files"""
print("Creating .csv pianoroll files...")
for filename in files:

    # Sampling rate: default 1 per sec
    fs = 1

    df = datp.midfile_to_piano_roll("C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly\\" + filename, fs)
    df.to_csv("C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly\\piano_roll_fs_1\\"+filename[:-3]+"csv")

""" Step 3: Remove large pauses and make transposed duplicates to increase training size """
files = [f for f in os.listdir('../datasets/training/voicesonly/piano_roll_fs_1') if os.path.isfile(os.path.join('../datasets/training/voicesonly/piano_roll_fs_1', f))]

print("Contracting verses...")
for filename in files:
    matrix = np.genfromtxt('../datasets/training/voicesonly/piano_roll_fs_1/' + filename, delimiter=',')
    matrix = matrix.transpose()
    indexes = []

    for i, entry in enumerate(matrix[1:, 1:]):
        if all(e == 0 for e in entry) and all(m == 0 for m in matrix[i, 1:]):
            indexes.append(i+1)

    matrix = np.delete(matrix, indexes, axis=0)
    matrix = matrix.transpose()
    np.savetxt("../datasets/training/voicesonly/piano_roll_fs_1/" + filename, matrix.astype(int), fmt='%i', delimiter=",")

    # Create transposed versions for more training data
    for i in range(1, 7):
        subset = copy.deepcopy(matrix[1:, 1:])
        transposed = np.roll(subset, i, axis=0)
        new_matrix = copy.deepcopy(matrix)
        new_matrix[1:, 1:] = transposed
        np.savetxt("../datasets/training/voicesonly/piano_roll_fs_1/" + filename[:-4] + str(i) + ".csv", new_matrix.astype(int), fmt='%i', delimiter=",")

        subset = copy.deepcopy(matrix[1:, 1:])
        transposed = np.roll(subset, -i, axis=0)
        new_matrix = copy.deepcopy(matrix)
        new_matrix[1:, 1:] = transposed
        np.savetxt("../datasets/training/voicesonly/piano_roll_fs_1/" + filename[:-4] + str(-i) + ".csv", new_matrix.astype(int), fmt='%i', delimiter=",")

print("Done")