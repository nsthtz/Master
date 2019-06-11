import datapreparation as datp
import os

#datp.midfile_to_piano_roll("C:\DeepLearning\Master\\neuralnet\datasets\\training\\",fs=5)
datp.midfile_to_piano_roll("C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly\AC-11-Wann.mid",fs=1)

#
#for filename in os.listdir("C:\DeepLearning\Master\\neural-composer-assignement\datasets\\training\\voicesonly"):
#    if os.path.isfile(filename):
#        datp.midfile_to_piano_roll("C:\DeepLearning\Master\\neural-composer-assignement\datasets\\training\\voicesonly\\"+filename, fs=1)

""""
files = [f for f in os.listdir('C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly') if os.path.isfile(os.path.join('C:\DeepLearning\Master\\neural-composer-assignement\datasets\\training\\voicesonly', f))]

print(files)
for filename in files:
    datp.midfile_to_piano_roll("C:\DeepLearning\Master\\neuralnet\datasets\\training\\voicesonly\\" + filename, fs=1)
"""
