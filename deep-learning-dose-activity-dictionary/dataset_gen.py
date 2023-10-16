import os
import numpy as np

# Directory where raw files are located
raw_directory = "/home/pablocabrales/phd/prototwin/deep-learning-dose-activity-dictionary/data/Prostata"

# Creating directories (if they don't already exist) for input and output data
input_dir = "/home/pablocabrales/phd/prototwin/deep-learning-dose-activity-dictionary/data/dataset_1/input"
output_dir = "/home/pablocabrales/phd/prototwin/deep-learning-dose-activity-dictionary/data/dataset_1/output"

if not os.path.exists(input_dir):
    os.makedirs(input_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Adding the activities to generate input images and moving doses to generate output images
for folder_name in os.listdir(raw_directory):
    beam_id = folder_name[0:4]
    folder_name = raw_directory + '/' + folder_name
    total_activation = 0

    with open(folder_name + '/Dose.raw', 'rb') as f:
        dose = np.frombuffer(f.read(), dtype=np.float32).reshape((150, 60, 70), order='F')
    np.save(output_dir + '/' + beam_id + '.npy', dose)

    for isotope in ['C11', 'O15', 'F18', 'N13']:
        with open(folder_name + '/' + isotope +'.raw', 'rb') as f:
            isotope_activation = np.frombuffer(f.read(), dtype=np.float32)
            total_activation += isotope_activation
    activation_volume = total_activation.reshape((150, 60, 70), order='F')
    np.save(input_dir + '/' + beam_id + '.npy', activation_volume)
