''' This script generates a population of xcat phantoms with different parameters
'''
import subprocess
import random

# Define the possible parameters for the population
param_dict = {
    ' --gender ' : ['0 --organ_file vmale50.nrb --heart_base vmale50_heart.nrb',
                   '1 --organ_file vfemale50.nrb --heart_base vfemale50_heart.nrb'],
    ' --phantom_height_scale ' : ['0.5', '0.8', '0.9', '1', '1.1', '1.2'],
    ' --phantom_long_axis_scale ' : ['0.8 --phantom_short_axis_scale 0.8',
                                    '0.9 --phantom_short_axis_scale 0.9',
                                    '1 --phantom_short_axis_scale 1',
                                    '1.1 --phantom_short_axis_scale 1.1',
                                    '1.2 --phantom_short_axis_scale 1.2']
}

# File names for the parameter file and the output file
param_file = '"C:\\Program Files (x86)\\xcat_v2\\Setup1\\general.samp.par"'
output_file_base  = ' "C:\\Users\\pablo\\Dropbox\\Carpeta de Pablo\\\
Dropbox\\Personal\\PhD\\xcat\\data\\subj_'

# Base command for xcat, including the parameter file and the output type file
command_base = 'dxcat2 general.samp.par --act_phan_each 0'

# Directory where xcat is installed
working_directory = 'C:\\Program Files (x86)\\xcat_v2\\Setup1'

N_subjects = 1  # Number of subjects to generate

# Generating the population
print('Generating population...')
for i in range(N_subjects):
    command = command_base
    # Iterate over each parameter
    for j in range(len(param_dict)):
        key_param = list(param_dict.keys())[j]
        # Add the randomized parameter to the command
        command += key_param + param_dict[key_param][random.randint(0, len(param_dict[key_param])-1)]
    command += output_file_base + str(i) + '"'
    # Write the command to a file
    with open(output_file_base[2:] + str(i) + '_command.txt', 'w') as f:
        f.write(command)
    # Run the command
    process = subprocess.run(command, shell=True, check=True, cwd=working_directory)

    print("\nGenerated subject " + str(i))