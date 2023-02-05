

import glob
import pickle

model_path = '/home/mahir/Github/PGP/PGP_lr-scheduler/archive/data.pkl'


with open(model_path, 'rb') as pickle_file:
    content = pickle.load(pickle_file)
