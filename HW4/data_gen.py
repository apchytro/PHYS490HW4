import numpy as np
import pandas as pd
from math import sqrt
import os


# Data class from data_gen.py
# Handles reading data from datafile 


class Data():
    
    def __init__(self, datafile):
        # Get directory 
        directory = os.getcwd()
        data_directory = os.path.join(directory, datafile)
        # Read data using pandas and storing as float 32
        data = pd.read_csv (r'{0}'.format(data_directory), delimiter = ' ', header = None, dtype = np.float32)   
        # Convert to numpy array
        data_array = np.array(data)
        # Normalize data
        train_image = data_array[:,:-1]/255
        
        # Store images
        # Images are stored as a (num, 1, pixels, pixels) array
        self.x_train = np.reshape(train_image,(train_image.shape[0],1,int(sqrt(train_image.shape[1])),int(sqrt(train_image.shape[1]))))

