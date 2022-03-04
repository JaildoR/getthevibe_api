###

### Imports
# General libraries
import numpy as np
import pandas as pd

def get_data():
    """fuction to get the training data (or a portion of it) from data folder"""
    image_df = pd.read_csv("data/fer2013.csv")
    return image_df

get_data
