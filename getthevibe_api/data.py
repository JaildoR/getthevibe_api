import pandas as pd

from google.cloud import storage
# from TaxiFareModel.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH
from getthevibe_api.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH

def get_data_from_gcp(optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"

    df = pd.read_csv(path)
    return df
