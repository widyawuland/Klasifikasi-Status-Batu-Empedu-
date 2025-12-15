# Download dan Load Dataset

from google.colab import drive
drive.mount('/content/drive')

#Download dataset Gallstone dari kaggle(https://www.kaggle.com/datasets/xixama/gallstone-dataset-uci?resource=download)
import kagglehub

# Download latest version
path = kagglehub.dataset_download("xixama/gallstone-dataset-uci")

print("Path to dataset files:", path)

#Load Dataset
import numpy as np
import pandas as pd

data=pd.read_csv("/kaggle/input/gallstone-dataset-uci/gallstone_.csv")

print(data)