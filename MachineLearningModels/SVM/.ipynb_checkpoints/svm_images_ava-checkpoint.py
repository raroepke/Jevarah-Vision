
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
# %matplotlib inline

import pandas as pd
import numpy as np

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

df = pd.read_csv("C:/Users/ava/Desktop/KSU/Year 4/Spring 2020/Senior Project/normalized_data2.csv")

labels = df['diagnosis']

print(labels)

def get_image(row_id, root="C:/Users/ava/Desktop/KSU/Year 4/Spring 2020/Senior Project/removed_hair_images2"):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)

# # subset the dataframe to just Apis (genus is 0.0) get the value of the sixth item in the index
# apis_row = labels[labels.genus == 0.0].index[5]

# # show the corresponding image of an Apis
# plt.imshow(get_image(apis_row))
# plt.show()

# # subset the dataframe to just Bombus (genus is 1.0) get the value of the sixth item in the index
# bombus_row = labels[labels.genus == 1.0].index[6]

# plt.imshow(get_image(bombus_row))
# plt.show()

# # load a bombus image using our get_image function and bombus_row from the previous cell
# bombus = get_image(bombus_row)

# print('Color bombus image has shape: ', bombus)

# # convert the bombus image to greyscale
# grey_bombus = rgb2grey(bombus)

# plt.imshow(grey_bombus, cmap=mpl.cm.gray)

# print('Greyscale bombus image has shape: ', grey_bombus)