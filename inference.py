import os

from uggnet import Ugg

from keras import backend as K
from keras import callbacks
import numpy as np
import glob
from PIL import Image

import matplotlib.pyplot as ply
import scipy.io
# model
model = Ugg()
model.load_weights('models/uggnet_model.hdf5')


x_batch = []
im = Image.open('sample_images/29030.jpg')
im = im.resize((320,320))
im = np.array(im, dtype=np.float32)
im = im[..., ::-1]  # RGB 2 BGR
R = im[..., 0].mean()
G = im[..., 1].mean()
B = im[..., 2].mean()
im[..., 0] -= R
im[..., 1] -= G
im[..., 2] -= B
x_batch.append(im)
x_batch = np.array(x_batch, np.float32)
prediction = model.predict(x_batch)
pred = np.reshape(prediction[0],(320,320))
ply.imshow(pred)
ply.show()
