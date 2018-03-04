__author__ = 'ahmaddorri'

import matplotlib.image as img
import numpy as np
import random
a = "2.png"
image = img.imread(a)
data = []
data.append(image)
data=np.array(data)
print(data.shape)



# img = data
# print(img.shape)
# print(img)
# print("-----------------------")
# img32=img[:,0:32,0:32,:]
# print(img32.shape)
#
# import matplotlib.pyplot as plt
#
# def disp_sample_dataset(dataset):
#     plt.imshow(dataset[0])
#     plt.show()
# disp_sample_dataset(img32)



# from PIL import Image
# im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))

# from pylab import *
# A = img32
# figure(1)
# imshow(A, interpolation='nearest')
# grid(True)
