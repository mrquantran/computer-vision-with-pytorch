import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/nezuko.jpg'
print('dir_path:', dir_path)
assert os.path.exists(dir_path)

img = cv2.imread(dir_path)
print(img.shape)
# crop image
img = img[0:250, 0:500]


# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')

# convert image into 25x25 array 
img_gray_small = cv2.resize(gray, (25, 25))
plt.imshow(img_gray_small, cmap='gray')
print(img_gray_small)

# show image
plt.show()
