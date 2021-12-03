import pickle
import cv2
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# change this to local folder holding all images
path = "/Users/yunhaowang/Desktop/images/mini"

# can replace to different dataset
with open('test.pkl', 'rb') as f:
    data = pickle.load(f)


img_data = data['image_data']

for i in range(len(img_data)):

    # RGB_img shape: (84, 84, 3), each range from 0 to 255
    RGB_img = cv2.cvtColor(img_data[i], cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(path , 'test'+str(i)+'.jpg'), RGB_img)
