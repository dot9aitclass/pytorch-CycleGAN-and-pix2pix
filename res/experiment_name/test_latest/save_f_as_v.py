import cv2
import numpy as np
import glob
from natsort import natsorted



img_array = []
file_array=glob.glob('images/*.png')
file_array=[fr for fr in file_array if 'real' in fr]
file_array=[fr.replace('_',"") for fr in file_array]
file_array=natsorted(file_array)
file_array=[fr.replace('frame',"frame_") for fr in file_array]
file_array=[fr.replace('real',"_real_") for fr in file_array]
#print(file_array)

for filename in file_array :
    try:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    except:
        print(filename)
        continue
out = cv2.VideoWriter('project_real.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()