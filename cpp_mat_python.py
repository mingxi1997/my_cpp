import ctypes  
import cv2
import numpy as np
lib = ctypes.cdll.LoadLibrary ("/home/xu/cpp_work/opencv_demo/build/libDisplayImage.so")   # absolute path
print ('helloworld')

# #image data
src = cv2.imread("1.jpg") #0-gray

cols = src.shape[1]
rows = src.shape[0]
channels = 0
if 3==len(src.shape):
 	channels = 3	
src = np.asarray(src, dtype=np.uint8) 
src1 = src.ctypes.data_as(ctypes.c_char_p)
a = lib.readfrombuffer(src1,rows,cols,channels)
