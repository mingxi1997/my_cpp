import ctypes
import cv2
import numpy as np

img_root="t.jpg"
lib="/home/xu/cpp_work/yolov5/build/libyolov5.so"

img = cv2.imread(img_root) 

def yolov5_infer(img,lib):
    class StructPointer(ctypes.Structure):
            _fields_ = [("bbox",ctypes.c_int *4*256),
                        ("id",ctypes.c_float * 256),
                        ("conf",ctypes.c_float * 256),
                        ("nums",ctypes.c_int)
                        ]
    func = ctypes.cdll.LoadLibrary(lib)
    
    
    cols = img.shape[1]
    rows = img.shape[0]
    channels = 0
    if 3==len(img.shape):
     	channels = 3	
    # src = np.asarray(src, dtype=np.uint8) 
    img_c_type = img.ctypes.data_as(ctypes.c_char_p)
    
    
    func.infer.restype = ctypes.POINTER(StructPointer)
    p = func.infer(img_c_type,rows,cols,channels)
    
    
    nums=p.contents.nums
    print('detected {} boxes'.format(nums))
    
    ids=np.zeros(nums).astype(np.int32)
    confs=np.zeros(nums)
    bbox=np.zeros((nums,4)).astype(np.int32)
    
    for i in range(nums):
        ids[i]=p.contents.id[i]
        confs[i]=p.contents.conf[i]
        for j in range(4):
            bbox[i][j]=p.contents.bbox[i][j]

    return nums,ids,confs,bbox


nums,ids,confs,bbox=yolov5_infer(img,lib)
for i in range(nums):
    x1,y1,w,h=bbox[i]
    # cv2.rectangle(img, (350,169), (350+32,169+38), (255,255,0), 2)
    cv2.rectangle(img, (x1,y1), (x1+w,y1+h), (255,255,0), 2)
cv2.imshow('tt',img)
cv2.waitKey(0)











