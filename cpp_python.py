

import ctypes


func = ctypes.cdll.LoadLibrary("/home/xu/cpp_work/test.so")



class StructPointer(ctypes.Structure):
        _fields_ = [("name",ctypes.c_char *5),
                    ("age",ctypes.c_int),
                    ("arr",ctypes.c_int * 3),
                    ("arrTwo", (ctypes.c_int * 3)*2)]
 
sp=StructPointer()
sp.name=bytes('hello',encoding='utf-8')
sp.age=23
for i in range(3):
        sp.arr[i]=i
for i in range(2):
     for j in range(3):
          sp.arrTwo[i][j]=i+j   
func.testStruct.restype = ctypes.POINTER(StructPointer)
p = func.testStruct(sp)
print ('*' * 20)
print ("传递并返回结构体：\n 传递值：name:joe_   age:23   arr:0 1 2  arrTwo:0 1 2;1 2 3")



print("返回值为：name+add:",p.contents.name,'   age+5:',p.contents.age,
      '   arr+1:',p.contents.arr[0],p.contents.arr[1],p.contents.arr[2],
      '   arrTwo+2:',p.contents.arrTwo[0][0],p.contents.arrTwo[0][1],p.contents.arrTwo[0][2],
      p.contents.arrTwo[1][0],p.contents.arrTwo[1][1],p.contents.arrTwo[1][2])
