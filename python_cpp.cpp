#include <iostream>
#include <cstdlib>
#include <ctime>
#include<cstring>
using namespace std;

// 要生成和返回随机数的函数




typedef struct StructPointerTest{  
    char name[5];  
    int age;
	int arr[3];
	int arrTwo[2][3];
}StructPointerTest, *StructPointer; 
 
extern "C" 
StructPointer testStruct(StructPointerTest input){     
	int i,j;
    StructPointer p = (StructPointer)malloc(sizeof(StructPointerTest)); 
    strcpy(p->name ,"hello");
    p->age = 5;  
	for(i=0;i<3;i++)
		p->arr[i] =input.arr[i]+1;
	for(i=0;i<2;i++)
		for(j=0;j<3;j++)
			p->arrTwo[i][j] =input.arrTwo[i][j]+2;
    return p;   
}
