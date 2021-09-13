#include <opencv2/opencv.hpp>
#include<iostream>
#include <opencv2/dnn/dnn.hpp>




using namespace cv;
using namespace std;
 
extern "C"

int readfrombuffer(uchar* frame_data,int height, int width,int channels)
{
    if(channels == 3)
	{
        Mat img(height, width, CV_8UC3);
        uchar* ptr =img.ptr<uchar>(0);
        int count = 0;
        for (int row = 0; row < height; row++)
		{
             ptr = img.ptr<uchar>(row);
             for(int col = 0; col < width; col++)
			 {
	               for(int c = 0; c < channels; c++)
				   {
	                  ptr[col*channels+c] = frame_data[count];
	                  count++;
	                }
	    
		    }
		
        
		}

		imshow("读取视频", img);
		waitKey(0);	//延时30
	}
		return 0;
}
