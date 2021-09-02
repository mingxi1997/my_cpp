#include <fstream>
#include <iostream>
#include <sstream>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>
#include <ctime>

#include<vector>
#include<string>
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

float* PicToArray(std::string name,int inputN, int inputH ,int inputW ,int inputC){
      
        Mat image = imread(name);
        cvtColor(image, image, CV_BGR2RGB);
        resize(image, image, cv::Size(inputW, inputH));  
        Mat img2;
        image.convertTo(img2, CV_32F);

        std::vector<cv::Mat> channels;
        split(img2,channels);

        float* arrays = new float[inputN * inputH * inputW * inputC];
    

        for (int n=0;n<inputN;n++)
        {
            for (int c = 0; c < inputC; c++)
            {
            for (int i = 0; i< inputH; i++)
            {
            
                for (int j = 0; j<inputW; j++)
                    {
                    arrays[c*inputH*inputW+i*inputW+j] =  (channels[c].at<float>(i, j))/127-1;
                    
                    }
            }
            }
        }
        return arrays;



}




int main(int argc, char** argv) {
    

    std::string cached_path = "save.bin";
    std::ifstream fin(cached_path);
    std::string cached_engine = "";
    while (fin.peek() != EOF){ 
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();



    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    IExecutionContext *context = engine->createExecutionContext();



    int inputN=1 ;
    int inputH=224;
    int inputW=224;
    int inputC=3;
    int nOutputSize=1000;

    for (int i=0;i<10;i++){
    clock_t start = clock();

    std::cout << "start to read image..." << std::endl;
    std::string name="t.jpg";
    float* arrays=PicToArray(name,inputN, inputH ,inputW ,inputC);


    void* buffers[2] = { NULL, NULL };
    cudaMalloc(&buffers[0], inputN * inputH * inputW * inputC * sizeof(float));
    cudaMalloc(&buffers[1], inputN * nOutputSize * sizeof(float));


    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void *data = malloc(inputN * inputH * inputW * inputC *sizeof(float));
    memcpy(data, arrays, inputH * inputW * inputC * sizeof(float));

   
    cudaMemcpyAsync(buffers[0], data, inputN * inputH * inputW * inputC * sizeof(float), cudaMemcpyHostToDevice, stream);


    
     
    std::cout << "start to infer image..." << std::endl;
    // 推理
    context->enqueueV2(buffers, stream, nullptr);
    // 显存到内存
    float prob[nOutputSize];
    cudaMemcpyAsync(prob, buffers[1], inputN * nOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 同步结束，释放资源
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::cout << "image inference finished!" << std::endl;




    int index = 0;
    int max=0;
    for (int i = 0; i < nOutputSize; i++)
    {
        if (max < prob[i]) {
            max = prob[i];
            index = i;
        }
    }
    std::cout << "predict digit: " << index << std::endl;
    clock_t end   = clock();
    std::cout << "花费了" << (double)(end - start) / CLOCKS_PER_SEC << "秒" << std::endl;
    }
        


    return 0;
 }
