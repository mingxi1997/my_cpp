#include <fstream>
#include <iostream>
#include <sstream>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>

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
    std::string onnx_filename = "alexnet.onnx";
    IBuilder* builder = createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(onnx_filename.c_str(), 2);
/*
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    printf("tensorRT load onnx mnist model...\n");
*/
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext *context = engine->createExecutionContext();
    
    const char* input_blob_name = network->getInput(0)->getName();
    const char* output_blob_name = network->getOutput(0)->getName();
    /*printf("input_blob_name : %s \n", input_blob_name);
    printf("output_blob_name : %s \n", output_blob_name);
    */
    const int inputN = network->getInput(0)->getDimensions().d[0];
    const int inputC = network->getInput(0)->getDimensions().d[1];
    const int inputH = network->getInput(0)->getDimensions().d[2];
    const int inputW = network->getInput(0)->getDimensions().d[3];
    
    /*printf("inputH : %d,inputH : %d, inputW: %d \n", inputN,inputH, inputW);*/
    
    int* OutPut = network->getOutput(0)->getDimensions().d;

    void* buffers[2] = { NULL, NULL };
    int nBatchSize = inputN;
    int nOutputSize = 1000;
    cudaMalloc(&buffers[0], nBatchSize * inputH * inputW * inputC * sizeof(float));
    cudaMalloc(&buffers[1], nBatchSize * nOutputSize * sizeof(float));

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void *data = malloc(nBatchSize * inputH * inputW * inputC *sizeof(float));
    // memcpy(data, img2.ptr<float>(0), inputH * inputW * inputC * sizeof(float));

    std::cout << "start to read image..." << std::endl;
    std::string name="t.jpg";
    float* arrays=PicToArray(name,inputN, inputH ,inputW ,inputC);
    memcpy(data, arrays, inputH * inputW * inputC * sizeof(float));
    cudaMemcpyAsync(buffers[0], data, nBatchSize * inputH * inputW * inputC * sizeof(float), cudaMemcpyHostToDevice, stream);

    //save serialize model
    nvinfer1::IHostMemory *modelStream = engine->serialize();


    std::ofstream serialize_output_stream;

    // 將序列化的模型結果拷貝至serialize_str字符串
    std::string serialize_str;
    serialize_str.resize(modelStream->size());
    memcpy((void*)serialize_str.data(), modelStream->data(), modelStream->size());

    // 將serialize_str字符串的內容輸出至cached_model.bin文件
    serialize_output_stream.open("save.bin");
    serialize_output_stream << serialize_str;
    serialize_output_stream.close();



    
    return 0;
 }
