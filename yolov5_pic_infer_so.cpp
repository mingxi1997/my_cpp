#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>



// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;



void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}



typedef struct detectResult{  
    int bbox[4*256];  
    float id[256];
	float conf[256];
    int nums;
	
} detectResult,*StructPointer; 









extern "C" 

StructPointer  infer(uchar* frame_data,int height, int width,int channels) {
  
    cv::Mat img(height, width, CV_8UC3);
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


    cudaSetDevice(DEVICE);
    std::string wts_name = "";
    std::string engine_name = "my_yolo.engin";


    std::ifstream file(engine_name, std::ios::binary);
    
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();



    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));


   char buffer[40];
   

    if (img.empty()) std::cout<<"no input image"<<std::endl;
    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
        
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

        // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // std::vector<std::vector<Yolo::Detection>> batch_res(1);

    std::vector<Yolo::Detection>res;
    
    // auto& res = batch_res[0];
    nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
    


    // for (size_t j = 0; j < res.size(); j++) {
    //     cv::Rect r = get_rect(img, res[j].bbox);
    //     std::cout<<"get :"<<r<<std::endl;
    //     cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
    //     cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    // }
    // cv::imwrite("_t.jpg", img);
  

    
    // cv::imshow("test",img);
    // cv::waitKey(0);


    StructPointer p = (StructPointer)malloc(sizeof(detectResult));
    p->nums=0;
    for (int i = 0; i < res.size(); i++) {
        cv::Rect r = get_rect(img, res[i].bbox);
        p->bbox[i*4+0]=r.x;
        p->bbox[i*4+1]=r.y;
        p->bbox[i*4+2]=r.width;
        p->bbox[i*4+3]=r.height;
   
        p->id[i]=res[i].class_id;
        std::cout<< p->id[i]<<std::endl;

        p->conf[i]=res[i].conf;
        std::cout<<res[i].conf<<std::endl;

        p->nums++;

       
    }



