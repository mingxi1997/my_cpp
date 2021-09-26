import torch
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

weight_map=torch.load('lenet5.pth')
weight_names=list(weight_map.keys())



gLogger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(gLogger)
builder.max_batch_size = 2
config = builder.create_builder_config()
config.max_workspace_size = 1 << 15


network = builder.create_network()

data=network.add_input("data", trt.float32, (1, 32, 32))

conv1=network.add_convolution(data,6,(5,5),weight_map[weight_names[0]].cpu().numpy(),weight_map[weight_names[1]].cpu().numpy())
conv1.stride = (1, 1)
relu1 = network.add_activation(conv1.get_output(0),trt.ActivationType.RELU)
pool1=network.add_pooling(relu1.get_output(0),trt.PoolingType.AVERAGE,trt.DimsHW(2, 2),)



conv2=network.add_convolution(pool1.get_output(0),16,(5,5),weight_map[weight_names[2]].cpu().numpy(),weight_map[weight_names[3]].cpu().numpy())
conv2.stride = (1, 1)
relu2 = network.add_activation(conv2.get_output(0),trt.ActivationType.RELU)
pool2=network.add_pooling(relu2.get_output(0),trt.PoolingType.AVERAGE,trt.DimsHW(2, 2),)

fc1=network.add_fully_connected(pool2.get_output(0),120,weight_map[weight_names[4]].cpu().numpy(),weight_map[weight_names[5]].cpu().numpy())
relu3 = network.add_activation(fc1.get_output(0),trt.ActivationType.RELU)

fc2=network.add_fully_connected(relu3.get_output(0),84,weight_map[weight_names[6]].cpu().numpy(),weight_map[weight_names[7]].cpu().numpy())
relu4 = network.add_activation(fc2.get_output(0),trt.ActivationType.RELU)

fc3=network.add_fully_connected(relu4.get_output(0),10,weight_map[weight_names[8]].cpu().numpy(),weight_map[weight_names[9]].cpu().numpy())
relu5 = network.add_activation(fc3.get_output(0),trt.ActivationType.RELU)

prob = network.add_softmax(fc3.get_output(0))

prob.get_output(0).name = "prob"
network.mark_output(prob.get_output(0))

engine = builder.build_engine(network, config)

engine_path="lenet.engine"
with open(engine_path, "wb") as f:
    f.write(engine.serialize())












