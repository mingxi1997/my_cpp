import argparse
import os
import struct
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt




def load_weights(file):
    print(f"Loading weights: {file}")

    assert os.path.exists(file), 'Unable to load weight file.'

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map





def doInference(context, host_in, host_out, batchSize):
    engine = context.engine
    assert engine.num_bindings == 2

    devide_in = cuda.mem_alloc(host_in.nbytes)
    devide_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(devide_in), int(devide_out)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(devide_in, host_in, stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)
    stream.synchronize()


INPUT_H = 32
INPUT_W = 32
OUTPUT_SIZE = 10
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"
weight_path = "./lenet5.wts"
engine_path = "./lenet5.engine"
gLogger = trt.Logger(trt.Logger.INFO)
maxBatchSize=1
builder = trt.Builder(gLogger)
weight_map = load_weights(weight_path)
network = builder.create_network()
config = builder.create_builder_config()






data = network.add_input(INPUT_BLOB_NAME, trt.float32, (1, INPUT_H, INPUT_W))
assert data

conv1 = network.add_convolution(input=data,
                                num_output_maps=6,
                                kernel_shape=(5, 5),
                                kernel=weight_map["conv1.weight"],
                                bias=weight_map["conv1.bias"])
assert conv1
conv1.stride = (1, 1)
print(conv1.get_output(0).shape)
relu1 = network.add_activation(conv1.get_output(0),
                                type=trt.ActivationType.RELU)
assert relu1

pool1 = network.add_pooling(input=relu1.get_output(0),
                            window_size=trt.DimsHW(2, 2),
                            type=trt.PoolingType.AVERAGE)
assert pool1
pool1.stride = (2, 2)

conv2 = network.add_convolution(pool1.get_output(0), 16, trt.DimsHW(5, 5),
                                weight_map["conv2.weight"],
                                weight_map["conv2.bias"])
assert conv2
conv2.stride = (1, 1)
print(conv2.get_output(0).shape)
relu2 = network.add_activation(conv2.get_output(0),
                                type=trt.ActivationType.RELU)
assert relu2

pool2 = network.add_pooling(input=relu2.get_output(0),
                            window_size=trt.DimsHW(2, 2),
                            type=trt.PoolingType.AVERAGE)
assert pool2
pool2.stride = (2, 2)

fc1 = network.add_fully_connected(input=pool2.get_output(0),
                                  num_outputs=120,
                                  kernel=weight_map['fc1.weight'],
                                  bias=weight_map['fc1.bias'])
assert fc1
print(fc1.get_output(0).shape)

relu3 = network.add_activation(fc1.get_output(0),
                                type=trt.ActivationType.RELU)
assert relu3

fc2 = network.add_fully_connected(input=relu3.get_output(0),
                                  num_outputs=84,
                                  kernel=weight_map['fc2.weight'],
                                  bias=weight_map['fc2.bias'])
assert fc2
print(fc2.get_output(0).shape)

relu4 = network.add_activation(fc2.get_output(0),type=trt.ActivationType.RELU)
assert relu4


# a=weight_map['fc3.weight']
# b=weight_map['fc3.bias']

fc3 = network.add_fully_connected(input=relu4.get_output(0),
                                  num_outputs=OUTPUT_SIZE,
                                  kernel=weight_map['fc3.weight'],
                                  bias=weight_map['fc3.bias'])
assert fc3
print(fc3.get_output(0).shape)


prob = network.add_softmax(fc3.get_output(0))


assert prob

prob.get_output(0).name = OUTPUT_BLOB_NAME
network.mark_output(prob.get_output(0))


# Build engine
builder.max_batch_size = 2
config.max_workspace_size = 1 << 15
engine = builder.build_engine(network, config)

# # del network
# # del weight_map

# assert engine

engine_path="lenet.engine"
with open(engine_path, "wb") as f:
    f.write(engine.serialize())











