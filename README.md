# PytorchServing
An server some like tensorflow/serving which services Pytorch models.
Any question will be welcome:zhonghongxia@foxmail.com
This server uses glog, protobuf, grpc.
This server supports any dims tensor of type float, double, int32, int64(find in `proto/Tensor.proto`. Not support string yet.

# API
API is under `proto/Predict.proto` folder.

# Core
This server will periodically load the model from file-system. And can also be extend to support other storage platform by implement a new `Source` like `FileStorageToIStreamSource`. Find in `sources/FileStorageToIStreamSource.h`).


