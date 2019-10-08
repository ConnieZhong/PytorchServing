CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(PytorchServing)
SET(CMAKE_CXX_STANDARD 11)
INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/third_party/cpptoml/include
        ${PROJECT_SOURCE_DIR}/sources
        ${PROJECT_SOURCE_DIR}/core
        ${PROJECT_SOURCE_DIR}/util
        ${PROJECT_SOURCE_DIR}/proto
        ${PROJECT_SOURCE_DIR}/interface
        ${PROJECT_SOURCE_DIR}/third_party/libtorch/include
        ${PROJECT_SOURCE_DIR}/third_party/glog
        ${PROJECT_SOURCE_DIR}/third_party
        ${PROJECT_SOURCE_DIR}/third_party/libtorch/include/torch/csrc/api/include
        ${PROJECT_SOURCE_DIR}/third_party/cpptoml
)
LINK_DIRECTORIES(
        /usr/local/lib
        /usr/lib
        ${PROJECT_SOURCE_DIR}/third_party/glog
)

LINK_LIBRARIES(libglog.a)
LINK_LIBRARIES(libgrpc.so)
LINK_LIBRARIES(libgrpc++.so)

SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "$PWD/../third_party/libtorch") #must compile in 'build' dir
SET(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0") #这句必须要有，不然会找不到库
SET(THREADS_PREFER_PTHREAD_FLAG ON)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
SET(CMAKE_CXX_FLAGS "-O0 -Wall -g")
#SET(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0") #添加了这个会导致链接protobuf失败
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(Protobuf REQUIRED)

ADD_SUBDIRECTORY(core)
ADD_SUBDIRECTORY(sources)
ADD_SUBDIRECTORY(util)
ADD_SUBDIRECTORY(interface)
ADD_SUBDIRECTORY(proto)

AUX_SOURCE_DIRECTORY(. DIR_SRCS)
ADD_EXECUTABLE(PytorchServing ${DIR_SRCS})

TARGET_LINK_LIBRARIES(PytorchServing core sources util interface proto  Threads::Threads
        ${TORCH_LIBRARIES}
		${PROTOBUF_LIBRARIES}
    )

AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/test/client CLIENT_DIR_SRCS)
ADD_EXECUTABLE(ClientServing ${CLIENT_DIR_SRCS})
TARGET_LINK_LIBRARIES(ClientServing proto  Threads::Threads
		${PROTOBUF_LIBRARIES}
    )

