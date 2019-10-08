CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(PytorchServing)
AUX_SOURCE_DIRECTORY(. DIR_SRCS)


FIND_PACKAGE(Protobuf REQUIRED)

ADD_LIBRARY(proto SHARED ${DIR_SRCS})
target_link_libraries(proto
	${PROTOBUF_LIBRARIES}
       	)


#
#find_package(Protobuf REQUIRED)
#FILE(GLOB protofiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
#message(STATUS ${protofiles})
#PROTOBUF_GENERATE_CPP(PROTOSRCS PROTOHDRS ${protofiles})
#message(STATUS ${PROTOSRCS})
#add_library(proto STATIC ${PROTOSRCS} ${PROTOHDRS})
#target_link_libraries(proto protobuf)
#target_include_directories(proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

