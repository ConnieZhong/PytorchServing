#include <torch/script.h>
#include "Code.h"

namespace pytorch {
namespace serving {

void Code::initCodeMap() {
    Code::getInstance().codeMap[SUCCESS] = "SUCCESS";
    Code::getInstance().codeMap[NULL_PTR] = "NULL_PTR";
    Code::getInstance().codeMap[LOAD_MODEL_ERROR] = "LOAD_MODEL_ERROR";
    Code::getInstance().codeMap[INNER_ERROR] = "INNER_ERROR";
    Code::getInstance().codeMap[DO_LATER] = "DO_LATER";
    Code::getInstance().codeMap[FILE_ERROR] = "FILE_ERROR";
    Code::getInstance().codeMap[CLASS_NOT_REGISTERED] = "CLASS_NOT_REGISTERED";
    Code::getInstance().codeMap[INIT_CONFIG_ERROR] = "INIT_CONFIG_ERROR";
    Code::getInstance().codeMap[CONFIG_PARM_ERROR] = "CONFIG_PARM_ERROR";

    Code::getInstance().codeMap[DATA_NOT_FIND] = "DATA_NOT_FIND";
    Code::getInstance().codeMap[NO_DATA] = "NO_DATA";
    Code::getInstance().codeMap[ALREADY_EXIST] = "ALREADY_EXIST";


    //Code::getInstance().protoTypeMap[pytorchserving::DT_INVALID] = at::k
    Code::getInstance().protoTypeMap[pytorchserving::DT_FLOAT] = at::kFloat;
    Code::getInstance().protoTypeMap[pytorchserving::DT_DOUBLE] = at::kDouble;
    Code::getInstance().protoTypeMap[pytorchserving::DT_INT32] = at::kInt;
    Code::getInstance().protoTypeMap[pytorchserving::DT_INT64] = at::kLong;
    Code::getInstance().protoTypeMap[pytorchserving::DT_BOOL] = at::kBool;
    //Code::getInstance().protoTypeMap[pytorchserving::DT_STRING] = at::kByte;

    Code::getInstance().protoTypeSizeMap[pytorchserving::DT_FLOAT] = 4;
    Code::getInstance().protoTypeSizeMap[pytorchserving::DT_DOUBLE] = 8;
    Code::getInstance().protoTypeSizeMap[pytorchserving::DT_INT32] = 4;
    Code::getInstance().protoTypeSizeMap[pytorchserving::DT_INT64] = 8;
    Code::getInstance().protoTypeSizeMap[pytorchserving::DT_BOOL] = 1;


    Code::getInstance().protoReverseTypeMap[at::kFloat] = pytorchserving::DT_FLOAT;
    Code::getInstance().protoReverseTypeMap[at::kDouble] = pytorchserving::DT_DOUBLE;
    Code::getInstance().protoReverseTypeMap[at::kInt] = pytorchserving::DT_INT32;
    Code::getInstance().protoReverseTypeMap[at::kLong] = pytorchserving::DT_INT64;
    Code::getInstance().protoReverseTypeMap[at::kBool] = pytorchserving::DT_BOOL;
    //Code::getInstance().protoReverseTypeMap[at::kByte] = pytorchserving::DT_STRING;

}
}
}