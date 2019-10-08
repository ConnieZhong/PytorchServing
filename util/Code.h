//
// Created by conniezhong on 2019/8/12.
//

#ifndef PYTORCHSERVINCODE_H
#define PYTORCHSERVINCODE_H

#include <map>
#include <string>
#include <ATen/core/ScalarType.h>
#include "Predict.grpc.pb.h"


namespace pytorch {
namespace serving {


class Code {
public:
    static Code &getInstance() {
        static Code instance;
        return instance;
    }

    std::map<int32_t, std::string> codeMap;
    std::map<c10::ScalarType, pytorchserving::DataType> protoReverseTypeMap;
    std::map<pytorchserving::DataType, c10::ScalarType> protoTypeMap;
    std::map<pytorchserving::DataType, int8_t> protoTypeSizeMap;

    void initCodeMap();
};


const int32_t SUCCESS = 0;
const int32_t NULL_PTR = -1;
const int32_t LOAD_MODEL_ERROR = -2;
const int32_t INNER_ERROR = -3;
const int32_t DO_LATER = -4;
const int32_t FILE_ERROR = -5;
const int32_t CLASS_NOT_REGISTERED = -6;
const int32_t INIT_CONFIG_ERROR = -7;
const int32_t CONFIG_PARM_ERROR = -8;
const int32_t EXCEPTION = -9;
const int32_t INPUT_ERROR = -10;

const int32_t DATA_NOT_FIND = 1;
const int32_t NO_DATA = 2;
const int32_t ALREADY_EXIST = 3;
}
}

#endif //PYTORCHSERVINCODE_H
