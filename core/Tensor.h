//
// Created by conniezhong on 2019/8/21.
//

#ifndef PYTORCHSERVING_TENSOR_H
#define PYTORCHSERVING_TENSOR_H

#include <vector>
#include <cstdint>
#include <memory>

namespace pytorch {
namespace serving {


enum DataType : int {
    FLOAT = 1,
    INT32 = 2,
    DOUBLE = 3
};


class TensorShape {

public:
    TensorShape() = default;

    TensorShape(std::vector<int32_t> &&shape);

    std::vector<int32_t> shape_;
    int32_t numElement_;
};

class TensorBase {
protected:

};

template<class T>
class Tensor : public TensorBase {
public:
    Tensor() = default;

    Tensor(std::shared_ptr<T *> data, const pytorch::serving::DataType &type,
           const pytorch::serving::TensorShape &shape) :data_(data), dataType_(type), shape_(shape) {
        //TODO num ele inital
    }

private:
    TensorShape shape_;
    DataType dataType_;
    int32_t numElement_;
    std::shared_ptr<T *> data_;
};

}
}


#endif //PYTORCHSERVING_TENSOR_H
