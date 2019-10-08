//
// Created by conniezhong on 2019/8/21.
//

#include "Tensor.h"

namespace pytorch {
namespace serving {
TensorShape::TensorShape(std::vector<int32_t> &&shape) {
    shape_ = std::move(shape);
}

}
}