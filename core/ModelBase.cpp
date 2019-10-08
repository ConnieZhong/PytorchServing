//
// Created by conniezhong on 2019/8/8.
//

#include "ModelBase.h"

namespace pytorch {
namespace serving {
void ModelBase::setInitialInfo(const pytorch::serving::ModelID &&id, LoadMode loadMode) {
    modelID_ = std::move(id);
    loadMode_ = loadMode;
}

}
}