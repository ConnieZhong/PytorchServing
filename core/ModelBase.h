//
// Created by conniezhong on 2019/8/8.
//

#ifndef PYTORCHSERVING_MODEL_H
#define PYTORCHSERVING_MODEL_H

#include <memory>
#include "Type.h"
#include "Source.h"


namespace pytorch {
namespace serving {

class ModelBase {
public:
    ModelID modelID() {
        return modelID_;
    }

    LoadMode loadMode(){
        return loadMode_;
    }
public:
    virtual Status loadModel(shared_ptr<SourceBase> sourcePtr)=0;

    virtual Status predict(const std::shared_ptr<void> &input, std::shared_ptr<void> &out) = 0;

    virtual string debugString() const = 0;

    void setInitialInfo(const ModelID &&id, LoadMode loadMode);
    /*template<typename T, typename U>
    virtual Status predict(const T &input, U &output);*/ //TODO 看这里怎么弄

protected:
    ModelID modelID_;
    LoadMode loadMode_;
};
}
}


#endif //PYTORCHSERVING_MODEL_H
