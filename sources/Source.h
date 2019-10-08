//
// Created by conniezhong on 2019/8/8.
//

#ifndef PYTORCHSERVING_SOURCE_H
#define PYTORCHSERVING_SOURCE_H

#include <functional>
#include <vector>
#include <memory>
#include <atomic>
#include <cpptoml.h>

#include "Type.h"
#include "Code.h"
#include "Log.h"

namespace pytorch {
namespace serving {

class SourceBase : std::enable_shared_from_this<SourceBase> {
public:

    /*using SourceReadyCallback = std::function<Status()>;
    virtual void setSourceReadyCallback(SourceReadyCallback callback){
        sourceReadyCallback_ = callback;
    }

    //将自己挂载source准备好的队列中
    SourceReadyCallback sourceReadyCallback(){
        return sourceReadyCallback_;
    }*/

    //virtual Status sourceReadyCallBack() = 0;

    virtual Status tryLoadSource() = 0;

    virtual string debugString() const = 0;

    virtual void setInitialInfo(ModelID id, LoadMode mode, string path) {
        modelID_ = id;
        loadMode_ = mode;
        path_ = path;
    }

public:
    SourceBase();

    void setSourceReady() {
        sourceReady_ = 1;
    }

    int32_t sourceReady() {
        return sourceReady_;
    }

    ModelID modelID() {
        return modelID_;
    }


private:
    //SourceReadyCallback sourceReadyCallback_;
    atomic_int sourceReady_;

protected:
    ModelID modelID_;
    LoadMode loadMode_; //加载最新的还是指定的
    string path_;
};

template<typename T>
class Source : public SourceBase {
public:


public:
    virtual ~Source() = default;

    Source() {

    }

    Source(ModelID id) {
        modelID_ = id;
    }

    shared_ptr<T> sourceData() {
        return sourceData_;
    }

    Status setSource(shared_ptr<T> ptr) {
        if (sourceData_ != nullptr) {
            PSLOG(ERROR) << "null ptr" << endl;
            return Status(INNER_ERROR, "");
        }
        setSourceReady();
        sourceData_ = ptr;
        return Status(SUCCESS);
    }



private:
    shared_ptr<T> sourceData_;
};
}
}
#endif //PYTORCHSERVING_SOURCE_H
