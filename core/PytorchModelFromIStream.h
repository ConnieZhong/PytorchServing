//
// Created by conniezhong on 2019/8/12.
//

#ifndef PYTORCHSERVING_MODELFROMISSTREAM_H
#define PYTORCHSERVING_MODELFROMISSTREAM_H

#include <istream>
#include <torch/script.h>
#include "ModelBase.h"
#include "Predict.grpc.pb.h"
//TODO
//#include "torch/script.h"
#include "Log.h"
#include "Object.h"

namespace pytorch {
namespace serving {
class PytorchModelFromIStream : public ModelBase {
public:
    Status loadModel(shared_ptr<SourceBase> sourcePtr);

    Status predict(const std::shared_ptr<void> &input, std::shared_ptr<void> &out);

    std::string debugString() const;

    virtual  ~PytorchModelFromIStream() {
    }

private:

    torch::jit::script::Module module_;
};

}
}


#endif //PYTORCHSERVING_MODELFROMISSTREAM_H
