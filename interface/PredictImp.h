//
// Created by conniezhong on 2019/8/30.
//

#ifndef PYTORCHSERVING_PREDICTIMP_H
#define PYTORCHSERVING_PREDICTIMP_H

#include <Type.h>
#include <ModelManager.h>
#include "torch/script.h"
#include "Log.h"
#include "Predict.grpc.pb.h"

namespace pytorch {
namespace serving {

class PredictImp final : public pytorchserving::Predictor::Service {
    ::grpc::Status Predict(::grpc::ServerContext *context, const pytorchserving::PredictRequest *request,
                           pytorchserving::PredictResponse *reply) override {
        std::shared_ptr<pytorchserving::Tensor> inputs = make_shared<pytorchserving::Tensor>();
        *inputs = request->inputs();
        std::shared_ptr<void> output;

        Status s;
        switch ((request->model_spec().mode())) {
            case LOADMODE_LATEST: {
                s = ModelManager::getInstance().predictLatest(request->model_spec().model_name(), inputs, output);
                break;
            }
            case LOADMODE_SPECIFIC: {
                ModelID id(request->model_spec().model_name(), request->model_spec().version());
                s = ModelManager::getInstance().predictVersion(id, inputs, output);
                break;
            }
            default: {
                PSLOG(ERROR) << "unknown mode:" << request->model_spec().mode() << std::endl;
            }
        }

        //reply->model_spec() = request->model_spec();
        *reply->mutable_model_spec() = request->model_spec();
        reply->set_code(s.code);
        reply->set_message(s.msg);
        if (output != nullptr) {
            *(reply->mutable_outputs()) = *(static_pointer_cast<pytorchserving::Tensor>(output));
        }
        return ::grpc::Status(::grpc::OK, "");
    }
};
}
}


#endif //PYTORCHSERVING_PREDICTIMP_H
