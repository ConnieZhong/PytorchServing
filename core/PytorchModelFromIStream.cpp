//
// Created by conniezhong on 2019/8/12.
//

#include "PytorchModelFromIStream.h"

namespace pytorch {
namespace serving {

std::string PytorchModelFromIStream::debugString() const {
    return modelID_.getString();
}

Status PytorchModelFromIStream::loadModel(shared_ptr<SourceBase> sourcePtr) {

    //获取到source里面的数据
    if (sourcePtr == nullptr) {
        LOG(ERROR) << "null ptr while load" << endl;
        return Status(NULL_PTR);
    }

    modelID_ = sourcePtr->modelID();
    //load模型
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        std::shared_ptr<Source<std::istream>> sp = dynamic_pointer_cast<Source<std::istream>>(sourcePtr);
        if (sp == nullptr) {
            LOG(ERROR) << "inner err while load" << endl;
            return Status(INNER_ERROR);
        }

        module_ = torch::jit::load(*(sp->sourceData()));
    }
    catch (const c10::Error &e) {
        PSLOG(ERROR) << "load model err:" << sourcePtr->modelID() << " msg:" << e.what() << endl;
        return Status(LOAD_MODEL_ERROR, "load model err");
    }

    return Status(SUCCESS);

}

//Status PytorchModelFromIStream::predict(const std::vector<torch::jit::IValue> &input, at::Tensor &output) {
Status PytorchModelFromIStream::predict(const std::shared_ptr<void> &input, std::shared_ptr<void> &output) {
    try {
        IF_NULL_RETURN_INT(input, Status(NULL_PTR), "input is null");
        shared_ptr<pytorchserving::Tensor> driverInput;
        driverInput = static_pointer_cast<pytorchserving::Tensor>(input);

        std::vector<torch::jit::IValue> modelInput;
        auto it = Code::getInstance().protoTypeMap.find(driverInput->dtype());
        if (it == Code::getInstance().protoTypeMap.end()) {
            PSLOG(ERROR) << "type err:" << modelID_ << " type:" << driverInput->dtype() << endl;
            return Status(INPUT_ERROR);
        }
        vector<int64_t> sizesV;
        for (int i = 0; i < driverInput->tensor_shape().dim_size(); i++) {
            int64_t tmp = driverInput->tensor_shape().dim(i);
            sizesV.push_back(tmp);
        }
        c10::IntArrayRef sizes(sizesV);

        modelInput.push_back(
                torch::from_blob((long *) (const_cast<char *>(driverInput->data().data())), sizes, it->second));

        shared_ptr<std::vector<torch::jit::IValue> > inputs = make_shared<std::vector<torch::jit::IValue>>();

        at::Tensor driverOutput;
        driverOutput = module_.forward(modelInput).toTensor();
        c10::IntArrayRef dims = driverOutput.sizes();

        shared_ptr<pytorchserving::Tensor> realOut = make_shared<pytorchserving::Tensor>();

        size_t numEle = 1;
        realOut->set_dtype(driverInput->dtype());
        for (auto it:dims) {
            realOut->mutable_tensor_shape()->add_dim(it);
            numEle *= it;
            PSDLOG(INFO) << modelID_ << " out dim:" << it << " total num:" << dims.size() << endl;
        }
        PSDLOG(INFO) << modelID_ << " total output num ele:" << numEle << endl;
        realOut->set_data(driverOutput.unsafeGetTensorImpl()->data(),
                          numEle * Code::getInstance().protoTypeSizeMap[driverInput->dtype()]);
        output = realOut;
    } catch (std::runtime_error &e) {
        PSLOG(ERROR) << "exception during predict:" << e.what() << " " << modelID_ << endl;
        return Status(EXCEPTION);
    }
    return Status(SUCCESS);
}

}
}