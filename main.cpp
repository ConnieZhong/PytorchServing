#include <iostream>
#include <FileStorageToIStreamSource.h>
#include <PytorchModelFromIStream.h>
#include <Object.h>
#include "ModelManager.h"
#include "Log.h"
#include "ConfigManager.h"
#include <grpcpp/grpcpp.h>
#include <PredictImp.h>

using namespace pytorch::serving;

REGISTER_CLASS(FileStorageToIStreamSource);

REGISTER_CLASS(PytorchModelFromIStream);

int main() {
    google::LogToStderr();
    Code::getInstance().initCodeMap();
    Status s = ConfigManager::getInstance().init("./config/server.toml");
    IF_ERROR_RETURN_INT(s, "config init err");

    s = InitLog();
    IF_ERROR_RETURN_INT(s, "log init err");

    PSLOG(INFO) << "begin start pytorch serving" << std::endl;

    s = ModelManager::getInstance().Init();
    IF_ERROR_RETURN_INT(s, "modelmanager init err");

    s = ModelManager::getInstance().beginLoadSource();
    IF_ERROR_RETURN_INT(s, "begin load source err");

    s = ModelManager::getInstance().beginLoadModel();
    IF_ERROR_RETURN_INT(s, "begin load model err");


    std::string server_address("0.0.0.0:50051");
    PredictImp service;

    ::grpc::ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();

    while (1) {
        /*
        shared_ptr<std::vector<torch::jit::IValue> > inputs = make_shared<std::vector<torch::jit::IValue>>();
        //inputs->push_back(torch::ones({1, 3, 224, 224}))

        vector<long int> d(150528,1);
        inputs->push_back(torch::from_blob(d.data(), {1,3,224,224}, at::kFloat));
        std::shared_ptr<void> output;

        if (ModelManager::getInstance().predictLatest("simple_model", inputs, output).success()) {
            std::cout << static_pointer_cast<at::Tensor>(output)->slice(1, 0, 5) << '\n';
        }*/

        /*
        vector<float > d;
        d.push_back(9);
        d.push_back(3);
        d.push_back(1);

        pytorchserving::PredictRequest request;
        request.mutable_model_spec()->set_mode(pytorchserving::M_LATEST);
        request.mutable_model_spec()->set_model_name("simple_model");

        request.mutable_inputs()->set_data(d.data(), d.size() * 4);
        request.mutable_inputs()->set_dtype(pytorchserving::DT_FLOAT);
        request.mutable_inputs()->mutable_tensor_shape()->add_dim(1);
        request.mutable_inputs()->mutable_tensor_shape()->add_dim(3);

        std::shared_ptr<pytorchserving::Tensor> inputs = make_shared<pytorchserving::Tensor>();
        *inputs = request.inputs();
        std::shared_ptr<void> output;

        Status s = ModelManager::getInstance().predictLatest(request.model_spec().model_name(), inputs, output);

        PSLOG(INFO) << "get out put over" << endl;
        */
        sleep(3);
    }
    return 0;
}