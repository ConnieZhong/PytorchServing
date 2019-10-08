#include <unistd.h>
#include <grpcpp/grpcpp.h>
#include <vector>
#include <iostream>
#include <memory>
#include "Predict.grpc.pb.h"

using namespace ::grpc;
using namespace std;

int main() {
    std::shared_ptr<Channel> channel = CreateChannel("localhost:50051", InsecureChannelCredentials());
    if(channel == nullptr){
        cout << "null ptr" << endl;
        return -1;
    }
    std::unique_ptr<pytorchserving::Predictor::Stub> stub = pytorchserving::Predictor::NewStub(channel);

    cout << "begin client" << endl;
int i = 1;
    while (i++) {

        vector<float> d;
        d.push_back(i+1);
        d.push_back(i+2);
        d.push_back(i+3);


        pytorchserving::PredictRequest request;
        request.mutable_model_spec()->set_mode(pytorchserving::M_LATEST);
        request.mutable_model_spec()->set_model_name("simple_model");

        request.mutable_inputs()->set_data(d.data(), d.size() * 4);
        request.mutable_inputs()->set_dtype(pytorchserving::DT_FLOAT);
        request.mutable_inputs()->mutable_tensor_shape()->add_dim(1);
        request.mutable_inputs()->mutable_tensor_shape()->add_dim(3);

        pytorchserving::PredictResponse reply;
        ClientContext context;
        cout << "begin send requst..." << endl;

        ::grpc::Status status = stub->Predict(&context, request, &reply);
        cout << "received reply: status:" << status.ok() << " " << status.error_code()
             << " |" << status.error_message() << " | " << status.error_details() << endl;
        float *data = (float *) reply.outputs().data().c_str();
        int num = 1;
        for (int j = 0; j < reply.outputs().tensor_shape().dim_size(); ++j) {
            num *= reply.outputs().tensor_shape().dim(j);
        }
        for (int i = 0; i < num; ++i) {
            std::cout << "index:" << i << " value:" << data[i] << endl;
        }
        sleep(3);
        cout << "begin again" << endl;
    }
}