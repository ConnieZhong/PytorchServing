//
// Created by conniezhong on 2019/8/8.
//

#ifndef PYTORCHSERVING_CONFIGMANAGER_H
#define PYTORCHSERVING_CONFIGMANAGER_H

#include <vector>
#include <memory>
#include <mutex>
#include <Log.h>
#include <cpptoml.h>
#include "Type.h"

namespace pytorch {
namespace serving {
struct SourceInfo {
    std::string modelName;
    std::string pathName;
    std::string sourceClassName;//处理source的派生类的名字
    std::string modelClassName;//处理source的派生类的名字
    LoadMode loadMode;
    int32_t version;

    std::string getString() const {
        return "{modelName:" + modelName + " path:"
               + pathName + " loadMode:" + to_string(loadMode)
               + " sourceClassName:" + sourceClassName
               + " modelClassName:" + modelClassName + "}";
    }

    SourceInfo &operator=(const SourceInfo &s) {
        this->modelName = s.modelName;
        this->sourceClassName = s.sourceClassName;
        this->modelClassName = s.modelClassName;
        this->loadMode = s.loadMode;
        this->version = s.version;
        this->pathName = s.pathName;
        return *this;
    }
};

inline ostream &operator<<(ostream &out, const SourceInfo &a) {
    out << a.getString();
    return out;
}

class ConfigManager {
public:
    Status init(string path);

    string debugString() const;

    ConfigManager() :
            logToStdErr_(false),
            sourceFindLoopS_(10),
            loadModelLoops_(10),
            checkStableTotalTimes_(3) {
    }

public:
    static ConfigManager &getInstance() {
        static ConfigManager instance;
        return instance;
    }

    string logPath() {
        return logPath_;
    }

    bool logToStderr() {
        return logToStdErr_;
    }

    int32_t sourceFindLoopS() {
        return sourceFindLoopS_;
    }

    int32_t loadModelLoops() {
        return loadModelLoops_;
    }

    std::shared_ptr<std::vector<SourceInfo>> sourceInfoVecPtr() {
        return sourceInfoVecPtr_;
    }

    int32_t checkStableTotalTimes() {
        return checkStableTotalTimes_;
    }


private:
    string logPath_;
    bool logToStdErr_;


    int32_t sourceFindLoopS_;//多少s进行一次模型发现
    int32_t loadModelLoops_;//多少s进行一次模型加载

    std::mutex sourceInfoMutex_;
    std::shared_ptr<std::vector<SourceInfo>> sourceInfoVecPtr_;
    int32_t checkStableTotalTimes_;
};


inline Status ConfigManager::init(string path) {
    //读取文件
    shared_ptr<std::vector<SourceInfo>> sourcePtr = make_shared<std::vector<SourceInfo>>();
    IF_NULL_RETURN_STATUS(sourcePtr, Status(INNER_ERROR), "make ptr err");
    try {
        auto config = cpptoml::parse_file("./config/server.toml");
        IF_NULL_RETURN_STATUS(config, Status(INIT_CONFIG_ERROR), "init config err");
        for (auto it:*config) {
            PSDLOG(INFO) << it.first << endl;
        }

        if (config->contains("source_find_loops")) {
            sourceFindLoopS_ = *config->get_as<int32_t>("source_find_loops");
        }
        if (config->contains("load_model_loops")) {
            loadModelLoops_ = *config->get_as<int32_t>("load_model_loops");
        }

        if (config->contains("check_stable_times")) {
            checkStableTotalTimes_ = *config->get_as<int32_t>("check_stable_times");
        }


        std::shared_ptr<cpptoml::table_array> model = config->get_table_array("model-array");
        IF_NULL_RETURN_STATUS(model, Status(INIT_CONFIG_ERROR), "init config err");
        for (const auto &table : *model) {
            SourceInfo si;
            si.modelName = *table->get_as<std::string>("model_name");
            si.pathName = *table->get_as<std::string>("file_path");
            si.version = *table->get_as<int32_t>("version");
            si.loadMode = (LoadMode) (*table->get_as<int32_t>("load_mode"));
            si.sourceClassName = *table->get_as<std::string>("source_class_name");
            si.modelClassName = *table->get_as<std::string>("model_class_name");

            if (si.modelClassName == "") {
                IF_ERROR_RETURN(Status(CONFIG_PARM_ERROR), "model class name empty. get config err." + si.getString());
            }
            if (si.sourceClassName == "") {
                IF_ERROR_RETURN(Status(CONFIG_PARM_ERROR), "source class name empty. get config err." + si.getString());
            }
            if (si.modelName == "") {
                IF_ERROR_RETURN(Status(CONFIG_PARM_ERROR), "model name empty. get config err." + si.getString());
            }
            if (si.loadMode <= LOADMODE_UNKNOWN || si.loadMode > LOADMODE_SPECIFIC) {
                IF_ERROR_RETURN(Status(CONFIG_PARM_ERROR), "load mode err. get config err." + si.getString());
            }


            sourcePtr->push_back(si);
        }
        if (config->contains("log")) {
            logPath_ = *config->get_qualified_as<string>("log.log_path");
        }
        if (config->contains("log")) {
            logToStdErr_ = *config->get_qualified_as<bool>("log.log_to_stderr");
        }
    } catch (cpptoml::parse_exception &e) {
        PSDLOG(ERROR) << "init config err" << endl;
        return Status(INIT_CONFIG_ERROR);
    }
    {
        std::lock_guard<std::mutex> lock_guard1(sourceInfoMutex_);
        sourceInfoVecPtr_ = sourcePtr;
    }
    PSLOG(INFO) << debugString() << endl;
    return Status(SUCCESS);
}

//TODO 如何能够不做成inline？
inline string ConfigManager::debugString() const {
    std::stringstream ss;
    ss << "logPath:" << logPath_
       << " sourceFindLoops:" << sourceFindLoopS_
       << " loadModelLoops:" << loadModelLoops_
       << " checkStableTotalTimes" << checkStableTotalTimes_;
    ss << " sourceInfo: ";


    for (const auto &it:*sourceInfoVecPtr_) {
        ss << "[" << it << "]";
    }
    return ss.str();
}
}
}


#endif //PYTORCHSERVING_CONFIGMANAGER_H
