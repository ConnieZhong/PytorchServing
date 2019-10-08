//
// Created by conniezhong on 2019/8/8.
//

#ifndef PYTORCHSERVING_MODELMANAGER_H
#define PYTORCHSERVING_MODELMANAGER_H

#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <list>
#include <atomic>
#include "ModelBase.h"
#include "Type.h"
#include "Log.h"

namespace pytorch {
namespace serving {

struct WaitLoadItem {
    shared_ptr<SourceBase> sourcePtr_;
    shared_ptr<ModelBase> modelPtr_;

    WaitLoadItem() = default;

    WaitLoadItem(shared_ptr<SourceBase> sp, shared_ptr<ModelBase> mp)
            : sourcePtr_(sp), modelPtr_(mp) {

    }

    WaitLoadItem(const shared_ptr<SourceBase> &sourcePtr_);
};

inline bool operator==(const WaitLoadItem &a, const WaitLoadItem &b) {
    return (a.sourcePtr_ == b.sourcePtr_) && (a.modelPtr_ == b.modelPtr_);
}

class ModelManager {
public:
    //访问到的shared_ptr<model>必须要是线程安全的；一个版本在进程运行中只允许加载一次;
    static ModelManager &getInstance() {
        static ModelManager instance;
        return instance;
    }

    ModelManager() : loadFlag_(1) {
    }

    void terminateLoad() {
        loadFlag_ = 0;
    }

    Status getLatestVersion(std::string modelName, int32_t &version){
        std::lock_guard<std::mutex> lock1(latestModelMapMutex_);
        auto it = latestModelMap_.find(modelName);
        if (it == latestModelMap_.end()) {
            PSDLOG(INFO) << "model name :" << modelName << " not found in latest map" << endl;
            return Status(DATA_NOT_FIND);
        }
        version = it->second->modelID().version;
        return Status(SUCCESS);
    }

public:

    Status Init();

    Status beginLoadSource();

    Status beginLoadModel();

    Status addLatestModelOnce(ModelID id, shared_ptr<ModelBase> modePtr);

    Status popOneReadySource(WaitLoadItem &item);

    Status moveSourceToReadyQue(list<WaitLoadItem>::iterator &it);

    Status isModelExits(ModelID id, bool &exits);



    Status predictVersion(const ModelID &id, const std::shared_ptr<void> &input, std::shared_ptr<void> &out);

    Status predictLatest(const string &modelName, const std::shared_ptr<void> &input, std::shared_ptr<void> &out);


private:

private:
    map<ModelID, shared_ptr<ModelBase>> versionModelMap_; //需要加锁
    std::mutex versionModeMapMutex_;
    map<string, shared_ptr<ModelBase>> latestModelMap_; //需要加锁
    std::mutex latestModelMapMutex_;
    list <WaitLoadItem> sourceReadyList_;
    std::mutex sourceReadyMutex_;
    map<string, pair<shared_ptr<ModelBase>, shared_ptr<SourceBase>>> msMap_;// 存储不同的模型与资源的对应关系
    list <WaitLoadItem> sourceWaitLoadList_; //待加载的sources

    atomic_int loadFlag_;
};

inline Status ModelManager::predictVersion(const pytorch::serving::ModelID &id, const std::shared_ptr<void> &input,
                                    std::shared_ptr<void> &out) {
    std::lock_guard<std::mutex> lockGuard(versionModeMapMutex_);
    auto it = versionModelMap_.find(id);
    if (it == versionModelMap_.end()) {
        PSLOG(ERROR) << "not find id in versioned-map." << id << endl;
        return Status(DATA_NOT_FIND);
    }
    return it->second->predict(input, out);
}


inline Status ModelManager::predictLatest(const string &modelName, const std::shared_ptr<void> &input, std::shared_ptr<void> &out) {
    std::lock_guard<std::mutex> lockGuard(latestModelMapMutex_);
    auto it = latestModelMap_.find(modelName);
    if (it == latestModelMap_.end()) {
        PSLOG(ERROR) << "not find id in latest-map." << modelName << endl;
        return Status(DATA_NOT_FIND);
    }
    return it->second->predict(input, out);
}
}
}


#endif //PYTORCHSERVING_MODELMANAGER_H
