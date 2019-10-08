//
// Created by conniezhong on 2019/8/8.
//
#include <thread>
#include <algorithm>
#include "Object.h"
#include "ModelManager.h"
#include "ConfigManager.h"

namespace pytorch {
namespace serving {

Status ModelManager::Init() {
    //创建需要加载的source
    return Status(SUCCESS);
}

Status ModelManager::addLatestModelOnce(ModelID id, shared_ptr<pytorch::serving::ModelBase> modePtr) {
    IF_NULL_RETURN_STATUS(modePtr, Status(NULL_PTR), "add latest model err, null ptr." + id.getString());

    {
        std::lock_guard<std::mutex> lock(versionModeMapMutex_);
        auto it = versionModelMap_.find(id);
        if (it != versionModelMap_.end()) {
            PSDLOG(ERROR) << "model exits will not add to versioned-map again. id:" << id << endl;
            return Status(INNER_ERROR, "");
        }
        versionModelMap_[id] = modePtr;
        PSDLOG(INFO) << " ----------------TODO " << versionModelMap_.size() << endl;
    }

    {
        std::lock_guard<std::mutex> lock(latestModelMapMutex_);
        latestModelMap_[id.name] = modePtr;
    }

    if (modePtr->loadMode() == LOADMODE_LATEST) {
        //remove other models
        std::lock_guard<std::mutex> loack(versionModeMapMutex_);
        for (auto it = versionModelMap_.begin(); it != versionModelMap_.end(); it++) {
            if (it->second->modelID().name == id.name && it->second->modelID().version != id.version) {
                versionModelMap_.erase(it);
            }
        }
        PSDLOG(INFO) << " ------2----------TODO " << versionModelMap_.size() << endl;

    }

    PSLOG(INFO) << "add model to versioned- and latest-map success." << id.getString() << endl;
    return Status(SUCCESS, "");
}


Status ModelManager::beginLoadModel() {
    //创建线程 TODO 将线程管理起来
    //start a thread to load source
    PSLOG(INFO) << "model load thread begin work." << endl;
    std::thread t([this]() {
        while (loadFlag_) {
            WaitLoadItem it;
            Status s = popOneReadySource(it);
            while (s.success()) {
                IF_NULL_RETURN_INT(it.sourcePtr_, Status(NULL_PTR), " inner err");
                IF_NULL_RETURN_INT(it.modelPtr_, Status(NULL_PTR), " inner err");
                Status status = it.modelPtr_->loadModel(it.sourcePtr_);
                if (!status.success()) {
                    //TODO 上报或者重试
                    PSLOG(ERROR) << "load model err." << it.sourcePtr_->debugString() << endl;
                    s = popOneReadySource(it);
                    continue;
                }
                status = addLatestModelOnce(it.modelPtr_->modelID(), it.modelPtr_);
                if (!status.success()) {
                    //TODO 上报或者重试
                    PSLOG(ERROR) << "add model err." << it.sourcePtr_->debugString() << endl;
                    s = popOneReadySource(it);
                    continue;
                }
                PSLOG(INFO) << it.modelPtr_->debugString() << " load success";

                //TODO 删除历史的版本
                s = popOneReadySource(it);
            }
            sleep(ConfigManager::getInstance().loadModelLoops());
        }
        return SUCCESS;
    });
    t.detach();
    return Status(SUCCESS);
}


Status ModelManager::beginLoadSource() {
    //创建线程 TODO 将线程管理起来
    //start a thread to load source
    PSLOG(INFO) << "source load thread begin work." << endl;
    std::thread t([this]() {
        while (loadFlag_) {
            //get wait load model config from configure manager
            shared_ptr<std::vector<SourceInfo>> sIVP = ConfigManager::getInstance().sourceInfoVecPtr();
            if (sIVP == nullptr) {
                PSLOG(ERROR) << "get source info vec err" << endl;
                return Status(INNER_ERROR, "");
            }
            PSDLOG(INFO) << "config source info size:" << sIVP->size() << endl;
            std::vector<SourceInfo> needLoadVec;
            for (const auto &it :*sIVP) {
                //if model has been loaded, need not load again
                switch (it.loadMode) {
                    case LOADMODE_LATEST: {
                        break;
                    }
                    case LOADMODE_SPECIFIC: {
                        ModelID id(it.modelName, it.version);
                        bool exist = false;
                        Status s = isModelExits(id, exist);
                        IF_ERROR_RETURN(s, "get model exist error " + id.getString());
                        if (exist) {
                            PSDLOG(INFO) << it << " already load, will not reload source" << endl;
                            continue;
                        }
                        break;
                    }
                    default: {
                        PSDLOG(ERROR) << "unknown load mode " << it << endl;
                        IF_ERROR_RETURN(Status(LOAD_MODEL_ERROR), it.getString());
                    }
                }
                PSLOG(INFO) << "source of " << it << " will be probe" << endl;
                needLoadVec.push_back(it);
            }
            //new a driven source object and put it to source wait load que
            for (auto it: needLoadVec) {
                shared_ptr<SourceBase> sourcePtr = static_pointer_cast<SourceBase>(
                        CObjectFactory::createObject(it.sourceClassName));

                IF_NULL_CONTINUE(sourcePtr, Status(CLASS_NOT_REGISTERED), "null ptr.source class name "
                                                                          + it.sourceClassName +
                                                                          " has't be registered by REGISTER_CLASS or not exists");
                sourcePtr->setInitialInfo(ModelID(it.modelName, it.version), it.loadMode, it.pathName);

                shared_ptr<ModelBase> modelPtr = static_pointer_cast<ModelBase>(
                        CObjectFactory::createObject(it.modelClassName));

                IF_NULL_CONTINUE(modelPtr, Status(CLASS_NOT_REGISTERED), "null ptr. model class name "
                                                                         + it.modelClassName +
                                                                         " has't be registered by REGISTER_CLASS or not exists")
                modelPtr->setInitialInfo(std::move(ModelID(it.modelName, it.version)), it.loadMode);

                sourceWaitLoadList_.push_back(WaitLoadItem(sourcePtr, modelPtr));
                PSDLOG(INFO) << "add " << it << " to wait load list over" << endl;
            }
            for (auto it = sourceWaitLoadList_.begin(); it != sourceWaitLoadList_.end();) {
                Status s = it->sourcePtr_->tryLoadSource();
                if (s.code == ALREADY_EXIST) {
                    it = sourceWaitLoadList_.erase(it);
                    continue;
                }

                if (!s.success()) {
                    PSLOG(ERROR) << "load source:" << it->sourcePtr_->modelID() << " error" << endl;
                    it++;
                    continue;
                }

                if (it->sourcePtr_->sourceReady()) {
                    s = moveSourceToReadyQue(it);
                    IF_ERROR_CONTINUE(s, "source ready call back err");
                }
            }
            sleep(ConfigManager::getInstance().sourceFindLoopS());
        }
        return Status(SUCCESS);
    });
    t.detach();
    return Status(SUCCESS);

}

Status ModelManager::isModelExits(ModelID id, bool &exits) {
    exits = false;
    std::lock_guard<std::mutex> lock1(versionModeMapMutex_);
    auto it = versionModelMap_.find(id);
    exits = it != versionModelMap_.end();


    PSDLOG(INFO) << "there is----------------------2- " << versionModelMap_.size() << " in versionModelMap_" << endl;

    for (auto it : versionModelMap_) {
        PSDLOG(INFO) << "there is----------------------- " << it.first << " in versionModelMap_" << endl;
    }
    return Status(SUCCESS);
}


Status ModelManager::moveSourceToReadyQue(list<WaitLoadItem>::iterator &it) {
    if (it->sourcePtr_ == nullptr || it->modelPtr_ == nullptr) {
        PSLOG(ERROR) << "null ptr." << it->sourcePtr_->modelID().getString() << endl;
        return Status(NULL_PTR);
    }

    auto itt = std::find(sourceWaitLoadList_.begin(), sourceWaitLoadList_.end(), *it);
    if (itt == sourceWaitLoadList_.end()) {
        IF_ERROR_RETURN(Status(INNER_ERROR), "find wait ready source error");
    }

    {
        std::lock_guard<std::mutex> lock_guard1(sourceReadyMutex_);
        sourceReadyList_.push_back(*it);
    }
    PSLOG(INFO) << "move " << it->sourcePtr_->debugString() << " to source ready list over." << endl;
    it = sourceWaitLoadList_.erase(it);

    return Status(SUCCESS);
}

Status ModelManager::popOneReadySource(pytorch::serving::WaitLoadItem &item) {
    //获取一个ready的item，然后将其从ready里面删除
    std::lock_guard<std::mutex> lock1(sourceReadyMutex_);
    if (sourceReadyList_.empty()) {
        PSDLOG(INFO) << "there is no ready source now..." << endl;
        return Status(NO_DATA);
    }
    item = sourceReadyList_.front();
    sourceReadyList_.pop_front();
    return Status(SUCCESS);
}



}
}