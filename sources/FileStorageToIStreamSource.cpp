//
// Created by conniezhong on 2019/8/12.
//

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <algorithm>
#include <regex>

#include "FileStorageToIStreamSource.h"
#include "ConfigManager.h"


namespace pytorch {
namespace serving {

std::string FileStorageToIStreamSource::debugString() const {
    return "path:" + path_ + " modelID:" + modelID_.getString();
}

Status FileStorageToIStreamSource::getFileSize(string path, int32_t &size) {
    struct stat buf;
    if (int32_t ret = stat(path.c_str(), &buf) != 0) {
        LOG(ERROR) << "stat err:" << path << " ret:" << ret << endl;
        return ret;
    }
    size = buf.st_size;
    return Status(SUCCESS);
}

Status FileStorageToIStreamSource::checkStable(const string &path, bool &stable) {
    int32_t oldSize = 0;
    stable = true;
    int32_t totalTimes = ConfigManager::getInstance().checkStableTotalTimes();

    Status s = getFileSize(path, oldSize);
    if (s.code != SUCCESS) {
        LOG(ERROR) << "get file size err." << path << " status:" << s << endl;
        return s;
    }

    while (totalTimes--) {
        usleep(100);
        int32_t newSize = 0;
        s = getFileSize(path, newSize);
        if (s.code != SUCCESS) {
            LOG(ERROR) << "get file size err." << path << " status:" << s << endl;
            return s;
        }

        if (newSize != oldSize) {
            stable = false;
            break;
        }
    }
    return Status(SUCCESS);
}

Status FileStorageToIStreamSource::convertPathToIstream(string path, shared_ptr<std::ifstream> &out) {
    //读取文件
    if (out == nullptr) {
        out = make_shared<ifstream>();
    }

    //根据路径名字，来获取到对象
    out->open(path.c_str());
    if (!out->good()) {
        LOG(ERROR) << "open err." << path << endl;
        return Status(INNER_ERROR);
    }
    return Status(SUCCESS);
}

Status FileStorageToIStreamSource::getLatestVersionOnFS(string path, int32_t &name) {
    DIR *dir;
    struct dirent *ptr;
    if ((dir = opendir(path.c_str())) == NULL) {
        PSLOG(ERROR) << "open path err." << path << endl;
        return Status(FILE_ERROR);
    }
    std::regex rx("[0-9]+");
    vector<int32_t> files;//存放文件名
    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    //current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8) {    //file
            try {
                bool match = std::regex_match(ptr->d_name, rx);
                if (!match) {
                    PSLOG(WARNING) << "file name:" << ptr->d_name << " not numerical" << endl;
                    continue;
                }
            } catch (regex_error &e) {
                PSLOG(ERROR) << "file name not number." << ptr->d_name << endl;
                closedir(dir);
                return Status(FILE_ERROR);
            }
            files.push_back(atoi(ptr->d_name));
        }
    }
    closedir(dir);
    if (files.size() == 0) {
        PSLOG(ERROR) << "empty path" << endl;
        return Status(FILE_ERROR);
    }
    std::sort(files.begin(), files.end());
    name = files[files.size() - 1];
    return Status(SUCCESS);
}

Status FileStorageToIStreamSource::tryLoadSource() {
    int32_t latestVersionOnFS = 0;
    switch (loadMode_) {
        case LOADMODE_SPECIFIC: {
            latestVersionOnFS = modelID_.version;
            break;
        }
        case LOADMODE_LATEST: {
            Status s = getLatestVersionOnFS(path_, latestVersionOnFS);
            if (!s.success()) {
                PSLOG(ERROR) << "get latest file name err." << modelID_ << endl;
                return s;
            }
            break;

        }
        default: {
            PSLOG(ERROR) << "source mode err." << modelID_ << " load mode:" << loadMode_ << endl;
            return Status(INNER_ERROR);
        }
    }
    modelID_.version = latestVersionOnFS;

    int32_t v = 0;
    Status s = ModelManager::getInstance().getLatestVersion(modelID_.name, v);
    if (s.success() && v == modelID_.version) {//已经加载了相同版本的模型
        LOG(INFO) << "model:" << modelID_ << " already loaded, will not load again" << endl;
        return Status(ALREADY_EXIST);
    }

    PSDLOG(INFO) << "begin check stable." << modelID_ << endl;
    //判断文件大小是否稳定了
    bool stable = false;
    s = checkStable(path_ + "/" + to_string(latestVersionOnFS), stable);
    if (!s.success()) {
        PSLOG(ERROR) << "check stable err." << s << endl;
        return s;
    }

    if (!stable) {
        PSLOG(ERROR) << modelID_.getString() << " not stable now" << endl;
        return Status(DO_LATER);
    }

    shared_ptr<ifstream> ist = make_shared<ifstream>();
    s = convertPathToIstream(path_ + "/" + to_string(latestVersionOnFS), ist);
    if (!s.success()) {
        PSLOG(ERROR) << "get istream err." << modelID_ << endl;
        return Status(FILE_ERROR);
    }

    s = setSource(ist);
    if (!s.success()) {
        LOG(ERROR) << "set source error" << endl;
        return s;
    }
    return Status(SUCCESS);
}


}
}