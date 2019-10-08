//
// Created by conniezhong on 2019/8/12.
//

#ifndef PYTORCHSERVING_TYPE_H
#define PYTORCHSERVING_TYPE_H

#include <string>
#include <sstream>
#include "Code.h"

using namespace std;
namespace pytorch {
namespace serving {
struct ModelID {
    std::string name;
    int32_t version; //存放这个model对应的版本号 ，如果是-1则代表是最新版本
    std::string getString() const {
        return "{name: " + name + " version: " + std::to_string(version) + "}";
    }

    ModelID() {
    }

    ModelID(std::string n, int32_t v) : name(n), version(v) {
    }
};

inline bool operator==(const ModelID &a, const ModelID &b) {
    return a.version == b.version && a.name == b.name;
}

inline bool operator!=(const ModelID &a, const ModelID &b) {
    return !(a == b);
}

inline bool operator<(const ModelID &a, const ModelID &b) {
    const int strcmp_result = a.name.compare(b.name);
    if (strcmp_result != 0) {
        return strcmp_result < 0;
    }
    return a.version < b.version;
}

inline ostream &operator<<(ostream &out, const ModelID &a) {
    out << a.getString();
    return out;
}

struct Status {
    Status(int32_t c, std::string m = "") : code(c), msg(m) {

    }

    Status() : code(SUCCESS) {

    }

    std::string getString() const {
        return "code:" + Code::getInstance().codeMap[code] + " msg:" + msg;
    }

    bool success() {
        return code == SUCCESS;
    }

    int32_t code;
    std::string msg;
};

inline bool operator==(const Status &a, int32_t &b) {
    return a.code == b;
}

inline ostream &operator<<(ostream &out, const Status &a) {
    out << a.getString();
    return out;
}

const int32_t SOURCE_UNREADY = 0;
const int32_t SOURCE_READY = 1;

enum LoadMode {
    LOADMODE_UNKNOWN = 0,
    LOADMODE_LATEST = 1,
    LOADMODE_SPECIFIC = 2//指定版本
    //TODO 添加指定一系列版本
};

const int32_t LATEST_VERSION = -1;

}
}

#endif //PYTORCHSERVING_TYPE_H