//
// Created by conniezhong on 2019/8/12.
//

#ifndef PYTORCHSERVING_FILESTORAGETOISTREAMSOURCE_H
#define PYTORCHSERVING_FILESTORAGETOISTREAMSOURCE_H

#include "ModelManager.h"
#include "Source.h"
#include "Object.h"

namespace pytorch {
namespace serving {
class FileStorageToIStreamSource : public Source<std::istream> {
public:

    FileStorageToIStreamSource() {
    }

public:
    std::string debugString() const;

    //FileStorageToIStreamSource中的资源是一个文件路径
    Status tryLoadSource();

    Status checkStable(const string &path, bool &stable);

    Status convertPathToIstream(string path, shared_ptr<std::ifstream> &out);

    Status getLatestVersionOnFS(string path, int32_t &vname);

private:
    Status getFileSize(string path, int32_t &size);

private:

};

}
}


#endif //PYTORCHSERVING_FILESTORAGETOISTREAMSOURCE_H
