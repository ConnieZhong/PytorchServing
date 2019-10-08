//
// Created by conniezhong on 2019/8/12.
//

#include "Log.h"
#include "ConfigManager.h"

namespace pytorch {
namespace serving {
pytorch::serving::Status InitLog() {
    /*PSDLOG(INFO) << "log path------------" << ConfigManager::getInstance().logPath() << endl;
    google::InitGoogleLogging(ConfigManager::getInstance().logPath().c_str());
    */ //TODO
    if(ConfigManager::getInstance().logToStderr()){
        google::LogToStderr();
    }
    return Status(SUCCESS);
}
}
}