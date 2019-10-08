//
// Created by conniezhong on 2019/8/12.
//

#ifndef PYTORCHSERVING_LOG_H
#define PYTORCHSERVING_LOG_H

#include <iostream>
#include "glog/logging.h"
#include "Type.h"

using namespace google;
namespace pytorch {
namespace serving {
#define     PSLOG(severity)     LOG(severity) << "[" <<__FUNCTION__ << "]"
#define     PSDLOG(severity)    DLOG(severity) << "[" << __FUNCTION__ << "]"

/*
#define     PSLOG(severity)     std::cout << severity
#define     PSDLOG(severity)    std::cout << severity
*/
#define     IF_ERROR_RETURN(s, msg) do{                                \
                if(!s.success()){                                   \
                PSLOG(ERROR) << msg << " status:"<< s <<endl;       \
                return s;                                           \
            }}while(0);

#define     IF_ERROR_CONTINUE(s, msg)                                 \
                if(!s.success()){                                   \
                PSLOG(ERROR) << msg << " status:"<< s <<endl;       \
                continue;                                           \
            }


#define     IF_ERROR_RETURN_INT(s, msg) do{                                \
                if(!s.success()){                                   \
                PSLOG(ERROR) << msg << " status:"<< s <<endl;       \
                return s.code;                                           \
            }}while(0);

#define     IF_NULL_RETURN_INT(ptr, s, msg) do{                                \
                if(ptr == nullptr){                                   \
                PSLOG(ERROR) << msg << " status:"<< s <<endl;       \
                return s.code;                                           \
            }}while(0);


#define     IF_NULL_RETURN_STATUS(ptr, s, msg) do{                                \
                if(ptr == nullptr){                                   \
                PSLOG(ERROR) << msg << " status:"<< s <<endl;       \
                return s;                                           \
            }}while(0);


#define     IF_NULL_CONTINUE(ptr, s, msg) do{                                \
                if(ptr == nullptr){                                   \
                PSLOG(ERROR) << msg << " status:"<< s <<endl;       \
                continue;                                           \
            }}while(0);

pytorch::serving::Status InitLog();
}
}


#endif //PYTORCHSERVING_LOG_H
