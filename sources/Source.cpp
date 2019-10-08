//
// Created by conniezhong on 2019/8/8.
//

#include "ModelManager.h"
#include "Source.h"

namespace pytorch {
namespace serving {
SourceBase::SourceBase() : sourceReady_(SOURCE_READY), modelID_("", 0), loadMode_(LOADMODE_UNKNOWN) {
}


}
}