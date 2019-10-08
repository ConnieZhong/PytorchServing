//
// Created by conniezhong on 2019/8/15.
//

#ifndef PYTORCHSERVING_OBJECT_H
#define PYTORCHSERVING_OBJECT_H

#include <string>
#include <functional>
#include <map>
#include <memory>
#include <iostream>
#include <Log.h>

namespace pytorch {
namespace serving {
//根据名字生成类

//使用方法：1.REGISTER_CLASS(A) ; 2.shared_ptr<Base> a = static_pointer_cast<Base>(CObjectFactory::createObject("A")); // A继承自Base

using Constructor = std::function<std::shared_ptr<void>()>;

class CObjectFactory {
public:
    static void registerClass(std::string className, Constructor constructor) {
        PSLOG(INFO) << "class [" << className << "] registered";
        constructors()[className] = constructor;

    }

    static std::shared_ptr<void> createObject(const std::string &className) {
        Constructor constructor = NULL;

        if (constructors().find(className) != constructors().end()) {
            PSDLOG(INFO) << "class " << className << " find" << endl;
            constructor = constructors().find(className)->second;
        }

        if (constructor == NULL)
            return NULL;

        return constructor();
    }

private:
    inline static std::map<std::string, Constructor> &constructors() {
        static std::map<std::string, Constructor> instance;
        return instance;
    }
};


#define REGISTER_CLASS(class_name) \
class class_name##Helper { \
public: \
    class_name##Helper() \
    { \
        CObjectFactory::registerClass(#class_name, class_name##Helper::creatObjFunc); \
    } \
    static std::shared_ptr<void> creatObjFunc() \
    { \
        return make_shared<class_name>(); \
    } \
}; \
class_name##Helper class_name##helper;
}
}
#endif //PYTORCHSERVING_OBJECT_H
