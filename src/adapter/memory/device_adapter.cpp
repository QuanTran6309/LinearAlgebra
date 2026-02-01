#include "memory/adapter/device_adapter.hpp"

namespace LinearAlg {

DeviceAdapter::DeviceAdapter(const Device& device) : device(device){}

DeviceType DeviceAdapter::getDeviceType(){
    return this->device.type;
}

}