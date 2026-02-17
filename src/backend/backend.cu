#include "backend/backend.hpp"
#include "backend/cpu_backend.hpp"
#include "backend/gpu_backend.hpp"
#include <cstring>
#include "utils/logger.cuh"

namespace LinearAlgebra {

// Define static member variables
std::unique_ptr<Backend> Backend::cpuBackend = nullptr;
std::unique_ptr<Backend> Backend::gpuBackend = nullptr;

Backend* Backend::getBackend(const Device& device){
    if (device == Device::CPU){
        if (Backend::cpuBackend == nullptr){
            Backend::cpuBackend = std::make_unique<CPU_Backend>();
        }

        return Backend::cpuBackend.get();
    }
    else {
        if (Backend::gpuBackend == nullptr){
            Backend::gpuBackend = std::make_unique<GPU_Backend>();
        }

        return Backend::gpuBackend.get();
    }
}

// I copied this from ChatGPT - not fully understand its mechanics
Device Backend::getPtrDevice(const void* ptr) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        return Device::CPU;
    }

    #if CUDART_VERSION >= 10000
        if (attr.type == cudaMemoryTypeDevice) return Device::GPU;
        if (attr.type == cudaMemoryTypeManaged) return Device::GPU;
    #else
        if (attr.memoryType == cudaMemoryTypeDevice) return Device::GPU;
    #endif
        return Device::CPU;
}


void Backend::copy(void *srcPtr, void *destPtr, size_t bytes){
    const bool isDestPtrOnGPU = Backend::getPtrDevice(destPtr) == Device::GPU;
    const bool isSrcPtrOnGPU = Backend::getPtrDevice(srcPtr) == Device::GPU;

    // Both on GPU
    if (isDestPtrOnGPU && isSrcPtrOnGPU){
        CUDA_ERR_CHECK(cudaMemcpy(destPtr, srcPtr, bytes, cudaMemcpyDeviceToDevice));
    }

    // Dest on GPU - Src on CPU
    else if (isDestPtrOnGPU && !isSrcPtrOnGPU) {
        CUDA_ERR_CHECK(cudaMemcpy(destPtr, srcPtr, bytes, cudaMemcpyHostToDevice));
    }

    // Dest on CPU - Src on GPU
    else if (!isDestPtrOnGPU && isSrcPtrOnGPU) {
        CUDA_ERR_CHECK(cudaMemcpy(destPtr, srcPtr, bytes, cudaMemcpyDeviceToHost));
    }

    // Both on CPU
    else {
        std::memcpy(destPtr, srcPtr, bytes);
    }
}


}