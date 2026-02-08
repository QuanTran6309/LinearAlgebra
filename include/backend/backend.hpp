#ifndef BACKEND
#define BACKEND

#include <memory>
#include "type.hpp"
namespace LinearAlgebra {


    
class Backend {

private: 
    static std::unique_ptr<Backend> cpuBackend;
    static std::unique_ptr<Backend> gpuBackend;

public:

    // Common methods both cpu and gpu backend must implement
    virtual ~Backend() = default;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void **ptr) = 0;

    virtual void copy(void *srcPtr, void *destPtr, size_t bytes) = 0;

    virtual void add(void *dest, 
                     const void *src1, 
                     const void *src2, 
                     unsigned int numberOfEntries, 
                     const Type& type) = 0;

    virtual void mult(int m, int n, int k,
                      const void *src1,
                      const void *src2,
                      void *dest,
                      const Type& type) = 0;


    // Singleton pattern
    static Backend* getBackend(const Device& device);
    static Device getPtrDevice(const void* ptr);
};

}
#endif