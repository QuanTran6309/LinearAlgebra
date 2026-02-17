
#ifndef TENSOR_TYPE
#define TENSOR_TYPE

namespace LinearAlgebra {

// Supported tensor type of the library
enum Type {
    INT,    // 4 bytes
    FLOAT,  // 4 bytes
    DOUBLE  // 8 bytes
};

// Supported device to perform operations and store memory
enum Device {CPU, GPU};


// Get the size of the supported tensor type
inline unsigned int typeSize(Type type){
    switch (type) {
        case Type::INT: 
            return 4;
        case Type::FLOAT:
            return 4;
        case Type::DOUBLE:
            return 8;
        default: 
            return 0;
    }
}

template <typename Functor>
inline void functionTypeDispatcher(Type type, Functor&& functor){
    switch (type)
    {
    case Type::INT:
        /* code */
        functor(static_cast<int>(0));
        break;
    case Type::FLOAT:
        /* code */
        functor(static_cast<float>(0));
        break;
    case Type::DOUBLE:
        /* code */
        functor(static_cast<double>(0));
        break;
    default:
        break;
    }
}



}

#endif