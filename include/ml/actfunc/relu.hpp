#ifndef RELU
#define RELU
#include "actfunc.hpp"
namespace LinearAlg {


class Relu : public ActFunc {

public:
    Relu() = default;
    void compute(Tensor& input) override;
};

}
#endif