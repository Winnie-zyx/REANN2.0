#ifdef PAIR_CLASS

PairStyle(reann_eqz,PairREANN_EQZ)   //T=0 reann_eqz is the name in the input script

#else

#ifndef LMP_PAIR_REANN_EQZ_H
#define LMP_PAIR_REANN_EQZ_H

#include "pair.h"
#include <torch/torch.h>
#include <torch/script.h> 
#include <string>

namespace LAMMPS_NS 
{
    class PairREANN_EQZ : public Pair 
    { 
         public:
             torch::jit::script::Module module;
             PairREANN_EQZ(class LAMMPS *);
             virtual ~PairREANN_EQZ();
             virtual void compute(int, int);
             virtual void init_style();
             virtual double init_one(int, int);
             virtual void settings(int, char **);
             virtual void coeff(int, char **);
         protected:
             virtual void allocate();
             virtual int select_gpu();
             double cutoff; 
             double cutoffsq;
             int numele;
             //std::vector<std::string> element_str;
             std::vector<long> element_int;
             std::string datatype;
             torch::Dtype tensor_type=torch::kDouble;
             torch::TensorOptions option1=torch::TensorOptions().dtype(torch::kDouble);
             torch::TensorOptions option2=torch::TensorOptions().dtype(torch::kLong);
             torch::Tensor device_tensor=torch::empty(1);
    };
}

#endif
#endif
