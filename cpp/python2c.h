#ifndef __PYTHON2C_H__
#define __PYTHON2C_H__

#include <torch/csrc/api/include/torch/torch.h>

using namespace torch;

void sample_patch(Tensor im, Tensor pos, Tensor sample_sz, optional<torch::Tensor> output_sz = nullopt,
                  string mode = "replicate", float max_scale_change = false, bool is_mask = false);



#endif