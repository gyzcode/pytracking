#ifndef __PYTHON2C_H__
#define __PYTHON2C_H__

#include <torch/csrc/api/include/torch/torch.h>
#include <vector>

using namespace torch;
using namespace std;

void sample_patch(const Tensor& im, Tensor pos, Tensor sample_sz, Tensor& im_patch, Tensor& patch_coord,
                    optional<Tensor> output_sz = nullopt,
                    string mode = "replicate", float max_scale_change = false, bool is_mask = false);


class Transform
{
    /*Base data augmentation transform class.*/
protected:
    vector<int64_t> output_sz;
    vector<int64_t> shift;
public:
    Transform(vector<int64_t> output_sz = vector<int64_t>(), const vector<int64_t>& shift = vector<int64_t>({0, 0}));
    virtual ~Transform();
    virtual Tensor call(const Tensor& image, bool is_mask = false) = 0;
    Tensor crop_to_output(const Tensor& image);
};


class Identity: public Transform
{
    /*Identity transformation.*/
public:
    Identity(vector<int64_t> output_sz = vector<int64_t>(), const vector<int64_t>& shift = vector<int64_t>({0, 0}));
    virtual ~Identity();
    Tensor call(const Tensor& image, bool is_mask = false);
};


#endif