#include "python2c.h"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace torch::indexing;
using namespace std;
namespace F = torch::nn::functional;

void sample_patch(const Tensor& im, Tensor pos, Tensor sample_sz, Tensor& im_patch, Tensor& patch_coord,
                    optional<Tensor> output_sz,
                    string mode, float max_scale_change, bool is_mask)
{
    /* Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    */

    // if mode not in ['replicate', 'inside']:
    //     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    // copy and convert
    Tensor posl = pos.to(kLong).clone();

    string pad_mode = mode;

    // Get new sample size if forced inside the image
    if(mode == "inside" || mode == "inside_major") {
        pad_mode = "replicate";
        Tensor im_sz = torch::rand({im.size(2), im.size(3)});
        Tensor shrink_factor = sample_sz.to(kFloat) / im_sz;
        if (mode == "inside")
            shrink_factor = shrink_factor.max();
        else if (mode == "inside_major")
            shrink_factor = shrink_factor.min();
        shrink_factor.clamp_(1, max_scale_change);
        sample_sz = (sample_sz.to(kFloat) / shrink_factor).to(kLong);
    }

    // Compute pre-downsampling factor
    int df;
    float resize_factor;
    if (output_sz) {
        resize_factor = min(sample_sz.to(kFloat) / output_sz.value().to(kFloat)).item<float>();
        df = max((int)(resize_factor - 0.1f), 1);
    }
    else {
        df = 1;
    }

    Tensor sz = sample_sz.to(kFloat) / df;     // new size

    // Do downsampling
    Tensor im2;
    if (df > 1) {
        Tensor os = posl % df;              // offset
        posl = floor_divide(posl - os, df);     // new position
        im2 = im.index({"...", Slice(os[0].item<int>(), None, df), Slice(os[1].item<int>(), None, df)});
    }
    else {
        im2 = im;
    }

    // compute size to crop
    auto szl = torch::max(sz.round(), tensor(2.f)).to(kLong);

    // Extract top and bottom coordinates
    auto tl = posl - floor_divide(szl - 1, 2);
    auto br = posl + floor_divide(szl, 2) + 1;

    // Shift the crop to inside
    if (mode == "inside" or mode == "inside_major") {
        auto im2_sz = tensor({im2.sizes()[2], im2.sizes()[3]}).to(kLong);
        auto shift = (-tl).clamp(0) - (br - im2_sz).clamp(0);
        tl += shift;
        br += shift;

        auto outside = floor_divide((-tl).clamp(0) + (br - im2_sz).clamp(0) , 2);
        shift = (-tl - outside) * (outside > 0).to(kLong);
        tl += shift;
        br += shift;

        // Get image patch
        // im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]
    }

    // Get image patch
    Tensor im_patch1;
    if (!is_mask) {
        im_patch1 = F::pad(im2, F::PadFuncOptions({-tl[1].item<long>(), br[1].item<long>() - im2.sizes()[3], -tl[0].item<long>(), br[0].item<long>() - im2.sizes()[2]}).mode(torch::kReplicate)); //need attention
    }
    else {
        im_patch1 = F::pad(im2, F::PadFuncOptions({-tl[1].item<long>(), br[1].item<long>() - im2.sizes()[3], -tl[0].item<long>(), br[0].item<long>() - im2.sizes()[2]}));
    }

    // Get image coordinates
    patch_coord = df * torch::cat({tl, br}).view({1, 4});

    if (!output_sz || (im_patch1.sizes()[2] == output_sz.value()[0].item<int>() && im_patch1.sizes()[3] == output_sz.value()[1].item<int>())) {
        im_patch = im_patch1.clone();
        return;
    }
        
    // Resample
    Tensor tts = output_sz.value().to(kLong);
    vector<int64_t> v(tts.data<int64_t>(), tts.data<int64_t>() + tts.numel());
    if (!is_mask)
        im_patch = F::interpolate(im_patch1, F::InterpolateFuncOptions().size(v).mode(torch::kBilinear));
    else
        im_patch = F::interpolate(im_patch1, F::InterpolateFuncOptions().size(v).mode(torch::kNearest));

    return;
}



Transform::Transform(vector<int64_t> output_sz, const vector<int64_t>& shift)
{
    cout << "Transform" << endl;
    cout << output_sz.size() << endl;
    this->output_sz = output_sz;
    this->shift = shift;
}

Transform::~Transform()
{
}

Tensor Transform::crop_to_output(const Tensor& image)
{
    auto imsz = image.sizes().slice(2);
    cout << imsz << endl;

    float pad_h, pad_w;
    if (output_sz.empty()) {
        pad_h = 0;
        pad_w = 0;
    }
    else {
        pad_h = (output_sz[0] - imsz[0]) / 2;
        pad_w = (output_sz[1] - imsz[1]) / 2;
    }

    int pad_left = floor(pad_w) + shift[1];
    int pad_right = ceil(pad_w) - shift[1];
    int pad_top = floor(pad_h) + shift[0];
    int pad_bottom = ceil(pad_h) - shift[0];

    return F::pad(image, F::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom}).mode(kReplicate));
}


Identity::Identity(vector<int64_t> output_sz, const vector<int64_t>& shift):Transform(output_sz, shift)
{
    cout << "Identity" << endl;
}

Identity::~Identity()
{
}

Tensor Identity::call(const Tensor& image, bool is_mask)
{
    return crop_to_output(image);
}

