#include "python2c.h"
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace torch::indexing;
using namespace std;

// def sample_patch(im: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
//                  mode: str = 'replicate', max_scale_change=None, is_mask=False):
void sample_patch(torch::Tensor im, Tensor pos, Tensor sample_sz, optional<torch::Tensor> output_sz,
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
        cout << posl << endl;
        //im2 = im.index({"...", Slice(os[0].item<int>(), None, df), Slice(os[1].item<int>(), None, df)});
        im2 = im.index({"...", Slice(os[1].item<int>(), None, df)});
    }
    else {
        im2 = im;
    }

    int rows = im.sizes()[2];
    int cols = im.sizes()[3];
    Mat imgshow(Size(cols, rows), CV_8UC3, im.squeeze().permute({1, 2, 0}).data_ptr<uchar>());
    imshow("test", imgshow);
    waitKey();
        

//     # compute size to crop
//     szl = torch.max(sz.round(), torch.Tensor([2])).long()

//     # Extract top and bottom coordinates
    
//     # tl = posl - (szl - 1)/2
//     # br = posl + szl/2 + 1
//     tl = posl - torch.floor_divide(szl - 1, 2)
//     br = posl + torch.floor_divide(szl, 2) + 1

//     # Shift the crop to inside
//     if mode == 'inside' or mode == 'inside_major':
//         im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
//         shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
//         tl += shift
//         br += shift

//         outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
//         shift = (-tl - outside) * (outside > 0).long()
//         tl += shift
//         br += shift

//         # Get image patch
//         # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

//     # Get image patch
//     if not is_mask:
//         im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]), pad_mode)
//     else:
//         im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]))

//     # Get image coordinates
//     patch_coord = df * torch.cat((tl, br)).view(1,4)

//     if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
//         return im_patch.clone(), patch_coord

//     # Resample
//     if not is_mask:
//         im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
//     else:
//         im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')

//     return im_patch, patch_coord

}