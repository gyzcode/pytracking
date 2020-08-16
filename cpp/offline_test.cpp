#include "tracker.h"
#include "python2c.h"
#include <opencv2/opencv.hpp>
#include <torch/csrc/api/include/torch/torch.h>

using namespace torch;
using namespace torch::indexing;
using namespace cv;

int main()
{
    //test python2c
    Mat cvimg = imread("/home/gyz/dataset/otb100/Crossing/img/0001.jpg");
    Tensor im = torch::from_blob(cvimg.data, {cvimg.rows, cvimg.cols, 3}, kByte).permute({2, 0, 1}).unsqueeze(0).to(kFloat32);

    Tensor pos = tensor({176, 213});
    Tensor sample_sz = tensor({291.5, 291.5});
    Tensor output_sz = tensor({576.f, 576.f});
    Tensor im_patch;
    Tensor patch_coord;
    sample_patch(im, pos, sample_sz, im_patch, patch_coord, output_sz);

    vector<int64_t> aug_output_sz({288, 288});
    Identity transforms(aug_output_sz);
    Tensor im_patches = transforms.call(im_patch);

    int rows = im_patches.sizes()[2];
    int cols = im_patches.sizes()[3];
    im_patches = im_patches.to(kByte).squeeze().permute({1, 2, 0}).contiguous();
    Mat imgshow(Size(cols, rows), CV_8UC3, im_patches.data_ptr<uchar>());
    imshow("test", imgshow);
    waitKey();

    Tracker myTracker;
    myTracker.Load("/home/gyz/workzone/pytracking/pytracking/networks/dimp50_output.engine");
    Mat frame;
    String fn;
    //Rect2d roi(204,150,17,50);  //Crossing
    //int numFrame = 120;
    Rect2d roi(288,143,35,42);  //Boy
    int numFrame = 602;
    for(int i=1; i<=numFrame; i++) {
        fn = format("/home/gyz/dataset/otb100/Boy/img/%04d.jpg", i);
        frame = imread(fn);

        TickMeter tm;
        tm.start();
        if(i == 1) {
            myTracker.Init(frame, roi);
        }
        else {
            myTracker.Update(frame, roi);
        }
        tm.stop();
        cout << tm.getTimeMilli() << endl;

        rectangle(frame, roi, CV_RGB(255, 0, 0), 2);
        imshow("Display", frame);
        waitKey(1);
    }

    destroyAllWindows();


    return 0;
}