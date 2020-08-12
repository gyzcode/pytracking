#include "tracker.h"
#include <iostream>
#include <fstream>


using namespace std;
namespace F = torch::nn::functional;


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        //不提示INFO信息，只显示警告和错误
        if (severity != Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
}gLogger;




Tracker::Tracker()
{
    center_x = 0;
    center_y = 0;
    width = 0;
    height = 0;

    scale = 0.0;
    m_zSize = 0.0;
    m_xSize = 0.0;

    m_scales[0] = 0.964;
    m_scales[1] = 1.0;
    m_scales[2] = 1.0375;

    m_penalty = 0.96;

    // hanming_window = Hanming_weight(272, 272);

    torch::TensorOptions options;
    options= options.dtype(kFloat32).device(kCUDA);
    Tensor hann = torch::hann_window(272, options);
    mHannWindow = torch::ger(hann, hann);
    mHannWindow /= mHannWindow.sum();
    mHannWindow *= 0.176;

    m_zFeat1 = at::zeros({1 ,512 ,36 , 36}, kFloat32).to(kCUDA);
    m_zFeat2 = at::zeros({1 ,1024 ,18 , 18}, kFloat32).to(kCUDA);
    m_xFeat = at::zeros({3 ,256 ,22 , 22}, kFloat32).to(kCUDA);

    cudaStreamCreate(&m_stream);

}

Tracker::~Tracker()
{
}


void Tracker::Load(const String& fn)
{
    size_t size{ 0 };
    vector<char> trtModelStream_;
    ifstream file(fn.c_str(), ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);
        //cout << "size:" << trtModelStream_.size() << endl;
        file.read(trtModelStream_.data(), size);
        file.close();
    }
    //cout << "size" << size;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    mEngine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
    mContext = mEngine->createExecutionContext();

    // deal with first run slow issue
    Mat tmp(100, 100, CV_8UC3);
    Rect2d roi(40, 40, 20, 20);
    Init(tmp, roi);
    Update(tmp, roi);
}


void Tracker::Init(const Mat& img, const Rect2d& roi)
{
    // exemplar and search sizes
    float context = (roi.width + roi.height) * .5f;
    m_zSize = sqrt((roi.width + context) * (roi.height + context));
    m_xSize = m_zSize * 2.0f;

    // prepare input data
    Tensor tz;
    PreProcess(img, tz, roi, m_zSize, 127);

    // allocate buffers
    Dims inputDims = mEngine->getProfileDimensions(0, 0, OptProfileSelector::kOPT);
    // mContext->setBindingDimensions(0, inputDims);
    Dims outputDims1 = mContext->getBindingDimensions(1);
    Dims outputDims2 = mContext->getBindingDimensions(2);
 
    mDeviceBindings.clear();
    mDeviceBindings.emplace_back(tz.data_ptr());
    mDeviceBindings.emplace_back(m_zFeat1.data_ptr());
    mDeviceBindings.emplace_back(m_zFeat2.data_ptr());

    // Asynchronously enqueue the inference work
    mContext->enqueueV2(mDeviceBindings.data(), m_stream, nullptr);
    // Wait for the work in the m_stream to complete
    cudaStreamSynchronize(m_stream);

    mDeviceBindings[1] = m_xFeat.data_ptr();
}


void Tracker::Update(const Mat& img, Rect2d& roi)
{   
    Tensor txs[3];
    for (int i = 0; i < 3; i++) {
        PreProcess(img, txs[i], roi, m_xSize * m_scales[i], 255);
    }
    Tensor tx = cat({txs[0], txs[1], txs[2]});
    mDeviceBindings[0] = tx.data_ptr();

    // Asynchronously enqueue the inference work
    mContext->enqueueV2(mDeviceBindings.data(), m_stream, nullptr);
    // Wait for the work in the m_stream to complete
    cudaStreamSynchronize(m_stream);
    
    // cudaDeviceSynchronize();
    // TickMeter tm;
    // tm.start();
    // cross correlation
    Tensor response = F::conv2d(m_xFeat, m_zFeat1);
    // cudaDeviceSynchronize();
    // tm.stop();
    // cout << tm.getTimeMilli() << endl;

    // penalize scale changes
    response[0] *= m_penalty;
    response[2] *= m_penalty;

    // find scale
    int scaleId = floor_divide(argmax(response), (17*17)).item().to<int>();

    // upsample
    Tensor response1 = response[scaleId].unsqueeze(0);
    response1 = F::interpolate(response1, F::InterpolateFuncOptions().size(vector<int64_t>{272, 272}).mode(torch::kBicubic).align_corners(false)).squeeze();

    // peak location
    response1 = response1 * 0.824f + mHannWindow;
    int maxIdx = response1.argmax().item().to<int>();
    int maxIdxY = maxIdx / 272;
    int maxIdxX = maxIdx % 272;
    float dispx = maxIdxX - 271/2.0;
    float dispy = maxIdxY - 271/2.0;

    // update roi
    dispx /= 2.0;
    dispy /= 2.0;
    dispx = dispx * m_xSize / 255;
    dispy = dispy * m_xSize / 255;
    
    m_xSize *= m_scales[scaleId];

    roi.x += dispx;
    roi.y += dispy;
    roi.width *= m_scales[scaleId];
    roi.height *= m_scales[scaleId];
}


void Tracker::PreProcess(const Mat& src, Tensor& dst, const Rect2d& roi, int size, int outSize)
{
    // half
    int hw = roi.width / 2;
    int hh = roi.height / 2;
    int hs = size / 2;

    // roi center
    int cx = roi.x + hw;
    int cy = roi.y + hh;

    // new roi
    Rect newRoi(cx-hs, cy-hs, size, size);

    // left and top margin
    int left = max(0, hs - cx);
    int top = max(0, hs - cy);

    // intersection of new roi and src
    newRoi &= Rect(0, 0, src.cols, src.rows);

    // right and down margin
    int right = size - newRoi.width - left;
    int bottom = size - newRoi.height - top;
    
    // crop and pad
    Mat d;
    src(newRoi).copyTo(d);
    dst = at::from_blob(d.data, {1, newRoi.height, newRoi.width, 3}, torch::kUInt8).to(at::kCUDA);
    dst = dst.permute({0, 3, 1, 2}).to(kFloat32);
    dst = replication_pad2d(dst, {left, right, top, bottom});

    // resize
    dst = F::interpolate(dst, F::InterpolateFuncOptions().size(vector<int64_t>{outSize, outSize}).mode(torch::kBilinear).align_corners(false));
}