#include "preprocess.hpp"

void preprocess::gamma_correction(const cv::Mat &src, cv::Mat &dst, const float gamma)
{
    cv::Mat table(1, 256, CV_8U);
    uchar *p = table.ptr();
    for (uint32_t i = 0; i < 256; ++i) {
        p[i] = (uchar) (pow(i / 255.0, gamma) * 255);
    }

    cv::LUT(src, table, dst);
}

void preprocess::histeq(cv::Mat &src, cv::Mat &dst)
{
    //Convert the image from BGR to YCrCb color space
    cv::cvtColor(src, dst, cv::COLOR_BGR2YCrCb);

    //Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
    std::vector<cv::Mat> vec_channels;
    cv::split(dst, vec_channels); 

    //Equalize the histogram of only the Y channel 
    cv::equalizeHist(vec_channels[0], vec_channels[0]);

    //Merge 3 channels in the vector to form the color image in YCrCB color space.
    cv::merge(vec_channels, dst);         
    cv::cvtColor(dst, dst, cv::COLOR_YCrCb2BGR);
}

void preprocess::white_balance_transform(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat planes[NUM_CHANNELS];
    cv::split(src, planes);

    for(uint32_t ch_i=0;ch_i<NUM_CHANNELS;ch_i++) {
        cv::Ptr<cv::xphoto::WhiteBalancer> wb = cv::xphoto::createSimpleWB();
        wb->balanceWhite(planes[ch_i], planes[ch_i]); 
    }

    cv::merge(planes, NUM_CHANNELS, dst);
}

preprocess::preprocess()  { }
preprocess::~preprocess() { }
