#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xphoto/white_balance.hpp>

class preprocess {
private:
    static constexpr uint_fast32_t NUM_CHANNELS = 3;

public:
    preprocess();
    ~preprocess();
    void gamma_correction(const cv::Mat &src, cv::Mat &dst, const float gamma);
    void histeq(cv::Mat &src, cv::Mat &dst);
    void white_balance_transform(cv::Mat &src, cv::Mat &dst);
};
