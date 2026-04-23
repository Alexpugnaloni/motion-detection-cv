#include "sift.h"

void computeSIFT(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors
)
{
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}