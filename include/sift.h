#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

// Calcola keypoints e descrittori SIFT
void computeSIFT(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors
);