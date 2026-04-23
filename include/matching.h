#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::DMatch> matchFeatures(
    const cv::Mat& desc1,
    const cv::Mat& desc2
);