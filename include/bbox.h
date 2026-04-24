#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

cv::Rect computeBoundingBox(
    const std::vector<cv::Point2f>& points
);