#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Step 1: punti in movimento da un frame
std::vector<cv::Point2f> extractMovingPoints(
    const std::vector<cv::KeyPoint>& kp0,
    const std::vector<cv::KeyPoint>& kpi,
    const std::vector<cv::DMatch>& matches,
    float threshold
);

// Step 2: accumulo globale
void accumulateMovingPoints(
    std::vector<cv::Point2f>& all_points,
    const std::vector<cv::Point2f>& new_points
);

// Step 3: filtro semplice (cluster)
std::vector<cv::Point2f> filterPoints(
    const std::vector<cv::Point2f>& points,
    float radius
);