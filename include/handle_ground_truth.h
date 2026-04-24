#ifndef HANDLE_GROUND_TRUTH_H
#define HANDLE_GROUND_TRUTH_H

#include <opencv2/opencv.hpp>
#include <string>

/**
 * Handles Ground Truth data by reading coordinates from a file and optionally 
 * drawing them on an image.
 * * Format expected in the label file: <xmin> <ymin> <xmax> <ymax>
 * * @param img The OpenCV matrix (image) where the box will be drawn.
 * @param labelPath The filesystem path to the ground truth .txt file.
 * @param draw Boolean flag: if true, the function renders the GT box in RED.
 * @return cv::Rect The ground truth bounding box object for further processing (metrics).
 */
cv::Rect handleGroundTruth(cv::Mat& img, const std::string& labelPath, bool draw = true);

#endif // HANDLE_GROUND_TRUTH_H