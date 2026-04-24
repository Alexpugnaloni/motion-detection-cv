#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

/**
 * Reads Ground Truth from a file and optionally draws it on an image.
 * Format expected in file: <xmin> <ymin> <xmax> <ymax>
 * * @param img The image to draw on (pass a copy if you don't want to modify the original).
 * @param labelPath Path to the .txt file containing the ground truth coordinates.
 * @param draw If true, draws the red rectangle on the provided image.
 * @return cv::Rect The ground truth bounding box.
 */
cv::Rect handleGroundTruth(cv::Mat& img, const std::string& labelPath, bool draw = true) {
    cv::Rect gtBox(0, 0, 0, 0);
    std::ifstream file(labelPath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open label file at " << labelPath << std::endl;
        return gtBox;
    }

    int xmin, ymin, xmax, ymax;
    // According to your file 0000.txt: 212 240 280 282
    if (file >> xmin >> ymin >> xmax >> ymax) {
        // Convert [xmin, ymin, xmax, ymax] to [x, y, width, height]
        gtBox.x = xmin;
        gtBox.y = ymin;
        gtBox.width = xmax - xmin;
        gtBox.height = ymax - ymin;

        if (draw && !img.empty()) {
            // Ensure the image is BGR for colored drawing
            if (img.channels() == 1) {
                cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
            }
            // Draw Ground Truth in RED
            cv::rectangle(img, gtBox, cv::Scalar(0, 0, 255), 2);
            cv::putText(img, "GT", cv::Point(gtBox.x, gtBox.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
    }

    file.close();
    return gtBox;
}