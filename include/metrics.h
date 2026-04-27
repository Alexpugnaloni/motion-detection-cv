#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct CategoryMetrics {
    std::string category;
    double iou;
    bool correct;
};

struct EvaluationSummary {
    std::vector<CategoryMetrics> results;
    double meanIoU;
    double detectionAccuracy;
};

cv::Rect readBoxFromFile(const std::string& path);

std::vector<cv::Rect> readBoxesFromFile(const std::string& path);

double computeIoU(const cv::Rect& predictedBox, const cv::Rect& groundTruthBox);

bool isCorrectDetection(double iou, double threshold = 0.5);

EvaluationSummary evaluateDataset(
    const std::vector<std::string>& categories,
    const std::string& predictionsFolder,
    const std::string& labelsFolder
);

void printEvaluationSummary(const EvaluationSummary& summary);