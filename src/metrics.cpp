#include "metrics.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;

cv::Rect readBoxFromFile(const std::string& path)
{
    std::ifstream file(path);

    if (!file.is_open())
    {
        std::cerr << "Error: cannot open file " << path << std::endl;
        return cv::Rect();
    }

    int xmin, ymin, xmax, ymax;

    if (!(file >> xmin >> ymin >> xmax >> ymax))
    {
        std::cerr << "Error: invalid bounding box format in " << path << std::endl;
        return cv::Rect();
    }

    int width = xmax - xmin;
    int height = ymax - ymin;

    if (width <= 0 || height <= 0)
    {
        std::cerr << "Error: invalid bounding box dimensions in " << path << std::endl;
        return cv::Rect();
    }

    return cv::Rect(xmin, ymin, width, height);
}

std::vector<cv::Rect> readBoxesFromFile(const std::string& path)
{
    std::vector<cv::Rect> boxes;
    std::ifstream file(path);

    if (!file.is_open())
    {
        std::cerr << "Error: cannot open file " << path << std::endl;
        return boxes;
    }

    int xmin, ymin, xmax, ymax;

    while (file >> xmin >> ymin >> xmax >> ymax)
    {
        int width = xmax - xmin;
        int height = ymax - ymin;

        if (width > 0 && height > 0)
        {
            boxes.push_back(cv::Rect(xmin, ymin, width, height));
        }
    }

    return boxes;
}

double computeIoU(const cv::Rect& predictedBox, const cv::Rect& groundTruthBox)
{
    if (predictedBox.empty() || groundTruthBox.empty())
    {
        return 0.0;
    }

    cv::Rect intersection = predictedBox & groundTruthBox;

    double intersectionArea = static_cast<double>(intersection.area());
    double predictedArea = static_cast<double>(predictedBox.area());
    double groundTruthArea = static_cast<double>(groundTruthBox.area());

    double unionArea = predictedArea + groundTruthArea - intersectionArea;

    if (unionArea <= 0.0)
    {
        return 0.0;
    }

    return intersectionArea / unionArea;
}

bool isCorrectDetection(double iou, double threshold)
{
    return iou > threshold;
}

static std::vector<cv::Rect> readGroundTruthBoxesForCategory(
    const std::string& labelsFolder,
    const std::string& category
)
{
    std::vector<cv::Rect> gtBoxes;

    fs::path categoryPath = fs::path(labelsFolder) / category;

    if (!fs::exists(categoryPath))
    {
        std::cerr << "Warning: label folder not found for category " << category << std::endl;
        return gtBoxes;
    }

    for (const auto& entry : fs::recursive_directory_iterator(categoryPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
        {
            std::vector<cv::Rect> boxes = readBoxesFromFile(entry.path().string());
            gtBoxes.insert(gtBoxes.end(), boxes.begin(), boxes.end());
        }
    }

    return gtBoxes;
}

EvaluationSummary evaluateDataset(
    const std::vector<std::string>& categories,
    const std::string& predictionsFolder,
    const std::string& labelsFolder
)
{
    EvaluationSummary summary;
    summary.meanIoU = 0.0;
    summary.detectionAccuracy = 0.0;

    int correctDetections = 0;
    int totalElements = 0;
    double iouSum = 0.0;

    for (const std::string& category : categories)
    {
        std::string predictionPath = predictionsFolder + "/" + category + ".txt";

        std::vector<cv::Rect> predictedBoxes = readBoxesFromFile(predictionPath);
        std::vector<cv::Rect> gtBoxes = readGroundTruthBoxesForCategory(labelsFolder, category);

        double categoryBestIoU = 0.0;
        bool categoryCorrect = false;

        if (!predictedBoxes.empty() && !gtBoxes.empty())
        {
            for (const cv::Rect& gtBox : gtBoxes)
            {
                double bestIoUForGT = 0.0;

                for (const cv::Rect& predBox : predictedBoxes)
                {
                    double iou = computeIoU(predBox, gtBox);
                    bestIoUForGT = std::max(bestIoUForGT, iou);
                }

                categoryBestIoU = std::max(categoryBestIoU, bestIoUForGT);

                if (isCorrectDetection(bestIoUForGT))
                {
                    correctDetections++;
                    categoryCorrect = true;
                }

                totalElements++;
            }
        }
        else
        {
            totalElements++;
        }

        summary.results.push_back({category, categoryBestIoU, categoryCorrect});
        iouSum += categoryBestIoU;
    }

    if (!categories.empty())
    {
        summary.meanIoU = iouSum / static_cast<double>(categories.size());
    }

    if (totalElements > 0)
    {
        summary.detectionAccuracy =
            static_cast<double>(correctDetections) / static_cast<double>(totalElements);
    }

    return summary;
}

void printEvaluationSummary(const EvaluationSummary& summary)
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\n===== Evaluation Results =====\n";
    std::cout << std::left
              << std::setw(15) << "Category"
              << std::setw(12) << "IoU"
              << std::setw(12) << "Correct"
              << std::endl;

    std::cout << "--------------------------------------\n";

    for (const auto& result : summary.results)
    {
        std::cout << std::left
                  << std::setw(15) << result.category
                  << std::setw(12) << result.iou
                  << std::setw(12) << (result.correct ? "Yes" : "No")
                  << std::endl;
    }

    std::cout << "--------------------------------------\n";
    int correctCategories = 0;

for (const auto& result : summary.results)
{
    if (result.correct)
    {
        correctCategories++;
    }
}

double categoryAccuracy = 0.0;

if (!summary.results.empty())
{
    categoryAccuracy =
        static_cast<double>(correctCategories) /
        static_cast<double>(summary.results.size());
}

std::cout << "mIoU: " << summary.meanIoU << std::endl;
std::cout << "Detection accuracy: "
          << categoryAccuracy * 100.0 << "%" << std::endl;

std::cout << "DEBUG: correct categories = "
          << correctCategories << " / "
          << summary.results.size() << std::endl;
}