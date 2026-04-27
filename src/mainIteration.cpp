#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "sift.h"
#include "matching.h"
#include "motion.h"
#include "bbox.h"
#include "handle_ground_truth.h"

namespace fs = std::filesystem;

// Disegna tutte le ground truth della categoria.
// Uso recursive_directory_iterator perché squirrel ha più sottocartelle.
void drawGroundTruthBoxes(cv::Mat& image, const std::string& category)
{
    std::string labelFolder = "../labels/" + category + "/";

    if (!fs::exists(labelFolder))
    {
        std::cout << "Warning: label folder not found for " << category << std::endl;
        return;
    }

    for (const auto& entry : fs::recursive_directory_iterator(labelFolder))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".txt")
        {
            continue;
        }

        std::ifstream file(entry.path());

        if (!file.is_open())
        {
            std::cout << "Warning: cannot open label file: "
                      << entry.path().string() << std::endl;
            continue;
        }

        int xmin, ymin, xmax, ymax;

        if (file >> xmin >> ymin >> xmax >> ymax)
        {
            cv::Rect gt_bbox(xmin, ymin, xmax - xmin, ymax - ymin);

            if (gt_bbox.width > 0 && gt_bbox.height > 0)
            {
                cv::rectangle(image, gt_bbox, cv::Scalar(0, 0, 255), 2);

                cv::putText(
                    image,
                    "GT",
                    cv::Point(gt_bbox.x, std::max(0, gt_bbox.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 255),
                    1
                );
            }
        }

        file.close();
    }
}

bool processCategory(const std::string& category)
{
    std::cout << "\n==============================" << std::endl;
    std::cout << "Processing category: " << category << std::endl;
    std::cout << "==============================" << std::endl;

    std::string path = "../data/" + category + "/";

    std::vector<cv::Mat> frames = loadImages(path);

    if (frames.empty())
    {
        std::cout << "Errore: nessun frame trovato per " << category << std::endl;
        return false;
    }

    // ==============================
    // SIFT frame 0
    // ==============================
    std::vector<cv::KeyPoint> kp0;
    cv::Mat desc0;

    computeSIFT(frames[0], kp0, desc0);

    if (desc0.empty())
    {
        std::cout << "Errore: descrittori vuoti nel primo frame di "
                  << category << std::endl;
        return false;
    }

    // ==============================
    // PARAMETRI
    // ==============================
    float threshold = 5.0f;
    float cluster_radius = 35.0f;
    int density_threshold = 12;

    std::vector<cv::Point2f> all_moving_points;

    // ==============================
    // LOOP SU TUTTI I FRAME
    // ==============================
    for (int i = 1; i < static_cast<int>(frames.size()); i++)
    {
        std::vector<cv::KeyPoint> kpi;
        cv::Mat desci;

        computeSIFT(frames[i], kpi, desci);

        if (desci.empty())
        {
            continue;
        }

        std::vector<cv::DMatch> matches =
            matchFeatures(desc0, desci);

        std::vector<cv::Point2f> moving_points =
            extractMovingPoints(kp0, kpi, matches, threshold);

        accumulateMovingPoints(all_moving_points, moving_points);
    }

    std::cout << "Totale punti accumulati: "
              << all_moving_points.size() << std::endl;

    // ==============================
    // FILTRO SPAZIALE BASE
    // ==============================
    std::vector<cv::Point2f> filtered_points =
        filterPoints(all_moving_points, 100.0f);

    // ==============================
    // DENSITY CLUSTERING
    // ==============================
    std::vector<cv::Point2f> final_points;

    for (int i = 0; i < static_cast<int>(filtered_points.size()); i++)
    {
        int count = 0;

        for (int j = 0; j < static_cast<int>(filtered_points.size()); j++)
        {
            if (cv::norm(filtered_points[i] - filtered_points[j]) < cluster_radius)
            {
                count++;
            }
        }

        if (count > density_threshold && count < 200)
        {
            final_points.push_back(filtered_points[i]);
        }
    }

    std::cout << "Punti finali (cluster): "
              << final_points.size() << std::endl;

    // ==============================
    // IMAGE 1: PUNTI IDENTIFICATI
    // ==============================
    cv::Mat points_img = frames[0].clone();

    for (const auto& p : final_points)
    {
        cv::circle(points_img, p, 3, cv::Scalar(0, 255, 255), -1);
    }

    std::string pointsOutputPath =
        "../images_output/" + category + "_points.png";

    if (!cv::imwrite(pointsOutputPath, points_img))
    {
        std::cout << "Errore: impossibile salvare immagine "
                  << pointsOutputPath << std::endl;
        return false;
    }

    cv::imshow("IDENTIFIED POINTS - " + category, points_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // ==============================
    // BOUNDING BOX
    // ==============================
    cv::Rect bbox;

    if (!final_points.empty())
    {
        bbox = computeBoundingBox(final_points);

        float xmin = 1e9f;
        float xmax = -1e9f;
        float ymin = 1e9f;
        float ymax = -1e9f;

        for (const auto& p : final_points)
        {
            xmin = std::min(xmin, p.x);
            xmax = std::max(xmax, p.x);
            ymin = std::min(ymin, p.y);
            ymax = std::max(ymax, p.y);
        }

        float range_x = xmax - xmin;
        float range_y = ymax - ymin;

        // Espansione proporzionale della bounding box
        bbox.x = static_cast<int>(bbox.x - 0.2f * range_x);
        bbox.y = static_cast<int>(bbox.y - 0.2f * range_y);
        bbox.width = static_cast<int>(bbox.width + 0.4f * range_x);
        bbox.height = static_cast<int>(bbox.height + 0.4f * range_y);

        // Clamp dentro l'immagine
        bbox.x = std::max(0, bbox.x);
        bbox.y = std::max(0, bbox.y);

        bbox.width = std::min(frames[0].cols - bbox.x, bbox.width);
        bbox.height = std::min(frames[0].rows - bbox.y, bbox.height);

        if (bbox.width < 0)
        {
            bbox.width = 0;
        }

        if (bbox.height < 0)
        {
            bbox.height = 0;
        }
    }
    else
    {
        std::cout << "Warning: nessun punto finale trovato per "
                  << category << std::endl;

        bbox = cv::Rect(0, 0, 0, 0);
    }

    // ==============================
    // IMAGE 2: BOUNDING BOX
    // ==============================
    cv::Mat boxes_img = frames[0].clone();

    // Predicted bbox verde
    if (bbox.width > 0 && bbox.height > 0)
    {
        cv::rectangle(boxes_img, bbox, cv::Scalar(0, 255, 0), 2);

        cv::putText(
            boxes_img,
            "PRED",
            cv::Point(bbox.x, std::max(0, bbox.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            1
        );
    }

    // Ground truth rossa
    drawGroundTruthBoxes(boxes_img, category);

    std::string boxesOutputPath =
        "../images_output/" + category + "_boxes.png";

    if (!cv::imwrite(boxesOutputPath, boxes_img))
    {
        std::cout << "Errore: impossibile salvare immagine "
                  << boxesOutputPath << std::endl;
        return false;
    }

    cv::imshow("BOUNDING BOXES - " + category, boxes_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // ==============================
    // OUTPUT TXT
    // ==============================
    std::string outputPath = "../output/" + category + ".txt";
    std::ofstream out(outputPath);

    if (!out.is_open())
    {
        std::cout << "Errore: impossibile salvare " << outputPath << std::endl;
        return false;
    }

    out << bbox.x << " "
        << bbox.y << " "
        << bbox.x + bbox.width << " "
        << bbox.y + bbox.height << std::endl;

    out.close();

    std::cout << "Output salvato per " << category << std::endl;
    std::cout << "Punti salvati in: " << pointsOutputPath << std::endl;
    std::cout << "Box salvate in: " << boxesOutputPath << std::endl;

    return true;
}

int main()
{
    fs::create_directories("../output");
    fs::create_directories("../images_output");

    std::vector<std::string> categories = {
        "bird",
        "car",
        "frog",
        "sheep",
        "squirrel"
    };

    int successful = 0;

    for (const std::string& category : categories)
    {
        if (processCategory(category))
        {
            successful++;
        }
    }

    std::cout << "\n==============================" << std::endl;
    std::cout << "Processing completed." << std::endl;
    std::cout << "Successful categories: "
              << successful << " / " << categories.size() << std::endl;
    std::cout << "==============================" << std::endl;

    return 0;
}