#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "sift.h"
#include "matching.h"
#include "motion.h"
#include "bbox.h"
#include "handle_ground_truth.h"

int main()
{
    std::string path = "../data/frog/";
    std::string category = "frog";

    std::vector<cv::Mat> frames = loadImages(path);

    if (frames.empty())
    {
        std::cout << "Errore: nessun frame trovato!" << std::endl;
        return -1;
    }

    // ==============================
    // SIFT frame 0
    // ==============================
    std::vector<cv::KeyPoint> kp0;
    cv::Mat desc0;
    computeSIFT(frames[0], kp0, desc0);

    // ==============================
    // PARAMETRI
    // ==============================
    float threshold = 5.0f;      // movimento minimo
    float cluster_radius = 40.0; // densità locale
    int density_threshold = 15;  // punti minimi per cluster

    std::vector<cv::Point2f> all_moving_points;

    // ==============================
    // LOOP SU TUTTI I FRAME
    // ==============================
    for (int i = 1; i < frames.size(); i++)
    {
        std::vector<cv::KeyPoint> kpi;
        cv::Mat desci;

        computeSIFT(frames[i], kpi, desci);

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
    // 🔥 DENSITY CLUSTERING
    // ==============================
    std::vector<cv::Point2f> final_points;

    for (int i = 0; i < filtered_points.size(); i++)
    {
        int count = 0;

        for (int j = 0; j < filtered_points.size(); j++)
        {
            if (cv::norm(filtered_points[i] - filtered_points[j]) < cluster_radius)
            {
                count++;
            }
        }

        if (count > density_threshold)
        {
            final_points.push_back(filtered_points[i]);
        }
    }

    std::cout << "Punti finali (cluster): "
              << final_points.size() << std::endl;

    // ==============================
    // DEBUG
    // ==============================
    cv::Mat debug = frames[0].clone();

    for (const auto& p : final_points)
    {
        cv::circle(debug, p, 3, cv::Scalar(0, 255, 255), -1);
    }

    cv::imshow("FINAL CLUSTER", debug);
    cv::waitKey(0);

    // ==============================
    // BOUNDING BOX
    // ==============================
    cv::Rect bbox = computeBoundingBox(final_points);

    // ==============================
    // 🔥 PADDING (IMPORTANTE)
    // ==============================
    int padding = 15;

    bbox.x = std::max(0, bbox.x - padding);
    bbox.y = std::max(0, bbox.y - padding);
    bbox.width = std::min(frames[0].cols - bbox.x, bbox.width + 2 * padding);
    bbox.height = std::min(frames[0].rows - bbox.y, bbox.height + 2 * padding);

    // ==============================
    // VISUALIZZAZIONE
    // ==============================
    cv::Mat final_img = frames[0].clone();
    cv::rectangle(final_img, bbox, cv::Scalar(0, 255, 0), 2);

    // Load and draw the ground truth (RED) using the first frame's label
    std::string labelPath = "../labels/" + category + "/0000.txt";
    cv::Rect gt_bbox = handleGroundTruth(final_img, labelPath, true);

    cv::imshow("FINAL BOUNDING BOX", final_img);
    cv::waitKey(0);

    // ==============================
    // OUTPUT
    // ==============================
    std::ofstream out("../output/" + category + ".txt");

    out << bbox.x << " "
        << bbox.y << " "
        << bbox.x + bbox.width << " "
        << bbox.y + bbox.height << std::endl;

    out.close();

    cv::imwrite("../images_output/" + category + ".png", final_img);

    std::cout << "Output salvato!" << std::endl;

    return 0;
}