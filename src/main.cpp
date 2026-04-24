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
    float threshold = 5.0f;
    float cluster_radius = 35.0f;
    int density_threshold = 12;

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
    // DENSITY CLUSTERING
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

        if (count > density_threshold && count < 200)
        {
            final_points.push_back(filtered_points[i]);
        }
    }

    std::cout << "Punti finali (cluster): "
              << final_points.size() << std::endl;

    // ==============================
    // DEBUG CLUSTER
    // ==============================
    cv::Mat debug = frames[0].clone();

    for (const auto& p : final_points)
    {
        cv::circle(debug, p, 3, cv::Scalar(0, 255, 255), -1);
    }

    cv::imshow("FINAL CLUSTER", debug);
    cv::waitKey(0);

    // ==============================
    // BOUNDING BOX (percentili)
    // ==============================
    cv::Rect bbox = computeBoundingBox(final_points);

    // ==============================
    // 🔥 ESPANSIONE BASATA SU DISTRIBUZIONE
    // ==============================
    float xmin = 1e9, xmax = -1e9;
    float ymin = 1e9, ymax = -1e9;

    for (const auto& p : final_points)
    {
        xmin = std::min(xmin, p.x);
        xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }

    float range_x = xmax - xmin;
    float range_y = ymax - ymin;

    // espansione proporzionale
    bbox.x -= 0.2f * range_x;
    bbox.y -= 0.2f * range_y;
    bbox.width += 0.4f * range_x;
    bbox.height += 0.4f * range_y;

    // ==============================
    // CLAMP
    // ==============================
    bbox.x = std::max(0, bbox.x);
    bbox.y = std::max(0, bbox.y);

    bbox.width = std::min(frames[0].cols - bbox.x, bbox.width);
    bbox.height = std::min(frames[0].rows - bbox.y, bbox.height);

    // ==============================
    // VISUALIZZAZIONE
    // ==============================
    cv::Mat final_img = frames[0].clone();

    // Predicted bbox (verde)
    cv::rectangle(final_img, bbox, cv::Scalar(0, 255, 0), 2);
    cv::putText(final_img, "PRED", cv::Point(bbox.x, bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // Ground truth (rosso)
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