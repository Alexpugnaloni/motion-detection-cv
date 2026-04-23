#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "sift.h"
#include "matching.h"
#include "motion.h"

int main()
{
    // Path relativo (IMPORTANTE: siamo in build/)
    std::string path = "../data/sheep/";

    // Caricamento immagini
    std::vector<cv::Mat> frames = loadImages(path);

    std::cout << "Numero di frame caricati: " << frames.size() << std::endl;

    if (frames.empty())
    {
        std::cout << "Errore: nessun frame trovato!" << std::endl;
        return -1;
    }

    // ==============================
    // SIFT sul frame 0
    // ==============================
    std::vector<cv::KeyPoint> kp0;
    cv::Mat desc0;

    computeSIFT(frames[0], kp0, desc0);

    std::cout << "Numero keypoints frame 0: " << kp0.size() << std::endl;

    // Debug keypoints
    cv::Mat output;
    cv::drawKeypoints(frames[0], kp0, output);

    cv::imshow("SIFT Keypoints - Frame 0", output);
    cv::waitKey(0);

    // ==============================
    // PARAMETRI
    // ==============================
    float threshold = 4.0f;
    float radius = 70.0f;

    // ==============================
    // ACCUMULO GLOBALE
    // ==============================
    std::vector<cv::Point2f> all_moving_points;

    // ==============================
    // LOOP SU TUTTI I FRAME
    // ==============================
    for (int i = 1; i < frames.size(); i++)
    {
        std::vector<cv::KeyPoint> kpi;
        cv::Mat desci;

        computeSIFT(frames[i], kpi, desci);

        // Matching con frame 0
        std::vector<cv::DMatch> matches = matchFeatures(desc0, desci);

        std::cout << "Frame " << i << " -> matches: "
                  << matches.size() << std::endl;

        // Motion analysis
        std::vector<cv::Point2f> moving_points =
            extractMovingPoints(kp0, kpi, matches, threshold);

        std::cout << "Frame " << i << " -> moving points: "
                  << moving_points.size() << std::endl;

        // Accumulo globale
        accumulateMovingPoints(all_moving_points, moving_points);

        // ==============================
        // DEBUG MATCH
        // ==============================
        cv::Mat img_matches;
        cv::drawMatches(frames[0], kp0,
                        frames[i], kpi,
                        matches, img_matches);

        cv::imshow("Matches frame 0 - frame " + std::to_string(i), img_matches);

        // ==============================
        // DEBUG MOVIMENTO (FRAME CORRENTE)
        // ==============================
        cv::Mat debug = frames[0].clone();

        for (const auto& p : moving_points)
        {
            cv::circle(debug, p, 3, cv::Scalar(0, 0, 255), -1);
        }

        cv::imshow("Moving Points (Frame 0) - frame " + std::to_string(i), debug);

        cv::waitKey(30); // automatico
    }

    // ==============================
    // DEBUG ACCUMULO (NO FILTER)
    // ==============================
    std::cout << "Totale punti accumulati: "
              << all_moving_points.size() << std::endl;

    cv::Mat debug_all = frames[0].clone();

    for (const auto& p : all_moving_points)
    {
        cv::circle(debug_all, p, 2, cv::Scalar(255, 0, 0), -1);
    }

    cv::imshow("ALL MOVING POINTS (NO FILTER)", debug_all);
    cv::waitKey(0);

    // ==============================
    // 🔥 FILTRO (CLUSTER)
    // ==============================
    std::vector<cv::Point2f> filtered_points =
        filterPoints(all_moving_points, radius);

    std::cout << "Punti filtrati: "
              << filtered_points.size() << std::endl;

    // ==============================
    // DEBUG FILTRO
    // ==============================
    cv::Mat filtered_debug = frames[0].clone();

    for (const auto& p : filtered_points)
    {
        cv::circle(filtered_debug, p, 3, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("FILTERED POINTS", filtered_debug);
    cv::waitKey(0);

    return 0;
}