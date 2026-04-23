#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "sift.h"

int main()
{
    // Path relativo (IMPORTANTE: siamo in build/)
    std::string path = "../data/bird/";

    // Caricamento immagini
    std::vector<cv::Mat> frames = loadImages(path);

    std::cout << "Numero di frame caricati: " << frames.size() << std::endl;

    if (frames.empty())
    {
        std::cout << "Errore: nessun frame trovato!" << std::endl;
        return -1;
    }

    // === SIFT sul primo frame ===
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    computeSIFT(frames[0], keypoints, descriptors);

    std::cout << "Numero keypoints: " << keypoints.size() << std::endl;

    // Disegna keypoints
    cv::Mat output;
    cv::drawKeypoints(frames[0], keypoints, output);

    // Mostra immagine
    cv::imshow("SIFT Keypoints - Frame 0", output);
    cv::waitKey(0);

    return 0;
}