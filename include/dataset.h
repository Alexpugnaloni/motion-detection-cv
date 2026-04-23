#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Legge tutte le immagini in una cartella e le restituisce ordinate
std::vector<cv::Mat> loadImages(const std::string& folder_path);