#include "dataset.h"

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<cv::Mat> loadImages(const std::string& folder_path)
{
    std::vector<std::string> file_names;
    std::vector<cv::Mat> images;

    // Legge tutti i file nella cartella
    for (const auto& entry : fs::directory_iterator(folder_path))
    {
        file_names.push_back(entry.path().string());
    }

    // Ordina i file (IMPORTANTE: 0000.png → 0001.png → ...)
    std::sort(file_names.begin(), file_names.end());

    // Carica immagini
    for (const auto& file : file_names)
    {
        cv::Mat img = cv::imread(file);

        if (img.empty())
        {
            std::cout << "Errore nel caricamento: " << file << std::endl;
            continue;
        }

        images.push_back(img);
    }

    return images;
}