#include "matching.h"

std::vector<cv::DMatch> matchFeatures(
    const cv::Mat& desc1,
    const cv::Mat& desc2
) {
    std::vector<std::vector<cv::DMatch>> knn_matches;
    std::vector<cv::DMatch> good_matches;

    cv::BFMatcher matcher(cv::NORM_L2);

    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    const float ratio_thresh = 0.75f;

    for (const auto& pair : knn_matches) {
        if (pair.size() < 2) continue;

        const cv::DMatch& m = pair[0];
        const cv::DMatch& n = pair[1];

        if (m.distance < ratio_thresh * n.distance) {
            good_matches.push_back(m);
        }
    }

    return good_matches;
}