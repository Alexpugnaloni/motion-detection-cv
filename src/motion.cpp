#include "motion.h"

// ==============================
// STEP 1: estrazione movimento
// ==============================
std::vector<cv::Point2f> extractMovingPoints(
    const std::vector<cv::KeyPoint>& kp0,
    const std::vector<cv::KeyPoint>& kpi,
    const std::vector<cv::DMatch>& matches,
    float threshold
) {
    std::vector<cv::Point2f> moving_points;

    for (const auto& m : matches)
    {
        cv::Point2f p0 = kp0[m.queryIdx].pt;
        cv::Point2f pi = kpi[m.trainIdx].pt;

        float dist = cv::norm(pi - p0);

        if (dist > threshold)
        {
            moving_points.push_back(p0); // SEMPRE frame 0
        }
    }

    return moving_points;
}

// ==============================
// STEP 2: accumulo
// ==============================
void accumulateMovingPoints(
    std::vector<cv::Point2f>& all_points,
    const std::vector<cv::Point2f>& new_points
) {
    all_points.insert(all_points.end(), new_points.begin(), new_points.end());
}

// ==============================
// STEP 3: filtro semplice (centro + raggio)
// ==============================
std::vector<cv::Point2f> filterPoints(
    const std::vector<cv::Point2f>& points,
    float radius
) {
    std::vector<cv::Point2f> filtered;

    if (points.empty())
        return filtered;

    // Calcolo centro medio
    cv::Point2f center(0, 0);

    for (const auto& p : points)
    {
        center += p;
    }

    center.x /= points.size();
    center.y /= points.size();

    // Filtro per distanza dal centro
    for (const auto& p : points)
    {
        if (cv::norm(p - center) < radius)
        {
            filtered.push_back(p);
        }
    }

    return filtered;
}