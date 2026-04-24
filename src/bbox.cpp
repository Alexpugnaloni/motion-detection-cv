#include "bbox.h"

cv::Rect computeBoundingBox(
    const std::vector<cv::Point2f>& points
) {
    if (points.empty())
        return cv::Rect();

    float xmin = points[0].x;
    float xmax = points[0].x;
    float ymin = points[0].y;
    float ymax = points[0].y;

    for (const auto& p : points)
    {
        xmin = std::min(xmin, p.x);
        xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }

    return cv::Rect(
        cv::Point2f(xmin, ymin),
        cv::Point2f(xmax, ymax)
    );
}