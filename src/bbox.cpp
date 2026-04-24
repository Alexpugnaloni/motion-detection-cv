#include "bbox.h"

cv::Rect computeBoundingBox(
    const std::vector<cv::Point2f>& points
) {
    if (points.empty())
        return cv::Rect();

    std::vector<float> xs, ys;

    for (const auto& p : points)
    {
        xs.push_back(p.x);
        ys.push_back(p.y);
    }

    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());

    int n = xs.size();

    int low = n * 0.1;   // 10%
    int high = n * 0.9;  // 90%

    float xmin = xs[low];
    float xmax = xs[high];
    float ymin = ys[low];
    float ymax = ys[high];

    return cv::Rect(
        cv::Point2f(xmin, ymin),
        cv::Point2f(xmax, ymax)
    );
}
