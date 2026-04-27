#include <iostream>
#include <vector>
#include <string>

#include "metrics.h"

int main()
{
    std::vector<std::string> categories = {
        "bird",
        "car",
        "frog",
        "sheep",
        "squirrel"
    };

    std::string predictionsFolder = "../output";
    std::string labelsFolder = "../labels";

    EvaluationSummary summary = evaluateDataset(
        categories,
        predictionsFolder,
        labelsFolder
    );

    printEvaluationSummary(summary);

    return 0;
}