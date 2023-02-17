//
// Created by felix on 08.09.22.
//

#include <Dynamic.h>

int main() {
    Point initialPoint{31.36, 68.87};
    Point leftPoint{-80.36, 68.87};
    Point upPoint{-80.36, 160};
    Point rightPoint{31.36, 160};


    MovingGridConfig config;
    MovingGrid movingGrid(initialPoint, config);

    std::cout << movingGrid.getCovariance() << std::endl;

    movingGrid(leftPoint);

    std::cout << movingGrid.getCovariance() << std::endl;

    movingGrid(upPoint);

    std::cout << movingGrid.getCovariance() << std::endl;

    movingGrid(rightPoint);

    std::cout << movingGrid.getCovariance() << std::endl;

    movingGrid(initialPoint);

    std::cout << movingGrid.getCovariance() << std::endl;

    return 0;
}