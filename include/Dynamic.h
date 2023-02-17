//
// Created by felix on 08.09.22.
//

#ifndef SOUNDSPEED_ESTIMATOR_DYNAMIC_H
#define SOUNDSPEED_ESTIMATOR_DYNAMIC_H

#include <Typedefs.h>
#include <Grid.h>
#include <GaussianProcess.h>

#include <memory>
#include <utility>
#include <iostream>

inline int roundToMultiple(double x, int multiple)
{
    int lb = (static_cast<int>(x) / multiple) * multiple;
    int ub = lb + multiple;
    return (x - lb > ub - x) ? ub : lb;
}

struct MovingGridConfig {
    int stepSize = 10;
    int numSteps = 10;
    int numSubgridsPerDirection = 100;
};

enum class Move {
    Up,
    Right,
    Down,
    Left,
    None
};

class MetaGrid {
public:
    explicit MetaGrid(int gridsPerDirection) : rows_(gridsPerDirection), cols_(gridsPerDirection), currRow_(rows_/2 - 1), currCol_(cols_/2 - 1), grids_(rows_*cols_) {  }
    ~MetaGrid() = default;

    void moveN() { currRow_--; }
    void moveE() { currCol_++; }
    void moveS() { currRow_++; }
    void moveW() { currCol_--; }

    std::shared_ptr<Grid>& C() { return grids_[currRow_ * cols_ + currCol_]; }
    std::shared_ptr<Grid>& N() { return grids_[(currRow_ - 1) * cols_ + currCol_]; }
    std::shared_ptr<Grid>& E() { return grids_[currRow_ * cols_ + (currCol_ + 1)]; }
    std::shared_ptr<Grid>& S() { return grids_[(currRow_ + 1) * cols_ + currCol_]; }
    std::shared_ptr<Grid>& W() { return grids_[currRow_ * cols_ + (currCol_ - 1)]; }
    std::shared_ptr<Grid>& NE() { return grids_[(currRow_ - 1) * cols_ + (currCol_ + 1)]; }
    std::shared_ptr<Grid>& SE() { return grids_[(currRow_ + 1) * cols_ + (currCol_ + 1)]; }
    std::shared_ptr<Grid>& SW() { return grids_[(currRow_ + 1) * cols_ + (currCol_ - 1)]; }
    std::shared_ptr<Grid>& NW() { return grids_[(currRow_ - 1) * cols_ + (currCol_ - 1)]; }

    void C(std::initializer_list<Interval> &&intervalList) {
        grids_[currRow_ * cols_ + currCol_] = std::make_shared<Grid>(intervalList);
    }
    void N(std::initializer_list<Interval> &&intervalList) {
        grids_[(currRow_ - 1) * cols_ + currCol_] = std::make_shared<Grid>(intervalList);
    }
    void E(std::initializer_list<Interval> &&intervalList) {
        grids_[currRow_ * cols_ + (currCol_ + 1)] = std::make_shared<Grid>(intervalList);
    }
    void S(std::initializer_list<Interval> &&intervalList) {
        grids_[(currRow_ + 1) * cols_ + currCol_] = std::make_shared<Grid>(intervalList);
    }
    void W(std::initializer_list<Interval> &&intervalList) {
        grids_[currRow_ * cols_ + (currCol_ - 1)] = std::make_shared<Grid>(intervalList);
    }
    void NE(std::initializer_list<Interval> &&intervalList) {
        grids_[(currRow_ - 1) * cols_ + (currCol_ + 1)] = std::make_shared<Grid>(intervalList);
    }
    void SE(std::initializer_list<Interval> &&intervalList) {
        grids_[(currRow_ + 1) * cols_ + (currCol_ + 1)] = std::make_shared<Grid>(intervalList);
    }
    void SW(std::initializer_list<Interval> &&intervalList) {
        grids_[(currRow_ + 1) * cols_ + (currCol_ - 1)] = std::make_shared<Grid>(intervalList);
    }
    void NW(std::initializer_list<Interval> &&intervalList) {
        grids_[(currRow_ - 1) * cols_ + (currCol_ - 1)] = std::make_shared<Grid>(intervalList);
    }

    void C(const std::shared_ptr<Grid> &ptr) { grids_[currRow_ * cols_ + currCol_] = ptr; }
    void N(const std::shared_ptr<Grid> &ptr) { grids_[(currRow_ - 1) * cols_ + currCol_] = ptr; }
    void E(const std::shared_ptr<Grid> &ptr) { grids_[currRow_ * cols_ + (currCol_ + 1)] = ptr; }
    void S(const std::shared_ptr<Grid> &ptr) { grids_[(currRow_ + 1) * cols_ + currCol_] = ptr; }
    void W(const std::shared_ptr<Grid> &ptr) { grids_[currRow_ * cols_ + (currCol_ - 1)] = ptr; }
    void NE(const std::shared_ptr<Grid> &ptr) { grids_[(currRow_ - 1) * cols_ + (currCol_ + 1)] = ptr; }
    void SE(const std::shared_ptr<Grid> &ptr) { grids_[(currRow_ + 1) * cols_ + (currCol_ + 1)] = ptr; }
    void SW(const std::shared_ptr<Grid> &ptr) { grids_[(currRow_ + 1) * cols_ + (currCol_ - 1)] = ptr; }
    void NW(const std::shared_ptr<Grid> &ptr) { grids_[(currRow_ - 1) * cols_ + (currCol_ - 1)] = ptr; }

private:
    int rows_;
    int cols_;
    int currRow_;
    int currCol_;
    std::vector<std::shared_ptr<Grid>> grids_;
};

class MovingGrid {
public:
    MovingGrid(Point initialPoint, const MovingGridConfig &config) : currentPoint_(std::move(initialPoint)), config_(config),
                                                                     metaGrid_(config_.numSubgridsPerDirection) {
        if (config_.numSteps % 2 == 0)
            config_.numSteps--;
        halfGridSize_ = config_.stepSize * config_.numSteps / 2;

        const auto xCenter = roundToMultiple(currentPoint_(0), config_.stepSize / 2);
        const auto xLbLeft = xCenter - 3 * halfGridSize_;
        const auto xLbCenter = xCenter - halfGridSize_;
        const auto xUbCenter = xCenter + halfGridSize_;
        const auto xUbRight = xCenter + 3 * halfGridSize_;
        const auto yCenter = roundToMultiple(currentPoint_(1), config_.stepSize / 2);
        const auto yLbBottom = yCenter - 3 * halfGridSize_;
        const auto yLbCenter = yCenter - halfGridSize_;
        const auto yUbCenter = yCenter + halfGridSize_;
        const auto yUbTop = yCenter + 3 * halfGridSize_;

        currentCenter_ << xCenter, yCenter;
        currentGrid_ = Grid(std::initializer_list<Interval>{Interval(xLbLeft, config_.stepSize,xUbRight), Interval(yLbBottom, config_.stepSize, yUbTop)});
        currentCovariance_ = MatrixX::Identity(3 * config_.numSteps, 3 * config_.numSteps);

        // Build the active subgrids
        metaGrid_.C(std::initializer_list<Interval>{Interval(xLbCenter, config_.stepSize,xUbCenter), Interval(yLbCenter, config_.stepSize, yUbCenter)});
        metaGrid_.N(std::initializer_list<Interval>{Interval(xLbCenter, config_.stepSize,xUbCenter), Interval(yUbCenter, config_.stepSize, yUbTop)});
        metaGrid_.NE(std::initializer_list<Interval>{Interval(xUbCenter, config_.stepSize,xUbRight), Interval(yUbCenter, config_.stepSize, yUbTop)});
        metaGrid_.E(std::initializer_list<Interval>{Interval(xUbCenter, config_.stepSize,xUbRight), Interval(yLbCenter, config_.stepSize, yUbCenter)});
        metaGrid_.SE(std::initializer_list<Interval>{Interval(xUbCenter, config_.stepSize,xUbRight), Interval(yLbBottom, config_.stepSize, yLbCenter)});
        metaGrid_.S(std::initializer_list<Interval>{Interval(xLbCenter, config_.stepSize,xUbCenter), Interval(yLbBottom, config_.stepSize, yLbCenter)});
        metaGrid_.SW(std::initializer_list<Interval>{Interval(xLbLeft, config_.stepSize,xLbCenter), Interval(yLbBottom, config_.stepSize, yLbCenter)});
        metaGrid_.W(std::initializer_list<Interval>{Interval(xLbLeft, config_.stepSize,xLbCenter), Interval(yLbCenter, config_.stepSize, yUbCenter)});
        metaGrid_.NW(std::initializer_list<Interval>{Interval(xLbLeft, config_.stepSize,xLbCenter), Interval(yUbCenter, config_.stepSize, yUbTop)});

        // Link the covariances
        auto covC = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covN = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covNE = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covE = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covSE = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covS = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covSW = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covW = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covNW = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));

        // grid C: Location wrt current C -> Location wrt grid C

        // C: NW -> NW, N -> N, NE -> NE, W -> W, C -> C, E -> E, SW -> SW, S -> S, SE -> SE
        auto &ptrC = metaGrid_.C();
        ptrC->CovC = covC;
        ptrC->CovN = covN;
        ptrC->CovNE = covNE;
        ptrC->CovE = covE;
        ptrC->CovSE = covSE;
        ptrC->CovS = covS;
        ptrC->CovSW = covSW;
        ptrC->CovW = covW;
        ptrC->CovNW = covNW;

        // N: NW -> W, N -> C, NE -> E, W -> SW, C -> S, E -> SE
        auto &ptrN = metaGrid_.N();
        ptrN->CovW = covNW;
        ptrN->CovC = covN;
        ptrN->CovE = covNE;
        ptrN->CovSW = covW;
        ptrN->CovS = covC;
        ptrN->CovSE = covE;

        // NW: NW -> C, N -> E, W -> S, C -> SE
        auto &ptrNW = metaGrid_.NW();
        ptrNW->CovC = covNW;
        ptrNW->CovE = covN;
        ptrNW->CovS = covW;
        ptrNW->CovSE = covC;

        // NE: N -> W, NE -> C, C -> SW, E -> S
        auto &ptrNE = metaGrid_.NE();
        ptrNE->CovW = covN;
        ptrNE->CovC = covNE;
        ptrNE->CovSW = covC;
        ptrNE->CovS = covE;

        // W: NW -> N, N -> NE, W -> C, C -> E, SW -> S, S -> SE
        auto &ptrW = metaGrid_.W();
        ptrW->CovN = covNW;
        ptrW->CovNE = covN;
        ptrW->CovE = covC;
        ptrW->CovSE = covS;
        ptrW->CovS = covSW;
        ptrW->CovC = covW;

        // E: N -> NW, NW -> N, C -> W, E -> C, S -> SW, SE -> S
        auto &ptrE = metaGrid_.E();
        ptrE->CovN = covNE;
        ptrE->CovNW = covN;
        ptrE->CovW = covC;
        ptrE->CovSW = covS;
        ptrE->CovS = covSE;
        ptrE->CovC = covE;

        // SW: W -> N, C -> NE, SW -> C, S -> E
        auto &ptrSW = metaGrid_.SW();
        ptrSW->CovN = covW;
        ptrSW->CovNE = covC;
        ptrSW->CovE = covS;
        ptrSW->CovC = covSW;

        // S: W -> NW, C -> N, E -> NE, SW -> W, S -> C, SE -> E
        auto &ptrS = metaGrid_.S();
        ptrS->CovNW = covW;
        ptrS->CovN = covC;
        ptrS->CovNE = covE;
        ptrS->CovW = covSW;
        ptrS->CovC = covS;
        ptrS->CovE = covSE;

        // SE: C -> NW, E -> N, S -> W, SE -> C
        auto &ptrSE = metaGrid_.SE();
        ptrSE->CovN = covE;
        ptrSE->CovNW = covC;
        ptrSE->CovW = covS;
        ptrSE->CovC = covSE;
    }

    bool operator()(const Point &point) {
        Move moveRequest = getRequiredMove(point);
        if (moveRequest == Move::None)
            return false;


        //TODO: Update muX, sigmaXX for current grid before moving
        updateGrids();

        while (moveRequest != Move::None) {

            switch(moveRequest) {

                case Move::Up:
                    moveUp();
                    break;
                case Move::Right:
                    moveRight();
                    break;
                case Move::Down:
                    moveDown();
                    break;
                case Move::Left:
                    moveLeft();
                    break;
                case Move::None:
                    break;
            }

            moveRequest = getRequiredMove(point);
        }

        updateCovariance();

        return true;
    }

    Move getRequiredMove(const Point &point) {
        const auto dx = point(0) - currentCenter_(0);
        const auto dy = point(1) - currentCenter_(1);
        if (dy > halfGridSize_)
            return Move::Up;
        else if (dx > halfGridSize_)
            return Move::Right;
        else if (-dy > halfGridSize_)
            return Move::Down;
        else if (-dx > halfGridSize_)
            return Move::Left;
        else
            return Move::None;
    }

    void print() {
    }

    MatrixX getGrid() {
        return currentGrid_.getPermutations();
    }

    MatrixX getCovariance() {
        return currentCovariance_;
    }

    void attachGaussianProcess(const std::shared_ptr<GaussianProcess> &gaussianProcess) {
        GP = gaussianProcess;
    }
    
private:
    void updateGrids() {
        auto centerGrid = metaGrid_.C();

        // centerGrid->CovX exists for all X, after-move step took care of that
        auto &matPtrC = *centerGrid->CovC;
        matPtrC = std::make_unique<MatrixX>(currentCovariance_.block(config_.numSteps,config_.numSteps,config_.numSteps,config_.numSteps));
        auto &matPtrN = *centerGrid->CovN;
        matPtrN = std::make_unique<MatrixX>(currentCovariance_.block(0,config_.numSteps,config_.numSteps,config_.numSteps));
        auto &matPtrNE = *centerGrid->CovNE;
        matPtrNE = std::make_unique<MatrixX>(currentCovariance_.block(0,2*config_.numSteps,config_.numSteps,config_.numSteps));
        auto &matPtrE = *centerGrid->CovE;
        matPtrE = std::make_unique<MatrixX>(currentCovariance_.block(config_.numSteps,2*config_.numSteps,config_.numSteps,config_.numSteps));
        auto &matPtrSE = *centerGrid->CovSE;
        matPtrSE = std::make_unique<MatrixX>(currentCovariance_.block(2*config_.numSteps,2*config_.numSteps,config_.numSteps,config_.numSteps));
        auto &matPtrS = *centerGrid->CovS;
        matPtrS = std::make_unique<MatrixX>(currentCovariance_.block(2*config_.numSteps,config_.numSteps,config_.numSteps,config_.numSteps));
        auto &matPtrSW = *centerGrid->CovSW;
        matPtrSW = std::make_unique<MatrixX>(currentCovariance_.block(2*config_.numSteps,0,config_.numSteps,config_.numSteps));
        auto &matPtrW = *centerGrid->CovW;
        matPtrW = std::make_unique<MatrixX>(currentCovariance_.block(config_.numSteps,0,config_.numSteps,config_.numSteps));
        auto &matPtrNW = *centerGrid->CovNW;
        matPtrNW = std::make_unique<MatrixX>(currentCovariance_.block(0,0,config_.numSteps,config_.numSteps));
    }

    void updateCovariance() {
        auto centerGrid = metaGrid_.C();

        currentCovariance_.block(config_.numSteps,config_.numSteps,config_.numSteps,config_.numSteps) = **centerGrid->CovC;
        currentCovariance_.block(0,config_.numSteps,config_.numSteps,config_.numSteps) = **centerGrid->CovN;
        currentCovariance_.block(0,2*config_.numSteps,config_.numSteps,config_.numSteps) = **centerGrid->CovNE;
        currentCovariance_.block(config_.numSteps,2*config_.numSteps,config_.numSteps,config_.numSteps) = **centerGrid->CovE;
        currentCovariance_.block(2*config_.numSteps,2*config_.numSteps,config_.numSteps,config_.numSteps) = **centerGrid->CovSE;
        currentCovariance_.block(2*config_.numSteps,config_.numSteps,config_.numSteps,config_.numSteps) = **centerGrid->CovS;
        currentCovariance_.block(2*config_.numSteps,0,config_.numSteps,config_.numSteps) = **centerGrid->CovSW;
        currentCovariance_.block(config_.numSteps,0,config_.numSteps,config_.numSteps) = **centerGrid->CovW;
        currentCovariance_.block(0,0,config_.numSteps,config_.numSteps) = **centerGrid->CovNW;
    }

    void moveUp() {
        std::cout << "Moving up..." << std::endl;

        currentCenter_(1) += 2 * halfGridSize_;
        const auto xCenter = currentCenter_(0);
        const auto xLbLeft = xCenter - 3 * halfGridSize_;
        const auto xLbCenter = xCenter - halfGridSize_;
        const auto xUbCenter = xCenter + halfGridSize_;
        const auto xUbRight = xCenter + 3 * halfGridSize_;
        const auto yCenter = currentCenter_(1);
        //const auto yLbBottom = yCenter - 3 * halfGridSize_;
        //const auto yLbCenter = yCenter - halfGridSize_;
        const auto yUbCenter = yCenter + halfGridSize_;
        const auto yUbTop = yCenter + 3 * halfGridSize_;

        metaGrid_.moveN();

        auto ptrW = metaGrid_.W();
        auto ptrC = metaGrid_.C();
        auto ptrE = metaGrid_.E();
        auto covW = ptrW->CovC;
        auto covC = ptrC->CovC;
        auto covE = ptrE->CovC;
        auto covNW = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covN = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covNE = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));

        std::shared_ptr<Grid> updatePtr;

        updatePtr = metaGrid_.NW();
        if (!updatePtr)
            metaGrid_.NW(std::initializer_list<Interval>{Interval(xLbLeft, config_.stepSize, xLbCenter),
                                                         Interval(yUbCenter, config_.stepSize, yUbTop)});
        else
            covNW = updatePtr->CovC;

        updatePtr = metaGrid_.N();
        if (!updatePtr)
            metaGrid_.N(std::initializer_list<Interval>{Interval(xLbCenter, config_.stepSize, xUbCenter),
                                                        Interval(yUbCenter, config_.stepSize, yUbTop)});
        else
            covN = updatePtr->CovC;

        updatePtr = metaGrid_.NE();
        if (!updatePtr)
            metaGrid_.NE(std::initializer_list<Interval>{Interval(xUbCenter, config_.stepSize, xUbRight),
                                                         Interval(yUbCenter, config_.stepSize, yUbTop)});
        else
            covNE = updatePtr->CovC;

        // Link middle row with upper row
        ptrW->CovN = covNW;
        ptrW->CovNE = covN;
        ptrC->CovNW = covNW;
        ptrC->CovN = covN;
        ptrC->CovNE = covNE;
        ptrE->CovNW = covN;
        ptrE->CovN = covNE;

        // N: NW -> W, N -> C, NE -> E, W -> SW, C -> S, E -> SE
        auto ptrN = metaGrid_.N();
        ptrN->CovW = covNW;
        ptrN->CovC = covN;
        ptrN->CovE = covNE;
        ptrN->CovSW = covW;
        ptrN->CovS = covC;
        ptrN->CovSE = covE;

        // NW: NW -> C, N -> E, W -> S, C -> SE
        auto ptrNW = metaGrid_.NW();
        ptrNW->CovC = covNW;
        ptrNW->CovE = covN;
        ptrNW->CovS = covW;
        ptrNW->CovSE = covC;

        // NE: N -> W, NE -> C, C -> SW, E -> S
        auto ptrNE = metaGrid_.NE();
        ptrNE->CovW = covN;
        ptrNE->CovC = covNE;
        ptrNE->CovSW = covC;
        ptrNE->CovS = covE;
    }

    void moveRight() {
        std::cout << "Moving right..." << std::endl;

        currentCenter_(0) += 2 * halfGridSize_;
        const auto xCenter = currentCenter_(0);
        //const auto xLbLeft = xCenter - 3 * halfGridSize_;
        //const auto xLbCenter = xCenter - halfGridSize_;
        const auto xUbCenter = xCenter + halfGridSize_;
        const auto xUbRight = xCenter + 3 * halfGridSize_;
        const auto yCenter = currentCenter_(1);
        const auto yLbBottom = yCenter - 3 * halfGridSize_;
        const auto yLbCenter = yCenter - halfGridSize_;
        const auto yUbCenter = yCenter + halfGridSize_;
        const auto yUbTop = yCenter + 3 * halfGridSize_;

        metaGrid_.moveE();

        auto ptrN = metaGrid_.N();
        auto ptrC = metaGrid_.C();
        auto ptrS = metaGrid_.S();
        auto covN = ptrN->CovC;
        auto covC = ptrC->CovC;
        auto covS = ptrS->CovC;
        auto covNE = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covE = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covSE = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));

        std::shared_ptr<Grid> updatePtr;

        updatePtr = metaGrid_.NE();
        if (!updatePtr)
            metaGrid_.NE(std::initializer_list<Interval>{Interval(xUbCenter, config_.stepSize,xUbRight), Interval(yUbCenter, config_.stepSize, yUbTop)});
        else
            covNE = updatePtr->CovC;

        updatePtr = metaGrid_.E();
        if (!updatePtr)
            metaGrid_.E(std::initializer_list<Interval>{Interval(xUbCenter, config_.stepSize,xUbRight), Interval(yLbCenter, config_.stepSize, yUbCenter)});
        else
            covE = updatePtr->CovC;

        updatePtr = metaGrid_.SE();
        if (!updatePtr)
            metaGrid_.SE(std::initializer_list<Interval>{Interval(xUbCenter, config_.stepSize,xUbRight), Interval(yLbBottom, config_.stepSize, yLbCenter)});
        else
            covSE = updatePtr->CovC;

        // Link middle column with right column
        ptrN->CovE = covNE;
        ptrN->CovSE = covE;
        ptrC->CovNE = covNE;
        ptrC->CovE = covE;
        ptrC->CovSE = covSE;
        ptrS->CovNE = covE;
        ptrS->CovE = covSE;

        // NE: N -> W, NE -> C, C -> SW, E -> S
        auto ptrNE = metaGrid_.NE();
        ptrNE->CovW = covN;
        ptrNE->CovC = covNE;
        ptrNE->CovSW = covC;
        ptrNE->CovS = covE;

        // E: N -> NW, NW -> N, C -> W, E -> C, S -> SW, SE -> S
        auto ptrE = metaGrid_.E();
        ptrE->CovN = covNE;
        ptrE->CovNW = covN;
        ptrE->CovW = covC;
        ptrE->CovSW = covS;
        ptrE->CovS = covSE;
        ptrE->CovC = covE;

        // SE: C -> NW, E -> N, S -> W, SE -> C
        auto ptrSE = metaGrid_.SE();
        ptrSE->CovN = covE;
        ptrSE->CovNW = covC;
        ptrSE->CovW = covS;
        ptrSE->CovC = covSE;

    }

    void moveDown() {
        std::cout << "Moving down..." << std::endl;

        currentCenter_(1) -= 2 * halfGridSize_;
        const auto xCenter = currentCenter_(0);
        const auto xLbLeft = xCenter - 3 * halfGridSize_;
        const auto xLbCenter = xCenter - halfGridSize_;
        const auto xUbCenter = xCenter + halfGridSize_;
        const auto xUbRight = xCenter + 3 * halfGridSize_;
        const auto yCenter = currentCenter_(1);
        const auto yLbBottom = yCenter - 3 * halfGridSize_;
        const auto yLbCenter = yCenter - halfGridSize_;
        //const auto yUbCenter = yCenter + halfGridSize_;
        //const auto yUbTop = yCenter + 3 * halfGridSize_;

        metaGrid_.moveS();

        auto ptrW = metaGrid_.W();
        auto ptrC = metaGrid_.C();
        auto ptrE = metaGrid_.E();
        auto covW = ptrW->CovC;
        auto covC = ptrC->CovC;
        auto covE = ptrE->CovC;
        auto covSW = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covS = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covSE = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));

        std::shared_ptr<Grid> updatePtr;

        updatePtr = metaGrid_.SW();
        if (!updatePtr)
            metaGrid_.SW(std::initializer_list<Interval>{Interval(xLbLeft, config_.stepSize,xLbCenter), Interval(yLbBottom, config_.stepSize, yLbCenter)});
        else
            covSW = updatePtr->CovC;

        updatePtr = metaGrid_.S();
        if (!updatePtr)
            metaGrid_.S(std::initializer_list<Interval>{Interval(xLbCenter, config_.stepSize,xUbCenter), Interval(yLbBottom, config_.stepSize, yLbCenter)});
        else
            covS = updatePtr->CovC;

        updatePtr = metaGrid_.SE();
        if (!updatePtr)
            metaGrid_.SE(std::initializer_list<Interval>{Interval(xUbCenter, config_.stepSize,xUbRight), Interval(yLbBottom, config_.stepSize, yLbCenter)});
        else
            covSE = updatePtr->CovC;

        // Link middle row with lower row
        ptrW->CovS = covSW;
        ptrW->CovSE = covS;
        ptrC->CovSW = covSW;
        ptrC->CovS = covS;
        ptrC->CovSE = covSE;
        ptrE->CovSW = covS;
        ptrE->CovS = covSE;

        // SW: W -> N, C -> NE, SW -> C, S -> E
        auto ptrSW = metaGrid_.SW();
        ptrSW->CovN = covW;
        ptrSW->CovNE = covC;
        ptrSW->CovE = covS;
        ptrSW->CovC = covSW;

        // S: W -> NW, C -> N, E -> NE, SW -> W, S -> C, SE -> E
        auto ptrS = metaGrid_.S();
        ptrS->CovNW = covW;
        ptrS->CovN = covC;
        ptrS->CovNE = covE;
        ptrS->CovW = covSW;
        ptrS->CovC = covS;
        ptrS->CovE = covSE;

        // SE: C -> NW, E -> N, S -> W, SE -> C
        auto ptrSE = metaGrid_.SE();
        ptrSE->CovN = covE;
        ptrSE->CovNW = covC;
        ptrSE->CovW = covS;
        ptrSE->CovC = covSE;
    }

    void moveLeft() {
        std::cout << "Moving left..." << std::endl;

        currentCenter_(0) -= 2 * halfGridSize_;
        const auto xCenter = currentCenter_(0);
        const auto xLbLeft = xCenter - 3 * halfGridSize_;
        const auto xLbCenter = xCenter - halfGridSize_;
        //const auto xUbCenter = xCenter + halfGridSize_;
        //const auto xUbRight = xCenter + 3 * halfGridSize_;
        const auto yCenter = currentCenter_(1);
        const auto yLbBottom = yCenter - 3 * halfGridSize_;
        const auto yLbCenter = yCenter - halfGridSize_;
        const auto yUbCenter = yCenter + halfGridSize_;
        const auto yUbTop = yCenter + 3 * halfGridSize_;

        metaGrid_.moveW();

        auto ptrN = metaGrid_.N();
        auto ptrC = metaGrid_.C();
        auto ptrS = metaGrid_.S();
        auto covN = ptrN->CovC;
        auto covC = ptrC->CovC;
        auto covS = ptrS->CovC;
        auto covNW = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covW = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));
        auto covSW = std::make_shared<std::unique_ptr<MatrixX>>(std::make_unique<MatrixX>(MatrixX::Zero(config_.numSteps, config_.numSteps)));

        std::shared_ptr<Grid> updatePtr;

        updatePtr = metaGrid_.NW();
        if (!updatePtr)
            metaGrid_.NW(std::initializer_list<Interval>{Interval(xLbLeft, config_.stepSize,xLbCenter), Interval(yUbCenter, config_.stepSize, yUbTop)});
        else
            covNW = updatePtr->CovC;

        updatePtr = metaGrid_.W();
        if (!updatePtr)
            metaGrid_.W(std::initializer_list<Interval>{Interval(xLbLeft, config_.stepSize,xLbCenter), Interval(yLbCenter, config_.stepSize, yUbCenter)});
        else
            covW = updatePtr->CovC;

        updatePtr = metaGrid_.SW();
        if (!updatePtr)
            metaGrid_.SW(std::initializer_list<Interval>{Interval(xLbLeft, config_.stepSize,xLbCenter), Interval(yLbBottom, config_.stepSize, yLbCenter)});
        else
            covSW = updatePtr->CovC;

        // Link middle column with left column
        ptrN->CovW = covNW;
        ptrN->CovSW = covW;
        ptrC->CovNW = covNW;
        ptrC->CovW = covW;
        ptrC->CovSW = covSW;
        ptrS->CovNW = covW;
        ptrS->CovW = covSW;

        // NW: NW -> C, N -> E, W -> S, C -> SE
        auto ptrNW = metaGrid_.NW();
        ptrNW->CovC = covNW;
        ptrNW->CovE = covN;
        ptrNW->CovS = covW;
        ptrNW->CovSE = covC;

        // W: NW -> N, N -> NE, W -> C, C -> E, SW -> S, S -> SE
        auto ptrW = metaGrid_.W();
        ptrW->CovN = covNW;
        ptrW->CovNE = covN;
        ptrW->CovE = covC;
        ptrW->CovSE = covS;
        ptrW->CovS = covSW;
        ptrW->CovC = covW;

        // SW: W -> N, C -> NE, SW -> C, S -> E
        auto ptrSW = metaGrid_.SW();
        ptrSW->CovN = covW;
        ptrSW->CovNE = covC;
        ptrSW->CovE = covS;
        ptrSW->CovC = covSW;
    }

    int halfGridSize_;
    Point currentPoint_;
    Point currentCenter_;
    Grid currentGrid_;
    MatrixX currentCovariance_;
    VectorX currentMean_;
    MovingGridConfig config_;
    std::shared_ptr<GaussianProcess> GP;
    MetaGrid metaGrid_;
};

#endif //SOUNDSPEED_ESTIMATOR_DYNAMIC_H
