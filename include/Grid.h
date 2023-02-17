//
// Created by haoming on 25.08.22.
//

#ifndef SOUNDSPEED_ESTIMATOR_GRID_H
#define SOUNDSPEED_ESTIMATOR_GRID_H

#include <Typedefs.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <iostream>

struct Digit {
    std::vector<double>::const_iterator begin;
    std::vector<double>::const_iterator end;
    std::vector<double>::const_iterator current;
};

class Interval {

public:
    Interval(double x0, double dX, double xN) : start(x0), step(dX), end(xN) { };

    [[nodiscard]] std::vector<double> generate() const {
        int steps = static_cast<int>((end - start) / step);
        std::vector<double> bases(steps);
        std::generate(bases.begin(), bases.end(), [&, n=0] () mutable -> double {
            return start + step * n++;
        });
        return bases;
    }

private:
    double start;
    double step;
    double end;

};

class BaseGrid {

using CovPtr = std::shared_ptr<std::unique_ptr<MatrixX>>;

public:
    BaseGrid() = default;
    BaseGrid(std::initializer_list<Interval> intervalList) {
        for (const auto &interval : intervalList)
            bases_.push_back(interval.generate());
        for (const auto &base : bases_)
            dims_.push_back(base.size());
    }

    [[nodiscard]] BaseVectors getBases() const {
        return bases_;
    }

    void print(std::string &&name) const {
        std::cout << name << std::endl;
        if (CovC) {
            if (*CovC)
                std::cout << "CovC initialized" << std::endl;
            else
                std::cout << "CovC empty unique_ptr" << std::endl;
        }
        if (CovN) {
            if (*CovN)
                std::cout << "CovN initialized" << std::endl;
            else
                std::cout << "CovN empty unique_ptr" << std::endl;
        }
        if (CovNE) {
            if (*CovNE)
                std::cout << "CovNE initialized" << std::endl;
            else
                std::cout << "CovNE empty unique_ptr" << std::endl;
        }
        if (CovE) {
            if (*CovE)
                std::cout << "CovE initialized" << std::endl;
            else
                std::cout << "CovE empty unique_ptr" << std::endl;
        }
        if (CovSE) {
            if (*CovSE)
                std::cout << "CovSE initialized" << std::endl;
            else
                std::cout << "CovSE empty unique_ptr" << std::endl;
        }
        if (CovS) {
            if (*CovS)
                std::cout << "CovS initialized" << std::endl;
            else
                std::cout << "CovS empty unique_ptr" << std::endl;
        }
        if (CovSW) {
            if (*CovSW)
                std::cout << "CovSW initialized" << std::endl;
            else
                std::cout << "CovSW empty unique_ptr" << std::endl;
        }
        if (CovW) {
            if (*CovW)
                std::cout << "CovW initialized" << std::endl;
            else
                std::cout << "CovW empty unique_ptr" << std::endl;
        }
        if (CovNW) {
            if (*CovNW)
                std::cout << "CovNW initialized" << std::endl;
            else
                std::cout << "CovNW empty unique_ptr" << std::endl;
        }
    }

    CovPtr CovC;
    CovPtr CovN;
    CovPtr CovNE;
    CovPtr CovE;
    CovPtr CovSE;
    CovPtr CovS;
    CovPtr CovSW;
    CovPtr CovW;
    CovPtr CovNW;
    VectorX Mu;

protected:
    BaseVectors bases_;
    std::vector<int> dims_;
private:
    MatrixX sigmaXX;
    VectorX muX;
};

class Grid : public BaseGrid{

public:

    Grid() = default;
    Grid(std::initializer_list<Interval> intervalList) : BaseGrid(intervalList), isCached_(false), gridDim_(intervalList.size()) {
        gridSize_ = std::accumulate(bases_.begin(),bases_.end(), 1, [](long accumulator, const std::vector<double> &base) -> long {
            return accumulator * base.size();
        });
        cartesianProduct_.resize(gridDim_, gridSize_);
    }

    MatrixX& getPermutations() {
        if (!isCached_)
            computeCartesianProduct();
        return cartesianProduct_;
    }

protected:
    bool isCached_;
    unsigned long gridDim_;
    long gridSize_;
    MatrixX cartesianProduct_;

    void computeCartesianProduct() {
        // Set all iterators to the beginning
        std::vector<Digit> digits;
        for (const auto &base : bases_) {
            Digit digit = {base.begin(), base.end(), base.begin()};
            digits.push_back(digit);
        }

        int colIdx = 0;

        while (true) {
            // Construct the first row vector by pulling
            // out the element of each vector via the iterator.
            int rowIdx = 0;
            for (const auto &digit : digits) {
                cartesianProduct_(rowIdx, colIdx) = *digit.current;
                rowIdx++;
            }

            // Increment the rightmost one, and repeat.

            // When we reach the end, reset that one to the beginning and
            // increment the next-to-last one. We can get the "next-to-last"
            // iterator by pulling it out of the neighboring element in the
            // vector of iterators.
            for(auto it = digits.begin(); ; ) {
                ++(it->current);
                if(it->current == it->end) {
                    if(it+1 == digits.end()) {
                        // last digit, roll over
                        return;
                    } else {
                        // cascade
                        it->current = it->begin;
                        ++it;
                    }
                } else {
                    // normal
                    break;
                }
            }
            colIdx++;
        }
        isCached_ = true;
    }
};

class InterpGrid : public Grid{

public:

    InterpGrid(std::initializer_list<Interval> intervalList, MatrixX data) : Grid(intervalList), data_(std::move(data)) { }

    [[nodiscard]] const MatrixX& getData() const {
        return data_;
    }

    template<typename... Args>
    double operator()(Args... args) {
        // Performs matrik-like interpolation where dimensions are rows, cols
        // For 3D we might want to work with a matrix with dimensions <D1*D3, D2> and take blocks <D1, D2> that represent planes in our cube
        return interpolate(std::forward<Args>(args)...);
    }

private:
    MatrixX data_;

    double interpolate(double x) {
        // Implements linear interpolation
        auto &xBase = bases_[0];
        const auto [ix1, ix2] = getRanges(x, 0);
        return data_(ix1) + (data_(ix2) - data_(ix1)) * (x - xBase[ix1]) / (xBase[ix2] - xBase[ix1]);
    }

    double interpolate(double x, double y) {
        // Implements bilinear interpolation
        const auto [ix1, ix2] = getRanges(x, 0);
        const auto [iy1, iy2] = getRanges(y, 1);
        auto &xBase = bases_[0];
        auto &yBase = bases_[1];
        const auto x1 = xBase[ix1];
        const auto x2 = xBase[ix2];
        const auto y1 = xBase[iy1];
        const auto y2 = xBase[iy2];

        VectorX Q(gridDim_ * gridDim_);
        Q << data_(ix1, iy1),
                data_(ix1, iy2),
                data_(ix2, iy1),
                data_(ix2, iy2);

        MatrixX X(gridDim_ * gridDim_, gridDim_ * gridDim_);
        X <<  x2*y2, -x2*y1, -x1*y2, x1*y1,
                -y2,     y1,     y2,   -y1,
                -x2,     x2,     x1,   -x1,
                1.0,   -1.0,   -1.0,   1.0;

        VectorX coeffs = 1.0 / ((x2 - x1) * (y2 - y1)) * X * Q;

        return coeffs(0) + coeffs(1) * x + coeffs(2) * y + coeffs(3) * x*y;
    }

    double interpolate(double x, double y, double z) {
        // Implements trilinear interpolation
        const auto [ix1, ix2] = getRanges(x, 0);
        const auto [iy1, iy2] = getRanges(y, 1);
        const auto [iz1, iz2] = getRanges(z, 2);
        auto &xBase = bases_[0];
        auto &yBase = bases_[1];
        auto &zBase = bases_[2];
        auto &xDim = dims_[0];
        auto &yDim = dims_[1];
        auto &zDim = dims_[2];
        const auto x1 = xBase[ix1];
        const auto x2 = xBase[ix2];
        const auto y1 = xBase[iy1];
        const auto y2 = xBase[iy2];
        const auto z1 = xBase[iz1];
        const auto z2 = xBase[iz2];

        const auto xd = (x - x1) / (x2 - x1);
        const auto yd = (y - y1) / (y2 - y1);
        const auto zd = (z - z1) / (z2 - z1);

        // 3D data is stored where 2D layers are concatenated horizontally yielding a long matrix with cols >> rows
        const MatrixX lowerPlane = data_.block(0, iz1*xDim, yDim, xDim);
        const MatrixX upperPlane = data_.block(0, iz2*xDim, yDim, xDim);

        const auto c000 = lowerPlane(ix1, iy1);
        const auto c100 = lowerPlane(ix2, iy1);
        const auto c010 = lowerPlane(ix1, iy2);
        const auto c110 = lowerPlane(ix2, iy2);
        const auto c001 = upperPlane(ix1, iy1);
        const auto c101 = upperPlane(ix2, iy1);
        const auto c011 = upperPlane(ix1, iy2);
        const auto c111 = upperPlane(ix2, iy2);

        const auto c00 = c000 * (1 - xd) + c100 * xd;
        const auto c01 = c001 * (1 - xd) + c101 * xd;
        const auto c10 = c010 * (1 - xd) + c110 * xd;
        const auto c11 = c011 * (1 - xd) + c111 * xd;

        const auto c0 = c00 * (1 - yd) + c10 * yd;
        const auto c1 = c01 * (1 - yd) + c11 * yd;

        return c0 * (1 - zd) + c1 * zd;
    }

    std::pair<long, long> getRanges(double value, size_t dim) {
        auto &base = bases_[dim];
        auto itAfter = std::upper_bound(base.begin(), base.end(), value);
        if (itAfter == base.end())
            itAfter--;
        const auto idxAfter = std::distance(base.begin(), itAfter);
        return std::make_pair(idxAfter - 1, idxAfter);
    }
};

#endif //SOUNDSPEED_ESTIMATOR_GRID_H
