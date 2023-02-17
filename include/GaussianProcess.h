//
// Created by felix on 24.08.22.
//

#ifndef CTD_GAUSSIANPROCESS_H
#define CTD_GAUSSIANPROCESS_H

#endif //CTD_GAUSSIANPROCESS_H

#include <iostream>
#include <utility>
#include <vector>
#include <boost/optional.hpp>
#include <Typedefs.h>

struct GaussianProcessConfig {
    double alpha = 10.0;
    double scale = 10.0;
    VectorX initMean = (VectorX(1) << 0.0).finished();
    MatrixX initCov = (MatrixX(1, 1) << 1.0).finished();
    long dim = 1;
};

class Kernel {

public:
    virtual MatrixX operator()(const VectorX &x,
            const VectorX &xPrime,
            boost::optional<MatrixX&> H1 = boost::none,
            boost::optional<MatrixX&> H2 = boost::none) = 0;

private:
};

class GaussianKernel : public Kernel {

public:
    GaussianKernel(double alpha, double scale, long dim) : alpha_(alpha), alphaSquared_(alpha * alpha), scale_(scale), scaleSquared_(scale * scale), dim_(dim) { };
    GaussianKernel() : alpha_(0.0), alphaSquared_(0.0), scale_(0.0), scaleSquared_(0.0), dim_(0) { };
    ~GaussianKernel() = default;

    void setAlpha(double alpha) {
        alpha_ = alpha;
        alphaSquared_ = alpha * alpha;
    }

    void setScale(double scale) {
        scale_ = scale;
        scaleSquared_ = scale * scale;
    }

    void setDim(long dim) {
        dim_ = dim;
    }

    MatrixX operator()(const VectorX &x,
            const VectorX &xPrime,
            boost::optional<MatrixX&> H1 = boost::none,
            boost::optional<MatrixX&> H2 = boost::none) override {
        const auto expMat = std::exp(-0.5 / scaleSquared_ * (x - xPrime).squaredNorm()) * MatrixX::Identity(dim_, dim_);
        if (H1)
            *H1 = 2 * alpha_ * expMat; // dk/dalpha
        if (H2)
            *H2 = (alphaSquared_ / (scale_ * scaleSquared_)) * (x - xPrime).squaredNorm() * expMat; // dk/dscale
        return alphaSquared_ * expMat;
    }

private:
    double alpha_;
    double alphaSquared_;
    double scale_;
    double scaleSquared_;
    long dim_;
};

class GaussianProcess {

public:
    explicit GaussianProcess(const GaussianProcessConfig &config) : GaussianProcess(MatrixX(), config) { };
    GaussianProcess(MatrixX &&grid, GaussianProcessConfig config) : gridX(std::move(grid)), config_(std::move(config)), gpDim_(config_.dim), gridDim_(gridX.rows()), gridSize_(gridX.cols()) {
        kernel_.setAlpha(config_.alpha);
        kernel_.setScale(config_.scale);
        kernel_.setDim(config_.dim);
        initialize();
    };
    GaussianProcess(const MatrixX &grid, GaussianProcessConfig config) : gridX(grid), config_(std::move(config)), gpDim_(config_.dim), gridDim_(gridX.rows()), gridSize_(gridX.cols()) {
        kernel_.setAlpha(config_.alpha);
        kernel_.setScale(config_.scale);
        kernel_.setDim(config_.dim);
        initialize();
    };
    ~GaussianProcess() = default;

    void step(const VectorX &x, const VectorX &theta) {
        // Inference
        // DxDS = DxDS * DSxDS
        MatrixX J = k(x, gridX) * kInvXX;
        // Dx1 = Dx1 + DxDS * (DSx1 - DSx1)
        VectorX muPriorX = m(x) + J * (muX - mX);
        // DxD = DxD - DxDS * DSxD + DxDS * DSxDS * DSxD
        MatrixX sigmaPriorXX = k(x, x) - J * k(gridX, x) + J * sigmaXX * J.transpose();

        // Update
        // DSxD = DSxDS * DSxD * inv(DxD + DxD)
        MatrixX G = sigmaXX * J.transpose() * (sigmaPriorXX + config_.initCov).inverse();
        // DSx1 = DSx1 + DSxD * (Dx1 - Dx1)
        muX = muX + G * (theta - muPriorX);
        // DSxDS = DSxDS - DSxD * DxDS * DSxDS
        sigmaXX = sigmaXX - G * J * sigmaXX;
    }

    void step(VectorX&& x, VectorX&& theta) {
        // Inference
        // DxDS = DxDS * DSxDS
        MatrixX J = k(x, gridX) * kInvXX;
        // Dx1 = Dx1 + DxDS * (DSx1 - DSx1)
        VectorX muPriorX = m(x) + J * (muX - mX);
        // DxD = DxD - DxDS * DSxD + DxDS * DSxDS * DSxD
        MatrixX sigmaPriorXX = k(x, x) - J * k(gridX, x) + J * sigmaXX * J.transpose();

        // Update
        // DSxD = DSxDS * DSxD * inv(DxD + DxD)
        MatrixX G = sigmaXX * J.transpose() * (sigmaPriorXX + config_.initCov).inverse();
        // DSx1 = DSx1 + DSxD * (Dx1 - Dx1)
        muX += G * (theta - muPriorX);
        // DSxDS = DSxDS - DSxD * DxDS * DSxDS
        sigmaXX -= G * J * sigmaXX;
    }

    [[nodiscard]] MatrixX getExpectation() const {
        MatrixX expMap(gpDim_, gridSize_);
        for (int i = 0; i < gridSize_; ++i)
            expMap.col(i) = muX.block(i * gpDim_, 0, gpDim_, 1);
        return expMap;
    }

    [[nodiscard]] VectorX getMean() const {
        return muX;
    }

    [[nodiscard]] MatrixX getCovariance() const {
        return sigmaXX;
    }

    [[nodiscard]] MatrixX getKernel() const {
        return kXX;
    }

    void setMean(const VectorX &mean) {
        muX = mean;
    }

    void setCovariance(const MatrixX &covariance) {
        sigmaXX = covariance;
    }

private:

    GaussianKernel kernel_; // D -> D
    MatrixX gridX; // [[11, 12, ..., 1D], ..., [M1, M2, ..., MD]]'
    VectorX mX; // [11, 12, ..., 1D, ..., M1, M2, ..., MD]'
    MatrixX kXX; // MD x MD matrix with DxD blocks for k(X, X')
    MatrixX kInvXX; // MD x MD matrix with DxD blocks for k(X, X')
    VectorX muX; // [11, 12, ..., 1D, ..., M1, M2, ..., MD]'
    MatrixX sigmaXX; // MD x MD matrix with DxD blocks for k(X, X')
    GaussianProcessConfig config_;
    long gpDim_;
    long gridDim_;
    long gridSize_;

    void initialize() {
        // mX
        mX = VectorX(gpDim_ * gridSize_);
        for (int i = 0; i < gridSize_; ++i) {
            mX.block(i * gpDim_, 0, gpDim_, 1) = config_.initMean;
        }
        // muX
        muX = mX;

        kXX = MatrixX(gpDim_ * gridSize_, gpDim_ * gridSize_);

        // gotta go fast
        for (int i = 0; i < gridSize_; ++i) {
            for (int j = 0; j < gridSize_; ++j) {
                // kXXinv
                kXX.block(i * gpDim_, j * gpDim_, gpDim_, gpDim_) = kernel_(gridX.col(i), gridX.col(j));
            }
        }

        // sigmaXX
        sigmaXX = kXX;
        kInvXX = kXX.llt().solve(MatrixX::Identity(gpDim_ * gridSize_, gpDim_ * gridSize_));

    }

    VectorX m(const VectorX &x) {
        return config_.initMean; // Dx1
    }

    MatrixX k(const VectorX &x, const VectorX &xPrime) {
        return kernel_(x, xPrime); // DxD
    }

    MatrixX k(const MatrixX &X, const VectorX &x) {
        MatrixX kXx(gpDim_ * gridSize_, gpDim_);
        for (int i = 0; i < gridSize_; ++i)
            kXx.block(i * gpDim_, 0, gpDim_, gpDim_) = kernel_(X.col(i), x);
        return kXx; // DSxD
    }

    MatrixX k(const VectorX &x, const MatrixX &X) {
        MatrixX kxX(gpDim_, gpDim_ * gridSize_);
        for (int i = 0; i < gridSize_; ++i)
            kxX.block(0, i * gpDim_, gpDim_, gpDim_) = kernel_(x, X.col(i));
        return kxX; // DxDS
    }

};