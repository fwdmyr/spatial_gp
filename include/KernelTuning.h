//
// Created by felix on 07.09.22.
//

#ifndef SOUNDSPEED_ESTIMATOR_KERNELTUNING_H
#define SOUNDSPEED_ESTIMATOR_KERNELTUNING_H

#include <Typedefs.h>
#include <ceres/ceres.h>

class FirstOrderGaussianKernel final : public ceres::FirstOrderFunction {
public:
    ~FirstOrderGaussianKernel() override = default;

    bool Evaluate(const double *parameters, double *cost, double *gradient) const override {
        if (!isInitialized_)
            throw std::runtime_error("FirstOrderGaussianKernel not initialized!");

        const double alpha = parameters[0];
        const double scale = parameters[1];

        GaussianKernel kernel(alpha, scale, gpDim_);
        MatrixX kXX(gpDim_ * gridSize_, gpDim_ * gridSize_);
        MatrixX H1(gpDim_ * gridSize_, gpDim_ * gridSize_);
        MatrixX H2(gpDim_ * gridSize_, gpDim_ * gridSize_);

        for (int i = 0; i < gridSize_; ++i) {
            for (int j = 0; j < gridSize_; ++j) {
                MatrixX H1Block(gpDim_, gpDim_);
                MatrixX H2Block(gpDim_, gpDim_);
                // kXXinv
                kXX.block(i * gpDim_, j * gpDim_, gpDim_, gpDim_) = kernel(X_.col(i), X_.col(j), H1Block, H2Block);
                H1.block(i * gpDim_, j * gpDim_, gpDim_, gpDim_) = H1Block;
                H2.block(i * gpDim_, j * gpDim_, gpDim_, gpDim_) = H2Block;
            }
        }

        MatrixX kInvXX = kXX.llt().solve(MatrixX::Identity(kXX.rows(), kXX.cols()));

        cost[0] = 0.5 * std::log(kXX.determinant()) + 0.5 * y_.transpose() * kInvXX * y_ + 0.5 * y_.size() * std::log(2 * M_PI);
        if (gradient) {
            gradient[0] = 0.5 * (kInvXX * H1).trace() - 0.5 * y_.transpose() * kInvXX * H1 * kInvXX * y_;
            gradient[1] = 0.5 * (kInvXX * H2).trace() - 0.5 * y_.transpose() * kInvXX * H2 * kInvXX * y_;
        }
        return true;
    }

    [[nodiscard]] int NumParameters() const override {return 2;}

    void initialize(const MatrixX &X, const VectorX &y, long gpDim) {
        gridDim_ = X.rows();
        gridSize_ = X.cols();
        gpDim_ = gpDim;
        assert(y.size() == gridSize_);
        X_ = X;
        y_ = y;
        isInitialized_ = true;
    }

private:
    MatrixX X_;
    VectorX y_;
    long gpDim_;
    long gridDim_;
    long gridSize_;
    bool isInitialized_{false};
};

#endif //SOUNDSPEED_ESTIMATOR_KERNELTUNING_H
