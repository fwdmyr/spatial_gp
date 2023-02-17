//
// Created by felix on 25.08.22.
//

#ifndef SOUNDSPEED_ESTIMATOR_TYPEDEFS_H
#define SOUNDSPEED_ESTIMATOR_TYPEDEFS_H

#include <Eigen/Dense>

template<int Rows, int Cols>
using Matrix = Eigen::Matrix<double, Rows, Cols>;
using MatrixX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
template<int Rows>
using DiagonalMatrix = Eigen::DiagonalMatrix<double, Rows>;
using DiagonalMatrixX = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
template<int Rows>
using Vector = Eigen::Matrix<double, Rows, 1>;
using VectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Point = Eigen::Matrix<double, 2, 1>;
using BaseVectors = std::vector<std::vector<double>>;

#endif //SOUNDSPEED_ESTIMATOR_TYPEDEFS_H
