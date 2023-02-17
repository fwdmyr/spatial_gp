//
// Created by haoming on 25.08.22.
//

#ifndef SOUNDSPEED_ESTIMATOR_DATA_H
#define SOUNDSPEED_ESTIMATOR_DATA_H

#include <Typedefs.h>

#include <fstream>

template<typename M>
M readCSV (const std::string &path) {
    std::ifstream ifs;
    ifs.open(path);
    std::string line;
    std::vector<double> values;
    size_t rows = 0;
    while (std::getline(ifs, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

void writeCSV(const std::string &path, const MatrixX &data) {
    std::ofstream ofs;
    ofs.open(path);
    for (int i = 0; i < data.rows(); ++i) {
        std::stringstream ss;
        for (int j = 0; j < data.cols(); ++j) {
            ss << data(i, j) << ',';
        }
        auto line = ss.str();
        line.pop_back();
        line += '\n';
        ofs << line;
    }
}

#endif //SOUNDSPEED_ESTIMATOR_DATA_H
