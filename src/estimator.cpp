//
// Created by felix on 24.08.22.
//

#include <GaussianProcess.h>
#include <KernelTuning.h>
#include <Grid.h>
#include <Data.h>

/*
 * Salinity:
 *   Range:
 *     3-68 mS/cm (1)
 *     0.1-6 mS/cm (2)
 *   Resolution:
 *     0.02 g/l (1)
 *     0.0005 g/l (2)
 *   Accuracy:
 *     +/-1 g/l (1)
 *     +/-0.1 g/l (2)
 * Temperature:
 *   Range:
 *     -1-40 °C
 *   Resolution:
 *     0.032 °C
 *   Accuracy:
 *     +/-0.1 °C
 * Depth:
 *   Range:
 *     0.1-100 m
 *     5-500 m
 *     5-1200 m
 *     10-2400 m
 *   Resolution:
 *     0.0003 * Range
 *   Accuracy:
 *     +/-0.006 * Range
 *
 */

constexpr int STEPS(double D0, double DD, double DN) {
    return static_cast<int>((DN - D0) / DD) + 1;
}

constexpr long DIM = 1;
constexpr long GRIDDIM = 2;

constexpr double X0 = 0.0;
constexpr double DX = 10.0;
constexpr double XN = 100.0;

constexpr double Y0 = 0.0;
constexpr double DY = 10.0;
constexpr double YN = 100.0;

constexpr long GRIDSIZE = STEPS(X0, DX, XN) * STEPS(Y0, DY, YN);

int main() {
    /**
    MatrixX testData = MatrixX::Ones(2, 4);
    testData.block(0, 0, 2, 1) = 2 * MatrixX::Ones(2, 1);
    testData.block(0, 2, 2, 1) = 2 * MatrixX::Ones(2, 1);
    std::cout << testData << std::endl;
    InterpGrid testGrid({Interval(0.0, 1.0, 2.0), Interval(0.0, 1.0, 2.0), Interval(0.0, 1.0, 2.0)}, testData);
    testGrid.print("testGrid");

    std::cout << testGrid(0.75, 0.75, 0.8) << std::endl;

    return 0;
    */
    /** Build the grid */
    Interval FirstDimension(X0, DX, XN);
    Interval SecondDimension(Y0, DY, YN);

    // Load the GT data
    auto temperatureData = readCSV<MatrixX>("/home/haoming/CTD/temperature_slice.txt");
    temperatureData = temperatureData.topLeftCorner(STEPS(X0, DX, XN), STEPS(Y0, DY, YN));
    auto salinityData = readCSV<MatrixX>("/home/haoming/CTD/salinity_slice.txt");
    salinityData = salinityData.topLeftCorner(STEPS(X0, DX, XN), STEPS(Y0, DY, YN));
    assert(temperatureData.size() == GRIDSIZE);
    assert(salinityData.size() == GRIDSIZE);

    //Grid<GRIDDIM, GRIDSIZE> temperatureGrid({FirstDimension, SecondDimension}, temperatureData);
    InterpGrid salinityGrid({FirstDimension, SecondDimension}, salinityData);
    InterpGrid temperatureGrid({FirstDimension, SecondDimension}, temperatureData);

    constexpr double alphaZero = 10.0;
    constexpr double scaleZero = 10.0;

    double parameters[2] = {alphaZero, scaleZero};
    auto fcnPtr = new FirstOrderGaussianKernel();
    const auto gridPoints = salinityGrid.getPermutations();
    Vector<GRIDSIZE> dataPoints;
    for (int i = 0; i < GRIDSIZE; ++i) {
        dataPoints(i) = salinityGrid(gridPoints(0, i), gridPoints(1, i));
    }
    fcnPtr->initialize(gridPoints, dataPoints, DIM);
    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = false;
    ceres::GradientProblemSolver::Summary summary;
    ceres::GradientProblem problem(fcnPtr);
    ceres::Solve(options, problem, parameters, &summary);
    std::cout << summary.FullReport() << std::endl;
    std::cout << "Initial alpha: " << alphaZero << " scale: " << scaleZero << std::endl;
    std::cout << "Final   alpha: " << parameters[0] << " scale: " << parameters[1]
              << std::endl;

    return 0;

    // Build the trajectory
    // Trajectory will be 2 sine waves starting at [1, 50] and ending at [99, 50] with amplitude 49
    auto xMin = 1.0;
    auto xMax = 99.0;
    auto yMean = 50.0;
    auto yAmplitude = 49.0;
    auto dx = 0.1;
    std::vector<Vector<GRIDDIM>> trajectory(static_cast<int>((xMax - xMin) / dx) + 1);
    std::generate(trajectory.begin(), trajectory.end(), [&, n=0]() mutable -> Vector<GRIDDIM> {
        auto xBase = xMin + dx * n++;
        auto yBase = yMean + yAmplitude * std::sin(4 * M_PI * (xBase - xMin) / (xMax - xMin));
        return {xBase, yBase};
    });

    // Configure the GP
    GaussianProcessConfig Config;
    Config.alpha = parameters[0];
    Config.scale = parameters[1];
    Config.initMean = Vector<DIM>(30.0);
    Config.initCov = Matrix<DIM, DIM>(1.0*1.0);
    Config.dim = DIM;

    GaussianProcess GP(std::move(salinityGrid.getPermutations()), std::move(Config));

    auto expectationBefore = GP.getExpectation();
    Eigen::Map<Matrix<STEPS(X0, DX, XN), STEPS(Y0, DY, YN)>> expectationBeforeGrid(expectationBefore.data());
    std::cout << "Ground Truth:" << std::endl;
    std::cout << salinityGrid.getData() << std::endl;
    std::cout << "Expectation Before:" << std::endl;
    std::cout << expectationBeforeGrid << std::endl;

    // Simulate for salinity
    for (const auto &point : trajectory) {
        Vector<DIM> obs(salinityGrid(point(1), point(0))); // matrix-style indexing
        GP.step(point, obs);
    }

    auto expectationAfter = GP.getExpectation();
    Eigen::Map<Matrix<STEPS(X0, DX, XN), STEPS(Y0, DY, YN)>> expectationAfterGrid(expectationAfter.data());
    std::cout << "Expectation After:" << std::endl;
    std::cout << expectationAfterGrid << std::endl;

    writeCSV("/home/haoming/CTD/salinity_slice_after.txt", expectationAfterGrid);
    return 0;
}