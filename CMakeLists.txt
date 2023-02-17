cmake_minimum_required(VERSION 3.13)
project(soundspeed_estimator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options( -Wall -Wextra -Wpedantic -std=c++17)
endif()

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 3.3.5 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

include_directories(
    ${Boost_INCLUDE_DIR}
    include
)

#add_library(${PROJECT_NAME} 
#        )

#target_link_libraries(${PROJECT_NAME}
#        Eigen3::Eigen
#)

add_executable(estimator src/estimator.cpp)
add_executable(moving src/moving.cpp)

target_link_libraries(estimator
        #${PROJECT_NAME}
        Eigen3::Eigen
        Ceres::ceres
        )

target_link_libraries(moving
        #${PROJECT_NAME}
        Eigen3::Eigen
        Ceres::ceres
        )

target_include_directories(estimator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
        )

target_include_directories(moving PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

target_compile_features(estimator PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
        
target_compile_features(moving PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

