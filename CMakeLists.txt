cmake_minimum_required(VERSION 3.5)

project(SetOfMandelbrot VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(CUDA REQUIRED)

include_directories(${CUDA_TOOLKIT_INCLUDE})
set(PROJECT_SOURCES
        main.cpp
        mandelbrotWidget.cpp
        mandelbrotWidget.h
        mandelbrotCuda.cu
        mandelbrotCuda.cuh
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
)

set(CUDA_SOURCES
    mandelbrotCuda.cu
    mandelbrotCuda.cuh
)
cuda_compile(CUDA_OBJECTS ${CUDA_SOURCES})

add_executable(SetOfMandelbrot
    ${PROJECT_SOURCES}
    ${CUDA_OBJECTS}
    )
target_link_libraries(SetOfMandelbrot PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${CUDA_LIBRARIES})

install(TARGETS SetOfMandelbrot
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
