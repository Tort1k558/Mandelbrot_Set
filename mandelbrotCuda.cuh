#pragma once
#include<stdio.h>
void mandelbrotCuda(int width, int height, double minX, double maxX, double minY, double maxY, int currentIterations, uchar4 *d_output);
