#include "mandelbrotWidget.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <QVBoxLayout>
#include "mandelbrotCuda.cuh"
#include <QDebug>
MandelbrotWidget::MandelbrotWidget(QWidget *parent) : QWidget(parent)
{
    setMouseTracking(true);
    m_minX = -2.0;
    m_maxX = 1.0;
    m_minY = -1.5;
    m_maxY = 1.5;
    m_currentIterations = 500;
    m_isDragging = false;
    m_dragStartX = 0;
    m_dragStartY = 0;
    m_fpsTimer.start();
}
void MandelbrotWidget::paintEvent(QPaintEvent *)
{
    m_frameCount++;
    int64_t elapsed = m_fpsTimer.elapsed();
    if (elapsed >= 1000) {
        m_fps = double(m_frameCount) * 1000.0 / elapsed;
        m_frameCount = 0;
        m_fpsTimer.restart();
    }

    QImage image(width(), height(), QImage::Format_RGB32);

    //cuda
    int width = this->size().width();
    int height = this->size().height();

    uchar4 *d_output;
    uchar4 *h_output = new uchar4[width * height];

    cudaMalloc(&d_output, width * height * sizeof(uchar4));
    mandelbrotCuda(width, height, m_minX, m_maxX, m_minY, m_maxY, m_currentIterations, d_output);

    cudaMemcpy(h_output, d_output, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uchar4 pixel = h_output[y * width + x];
            QRgb rgb = qRgb(pixel.x, pixel.y, pixel.z);
            image.setPixel(x, y, rgb);
        }
    }


    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.drawImage(0, 0, image);
    update();
}
void MandelbrotWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        m_isDragging = true;
        m_dragStartX = event->position().x();
        m_dragStartY = event->position().y();
    }
}
void MandelbrotWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        m_isDragging = false;
        double dx = map(m_dragStartX - event->position().x(), 0, width(), m_minX, m_maxX) - map(0, 0, width(), m_minX, m_maxX);
        double dy = map(m_dragStartY - event->position().y(), 0, height(), m_minY, m_maxY) - map(0, 0, height(), m_minY, m_maxY);
        m_minX += dx;
        m_maxX += dx;
        m_minY += dy;
        m_maxY += dy;
        update();
    }
}

void MandelbrotWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (m_isDragging) {
        double dx = map(m_dragStartX - event->position().x(), 0, width(), m_minX, m_maxX) - map(0, 0, width(), m_minX, m_maxX);
        double dy = map(m_dragStartY - event->position().y(), 0, height(), m_minY, m_maxY) - map(0, 0, height(), m_minY, m_maxY);
        m_minX += dx;
        m_maxX += dx;
        m_minY += dy;
        m_maxY += dy;
        update();
    }
}

void MandelbrotWidget::wheelEvent(QWheelEvent* event)
{
    double scale = 1.2;
    double scaleFactor = event->angleDelta().y() < 0 ? scale : 1.0 / scale;
    QPointF cursorPos = event->position();
    double centerA = map(cursorPos.x(), 0, width(), m_minX, m_maxX);
    double centerB = map(cursorPos.y(), 0, height(), m_minY, m_maxY);
    double width = m_maxX - m_minX;
    double height = m_maxY - m_minY;
    double newWidth = width * scaleFactor;
    double newHeight = height * scaleFactor;
    m_minX = centerA - newWidth / 2;
    m_maxX = centerA + newWidth / 2;
    m_minY = centerB - newHeight / 2;
    m_maxY = centerB + newHeight / 2;
    update();
}
double MandelbrotWidget::map(double value, double inMin, double inMax, double outMin, double outMax)
{
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

int MandelbrotWidget::mandelbrot(double a, double b)
{
    double x = 0.0;
    double y = 0.0;
    int n = 0;
    while (x * x + y * y <= 4.0 && n < m_currentIterations) {
        double xtemp = x * x - y * y + a;
        y = 2.0 * x * y + b;
        x = xtemp;
        n++;
    }
    return n;
}
void MandelbrotWidget::setIterations(int count)
{
    m_currentIterations = count;
}
