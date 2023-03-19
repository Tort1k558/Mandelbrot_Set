#pragma once
#include <QPainter>
#include <QMouseEvent>
#include <QTimer>
#include <QElapsedTimer>
#include <QWidget>
#include <QSlider>
#include <QLabel>
class MandelbrotWidget : public QWidget
{
    Q_OBJECT
public:
    MandelbrotWidget(QWidget *parent = nullptr);
    void setIterations(int count);
    int getIterations(){return m_currentIterations;}
    double getFPS(){return m_fps;}
protected:
    void paintEvent(QPaintEvent *) override;
    void mousePressEvent(QMouseEvent *event ) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    double map(double value, double inMin, double inMax, double outMin, double outMax);
    int mandelbrot(double a, double b);

    double m_minX;
    double m_maxX;
    double m_minY;
    double m_maxY;
    int m_currentIterations;
    bool m_isDragging;
    int m_dragStartX;
    int m_dragStartY;
    QElapsedTimer m_fpsTimer;
    int m_frameCount;
    double m_fps;
};
