#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "mandelbrotWidget.h"
#include<QVBoxLayout>
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFixedSize(800,800);
    m_mandelbrot = new MandelbrotWidget();

    ui->lineEdit->setText(QString("Iterations: ")+QString::number(m_mandelbrot->getIterations()));
    ui->horizontalSlider->setValue(m_mandelbrot->getIterations());
    ui->verticalLayout_3->addWidget(m_mandelbrot);
    m_timer = new QTimer(this);
    connect(m_timer,&QTimer::timeout,this,&MainWindow::updateTitleFPS);
    m_timer->start(0);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::updateTitleFPS()
{
    this->setWindowTitle(QString("FPS: ")+QString::number(m_mandelbrot->getFPS()));
}
void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    m_mandelbrot->setIterations(value);
    ui->lineEdit->setText(QString("Iterations: ")+QString::number(m_mandelbrot->getIterations()));
}

