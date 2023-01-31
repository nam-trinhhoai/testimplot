#include "logstats.h"
#include "ui_logstats.h"

LogStats::LogStats(QString prefix, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LogStats)
{
    ui->setupUi(this);
    setPrefix(prefix);
}

LogStats::~LogStats()
{
    delete ui;
}

void LogStats::setMean(double val) {
    mean = val;
    ui->lineEditMean->setText(QString::number(mean));
}

void LogStats::setMin(double val) {
    min = val;
    ui->lineEditMin->setText(QString::number(min));
}

void LogStats::setMax(double val) {
    max = val;
    ui->lineEditMax->setText(QString::number((max)));
}

void LogStats::setStd(double val) {
    std = val;
    ui->lineEditStd->setText(QString::number((std)));
}

void LogStats::setPrefix(QString name) {
    prefix = name;
    QString prefix = this->prefix;
    if(!prefix.isEmpty() && !prefix.isNull()) {
        prefix = prefix+" ";
    }
    ui->labelMin->setText(prefix+"Min");
    ui->labelMax->setText(prefix+"Max");
    ui->labelMean->setText(prefix+"Mean");
    ui->labelStd->setText(prefix+"Std");
}

double LogStats::getMean() {
    return mean;
}

double LogStats::getMin() {
    return min;
}

double LogStats::getMax() {
    return max;
}

double LogStats::getStd() {
    return std;
}

QString LogStats::getPrefix() {
    return prefix;
}
