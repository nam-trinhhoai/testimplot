#ifndef LOGSTATS_H
#define LOGSTATS_H

#include <QWidget>
#include <QString>

namespace Ui {
class LogStats;
}

class LogStats : public QWidget
{
    Q_OBJECT

public:
    explicit LogStats(QString prefix="", QWidget *parent = 0);
    ~LogStats();
    void setMean(double);
    void setMin(double);
    void setMax(double);
    void setStd(double);

    double getMean();
    double getMin();
    double getMax();
    double getStd();

    void setPrefix(QString name);
    QString getPrefix();
private:
    Ui::LogStats *ui;

    double min=0;
    double max=0;
    double mean=0;
    double std=0;
    QString prefix;
};

#endif // LOGSTATS_H
