#ifndef CUSTOMCHARTVIEW_H
#define CUSTOMCHARTVIEW_H

#include <QWidget>
#include <QChartView>
#include <QResizeEvent>
#include <callout.h>
#include <QPointF>

class CustomChartView : public QChartView
{
    Q_OBJECT

public:
    //explicit CustomChartView(QWidget *parent = 0);
    explicit CustomChartView(QChart* chart, QWidget *parent = 0);
    ~CustomChartView();
    //void setContent(QWidget* w);

public slots:
    void tooltip(QPointF point, bool state);

protected:
    //virtual void resizeEvent(QResizeEvent *event) Q_DECL_OVERRIDE;

private:
    //QWidget* widget;
    Callout *m_tooltip = nullptr;
    QChart* m_chart = nullptr;

private:
};

#endif // CUSTOMCHARTVIEW_H
