#include "customchartview.h"
#include <QVBoxLayout>

/*CustomChartView::CustomChartView(QWidget *parent) :
    QWidget(parent)
{
    setSizePolicy( QSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum) );
    setLayout(new QVBoxLayout);
}*/

CustomChartView::CustomChartView(QChart* chart, QWidget *parent) :
    QChartView(chart, parent)
{
    //setSizePolicy( QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred) );
    m_chart = chart;
}

CustomChartView::~CustomChartView()
{

}

void CustomChartView::tooltip(QPointF point, bool state)
{
    if (m_tooltip == 0)
        m_tooltip = new Callout(m_chart);

    if (state) {
        m_tooltip->setText(QString("X: %1 \nY: %2 ").arg(point.x()).arg(point.y()));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
}

/*void CustomChartView::resizeEvent(QResizeEvent *event) {
    event->accept();

    QWidget::resizeEvent(event);

    widget->resize(QSize(event->size().width(), event->size().width()));
}

void CustomChartView::setContent(QWidget *w) {
    layout()->addWidget(w);
    this->widget = w;
}*/
