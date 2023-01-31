#ifndef HISTOGRAMWIDGET_H
#define HISTOGRAMWIDGET_H

#include <QWidget>

#include "qhistogram.h"
#include "lookuptable.h"

class HistogramWidget : public QWidget
{
	Q_OBJECT
public:
    HistogramWidget(QWidget* parent = 0);
    ~HistogramWidget();
signals:
	void rangeChanged(const QVector2D& range);
public:
    void setHistogramAndLookupTable(const QHistogram &histo, const QVector2D& restrictedRange, const LookupTable & table);
    void setHistogram(const QHistogram &histo,
    		const QVector2D &restrictedRange);
    void setRange(const QVector2D& range );

    void setUseLookupTable(bool useColorTable);
    void setDefaultColor(QColor color);

protected:
    virtual void paintEvent(QPaintEvent *);
    virtual void mousePressEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
private:
    void drawVArrow(QPainter *p);
    void drawHArrow(QPainter *p);

    void drawButton(QPainter *p,int pos);

    void updateIndividualRange(int x);
    void updateBothRange(int x);
    void notifyRangeChanged();
private:
    QHistogram m_histo;
    bool m_histoSet;
    double hranges[2];

    //Interactions
    bool m_minMoving = false;
    bool m_maxMoving = false;

    bool m_bothMoving = false;
    int  m_bothInitialPosition=0;

    double currentMin = 0;
    double currentMax = 0;


    int m_min = 0;
    int m_max = 255;

    int minHistoIndex = 0;
    int maxHistoIndex = 0;

    LookupTable m_currentTable;


    bool m_useLookupTable;
    QColor m_defaultColor;
};


#endif // HISTOGRAMWIDGET_H
