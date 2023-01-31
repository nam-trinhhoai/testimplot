/*
 * colortablewidget.h
 *
 *  Created on: 16 mai 2018
 *      Author: j0334308
 */

#ifndef COLORTABLE_WIDGET_H
#define COLORTABLE_WIDGET_H

#include <QWidget>

#include "qhistogram.h"
#include "lookuptable.h"
#include "abstractfct.h"


class LUTWidget : public QWidget{
	Q_OBJECT
public:
	LUTWidget(QWidget* parent = 0);
	virtual ~LUTWidget();

    void setHistogramAndLookupTable(const QHistogram &histo, const QVector2D& restrictedRange, const LookupTable & table);
    void setLookupTable( const LookupTable & table);

    void setLookupTableParam1(int val);
    void setLookupTableParam2(int val);

    void setTransfertFunctionType(AbstractFct::FUNCTION_TYPE);

    void setTransfertInverted(bool state);

    void razTransp();
   	void razFunction();

protected:
	void resizeEvent(QResizeEvent *event) override;
signals:
	void lookupTableChanged(const LookupTable& colorTable);
	void lookupTableFunctionParamsChanged(int p1, int p2);

	void sizeChanged();

protected:
    virtual void paintEvent(QPaintEvent *);
    virtual void mousePressEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
private:
    void drawVArrow(QPainter *p);
    void drawHArrow(QPainter *p);

    void updateTransparency(int x, int y);
    void updateFct(int x, int y);
private:
    QHistogram m_histo;
    bool m_histoSet;
    double hranges[2];
    QVector2D restrictedRange;

    bool m_alphaEditing = false;
    int m_alphaBeginEditPos=0;

    bool m_fctEditing = false;
    int m_fctBeginEditPosX=0;
    int m_fctBeginEditPosY=0;

    LookupTable m_currentTable;
};



#endif /* QTLARGEIMAGEVIEWER_SRC_HISTOGRAM_COLORTABLEWIDGET_H_ */
