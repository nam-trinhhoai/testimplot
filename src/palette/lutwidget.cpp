/*
 * colortablewidget.cpp
 *
 *  Created on: 16 mai 2018
 *      Author: j0334308
 */

#include "lutwidget.h"

#include <cmath>
#include <QImage>
#include <QPainter>
#include <QVector>
#include <QMouseEvent>
#include <QDebug>

#include "linearfct.h"
#include "lutrenderutil.h"


LUTWidget::LUTWidget(QWidget* parent ) :QWidget(parent)
{
	setMinimumSize(LUTRenderUtil::INITIAL_WIDTH, LUTRenderUtil::INITIAL_HEIGHT  );
}

LUTWidget::~LUTWidget() {

}

void LUTWidget::setLookupTableParam1(int val)
{
	m_currentTable.setFunctionParam1(val);
	update();
	emit lookupTableChanged(m_currentTable);
}
void LUTWidget::setLookupTableParam2(int val)
{
	m_currentTable.setFunctionParam2(val);
	update();
	emit lookupTableChanged(m_currentTable);
}


void LUTWidget::setTransfertFunctionType(AbstractFct::FUNCTION_TYPE type)
{
	m_currentTable.setFunctionType(type);
	update();
	emit lookupTableChanged(m_currentTable);
}

void LUTWidget::setTransfertInverted(bool state)
{
	m_currentTable.setFunctionInverted(state);
	update();
	emit lookupTableChanged(m_currentTable);
}

void LUTWidget::razTransp()
{
	m_currentTable.razTransp();
	update();
	emit lookupTableChanged(m_currentTable);
}

void LUTWidget::razFunction()
{
	m_currentTable.razFunction();
	update();
	emit lookupTableChanged(m_currentTable);
}


void LUTWidget::setHistogramAndLookupTable(const QHistogram &histo, const QVector2D& restrictedRange,const LookupTable & table)
{
	m_histo=histo;
	m_currentTable=table;
	this->restrictedRange = restrictedRange;

	double maxVal=0;
	double minVal=0;
	for(int i=0;i<QHistogram::HISTOGRAM_SIZE;i++)
	{
		double val=histo[i];
		minVal=std::min(val,minVal);
		maxVal=std::max(val,maxVal);
	}

	hranges[0]=minVal;
	hranges[1]=maxVal;

	QPalette pal = palette();
	// set black background
	pal.setColor(QPalette::Base, Qt::gray);
	setAutoFillBackground(true);

	m_histoSet=true;
	update();
}

void LUTWidget::setLookupTable( const LookupTable & table)
{
	m_currentTable=table;

	QPalette pal = palette();
	// set black background
	pal.setColor(QPalette::Base, Qt::gray);
	setAutoFillBackground(true);

	//?????????????m_histoSet=true;
	update();
}

void LUTWidget::paintEvent(QPaintEvent *)
{
    int iMaxY = height() - LUTRenderUtil::Y_OFFSET;

    QPainter p(this);

    p.setBrush( QColor(0xFF,0xFF,0xFF) );

    p.drawLine(LUTRenderUtil::X_MIN,iMaxY+2,width()-LUTRenderUtil::X_MIN,iMaxY+2);
    p.drawLine(LUTRenderUtil::X_MIN,iMaxY+2,LUTRenderUtil::X_MIN,10);

    LUTRenderUtil::drawVArrow(&p);
    LUTRenderUtil::drawHArrow(&p,size());

    if(!m_histoSet)return;

    int iH=iMaxY-10;

    double ratioLutHisto = m_currentTable.size() / (double) QHistogram::HISTOGRAM_SIZE;
    double rationHistoRange =  QHistogram::HISTOGRAM_SIZE /
    		(restrictedRange.y() - restrictedRange.x() + 1.);

    for ( int i = 0; i < QHistogram::HISTOGRAM_SIZE; i++ )
    {
        uint pos=(uint)(m_histo[i] * iH / (hranges[1]) + 0.5);
        double lutInd = i * ratioLutHisto;
		std::array<int, 4>  c;
		int alpha;
		c = m_currentTable.getColors(lutInd);
		alpha =m_currentTable.getAlpha(lutInd);

      	p.setPen( QColor(c[0],c[1],c[2],c[3]) );
      	p.setBrush( QColor(c[0],c[1],c[2],c[3]) );

      	int xMin=LUTRenderUtil::convertPositionFromHistogramToCanvas(i, QHistogram::HISTOGRAM_SIZE,width());
      	int xMax=LUTRenderUtil::convertPositionFromHistogramToCanvas(i+1, QHistogram::HISTOGRAM_SIZE,width());

      	p.drawRect(xMin,iMaxY- pos,xMax-xMin,pos);

        if(i!=QHistogram::HISTOGRAM_SIZE-1)
        {
        	std::array<int, 4>  nextC=m_currentTable.getColors(lutInd+1);
        	int nextAlpha =m_currentTable.getAlpha(lutInd+1);

			//Draw the Alpha (using the non converted color through LUT)
			p.setPen( Qt::red );

			uint yAlphaPosNext=(uint)(nextAlpha * iH/ 255.0 + 0.5);
			uint yAlphaPos=(uint)(alpha * iH/ 255.0 + 0.5);

			p.drawLine(xMin,iMaxY- yAlphaPos, xMax, iMaxY- yAlphaPosNext);
        }
    }
    p.setPen(Qt::yellow);
    m_currentTable.paintFunction(&p,size(),QHistogram::HISTOGRAM_SIZE);
}

void LUTWidget::updateTransparency(int x, int y) {

	int delta=x-m_alphaBeginEditPos;
	if(delta==0)return;

	int iMaxY = height() - LUTRenderUtil::Y_OFFSET;
	int iH=iMaxY-10;

	int alpha= (int)(255.0*(iMaxY - y)/iH);

	if(alpha<0)alpha=0;
	if(alpha>255)alpha=255;

	int minIndex=m_currentTable.size();
	int maxIndex=0;

	for(int i=0;i<std::abs(delta);i++)
	{
		int newX=0;
		if(delta>0)
			newX=m_alphaBeginEditPos+i;
		else
			newX=m_alphaBeginEditPos-i;

		int lutPos=LUTRenderUtil::convertPositionFromCanvasToLUT(newX,QHistogram::HISTOGRAM_SIZE,width(),m_currentTable.size());

		qDebug() << "x= " << x << " y= " << y << " alpha= " << alpha << " lutPos= " << lutPos;
		minIndex=qMin(minIndex,lutPos);
		maxIndex=qMax(maxIndex,lutPos);
	}
	qDebug() << "minIndex= " << minIndex << " maxIndex= " << maxIndex << " alpha= " << alpha;
	m_currentTable.setInterpolatedAlpha(minIndex,maxIndex,delta>0,alpha);

	m_alphaBeginEditPos=x;

	update();
	emit lookupTableChanged(m_currentTable);
}


void LUTWidget::updateFct(int x, int y) {

	int deltaX=x-m_fctBeginEditPosX;
	m_fctBeginEditPosX=x;

	int deltaY=y-m_fctBeginEditPosY;
	m_fctBeginEditPosY=y;

	//Convert the parameter in the display space (eg histogram)
	int p1Display=LUTRenderUtil::convertPositionFromLUTToCanvas(m_currentTable.getFunctionParam1(),QHistogram::HISTOGRAM_SIZE,width(),m_currentTable.size());
	int p2Display=LUTRenderUtil::convertPositionFromLUTToCanvas(m_currentTable.getFunctionParam2(),QHistogram::HISTOGRAM_SIZE,width(),m_currentTable.size());

	//Apply operation
	p1Display+=deltaX;
	p2Display+=deltaX;

	p1Display-=deltaY;
	p2Display+=deltaY;

	//Go back to the color table space
	int p1 = LUTRenderUtil::convertPositionFromCanvasToLUT(p1Display,QHistogram::HISTOGRAM_SIZE,width(),m_currentTable.size());
	int p2 = LUTRenderUtil::convertPositionFromCanvasToLUT(p2Display,QHistogram::HISTOGRAM_SIZE,width(),m_currentTable.size());
	m_currentTable.setFunctionParam1(p1);
	m_currentTable.setFunctionParam2(p2);

	update();

	emit lookupTableFunctionParamsChanged(p1,p2);
	emit lookupTableChanged(m_currentTable);
}

void LUTWidget::mouseReleaseEvent(QMouseEvent *event)
{
	if(m_alphaEditing)
	{
		//updateTransparency(event->x(),event->y());
		m_alphaEditing=false;
	}
	if(m_fctEditing)
	{
		//updateFct(event->x(),event->y());
		m_fctEditing=false;
	}

}

void LUTWidget::mouseMoveEvent(QMouseEvent *event)
{
	if(m_alphaEditing)
	{
		updateTransparency(event->x(),event->y());
	}
	if(m_fctEditing)
	{
		updateFct(event->x(),event->y());
	}
}

void LUTWidget::mousePressEvent ( QMouseEvent * event )
{
	//Use middle button to edit transparency
	if(event->button()==Qt::MouseButton::MiddleButton)
	{
		m_alphaEditing=true;
		m_alphaBeginEditPos=event->x();
	}

	if(event->button()==Qt::MouseButton::LeftButton)
	{
		m_fctEditing=true;
		m_fctBeginEditPosX=event->x();
		m_fctBeginEditPosY=event->y();
	}

}

void LUTWidget::resizeEvent(QResizeEvent *event)
{
	emit sizeChanged();
}



