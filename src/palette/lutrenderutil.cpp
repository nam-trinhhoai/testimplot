#include "lutrenderutil.h"

#include <QPainter>
#include <QPolygon>
#include <QPainterPath>
#include <QPen>


int LUTRenderUtil::getMaxWidth(int dimX)
{
	//Global Window dimension  is reduced from 3 time XMIN to preserve Lateral origin and a end point
	return dimX -3*LUTRenderUtil::X_MIN;
}


double LUTRenderUtil::convertPositionFromCanvasToHistogram(double pos, int histogramSize,int dimX)
{
	return histogramSize *(pos-LUTRenderUtil::X_MIN)/(double)getMaxWidth(dimX);
}

int LUTRenderUtil::convertPositionFromCanvasToHistogram(int pos, int histogramSize,int dimX)
{
	return (int)convertPositionFromCanvasToHistogram((double)pos,histogramSize,dimX);
}

double LUTRenderUtil::convertPositionFromHistogramToCanvas(double pos, int histogramSize,int dimX)
{
	return (pos/histogramSize)*getMaxWidth(dimX) +LUTRenderUtil::X_MIN;
}

int LUTRenderUtil::convertPositionFromHistogramToCanvas(int pos, int histogramSize,int dimX)
{
	return (int)convertPositionFromHistogramToCanvas((double)pos,histogramSize,dimX);
}


double LUTRenderUtil::convertPositionFromHistogramToLUT(double pos, int histogramSize,int lutSize)
{
	double colorTableRatio=(double)lutSize/histogramSize;
	return qMin(qMax(0.0,pos*colorTableRatio),(double)lutSize);
}
int LUTRenderUtil::convertPositionFromHistogramToLUT(int pos, int histogramSize,int lutSize)
{
	return (int)(convertPositionFromHistogramToLUT((double)pos,histogramSize,lutSize));
}


double LUTRenderUtil::convertPositionFromLUTToHistogram(double pos, int histogramSize,int lutSize)
{
	double colorTableRatio=(double)lutSize/histogramSize;
	return qMin(qMax(0.0,pos/colorTableRatio),(double)histogramSize);
}

int LUTRenderUtil::convertPositionFromLUTToHistogram(int pos, int histogramSize,int lutSize)
{
	return (int)convertPositionFromLUTToHistogram((double)pos,histogramSize,lutSize);
}

int LUTRenderUtil::convertPositionFromCanvasToLUT(int pos, int histogramSize,int dimX,int lutSize)
{
	double histoPos=convertPositionFromCanvasToHistogram((double)pos,histogramSize,dimX);
	return (int)convertPositionFromHistogramToLUT(histoPos,histogramSize,lutSize);
}
 int LUTRenderUtil::convertPositionFromLUTToCanvas(int pos, int histogramSize,int dimX,int lutSize)
 {
	 double histoPos=convertPositionFromLUTToHistogram((double)pos,histogramSize,lutSize);
	 return (int)convertPositionFromHistogramToCanvas(histoPos,histogramSize,dimX);
 }

void LUTRenderUtil::drawVArrow(QPainter *p)
{
	int x1=LUTRenderUtil::X_MIN;
	int y1=10;

	int size=LUTRenderUtil::ARROW_SIZE;

	QPolygon pol;
	pol<<QPoint(x1,y1)<<QPoint(x1-size/2,y1+size)<<QPoint(x1+size/2,y1+size);

	QPainterPath path;
	path.moveTo (pol.at(0));
	path.lineTo (pol.at(1));
	path.lineTo (pol.at(2));
	path.lineTo (pol.at(0));

	p->drawPolygon(pol);
	p->fillPath(path,QBrush(QColor("black")));
}

void LUTRenderUtil::drawHArrow(QPainter *p,QSize dims)
{
	int x1 = dims.width() - 10;
	int y1 = dims.height() - LUTRenderUtil::Y_OFFSET + 2;

	int size = LUTRenderUtil::ARROW_SIZE;

	QPolygon pol;
	pol<<QPoint(x1,y1)<<QPoint( x1 - size,y1 + size / 2)<<QPoint(x1 - size,y1 - size / 2);

	QPainterPath path;
	path.moveTo (pol.at(0));
	path.lineTo (pol.at(1));
	path.lineTo (pol.at(2));
	path.lineTo (pol.at(0));


	p->drawPolygon(pol);
	p->fillPath(path,QBrush(QColor("black")));
}

void LUTRenderUtil::drawButton(int pos,QPainter *p,QSize dims, int histogramSize) {

	int x1 = LUTRenderUtil::convertPositionFromHistogramToCanvas(pos,histogramSize,dims.width());
	int y1 = dims.height() - LUTRenderUtil::Y_OFFSET + 2;

	int size = LUTRenderUtil::ARROW_SIZE;

	QPolygon pol;
	pol<<QPoint(x1,y1)<<QPoint( x1 - size,y1 + size)<<QPoint(x1 + size,y1 + size);
	QPainterPath path;
	path.moveTo (pol.at(0));
	path.lineTo (pol.at(1));
	path.lineTo (pol.at(2));
	path.lineTo (pol.at(0));

	QPen pen;
	pen.setWidth(3);
	pen.setBrush(Qt::lightGray);
	p->setPen( pen );
	p->drawLine(x1, y1, x1, 10);
	p->drawPolygon(pol);
	p->fillPath(path,QBrush(QColor("gray")));
}

void LUTRenderUtil::drawRangeAndThresold(QPainter *p, double minV, double thresholdV, double maxV, QSize widgetSize) {
	p->drawText(LUTRenderUtil::X_MIN, 10, QString::number(minV));
	p->drawText(LUTRenderUtil::X_MIN + widgetSize.width()/2, 10, QString::number(thresholdV));
	p->drawText(widgetSize.width() - 30, 10, QString::number(maxV));
}
