#include "binfct.h"
#include <QDebug>

BinFct::BinFct(int p1, int p2,int colorTableSize, bool inverted):AbstractFct( p1, p2, colorTableSize, inverted){}
BinFct::BinFct():AbstractFct() {}
BinFct::BinFct(int colorTableSize):AbstractFct(colorTableSize) {
	reset();
}

BinFct::~BinFct() {}

void BinFct::reset() {
	setParam1(colorTableSize/4);
	setParam2(3*colorTableSize/4);
}

AbstractFct *  BinFct::clone(){
  return new BinFct(getParam1(),getParam2(),colorTableSize, inverted);
}

int  BinFct::get(int x) {
	int result = (x >= p1 && x <= p2) ? colorTableSize - 1 : 0;
	return isInverted() ? colorTableSize - 1 - result : result;
}

void  BinFct::paint(QPainter * e,const QSize &size, int histogramSize) {
	//Beware: p1 and P2 are expressed in the coloTable range
	//Display is done on top of the data histogram.
	//A scale need to be apply

	double displayP1=LUTRenderUtil::convertPositionFromLUTToCanvas(p1,histogramSize,size.width(),colorTableSize);
	double displayP2=LUTRenderUtil::convertPositionFromLUTToCanvas(p2,histogramSize,size.width(),colorTableSize);


	int iMaxY = size.height() - LUTRenderUtil::Y_OFFSET;
	int iH=iMaxY-10;

	int x1 = LUTRenderUtil::X_MIN;
	int y1 = (isInverted()) ? 10 : iMaxY;
	int x2 = displayP1;
	int y3 = (isInverted()) ?  iMaxY : 10;

	if (displayP1 > 0) {
	  e->drawLine(x1, y1, x2, y1);
	  e->drawLine(x2, y1, x2, y3);
	}

	int x3 = displayP2;
	e->drawLine(x2, y3, x3, y3);

	if (displayP2 < (colorTableSize - 1)) {
	  e->drawLine(x3, y3, x3, y1);
	  e->drawLine(x3, y1,LUTRenderUtil::getMaxWidth(size.width()), y1);
	}
}
