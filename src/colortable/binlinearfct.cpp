#include "binlinearfct.h"


BinLinearFct::BinLinearFct(int p1, int p2,int colorTableSize, bool inverted):AbstractFct( p1, p2, colorTableSize, inverted)
{

}


BinLinearFct::BinLinearFct():AbstractFct() {

}

BinLinearFct::BinLinearFct(int colorTableSize):AbstractFct(colorTableSize) {
	reset();
}

BinLinearFct::~BinLinearFct() {}

void BinLinearFct::reset() {
	setParam1(colorTableSize/4);
	setParam2(3*colorTableSize/4);
}

AbstractFct *  BinLinearFct::clone(){
  return new BinLinearFct(getParam1(),getParam2(),colorTableSize, inverted);
}

int  BinLinearFct::get(int x) {
	int result = 0;

	if (x < p1) {
	  result = 0;
	}
	else if (x > p2) {
	  result =colorTableSize - 1;
	}
	else {
	  result = x;
	}
	return ((isInverted()) ? (colorTableSize - 1) - result : result);
	}

void  BinLinearFct::paint(QPainter * e,const QSize &size, int histogramSize) {
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
	}

	int x3 = displayP2;
	e->drawLine(x2, y1, x3, y3);

	if (displayP2 < (colorTableSize - 1)) {
	  e->drawLine(x3, y3, x3, y1);
	  e->drawLine(x3, y1,LUTRenderUtil::getMaxWidth(size.width()), y1);
	}
}
