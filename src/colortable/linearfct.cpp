#include "linearfct.h"


LinearFct::LinearFct(int p1, int p2,int colorTableSize, bool inverted):AbstractFct( p1, p2, colorTableSize, inverted){}
LinearFct::LinearFct():AbstractFct() {}
LinearFct::LinearFct(int colorTableSize):AbstractFct(colorTableSize) {}
LinearFct::~LinearFct() {}

void LinearFct::reset() {
	setParam1(0);
	setParam2(colorTableSize);
}

AbstractFct *  LinearFct::clone(){
  return new LinearFct(getParam1(),getParam2(),colorTableSize, inverted);
}

int  LinearFct::get(int x) {
	int result = 0;

	if (x < p1) {
	  result = 0;
	}
	else if (x > p2) {
	  result =colorTableSize - 1;
	}
	else {
	  int delta=p2 - p1;
	  if(delta==0)
	  {
		  delta=1;
	  }
	  result = (x - p1) * (colorTableSize - 1) / delta;
	}
	return ((isInverted()) ? (colorTableSize - 1) - result : result);
	}

void  LinearFct::paint(QPainter * e,const QSize &size, int histogramSize) {
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

	if (displayP1 > 0) {
	  e->drawLine(x1, y1, x2, y1);
	}

	int x3 = displayP2;
	int y3 = (isInverted()) ?  iMaxY : 10;

	e->drawLine(x2, y1, x3, y3);

	if (displayP2 < (colorTableSize - 1)) {
	  e->drawLine(x3, y3,LUTRenderUtil::getMaxWidth(size.width()), y3);
	}

	int x4 = (displayP2 + displayP1) / 2;
	e->drawLine(x4, iMaxY-3*iH/8  , x4, iMaxY-5*iH/8);
}
