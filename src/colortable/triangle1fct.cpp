#include "triangle1fct.h"


Triangle1Fct::Triangle1Fct(int p1, int p2,int colorTableSize, bool inverted):AbstractFct( p1, p2, colorTableSize, inverted){}
Triangle1Fct::Triangle1Fct():AbstractFct() {}
Triangle1Fct::Triangle1Fct(int colorTableSize):AbstractFct(colorTableSize) {
	reset();
}

Triangle1Fct::~Triangle1Fct() {}

void Triangle1Fct::reset() {
	setParam1(colorTableSize/4);
	setParam2(3*colorTableSize/4);
}

AbstractFct *  Triangle1Fct::clone(){
  return new Triangle1Fct(getParam1(),getParam2(),colorTableSize, inverted);
}

int  Triangle1Fct::get(int x) {
	int param1 = getParam1();
	int xmin = x - param1;
	int result;

	int delta=p2-p1;
	if(delta==0)delta=1;

	if (xmin < 0) {
	  result = (delta - (-xmin % delta)) * (colorTableSize - 1) / delta;
	}
	else {
	  result = ((x - param1) % delta) * (colorTableSize - 1) / delta;
	}
	return ((isInverted()) ? colorTableSize - 1 - result : result);
}

void  Triangle1Fct::paint(QPainter * e,const QSize &size, int histogramSize) {
	//Beware: p1 and P2 are expressed in the coloTable range
	//Display is done on top of the data histogram.
	//A scale need to be apply


	int iMaxY = size.height() - LUTRenderUtil::Y_OFFSET;
	int iH=iMaxY-10;

	int x1 = LUTRenderUtil::X_MIN;
	int y1 = (isInverted()) ? 10 : iMaxY;
	int y2 = (isInverted()) ?  iMaxY : 10;

	int delta=p2-p1;
	int start = p1;
    int stop = start + delta;

    while (start < colorTableSize - 1) {
    LUTRenderUtil::convertPositionFromLUTToCanvas(p1,histogramSize,size.width(),colorTableSize);

	  int posStart = LUTRenderUtil::convertPositionFromLUTToCanvas(start,histogramSize,size.width(),colorTableSize);;
	  int posEnd = LUTRenderUtil::convertPositionFromLUTToCanvas(stop,histogramSize,size.width(),colorTableSize);;

	  e->drawLine(posStart, y1, posStart, y2);
	  e->drawLine(posStart, y2, posEnd, y1);
	  start = stop;
	  stop = start + delta;
	}

	start = p1 - delta;
	stop = p1;
	while (stop > 0) {
	  int posStart = LUTRenderUtil::convertPositionFromLUTToCanvas(start,histogramSize,size.width(),colorTableSize);;
	  int posEnd = LUTRenderUtil::convertPositionFromLUTToCanvas(stop,histogramSize,size.width(),colorTableSize);;
	  e->drawLine(posStart, y1, posStart, y2);
	  e->drawLine(posStart, y2, posEnd, y1);

	  stop = start;
	  start = stop - delta;
	}
}
