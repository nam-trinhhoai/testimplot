#include "triangle2fct.h"


Triangle2Fct::Triangle2Fct(int p1, int p2,int colorTableSize, bool inverted):AbstractFct( p1, p2, colorTableSize, inverted){}
Triangle2Fct::Triangle2Fct():AbstractFct() {}
Triangle2Fct::Triangle2Fct(int colorTableSize):AbstractFct(colorTableSize) {
	reset();
}

Triangle2Fct::~Triangle2Fct() {}

void Triangle2Fct::reset() {
	setParam1(colorTableSize/4);
	setParam2(3*colorTableSize/4);
}

AbstractFct *  Triangle2Fct::clone(){
  return new Triangle2Fct(getParam1(),getParam2(),colorTableSize, inverted);
}

int  Triangle2Fct::get(int x) {
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


void  Triangle2Fct::paint(QPainter * e,const QSize &size, int histogramSize) {
	//Beware: p1 and P2 are expressed in the coloTable range
	//Display is done on top of the data histogram.
	//A scale need to be apply

	double colorTableRatio=(double)colorTableSize/histogramSize;
	int iMaxY = size.height() - LUTRenderUtil::Y_OFFSET;
	int iH=iMaxY-10;

	int x1 = LUTRenderUtil::X_MIN;
	int y1 = (isInverted()) ? 10 : iMaxY;
	int y2 = (isInverted()) ?  iMaxY : 10;

	int delta=p2-p1;
	int start = p1;
    int stop = start + delta;

    while (start < colorTableSize - 1) {
    	int posStart = LUTRenderUtil::convertPositionFromLUTToCanvas(start,histogramSize,size.width(),colorTableSize);
    	int posEnd = LUTRenderUtil::convertPositionFromLUTToCanvas(stop,histogramSize,size.width(),colorTableSize);

        e->drawLine(posStart, y1, posEnd, y2);

        start = stop;
        stop = start + delta;

        posStart = LUTRenderUtil::convertPositionFromLUTToCanvas(start,histogramSize,size.width(),colorTableSize);
        posEnd = LUTRenderUtil::convertPositionFromLUTToCanvas(stop,histogramSize,size.width(),colorTableSize);

        e->drawLine(posStart, y2, posEnd, y1);

        start = stop;
        stop = start + delta;
     }

	 start = p1 - delta;
	 stop = p1;

	 while (stop > 0) {
		int posStart = LUTRenderUtil::convertPositionFromLUTToCanvas(start,histogramSize,size.width(),colorTableSize);
		int posEnd = LUTRenderUtil::convertPositionFromLUTToCanvas(stop,histogramSize,size.width(),colorTableSize);
		e->drawLine(posStart, y2, posEnd, y1);

		stop = start;
		start = stop - delta;

		posStart = LUTRenderUtil::convertPositionFromLUTToCanvas(start,histogramSize,size.width(),colorTableSize);
		posEnd = LUTRenderUtil::convertPositionFromLUTToCanvas(stop,histogramSize,size.width(),colorTableSize);

		e->drawLine(posStart, y1, posEnd, y2);

		stop = start;
		start = stop - delta;
	 }
}
