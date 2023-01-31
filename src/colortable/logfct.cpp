#include "logfct.h"
#include <cmath>

LogFct::LogFct(int p1, int p2,int colorTableSize, bool inverted):AbstractFct( p1, p2, colorTableSize, inverted){}
LogFct::LogFct():AbstractFct() {}
LogFct::LogFct(int colorTableSize):AbstractFct(colorTableSize) {
	reset();
}

LogFct::~LogFct() {}

void LogFct::reset() {
	setParam1(colorTableSize/4);
	setParam2(3*colorTableSize/4);
}

AbstractFct *  LogFct::clone(){
  return new LogFct(getParam1(),getParam2(),colorTableSize, inverted);
}

int  LogFct::get(int x) {
	int result = 0;
	int size = colorTableSize - 1;

	if (x < p1) {
	  result = 0;
	}
	else if (x > p2) {
	  result = size;
	}
	else {
	  int delta=p2 - p1;
	  if(delta==0)
	  {
		  delta=1;
	  }
	  result = (x - p1) * size / delta;
	  result = (int) std::round((std::log(result + 1) / std::log(size + 1)) * size);
	}
	return ((isInverted()) ? size - result : result);
}

std::vector<int> LogFct::discreteSegment(int x1, int y1, int x2, int y2)
{
    int born;
    int born1;
    int sens;
    int binc;

    int deb[2];
    int fin[2];

    int lon = 0;
    int size = 2 * (std::max(std::abs(x2 - x1), std::abs(y2 - y1)) + 1);

    std::vector<int> ixy(size);
    if (std::abs(y2 - y1) < std::abs(x2 - x1)) {
      born = 0;
      if (x1 < x2) sens = 1; else sens = -1;
    }else {
      born = 1;
      if (y1 < y2) sens = 1; else sens = -1;
    }

    born1 = 1 - born;

    deb[0] = x1;
    deb[1] = y1;
    fin[0] = x2;
    fin[1] = y2;

    for (binc = deb[born]; binc != fin[born] + sens; binc += sens) {
      ixy[lon + born] = binc;
      if ((lon - 2 + born) >= 0 && ixy[lon - 2 + born] == fin[born])
        ixy[lon + born1] = fin[born1];
      else {
        if (fin[born] == deb[born]) {
          ixy[lon + born1] = fin[born];
        }
        else {
          ixy[lon + born1] = (int)(((float) (fin[born1] * (binc - deb[born])- deb[born1] * (binc - fin[born])) / (fin[born] - deb[born])) + .5);
        }
      }
      lon = lon + 2;
    }

    return ixy;
  }


void  LogFct::paint(QPainter * e,const QSize &size, int histogramSize) {
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

	std::vector<int> segments = discreteSegment(x2, y1, x3, y3);
	int length = segments.size();
	QPoint points[length / 2];
	int l = length / 2;
	int top = isInverted() ? y3 : y1;
	int base = isInverted() ? y1 : y3;
	l = top - base;

	for (int i = 0, cp = 0; i < length; i += 2, cp++) {
	  int x = segments[i];
	  int y = segments[i + 1];
	  points[cp].setX(x);

	  int result = (y - top) * l / (base - top);
	  if (result != 0) result = (int) std::round(std::log(result) * (l / std::log(l)));
	  result = top - result;
	  points[cp].setY(result);
	}

	e->drawPolyline(points, length / 2);


	if (displayP2 < (colorTableSize - 1)) {
	  e->drawLine(x3, y3,LUTRenderUtil::getMaxWidth(size.width()), y3);
	}
}
