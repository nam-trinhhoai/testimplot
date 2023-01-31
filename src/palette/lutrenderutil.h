/*
 * colortablerenderutil.h
 *
 *  Created on: 16 mai 2018
 *      Author: j0334308
 */

#ifndef COLORTABLERENDERUTIL_H_
#define COLORTABLERENDERUTIL_H_

#include <QSize>
class QPainter;

class LUTRenderUtil
{
public:
	static const int X_MIN=10;
	static const int Y_OFFSET=15;
	static const int ARROW_SIZE=7;

	static const int INITIAL_HEIGHT=175;
	static const int INITIAL_WIDTH=300;


	static double convertPositionFromHistogramToLUT(double pos, int histogramSize,int lutSize);
	static double convertPositionFromLUTToHistogram(double pos, int histogramSize,int lutSize);

	static int convertPositionFromHistogramToLUT(int pos, int histogramSize,int lutSize);
	static int convertPositionFromLUTToHistogram(int pos, int histogramSize,int lutSize);


	static double convertPositionFromCanvasToHistogram(double pos, int histogramSize,int dimX);
	static double convertPositionFromHistogramToCanvas(double pos, int histogramSize,int dimX);

	static int convertPositionFromCanvasToHistogram(int pos, int histogramSize,int dimX);
	static int convertPositionFromHistogramToCanvas(int pos, int histogramSize,int dimX);


	static int convertPositionFromCanvasToLUT(int pos, int histogramSize,int dimX,int lutSize);
	static int convertPositionFromLUTToCanvas(int pos, int histogramSize,int dimX,int lutSize);

	static int getMaxWidth(int dimX);



	static void drawVArrow(QPainter *p);
	static void drawHArrow(QPainter *p,QSize dims);
	static void drawButton(int pos,QPainter *p,QSize dims, int histogramSize);
	static void drawRangeAndThresold(QPainter *p, double minV, double thresholdV, double maxV, QSize widgetSize);
};


#endif /* QTLARGEIMAGEVIEWER_SRC_HISTOGRAM_COLORTABLERENDERUTIL_H_ */
