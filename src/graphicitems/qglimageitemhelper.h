#ifndef QGLImageItemHelper_H
#define QGLImageItemHelper_H

class IGeorefImage;
#include "qvertex2D.h"

class QGLImageItemHelper
{
public:
	static void computeImageCorner(IGeorefImage *image,QVertex2D &v0, QVertex2D &v1,
			QVertex2D &v2, QVertex2D &V3);
	virtual ~QGLImageItemHelper();
private:
	QGLImageItemHelper(){}

};
#endif

