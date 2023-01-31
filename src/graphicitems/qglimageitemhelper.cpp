#include "qglimageitemhelper.h"
#include "igeorefimage.h"

void QGLImageItemHelper::computeImageCorner(IGeorefImage *image,QVertex2D &v0, QVertex2D &v1,
		QVertex2D &v2, QVertex2D &v3)
{
	double x,y;
	image->imageToWorld(0, image->height(), x, y);
	v0.position = QVector2D(x, y);
	v0.coords = QVector2D(0, 1);

	image->imageToWorld(0, 0, x, y);
	v1.position = QVector2D(x, y);
	v1.coords = QVector2D(0, 0);

	image->imageToWorld(image->width(), image->height(), x, y);
	v2.position = QVector2D(x, y);
	v2.coords = QVector2D(1, 1);

	image->imageToWorld(image->width(), 0, x, y);
	v3.position = QVector2D(x, y);
	v3.coords = QVector2D(1, 0);
}
