#include "igeorefgrid.h"

IGeorefGrid::IGeorefGrid() {

}

IGeorefGrid::~IGeorefGrid() {

}

QRectF IGeorefGrid::worldExtent(const IGeorefGrid*const image) {
	double ij[8];

	ij[0] = 0;
	ij[1] = 0;

	ij[2] = image->width()-1;
	ij[3] = 0;

	ij[4] = 0;
	ij[5] = image->height()-1;

	ij[6] = image->width()-1;
	ij[7] = image->height()-1;
	return computeRect(image,ij);
}

QRectF IGeorefGrid::computeRect(const IGeorefGrid *const image,double * ij)
{
	double xMin = std::numeric_limits<double>::max();
	double yMin = std::numeric_limits<double>::max();

	double xMax = std::numeric_limits<double>::min();
	double yMax = std::numeric_limits<double>::min();
	double x, y;
	for (int i = 0; i < 4; i++) {
		image->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);

		xMin = std::min(xMin, x);
		yMin = std::min(yMin, y);

		xMax = std::max(xMax, x);
		yMax = std::max(yMax, y);
	}

	return QRectF(xMin, yMin, xMax - xMin, yMax - yMin);

}

QRectF IGeorefGrid::imageToWorld(const IGeorefGrid * const image, const QRectF & rect)
{
	double x1,y1,x2,y2;
	rect.getCoords(&x1,&y1,&x2,&y2);

	double ij[]={x1,y1,x1,y2,x2,y1,x2,y2};
	return computeRect(image,ij);
}
