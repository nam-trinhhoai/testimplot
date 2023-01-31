#include "igeorefimage.h"

IGeorefImage::IGeorefImage() {
}

IGeorefImage::~IGeorefImage() {
}

bool IGeorefImage::value(const IGeorefImage *const image,double worldX, double worldY, int &i, int &j,
		double &value) {
	double di, dj;
	image->worldToImage(worldX, worldY, di, dj);

	i = (int) di;
	j = (int) dj;
	if (i < 0 || j < 0 || i >= image->width() || j >= image->height())
		return false;
	return image->valueAt(i,j,value);
}
