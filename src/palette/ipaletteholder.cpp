#include "ipaletteholder.h"

IPaletteHolder::IPaletteHolder() {

}

IPaletteHolder::~IPaletteHolder() {

}
QVector2D IPaletteHolder::smartAdjust(const QVector2D & range,const QHistogram & histo) {
	double borne = 0.005;
	double ratio = (range.y() - range.x())
			/ (histo.HISTOGRAM_SIZE - 1);

	double numPix = 0;
	for (int i = 0; i < histo.HISTOGRAM_SIZE; i++)
		numPix += histo[i];

	int i = 0;
	double sum = 0;
	while (sum < borne * numPix && i<histo.HISTOGRAM_SIZE) {
		i++;
		sum += histo[i];
	}

	double vMin = std::max(i * ratio + range.x(),
			(double) range.x());

	i = histo.HISTOGRAM_SIZE - 1;
	sum = 0;
	while (sum < borne * numPix && i>=0) {
		i--;
		sum += histo[i];
	}

	double vMax = std::min(i * ratio + range.x(),
			(double) range.y());
	if (vMax < vMin)
		vMax = vMin;

	return QVector2D{ (float) vMin, (float) vMax };

}
