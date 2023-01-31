#include "paletteholder.h"

PaletteHolder::PaletteHolder(float min, float max) {
    this->min = min;
    this->max = max;
}

QVector2D PaletteHolder::range() {
    QVector2D res(min, max);
    return res;
}

QVector2D PaletteHolder::dataRange() {
    QVector2D res(min, max);
    return res;
}

QVector2D PaletteHolder::rangeRatio() {
    QVector2D res;
    res.setX(min);
	if (max - min != 0) {
		res.setY(1.0f / (max - min));
	}
    return res;
}

QHistogram PaletteHolder::computeHistogram(const QVector2D &range, int nBuckets) {
    QHistogram res;
    res.setRange(range);
    return res;
}

bool PaletteHolder::hasNoDataValue() const {
	return false;
}

float PaletteHolder::noDataValue() const {
	return -9999.0;
}
