#ifndef PALETTEHOLDER_H
#define PALETTEHOLDER_H

#include "ipaletteholder.h"

class PaletteHolder : public IPaletteHolder {
public:
    PaletteHolder(float min, float max);

    virtual QHistogram computeHistogram(const QVector2D &range, int nBuckets) override ;

	virtual bool hasNoDataValue() const override;
	virtual float noDataValue() const override;

	virtual QVector2D dataRange() override;
	virtual QVector2D rangeRatio() override;
	virtual QVector2D range() override;
private:
    float min;
    float max;
};

#endif // PALETTEHOLDER_H
