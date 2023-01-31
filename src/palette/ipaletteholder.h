#ifndef QAbstractPaletteHolder_h
#define QAbstractPaletteHolder_h

#include "qhistogram.h"
#include <QVector2D>

class IPaletteHolder
{
public:
	virtual ~IPaletteHolder();

	static QVector2D smartAdjust(const QVector2D & range,const QHistogram & histo);

	virtual QHistogram computeHistogram(const QVector2D &range, int nBuckets)=0;

	virtual bool hasNoDataValue() const=0;
	virtual float noDataValue() const =0;

	virtual QVector2D dataRange()=0;
	virtual QVector2D rangeRatio()=0;
	virtual QVector2D range()=0;

protected:
	IPaletteHolder();

};


#endif

