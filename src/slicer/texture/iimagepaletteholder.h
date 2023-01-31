#ifndef IIMAGEPALTTEHOLDER_H
#define IIMAGEPALTTEHOLDER_H

#include "qglabstractfullimage.h"
#include "imageformats.h"
#include "lookuptable.h"

class IImagePaletteHolder : public QObject,
		public IPaletteHolder,
		public IGeorefImage {
	Q_OBJECT
public:
	IImagePaletteHolder(QObject* parent=nullptr);
	virtual ~IImagePaletteHolder();

	virtual size_t internalPointerSize() = 0;

	virtual void lockPointer() = 0;
	virtual void* backingPointer() = 0;
	virtual const void* constBackingPointer() = 0;
	virtual QByteArray getDataAsByteArray() = 0;
	virtual void unlockPointer() = 0;

	virtual ImageFormats::QColorFormat colorFormat() const = 0;
	virtual ImageFormats::QSampleType sampleType() const = 0;

	virtual LookupTable lookupTable() const = 0;
	virtual float opacity() const = 0;

signals:
	void opacityChanged();
	void rangeChanged();
	void lookupTableChanged();

	void dataChanged();

public slots:
	virtual void setOpacity(float value) = 0;
	virtual void setLookupTable(const LookupTable &table) = 0;
	virtual void setRange(const QVector2D &range) = 0;
};

#endif
