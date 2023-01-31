#ifndef CPUImagePaletteHolder_H
#define CPUImagePaletteHolder_H

#include <QWidget>
#include "qglabstractfullimage.h"
#include "imageformats.h"
#include "lookuptable.h"
#include "iimagepaletteholder.h"

class QOpenGLFunctions;
class QOpenGLTexture;
#include <QMutex>
class CPUImageBuffer;

class CPUImagePaletteHolder: public IImagePaletteHolder {
Q_OBJECT

Q_PROPERTY(int width READ width CONSTANT)
Q_PROPERTY(int height READ height CONSTANT)

Q_PROPERTY(LookupTable lookupTable READ lookupTable WRITE setLookupTable NOTIFY lookupTableChanged)
Q_PROPERTY(QVector2D rangeRatio READ rangeRatio NOTIFY rangeChanged)
Q_PROPERTY(float opacity READ opacity WRITE setOpacity NOTIFY opacityChanged)
public:
	CPUImagePaletteHolder(int width, int height,
			ImageFormats::QSampleType type = ImageFormats::QSampleType::INT16,
			const IGeorefImage *const transfoProvider = nullptr,
			QObject *parent = 0);
	virtual ~CPUImagePaletteHolder();

	virtual size_t internalPointerSize() override;

	//LUT handling
	virtual LookupTable lookupTable() const override;
	virtual float opacity() const override;

	bool hasNoDataValue() const;
	float noDataValue() const;

	virtual ImageFormats::QColorFormat colorFormat() const override;
	virtual ImageFormats::QSampleType sampleType() const override;

	bool value(double worldX, double worldY, int &i, int &j,
			double &value) const;

	//IPaletteHolder
	virtual QVector2D rangeRatio()  override;
	virtual QVector2D range()  override;
	virtual QVector2D dataRange()  override;
	virtual QHistogram computeHistogram(const QVector2D &range,
			int nBuckets) override;

	int width() const override;
	int height() const override;

	//IGeorefImage
	virtual void worldToImage(double worldX, double worldY, double &imageX,
			double &imageY) const override;
	virtual void imageToWorld(double imageX, double imageY, double &worldX,
			double &worldY) const override;
	virtual QMatrix4x4 imageToWorldTransformation() const override;

	QRectF worldExtent() const override;

	bool setValue(int i, int j, double value);
	bool valueAt(int i, int j, double &value) const override;
	void valuesAlongJ(int j, bool *valid, double *values) const override;
	void valuesAlongI(int i, bool *valid, double *values) const override;

	//Unsafe operation (use with a lot of caution)
	virtual void lockPointer() override;
	virtual void* backingPointer() override;
	virtual const void* constBackingPointer() override;
	virtual QByteArray getDataAsByteArray() override;
	virtual void unlockPointer() override;

	void updateTexture(const QByteArray&input, bool byteSwapAndTranspose);
	void updateTexture(const QByteArray& input, bool byteSwapAndTranspose,
			const QVector2D& cacheRange, const QHistogram& cacheHistogram=QHistogram());

signals:
	void opacityChanged(float);
	void opacityChanged();
	void lookupTableChanged(const LookupTable &table);
	void lookupTableChanged();
	void rangeChanged(const QVector2D& range);
	void rangeChanged();

public slots:
	virtual void setOpacity(float value) override;
	virtual void setLookupTable(const LookupTable &table) override;
	virtual void setRange(const QVector2D &range) override;

private:
	CPUImageBuffer *m_buffer;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_IMAGEITEM_GPUIMAGEPALETTEHOLDER_H_ */
