#ifndef CUDARGBINTERLEAVEDIMAGE_H
#define CUDARGBINTERLEAVEDIMAGE_H

#include <QObject>
#include <QMutex>

#include "qglabstractfullimage.h"
#include "imageformats.h"

#include <memory>

class QOpenGLFunctions;
class QOpenGLTexture;
class CUDARGBInterleavedImageBuffer;

class CUDARGBInterleavedImage;

class CUDARGBInterleavedImageHolder : public IPaletteHolder {
public:
	CUDARGBInterleavedImageHolder(CUDARGBInterleavedImage* image, int channelIndex);
	virtual ~CUDARGBInterleavedImageHolder();

	virtual QHistogram computeHistogram(const QVector2D &range, int nBuckets) override;

	virtual bool hasNoDataValue() const override;
	virtual float noDataValue() const override;

	virtual QVector2D dataRange() override;
	virtual QVector2D rangeRatio() override;
	virtual QVector2D range() override;

private:
	CUDARGBInterleavedImage* m_image = nullptr;
	int m_channelIndex = 0;
};

class CUDARGBInterleavedImage : public QObject,
		public IGeorefGrid {
	Q_OBJECT

	Q_PROPERTY(int width READ width CONSTANT)
	Q_PROPERTY(int height READ height CONSTANT)

	Q_PROPERTY(QVector2D redRangeRatio READ redRangeRatio NOTIFY redRangeChanged)
	Q_PROPERTY(QVector2D greenRangeRatio READ greenRangeRatio NOTIFY greenRangeChanged)
	Q_PROPERTY(QVector2D blueRangeRatio READ blueRangeRatio NOTIFY blueRangeChanged)
	Q_PROPERTY(float opacity READ opacity WRITE setOpacity NOTIFY opacityChanged)

public:
	CUDARGBInterleavedImage(int width, int height,
				ImageFormats::QSampleType type = ImageFormats::QSampleType::INT16,
				const IGeorefImage *const transfoProvider = nullptr,
				QObject *parent = 0);
	virtual ~CUDARGBInterleavedImage();

	size_t internalPointerSize();
	float opacity() const;

	bool hasNoDataValue() const;
	float noDataValue() const;

	ImageFormats::QColorFormat colorFormat() const;
	ImageFormats::QSampleType sampleType() const;

	bool setValue(int i, int j, double value);

	bool value(double worldX, double worldY, int channel, int &i, int &j,
			double &value) const;

	//IPaletteHolder
	QVector2D redRangeRatio();
	QVector2D redRange();
	QVector2D redDataRange();
	QVector2D greenRangeRatio();
	QVector2D greenRange();
	QVector2D greenDataRange();
	QVector2D blueRangeRatio();
	QVector2D blueRange();
	QVector2D blueDataRange();

	CUDARGBInterleavedImageHolder* redHolder();
	CUDARGBInterleavedImageHolder* greenHolder();
	CUDARGBInterleavedImageHolder* blueHolder();

	CUDARGBInterleavedImageHolder* holder(int i);

	int width() const override;
	int height() const override;

	//IGeorefImage
	virtual void worldToImage(double worldX, double worldY, double &imageX,
			double &imageY) const override;
	virtual void imageToWorld(double imageX, double imageY, double &worldX,
			double &worldY) const override;
	virtual QMatrix4x4 imageToWorldTransformation() const override;

	QRectF worldExtent() const override;

	bool valueAt(int i, int j, int channel, double &value) const;

	//Unsafe operation (use with a lot of caution)
	void lockPointer();
	void* backingPointer();
	const void* constBackingPointer();
	const QByteArray& byteArray();
	void unlockPointer();

	void updateTexture(const QByteArray& input, bool byteSwapAndTranspose);
	void updateTexture(const QByteArray& input, bool byteSwapAndTranspose,
			const QVector2D& redCacheRange, const QVector2D& greenCacheRange,
			const QVector2D& blueCacheRange);
	void updateTexture(const QByteArray& input, bool byteSwapAndTranspose,
			const QVector2D& redCacheRange, const QVector2D& greenCacheRange,
			const QVector2D& blueCacheRange, const QHistogram& redHistogram,
			const QHistogram& greenHistogram, const QHistogram& blueHistogram);

	QVector<IPaletteHolder*> holders() const;

	std::vector<QHistogram> computeHistogram(const QVector2D &redRange,
			const QVector2D &greenRange, const QVector2D &blueRange, int nBuckets);

signals:
	void opacityChanged(float val);
	void redRangeChanged(const QVector2D &range);
	void greenRangeChanged(const QVector2D &range);
	void blueRangeChanged(const QVector2D &range);
	void rangeChanged(unsigned int ,const QVector2D & );

	void dataChanged();

public slots:
	void setOpacity(float value);
	void setRedRange(const QVector2D &range);
	void setGreenRange(const QVector2D &range);
	void setBlueRange(const QVector2D &range);
	void setRange(unsigned int ,const QVector2D & );

private:
	CUDARGBInterleavedImageBuffer *m_buffer;

	std::unique_ptr<CUDARGBInterleavedImageHolder> m_redHolder, m_greenHolder, m_blueHolder;
};

#endif
