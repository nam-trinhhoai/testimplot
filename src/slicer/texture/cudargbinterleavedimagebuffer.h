#ifndef CUDARGBINTERLEAVEDIMAGEBUFFER_H
#define CUDARGBINTERLEAVEDIMAGEBUFFER_H

#include <QReadWriteLock>
#include <QVector2D>

#include <vector>

#include "igeorefimage.h"
#include "imageformats.h"
#include "qhistogram.h"

class IGeorefImage;

//This class holds the buffers of CudaImage.
//It's a way to delegate Mutex protection to this object to handle multi threading update
class CUDARGBInterleavedImageBuffer: public QObject {
Q_OBJECT
public:
	CUDARGBInterleavedImageBuffer(int width, int height, ImageFormats::QSampleType type,
			const IGeorefImage * const transfoProvider, QObject *parent);
	virtual ~CUDARGBInterleavedImageBuffer();

	int width();
	int height();

	//CAUTION:Thread unsafe!!!!
	void lock();
	void* backingPointer();
	const void* constBackingPointer();
	const QByteArray& byteArray();
	void unlock();

	float opacity();
	void setOpacity(float value);

	bool hasNoDataValue();
	float noDataValue();

	QVector2D redRangeRatio();
	QVector2D redRange();
	QVector2D redDataRange();
	void setRedRange(const QVector2D &range);

	QVector2D greenRangeRatio();
	QVector2D greenRange();
	QVector2D greenDataRange();
	void setGreenRange(const QVector2D &range);

	QVector2D blueRangeRatio();
	QVector2D blueRange();
	QVector2D blueDataRange();
	void setBlueRange(const QVector2D &range);

	ImageFormats::QColorFormat colorFormat();
	ImageFormats::QSampleType sampleType();

	void updateTexture(const QByteArray& input, bool byteSwapAndTranspose);
	void updateTexture(const QByteArray& input, bool byteSwapAndTranspose,
			const QVector2D& redCacheRange, const QVector2D& greenCacheRange,
			const QVector2D& blueCacheRange);
	void updateTexture(const QByteArray& input, bool byteSwapAndTranspose,
			const QVector2D& redCacheRange, const QVector2D& greenCacheRange,
			const QVector2D& blueCacheRange, const QHistogram& redHistogram,
			const QHistogram& greenHistogram, const QHistogram& blueHistogram);

	void worldToImage(double worldX, double worldY, double &imageX,
			double &imageY);
	void imageToWorld(double imageX, double imageY, double &worldX,
			double &worldY);
	QMatrix4x4 imageToWorldTransformation();
	bool valueAt(int i, int j, int channel, double &value);
	QRectF worldExtent();

	bool setValue(int i, int j, double value);
	bool value(double worldX, double worldY, int channel, int &i, int &j, double &value);

	size_t internalPointerSizeSafe();

	std::vector<QHistogram> computeHistogram(const QVector2D &redRange,
			const QVector2D &greenRange, const QVector2D &blueRange, int nBuckets);
private:
	size_t internalBufferSize();
	void updateInternalBuffers();

	// return a vector of 3 QHistogram
	std::vector<QVector2D> computeRangeUnsafe();

	void updateRedRangeRatioUnsafe();
	void updateGreenRangeRatioUnsafe();
	void updateBlueRangeRatioUnsafe();
	void updateRangeRatioUnsafe(const QVector2D& range, QVector2D& rangeRatio);
	void initRangeUnsafe();
	void resetRangeAndHistogramUnsafe();

	// return a vector of 3 QHistogram
	std::vector<QHistogram> computeHistogramUnsafe(const QVector2D &redRange,
			const QVector2D &greenRange, const QVector2D &blueRange, int nBuckets);

	void allocateInternalBufferOnDemandUnsafe();
private:
	template<typename InputType>
	struct ComputeRangeUnsafeKernel {
		static void run(CUDARGBInterleavedImageBuffer* obj, float& redX, float& redY,
				float& greenX, float& greenY, float& blueX, float& blueY);
	};
	template<typename InputType>
	struct ComputeHistogramUnsafeKernel {
		static void run(CUDARGBInterleavedImageBuffer* obj, QHistogram& redHist,
				QHistogram& greenHist, QHistogram& blueHist, const QVector2D &redRange,
				const QVector2D &greenRange, const QVector2D &blueRange);
	};
	template<typename InputType>
	struct ByteSwapAndTransposeImageDataKernel {
		static void run(CUDARGBInterleavedImageBuffer* obj, const void* swappingPointer);
	};

	QReadWriteLock m_lock;

	float m_opacity;

	float m_noDataValue;
	bool m_hasNodataValue;

	QVector2D m_redRangeRatio;
	QVector2D m_redRange;
	QVector2D m_redDataRange;
	QVector2D m_greenRangeRatio;
	QVector2D m_greenRange;
	QVector2D m_greenDataRange;
	QVector2D m_blueRangeRatio;
	QVector2D m_blueRange;
	QVector2D m_blueDataRange;
	bool m_dataRangeComputed;

	QHistogram m_redCachedHistogram;
	QHistogram m_greenCachedHistogram;
	QHistogram m_blueCachedHistogram;

	ImageFormats::QColorFormat m_colorFormat;
	ImageFormats::QSampleType m_samplType;

	const IGeorefImage * const m_externalTransfoProvider;
	unsigned int m_width, m_height;

	QByteArray m_backingPointer;
};

#endif
