#ifndef CpuImageBuffer_H
#define CpuImageBuffer_H

#include <QReadWriteLock>
#include <QVector2D>
#include <QByteArray>

#include "lookuptable.h"
#include "ipaletteholder.h"
#include "igeorefimage.h"
#include "imageformats.h"

class IGeorefImage;

//This class holds the buffers of CudaImage.
//It's a way to delegate Mutex protection to this object to handle multi threading update
class CPUImageBuffer: public QObject {
Q_OBJECT
public:
	CPUImageBuffer(int width, int height, ImageFormats::QSampleType type,
			const IGeorefImage * const transfoProvider, QObject *parent);
	virtual ~CPUImageBuffer();

	int width();
	int height();

	//CAUTION:Thread unsafe!!!!
	void lock();
	void* backingPointer();
	const void* constBackingPointer();
	const QByteArray& byteArray();
	void unlock();

	LookupTable lookupTable();
	void setLookupTable(const LookupTable &table);

	float opacity();
	void setOpacity(float value);

	bool hasNoDataValue();
	float noDataValue();

	QVector2D rangeRatio();
	QVector2D range();
	QVector2D dataRange();
	void setRange(const QVector2D &range);

	ImageFormats::QColorFormat colorFormat();
	ImageFormats::QSampleType sampleType();

	QHistogram computeHistogram(const QVector2D &range, int nBuckets);

	void updateTexture(const QByteArray& input, bool byteSwapAndTranspose);
	void updateTexture(const QByteArray& input, bool byteSwapAndTranspose,
			const QVector2D& cacheRange, const QHistogram& cacheHistogram=QHistogram());

	void worldToImage(double worldX, double worldY, double &imageX,
			double &imageY);
	void imageToWorld(double imageX, double imageY, double &worldX,
			double &worldY);
	QMatrix4x4 imageToWorldTransformation();
	bool setValue(int i, int j, double value);
	bool valueAt(int i, int j, double &value);
	void valuesAlongJ(int j, bool *valid, double *values);
	void valuesAlongI(int i, bool *valid, double *values);
	QRectF worldExtent();

	bool value(double worldX, double worldY, int &i, int &j, double &value);

	size_t internalPointerSizeSafe();
private:
	size_t internalBufferSize();
	void updateInternalBuffers();

	QVector2D computeRangeUnsafe();

	void updateRangeRatioUnsafe();
	void initRangeUnsafe();
	void resetRangeAndHistogramUnsafe();

	QHistogram computeHistogramUnsafe(const QVector2D &range, int nBuckets);

	void allocateInternalBufferOnDemandUnsafe();
private:
	template<typename InputType>
	struct ComputeRangeUnsafeKernel {
		static void run(CPUImageBuffer* obj, float& x, float& y);
	};
	template<typename InputType>
	struct ComputeHistogramUnsafeKernel {
		static void run(CPUImageBuffer* obj, unsigned int *hist, const QVector2D &range);
	};
	template<typename InputType>
	struct ByteSwapAndTransposeImageDataKernel {
		static void run(CPUImageBuffer* obj, const void* swappingPointer);
	};

	QReadWriteLock m_lock;
	LookupTable m_lookupTable;

	float m_opacity;

	float m_noDataValue;
	bool m_hasNodataValue;

	QVector2D m_rangeRatio;
	QVector2D m_range;
	QVector2D m_dataRange;
	bool m_dataRangeComputed;

	QHistogram m_cachedHistogram;

	ImageFormats::QColorFormat m_colorFormat;
	ImageFormats::QSampleType m_samplType;

	const IGeorefImage * const m_externalTransfoProvider;
	unsigned int m_width, m_height;

	QByteArray m_backingPointer;
};

#endif
