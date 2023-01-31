#include "cudargbinterleavedimagebuffer.h"
#include <QDebug>
#include <iostream>
#include <QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QElapsedTimer>
#include <QOpenGLPixelTransferOptions>

#include "cuda_common_helpers.h"
#include "cuda_algo.h"
#include "cuda_volume.h"
#include "texturehelper.h"
#include "cudaimagebuffer.h"
#include "sampletypebinder.h"

#include <omp.h>

CUDARGBInterleavedImageBuffer::CUDARGBInterleavedImageBuffer(int width, int height,
		ImageFormats::QSampleType type,
		const IGeorefImage *const transfoProvider, QObject *parent) :
		QObject(parent), m_externalTransfoProvider(transfoProvider) {
	m_width = width;
	m_height = height;

	m_samplType = type;
	m_colorFormat = ImageFormats::QColorFormat::RGB_INTERLEAVED;

	m_opacity = 1.0;
	m_noDataValue = 0;
	m_hasNodataValue = false;

	m_redRangeRatio = QVector2D(0, 1);
	m_redRange = QVector2D(0, 1);
	m_redDataRange = QVector2D(0, 1);

	m_greenRangeRatio = QVector2D(0, 1);
	m_greenRange = QVector2D(0, 1);
	m_greenDataRange = QVector2D(0, 1);

	m_blueRangeRatio = QVector2D(0, 1);
	m_blueRange = QVector2D(0, 1);
	m_blueDataRange = QVector2D(0, 1);
	m_dataRangeComputed = false;
}

void CUDARGBInterleavedImageBuffer::allocateInternalBufferOnDemandUnsafe()
{
	if(m_backingPointer.size()!=0)
		return;

	size_t pointerSize = internalBufferSize();

	m_backingPointer.resize(pointerSize);
	memset(m_backingPointer.data(), 0, pointerSize);
}

void CUDARGBInterleavedImageBuffer::resetRangeAndHistogramUnsafe() {
	m_dataRangeComputed = false;
	m_redCachedHistogram=QHistogram();
	m_greenCachedHistogram=QHistogram();
	m_blueCachedHistogram=QHistogram();
}

void CUDARGBInterleavedImageBuffer::initRangeUnsafe() {
	if (m_dataRangeComputed)
		return;

	std::vector<QVector2D> ranges = computeRangeUnsafe();
	m_redDataRange = ranges[0];
	m_greenDataRange = ranges[1];
	m_blueDataRange = ranges[2];
	if (m_redCachedHistogram.range() != m_redDataRange ||
			m_greenCachedHistogram.range() != m_greenDataRange ||
			m_blueCachedHistogram.range() != m_blueDataRange) {
		std::vector<QHistogram> histos = computeHistogramUnsafe(m_redDataRange,
				m_greenDataRange, m_blueDataRange, QHistogram::HISTOGRAM_SIZE);
		m_redCachedHistogram = histos[0];
		m_greenCachedHistogram = histos[1];
		m_blueCachedHistogram = histos[2];
	}
	m_redRange = IPaletteHolder::smartAdjust(m_redDataRange, m_redCachedHistogram);
	m_greenRange = IPaletteHolder::smartAdjust(m_greenDataRange, m_greenCachedHistogram);
	m_blueRange = IPaletteHolder::smartAdjust(m_blueDataRange, m_blueCachedHistogram);
	updateRedRangeRatioUnsafe();
	updateGreenRangeRatioUnsafe();
	updateBlueRangeRatioUnsafe();
	m_dataRangeComputed = true;
}

void CUDARGBInterleavedImageBuffer::updateRedRangeRatioUnsafe() {
	updateRangeRatioUnsafe(m_redRange, m_redRangeRatio);
}

void CUDARGBInterleavedImageBuffer::updateGreenRangeRatioUnsafe() {
	updateRangeRatioUnsafe(m_greenRange, m_greenRangeRatio);
}

void CUDARGBInterleavedImageBuffer::updateBlueRangeRatioUnsafe() {
	updateRangeRatioUnsafe(m_blueRange, m_blueRangeRatio);
}

void CUDARGBInterleavedImageBuffer::updateRangeRatioUnsafe(const QVector2D& range, QVector2D& rangeRatio) {
	rangeRatio.setX(range.x());
	if (range.y() - range.x() != 0) {
		rangeRatio.setY(1.0f / (range.y() - range.x()));
	}
}

QVector2D CUDARGBInterleavedImageBuffer::redDataRange() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_redDataRange;
	m_lock.unlock();
	return r;
}

QVector2D CUDARGBInterleavedImageBuffer::redRange() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_redRange;
	m_lock.unlock();
	return r;
}

QVector2D CUDARGBInterleavedImageBuffer::redRangeRatio() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_redRangeRatio;
	m_lock.unlock();
	return r;
}

void CUDARGBInterleavedImageBuffer::setRedRange(const QVector2D &range) {
	m_lock.lockForWrite();
	m_redRange = range;
	updateRedRangeRatioUnsafe();
	m_lock.unlock();
}

QVector2D CUDARGBInterleavedImageBuffer::greenDataRange() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_greenDataRange;
	m_lock.unlock();
	return r;
}

QVector2D CUDARGBInterleavedImageBuffer::greenRange() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_greenRange;
	m_lock.unlock();
	return r;
}

QVector2D CUDARGBInterleavedImageBuffer::greenRangeRatio() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_greenRangeRatio;
	m_lock.unlock();
	return r;
}

void CUDARGBInterleavedImageBuffer::setGreenRange(const QVector2D &range) {
	m_lock.lockForWrite();
	m_greenRange = range;
	updateGreenRangeRatioUnsafe();
	m_lock.unlock();
}

QVector2D CUDARGBInterleavedImageBuffer::blueDataRange() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_blueDataRange;
	m_lock.unlock();
	return r;
}

QVector2D CUDARGBInterleavedImageBuffer::blueRange() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_blueRange;
	m_lock.unlock();
	return r;
}

QVector2D CUDARGBInterleavedImageBuffer::blueRangeRatio() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_blueRangeRatio;
	m_lock.unlock();
	return r;
}

void CUDARGBInterleavedImageBuffer::setBlueRange(const QVector2D &range) {
	m_lock.lockForWrite();
	m_blueRange = range;
	updateBlueRangeRatioUnsafe();
	m_lock.unlock();
}

ImageFormats::QColorFormat CUDARGBInterleavedImageBuffer::colorFormat() {
	m_lock.lockForRead();
	ImageFormats::QColorFormat f = m_colorFormat;
	m_lock.unlock();
	return f;
}

ImageFormats::QSampleType CUDARGBInterleavedImageBuffer::sampleType() {
	m_lock.lockForRead();
	ImageFormats::QSampleType type = m_samplType;
	m_lock.unlock();
	return type;
}

void CUDARGBInterleavedImageBuffer::lock() {
	m_lock.lockForWrite();
}

void* CUDARGBInterleavedImageBuffer::backingPointer() {
	allocateInternalBufferOnDemandUnsafe();
	return m_backingPointer.data();
}

const void* CUDARGBInterleavedImageBuffer::constBackingPointer() {
	allocateInternalBufferOnDemandUnsafe();
	return m_backingPointer.constData();
}

const QByteArray& CUDARGBInterleavedImageBuffer::byteArray() {
	allocateInternalBufferOnDemandUnsafe();
	return m_backingPointer;
}

void CUDARGBInterleavedImageBuffer::unlock() {
	m_lock.unlock();
}

float CUDARGBInterleavedImageBuffer::opacity() {
	m_lock.lockForRead();
	float val = m_opacity;
	m_lock.unlock();
	return val;
}

void CUDARGBInterleavedImageBuffer::setOpacity(float value) {
	m_lock.lockForWrite();
	m_opacity = value;
	m_lock.unlock();
}

bool CUDARGBInterleavedImageBuffer::hasNoDataValue() {
	m_lock.lockForRead();
	bool val = m_hasNodataValue;
	m_lock.unlock();
	return val;
}
float CUDARGBInterleavedImageBuffer::noDataValue() {
	m_lock.lockForRead();
	float val = m_noDataValue;
	m_lock.unlock();
	return val;
}

int CUDARGBInterleavedImageBuffer::width() {
	m_lock.lockForRead();
	int val = m_width;
	m_lock.unlock();
	return val;
}
int CUDARGBInterleavedImageBuffer::height() {
	m_lock.lockForRead();
	int val = m_height;
	m_lock.unlock();
	return val;
}

CUDARGBInterleavedImageBuffer::~CUDARGBInterleavedImageBuffer() {
	// to allow on going process to finish
	m_lock.lockForWrite();
	m_lock.unlock();
}

size_t CUDARGBInterleavedImageBuffer::internalBufferSize() {
	size_t pointerSize = 3 * m_width * m_height * m_samplType.byte_size();
	return pointerSize;
}

size_t CUDARGBInterleavedImageBuffer::internalPointerSizeSafe() {
	m_lock.lockForRead();
	size_t pointerSize = internalBufferSize();
	m_lock.unlock();
	return pointerSize;
}

template<typename InputType>
void CUDARGBInterleavedImageBuffer::ByteSwapAndTransposeImageDataKernel<InputType>::run(
		CUDARGBInterleavedImageBuffer* obj, const void* _swappingPointer) {
//	byteSwapAndTransposeImageData((InputType*) obj->m_cudaPointer,
//					(InputType*) swappingPointer, obj->m_width, obj->m_height, 3);
	const InputType* swappingPointer = static_cast<const InputType*>(_swappingPointer);
	InputType* backingPointer = static_cast<InputType*>(static_cast<void*>(obj->m_backingPointer.data()));
#pragma omp parallel for
	for (long j=0; j<obj->m_height; j++) {
		for (long i=0; i<obj->m_width; i++) {
			for (long c = 0; c<3; c++) {
				backingPointer[(j*obj->m_width + i)*3 + c] = swappingPointer[(obj->m_height*i + j)*3 + c];
			}
		}
	}
}

void CUDARGBInterleavedImageBuffer::updateTexture(const QByteArray& input, bool byteSwapAndTranspose) {
	m_lock.lockForWrite();
	allocateInternalBufferOnDemandUnsafe();
	size_t pointerSize = internalBufferSize();
	bool doByteSwap = byteSwapAndTranspose;

	if (doByteSwap) {
		const void *swappingPointer = input.constData();
		SampleTypeBinder binder(m_samplType);
		binder.bind<ByteSwapAndTransposeImageDataKernel>(this, swappingPointer);
	} else {
		m_backingPointer = input;
	}

	resetRangeAndHistogramUnsafe();
	m_lock.unlock();
}

void CUDARGBInterleavedImageBuffer::updateTexture(const QByteArray& input, bool byteSwapAndTranspose,
		const QVector2D& redCacheRange, const QVector2D& greenCacheRange, const QVector2D& blueCacheRange) {
	m_lock.lockForWrite();
	allocateInternalBufferOnDemandUnsafe();
	size_t pointerSize = internalBufferSize();
	bool doByteSwap = byteSwapAndTranspose;

	if (doByteSwap) {
		const void *swappingPointer = input.constData();
		SampleTypeBinder binder(m_samplType);
		binder.bind<ByteSwapAndTransposeImageDataKernel>(this, swappingPointer);
	} else {
		m_backingPointer = input;
	}

	resetRangeAndHistogramUnsafe();
	m_redDataRange = redCacheRange;
	m_redRange = m_redDataRange;
	m_greenDataRange = greenCacheRange;
	m_greenRange = m_greenDataRange;
	m_blueDataRange = blueCacheRange;
	m_blueRange = m_blueDataRange;
	m_redCachedHistogram = QHistogram();
	m_greenCachedHistogram = QHistogram();
	m_blueCachedHistogram = QHistogram();
	updateRedRangeRatioUnsafe();
	updateGreenRangeRatioUnsafe();
	updateBlueRangeRatioUnsafe();
	m_dataRangeComputed = true;
	m_lock.unlock();
}

void CUDARGBInterleavedImageBuffer::updateTexture(const QByteArray& input, bool byteSwapAndTranspose,
		const QVector2D& redCacheRange, const QVector2D& greenCacheRange, const QVector2D& blueCacheRange,
		const QHistogram& redHistogram, const QHistogram& greenHistogram, const QHistogram& blueHistogram) {
	m_lock.lockForWrite();
	allocateInternalBufferOnDemandUnsafe();
	size_t pointerSize = internalBufferSize();
	bool doByteSwap = byteSwapAndTranspose;

	if (doByteSwap) {
		const void *swappingPointer = input.constData();
		SampleTypeBinder binder(m_samplType);
		binder.bind<ByteSwapAndTransposeImageDataKernel>(this, swappingPointer);
	} else {
		m_backingPointer = input;
	}

	resetRangeAndHistogramUnsafe();
	m_redDataRange = redCacheRange;
	m_redRange = m_redDataRange;
	m_greenDataRange = greenCacheRange;
	m_greenRange = m_greenDataRange;
	m_blueDataRange = blueCacheRange;
	m_blueRange = m_blueDataRange;
	m_redCachedHistogram = redHistogram;
	m_greenCachedHistogram = greenHistogram;
	m_blueCachedHistogram = blueHistogram;
	updateRedRangeRatioUnsafe();
	updateGreenRangeRatioUnsafe();
	updateBlueRangeRatioUnsafe();
	m_dataRangeComputed = true;
	m_lock.unlock();
}

template<typename InputType>
void CUDARGBInterleavedImageBuffer::ComputeRangeUnsafeKernel<InputType>::run(CUDARGBInterleavedImageBuffer* obj,
		float& redX, float& redY, float& greenX, float& greenY, float& blueX, float& blueY) {
	redX = std::numeric_limits<InputType>::max();
	redY = std::numeric_limits<InputType>::lowest();
	greenX = std::numeric_limits<InputType>::max();
	greenY = std::numeric_limits<InputType>::lowest();
	blueX = std::numeric_limits<InputType>::max();
	blueY = std::numeric_limits<InputType>::lowest();
//	computeMinMaxOptimizedMultiChannels((InputType*) obj->m_cudaPointer, obj->m_width * obj->m_height,
//			ranges.data(), countChannels);
	InputType* backingPointer = static_cast<InputType*>(static_cast<void*>(obj->m_backingPointer.data()));
#pragma omp parallel for
	for (long j=0; j<obj->m_height; j++) {
		for (long i=0; i<obj->m_width; i++) {
			InputType valRed = backingPointer[(j*obj->m_width + i)*3];
			InputType valGreen = backingPointer[(j*obj->m_width + i)+1];
			InputType valBlue = backingPointer[(j*obj->m_width + i)*3+2];
			if (valRed<redX) {
#pragma omp critical
				if (valRed<redX) {
					redX = valRed;
				}
			}
			if (valRed>redY) {
#pragma omp critical
				if (valRed>redY) {
					redY = valRed;
				}
			}
			if (valGreen<greenX) {
#pragma omp critical
				if (valGreen<greenX) {
					greenX = valGreen;
				}
			}
			if (valGreen>greenY) {
#pragma omp critical
				if (valGreen>greenY) {
					greenY = valGreen;
				}
			}
			if (valBlue<blueX) {
#pragma omp critical
				if (valBlue<blueX) {
					blueX = valBlue;
				}
			}
			if (valBlue>blueY) {
#pragma omp critical
				if (valBlue>blueY) {
					blueY = valBlue;
				}
			}
		}
	}
}

std::vector<QVector2D> CUDARGBInterleavedImageBuffer::computeRangeUnsafe() {
	allocateInternalBufferOnDemandUnsafe();
	float redX, redY, greenX, greenY, blueX, blueY;
	SampleTypeBinder binder(m_samplType);
	binder.bind<ComputeRangeUnsafeKernel>(this, redX, redY, greenX, greenY, blueX, blueY);

	std::vector<QVector2D> out;
	out.push_back(QVector2D(redX, redY));
	out.push_back(QVector2D(greenX, greenY));
	out.push_back(QVector2D(blueX, blueY));
	return out;
}

template<typename InputType>
void CUDARGBInterleavedImageBuffer::ComputeHistogramUnsafeKernel<InputType>::run(
		CUDARGBInterleavedImageBuffer* obj, QHistogram& redHist,
		QHistogram& greenHist, QHistogram& blueHist, const QVector2D &redRange,
		const QVector2D &greenRange, const QVector2D &blueRange) {
	bool computeRedHisto = true;
	if (qFuzzyCompare(redRange.y(), redRange.x())) {
		for (int i=0; i<256; i++) {
			redHist[i] = 0;
		}
		computeRedHisto = false;
	}
	bool computeGreenHisto = true;
	if (qFuzzyCompare(greenRange.y(), greenRange.x())) {
		for (int i=0; i<256; i++) {
			greenHist[i] = 0;
		}
		computeGreenHisto = false;
	}
	bool computeBlueHisto = true;
	if (qFuzzyCompare(blueRange.y(), blueRange.x())) {
		for (int i=0; i<256; i++) {
			blueHist[i] = 0;
		}
		computeBlueHisto = false;
	}
	if (!computeRedHisto && !computeGreenHisto && !computeBlueHisto) {
		return;
	}

	int threadCount = omp_get_max_threads();
	std::vector<std::vector<std::vector<unsigned int>>> hists;
	std::vector<unsigned int> initHist;
	initHist.resize(256, 0);
	std::vector<std::vector<unsigned int>> initHists;
	initHists.resize(3, initHist);
	hists.resize(threadCount, initHists);
	//qDebug()<<"ComputeHistogramUnsafeKernel";
	std::vector<QVector2D> ranges;
	ranges.push_back(redRange);
	ranges.push_back(greenRange);
	ranges.push_back(blueRange);
	std::vector<bool> valid;
	valid.push_back(computeRedHisto);
	valid.push_back(computeGreenHisto);
	valid.push_back(computeBlueHisto);

	const InputType* backingPointer = static_cast<const InputType*>(static_cast<const void*>(obj->m_backingPointer.constData()));

	#pragma omp parallel
	{
		int threadId = omp_get_thread_num();
		if (threadId<threadCount) {
			std::vector<std::vector<unsigned int>>& histThread = hists[threadId];
			#pragma omp for
			for (long j=0; j<obj->m_height; j++) {
				for (long c=0; c<3; c++) {
					if (valid[c]) {
						for (long i=0; i<obj->m_width; i++) {
							float val = backingPointer[(j*obj->m_width + i)*3+c];
							int index = 0;
							if (val>ranges[c].y()) {
								index = 255;
							} else if (val<ranges[c].x()) {
								index = 0;
							} else {
								index = (val - ranges[c].x()) / (ranges[c].y() - ranges[c].x()) * 255;
							}
							histThread[c][index]++;
						}
					}
				}
			}
		}
	}

	for (int c=0; c<3; c++) {
		if (valid[c]) {
			QHistogram* hist;
			if (c==0) {
				hist = &redHist;
			} else if (c==1) {
				hist = &greenHist;
			} else {
				hist = &blueHist;
			}
			for (int i=0; i<256; i++) {
				unsigned int count = 0;
				for (int j=0; j<threadCount; j++) {
					count += hists[j][c][i];
				}
				(*hist)[i] = count;
			}
		}
	}
}

std::vector<QHistogram> CUDARGBInterleavedImageBuffer::computeHistogramUnsafe(const QVector2D &redRange,
		const QVector2D &greenRange, const QVector2D &blueRange, int nBuckets) {
	allocateInternalBufferOnDemandUnsafe();

	if (nBuckets != 256)
		std::cerr
				<< "Histogram length is not compatible with QGLGraphicsItem (expected 256 got "
				<< nBuckets << ")" << std::endl;

	int hSize = 256;
	size_t length = hSize * sizeof(unsigned int);

	QHistogram redHisto, greenHisto, blueHisto;

	SampleTypeBinder binder(m_samplType);
	binder.bind<ComputeHistogramUnsafeKernel>(this, redHisto, greenHisto, blueHisto,
			redRange, greenRange, blueRange);

	redHisto.setRange(redRange);
	greenHisto.setRange(greenRange);
	blueHisto.setRange(blueRange);

	std::vector<QHistogram> histos;
	histos.push_back(redHisto);
	histos.push_back(greenHisto);
	histos.push_back(blueHisto);
	return histos;
}

std::vector<QHistogram> CUDARGBInterleavedImageBuffer::computeHistogram(const QVector2D &redRange,
		const QVector2D &greenRange, const QVector2D &blueRange,
		int nBuckets) {
	m_lock.lockForRead();
	std::vector<QHistogram> histos;
	if (m_redCachedHistogram.range() != redRange ||
			m_greenCachedHistogram.range() != greenRange ||
			m_blueCachedHistogram.range() != blueRange) {
		histos = computeHistogramUnsafe(redRange, greenRange, blueRange, nBuckets);
		m_redCachedHistogram = histos[0];
		m_greenCachedHistogram = histos[1];
		m_blueCachedHistogram = histos[2];
	} else {
		histos.push_back(m_redCachedHistogram);
		histos.push_back(m_greenCachedHistogram);
		histos.push_back(m_blueCachedHistogram);
	}

	m_lock.unlock();
	return histos;
}

void CUDARGBInterleavedImageBuffer::worldToImage(double worldX, double worldY, double &imageX,
		double &imageY) {
	m_lock.lockForRead();
	if (m_externalTransfoProvider != nullptr) {
		m_externalTransfoProvider->worldToImage(worldX, worldY, imageX, imageY);
	} else {
		imageX = worldX;
		imageY = worldY;
	}
	m_lock.unlock();
}

void CUDARGBInterleavedImageBuffer::imageToWorld(double imageX, double imageY, double &worldX,
		double &worldY) {
	m_lock.lockForRead();
	if (m_externalTransfoProvider != nullptr) {
		m_externalTransfoProvider->imageToWorld(imageX, imageY, worldX, worldY);
	} else {
		worldX = imageX;
		worldY = imageY;
	}
	m_lock.unlock();
}
QMatrix4x4 CUDARGBInterleavedImageBuffer::imageToWorldTransformation() {
	QMatrix4x4 id;
	id.setToIdentity();
	m_lock.lockForRead();
	if (m_externalTransfoProvider != nullptr)
		id = m_externalTransfoProvider->imageToWorldTransformation();
	m_lock.unlock();
	return id;
}

bool CUDARGBInterleavedImageBuffer::valueAt(int i, int j, int channel, double &value) {
	if (channel<0 || channel>2) {
		return false;
	}
	m_lock.lockForRead();
	allocateInternalBufferOnDemandUnsafe();
	bool val = TextureHelper::valueAt(m_backingPointer.constData(), i*3+channel+channel, j, m_width*3,
			m_samplType, value);
	m_lock.unlock();
	return val;
}

bool CUDARGBInterleavedImageBuffer::setValue(int i, int j, double value)
{
	bool val;
	for (int channel=0; channel <3; channel ++)
	{
		m_lock.lockForRead();
		allocateInternalBufferOnDemandUnsafe();
		bool val = TextureHelper::setValue(m_backingPointer.data(), i*3+channel+channel, j, m_width*3,
				m_samplType, value);
		m_lock.unlock();
	}
	return val;
}

bool CUDARGBInterleavedImageBuffer::value(double worldX, double worldY, int channel, int &i, int &j,
		double &value) {
	if (channel<0 || channel>2) {
		return false;
	}
	//Fully reimplemented to lock while gathering data information to avoid an asynchronous in beteween and still be coherent
	m_lock.lockForRead();
	allocateInternalBufferOnDemandUnsafe();
	double di, dj;
	if (m_externalTransfoProvider != nullptr) {
		m_externalTransfoProvider->worldToImage(worldX, worldY, di, dj);
	} else {
		di = worldX;
		dj = worldY;
	}

	i = (int) di;
	j = (int) dj;

	bool val = false;
	if (i >= 0 && j >= 0 && i < m_width && j < m_height) {
		val = TextureHelper::valueAt(m_backingPointer.constData(), i*3+channel, j, m_width*3,
				m_samplType, value);
	}

	m_lock.unlock();
	return val;
}

QRectF CUDARGBInterleavedImageBuffer::worldExtent() {
	m_lock.lockForRead();
	double ij[8] = { 0.0, 0.0, (double) m_width, 0.0, 0.0, (double) m_height,
			(double) m_width, (double) m_height };

	double xMin = std::numeric_limits<double>::max();
	double yMin = std::numeric_limits<double>::max();

	double xMax = std::numeric_limits<double>::min();
	double yMax = std::numeric_limits<double>::min();
	double x, y;
	for (int i = 0; i < 4; i++) {
		x = ij[2 * i];
		y = ij[2 * i + 1];
		if (m_externalTransfoProvider != nullptr)
			m_externalTransfoProvider->imageToWorld(ij[2 * i], ij[2 * i + 1], x,
					y);
		imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);

		xMin = std::min(xMin, x);
		yMin = std::min(yMin, y);

		xMax = std::max(xMax, x);
		yMax = std::max(yMax, y);
	}
	QRectF val(xMin, yMin, xMax - xMin, yMax - yMin);
	m_lock.unlock();
	return val;
}

