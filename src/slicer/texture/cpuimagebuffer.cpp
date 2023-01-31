#include "cpuimagebuffer.h"
#include <QDebug>
#include <iostream>
#include <QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QElapsedTimer>
#include <QOpenGLPixelTransferOptions>
#include <omp.h>

#include "texturehelper.h"
#include "cpuimagebuffer.h"
#include "colortableregistry.h"
#include "sampletypebinder.h"

CPUImageBuffer::CPUImageBuffer(int width, int height,
		ImageFormats::QSampleType type,
		const IGeorefImage *const transfoProvider, QObject *parent) :
		QObject(parent), m_externalTransfoProvider(transfoProvider) {
	m_width = width;
	m_height = height;

	m_samplType = type;
	m_colorFormat = ImageFormats::QColorFormat::GRAY;

	m_opacity = 1.0;
	m_noDataValue = 0;
	m_hasNodataValue = false;

	m_rangeRatio = QVector2D(0, 1);
	m_range = QVector2D(0, 1);
	m_dataRange = QVector2D(0, 1);
	m_dataRangeComputed = false;
	m_lookupTable = ColorTableRegistry::DEFAULT();
}

void CPUImageBuffer::allocateInternalBufferOnDemandUnsafe()
{
	if(m_backingPointer!=nullptr)
		return;

	size_t pointerSize = internalBufferSize();
	m_backingPointer.resize(pointerSize);
	/*qDebug() << "CPU try allocate "<<pointerSize << ", allocate " <<
			m_backingPointer.size() <<  ", diff " <<
			pointerSize-m_backingPointer.size();*/
	memset(m_backingPointer.data(), 0, pointerSize);
}

LookupTable CPUImageBuffer::lookupTable() {
	m_lock.lockForRead();
	LookupTable internalLut = m_lookupTable;
	m_lock.unlock();
	return internalLut;
}

void CPUImageBuffer::setLookupTable(const LookupTable &table) {
	m_lock.lockForWrite();
	m_lookupTable = table;
	m_lock.unlock();
}


void CPUImageBuffer::resetRangeAndHistogramUnsafe() {
	m_dataRangeComputed = false;
	m_cachedHistogram=QHistogram();
}

void CPUImageBuffer::initRangeUnsafe() {
	if (m_dataRangeComputed)
		return;

	m_dataRange = computeRangeUnsafe();
	if (m_cachedHistogram.range() != m_dataRange)
		m_cachedHistogram = computeHistogramUnsafe(m_dataRange,
				QHistogram::HISTOGRAM_SIZE);
	m_range = IPaletteHolder::smartAdjust(m_dataRange, m_cachedHistogram);
	updateRangeRatioUnsafe();
	m_dataRangeComputed = true;
}

void CPUImageBuffer::updateRangeRatioUnsafe() {
	m_rangeRatio.setX(m_range.x());
	if (m_range.y() - m_range.x() != 0) {
		m_rangeRatio.setY(1.0f / (m_range.y() - m_range.x()));
	}
}


QVector2D CPUImageBuffer::dataRange() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_dataRange;
	m_lock.unlock();
	return r;
}

QVector2D CPUImageBuffer::range() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_range;
	m_lock.unlock();
	return r;
}

QVector2D CPUImageBuffer::rangeRatio() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_rangeRatio;
	m_lock.unlock();
	return r;
}

void CPUImageBuffer::setRange(const QVector2D &range) {
	m_lock.lockForWrite();
	m_range = range;
	updateRangeRatioUnsafe();
	m_lock.unlock();
}

ImageFormats::QColorFormat CPUImageBuffer::colorFormat() {
	m_lock.lockForRead();
	ImageFormats::QColorFormat f = m_colorFormat;
	m_lock.unlock();
	return f;
}

ImageFormats::QSampleType CPUImageBuffer::sampleType() {
	m_lock.lockForRead();
	ImageFormats::QSampleType type = m_samplType;
	m_lock.unlock();
	return type;
}

void CPUImageBuffer::lock() {
	m_lock.lockForWrite();
}

void* CPUImageBuffer::backingPointer() {
	allocateInternalBufferOnDemandUnsafe();
	return static_cast<void*>(m_backingPointer.data());
}

const void* CPUImageBuffer::constBackingPointer() {
	allocateInternalBufferOnDemandUnsafe();
	return static_cast<const void*>(m_backingPointer.constData());
}

const QByteArray& CPUImageBuffer::byteArray() {
	allocateInternalBufferOnDemandUnsafe();
	return m_backingPointer;
}

void CPUImageBuffer::unlock() {
	m_lock.unlock();
}

float CPUImageBuffer::opacity() {
	m_lock.lockForRead();
	float val = m_opacity;
	m_lock.unlock();
	return val;
}

void CPUImageBuffer::setOpacity(float value) {
	m_lock.lockForWrite();
	m_opacity = value;
	m_lock.unlock();
}

bool CPUImageBuffer::hasNoDataValue() {
	m_lock.lockForRead();
	bool val = m_hasNodataValue;
	m_lock.unlock();
	return val;
}
float CPUImageBuffer::noDataValue() {
	m_lock.lockForRead();
	float val = m_noDataValue;
	m_lock.unlock();
	return val;
}

int CPUImageBuffer::width() {
	m_lock.lockForRead();
	int val = m_width;
	m_lock.unlock();
	return val;
}
int CPUImageBuffer::height() {
	m_lock.lockForRead();
	int val = m_height;
	m_lock.unlock();
	return val;
}

CPUImageBuffer::~CPUImageBuffer() {
	m_lock.lockForWrite();
	m_lock.unlock();
}

size_t CPUImageBuffer::internalBufferSize() {
	size_t pointerSize = m_width * m_height * m_samplType.byte_size();
	return pointerSize;
}

size_t CPUImageBuffer::internalPointerSizeSafe() {
	m_lock.lockForRead();
	size_t pointerSize = internalBufferSize();
	m_lock.unlock();
	return pointerSize;
}

template<typename InputType>
void CPUImageBuffer::ByteSwapAndTransposeImageDataKernel<InputType>::run(
		CPUImageBuffer* obj, const void* _swappingPointer) {
//	byteSwapAndTransposeImageData((InputType*) obj->m_cudaPointer,
//					(InputType*) swappingPointer, obj->m_width, obj->m_height);
	const InputType* swappingPointer = static_cast<const InputType*>(_swappingPointer);
	InputType* backingPointer = static_cast<InputType*>(static_cast<void*>(obj->m_backingPointer.data()));
#pragma omp parallel for
	for (long j=0; j<obj->m_height; j++) {
		for (long i=0; i<obj->m_width; i++) {
			backingPointer[j*obj->m_width + i] = swappingPointer[obj->m_height*i + j];
		}
	}
}

void CPUImageBuffer::updateTexture(const QByteArray& input, bool byteSwapAndTranspose) {
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

void CPUImageBuffer::updateTexture(const QByteArray& input, bool byteSwapAndTranspose, const QVector2D& cacheRange,
		const QHistogram& cacheHistogram) {
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
	m_dataRange = cacheRange;
	m_range = m_dataRange;
	m_cachedHistogram = cacheHistogram;
	updateRangeRatioUnsafe();
	m_dataRangeComputed = true;
	m_lock.unlock();
}

template<typename InputType>
void CPUImageBuffer::ComputeRangeUnsafeKernel<InputType>::run(CPUImageBuffer* obj, float& x, float& y) {
	//computeMinMaxOptimized((InputType*) obj->m_cudaPointer, obj->m_width * obj->m_height, x, y);
	x = std::numeric_limits<InputType>::max();
	y = std::numeric_limits<InputType>::lowest();
	InputType* backingPointer = static_cast<InputType*>(static_cast<void*>(obj->m_backingPointer.data()));
#pragma omp parallel for
	for (long j=0; j<obj->m_height; j++) {
		for (long i=0; i<obj->m_width; i++) {
			InputType val = backingPointer[j*obj->m_width + i];
			if (val<x) {
#pragma omp critical
				if (val<x) {
					x = val;
				}
			}
			if (val>y) {
#pragma omp critical
				if (val>y) {
					y = val;
				}
			}
		}
	}
}

QVector2D CPUImageBuffer::computeRangeUnsafe() {
	allocateInternalBufferOnDemandUnsafe();
	float x, y;
	SampleTypeBinder binder(m_samplType);
	binder.bind<ComputeRangeUnsafeKernel>(this, x, y);

	return QVector2D(x, y);
}

template<typename InputType>
void CPUImageBuffer::ComputeHistogramUnsafeKernel<InputType>::run(CPUImageBuffer* obj, unsigned int *hist, const QVector2D &range) {
//	computeImageHistogram((InputType*) obj->m_cudaPointer, hist,
//					obj->m_width, obj->m_height, range.x(), 1 / (range.y() - range.x()));
	if (qFuzzyCompare(range.y(), range.x())) {
		memset(hist, 0, sizeof(unsigned int) * 256);
		return;
	}

	int threadCount = omp_get_max_threads();
	std::vector<std::vector<unsigned int>> hists;
	std::vector<unsigned int> initHist;
	initHist.resize(256, 0);
	hists.resize(threadCount, initHist);
	//qDebug()<<"ComputeHistogramUnsafeKernel";

	const InputType* backingPointer = static_cast<const InputType*>(static_cast<const void*>(obj->m_backingPointer.constData()));

	#pragma omp parallel
	{
		int threadId = omp_get_thread_num();
		if (threadId<threadCount) {
			std::vector<unsigned int>& histThread = hists[threadId];
			#pragma omp for
			for (long j=0; j<obj->m_height; j++) {
				for (long i=0; i<obj->m_width; i++) {
					float val = backingPointer[j*obj->m_width + i];
					int index = 0;
					if (val>range.y()) {
						index = 255;
					} else if (val<range.x()) {
						index = 0;
					} else {
						index = (val - range.x()) / (range.y() - range.x()) * 255;
					}
					histThread[index]++;
				}
			}
		}
	}

	for (int i=0; i<256; i++) {
		unsigned int count = 0;
		for (int j=0; j<threadCount; j++) {
			count += hists[j][i];
		}
		hist[i] = count;
	}
}

QHistogram CPUImageBuffer::computeHistogramUnsafe(const QVector2D &range,
		int nBuckets) {
	allocateInternalBufferOnDemandUnsafe();

	if (nBuckets != 256)
		std::cerr
				<< "Histogram length is not compatible with QGLGraphicsItem (expected 256 got "
				<< nBuckets << ")" << std::endl;

	int hSize = 256;
	size_t length = hSize * sizeof(unsigned int);

	std::vector<unsigned int> hh;
	hh.resize(256);

	SampleTypeBinder binder(m_samplType);
	binder.bind<ComputeHistogramUnsafeKernel>(this, (unsigned int*) hh.data(), range);

	QHistogram histo;
	for (int i = 0; i < 256; i++) {
		histo[i] = hh[i];
	}
	histo.setRange(range);

	return histo;
}

QHistogram CPUImageBuffer::computeHistogram(const QVector2D &range,
		int nBuckets) {
	m_lock.lockForRead();
	QHistogram histo;
	if (m_cachedHistogram.range() != range)
		m_cachedHistogram = computeHistogramUnsafe(range, nBuckets);
	histo = m_cachedHistogram;

	m_lock.unlock();
	return histo;
}

void CPUImageBuffer::worldToImage(double worldX, double worldY, double &imageX,
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

void CPUImageBuffer::imageToWorld(double imageX, double imageY, double &worldX,
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
QMatrix4x4 CPUImageBuffer::imageToWorldTransformation() {
	QMatrix4x4 id;
	id.setToIdentity();
	m_lock.lockForRead();
	if (m_externalTransfoProvider != nullptr)
		id = m_externalTransfoProvider->imageToWorldTransformation();
	m_lock.unlock();
	return id;
}

bool CPUImageBuffer::setValue(int i, int j, double value) {
	m_lock.lockForWrite();
	bool val = TextureHelper::setValue(m_backingPointer.data(), i, j, m_width,
			m_samplType, value);
	m_lock.unlock();
	return true;
}

bool CPUImageBuffer::valueAt(int i, int j, double &value) {
	m_lock.lockForRead();
	allocateInternalBufferOnDemandUnsafe();
	bool val = TextureHelper::valueAt(m_backingPointer.constData(), i, j, m_width,
			m_samplType, value);
	m_lock.unlock();
	return val;
}

void CPUImageBuffer::valuesAlongJ(int j, bool *valid, double *values) {
	m_lock.lockForRead();
	allocateInternalBufferOnDemandUnsafe();
	TextureHelper::valuesAlongJ(m_backingPointer.constData(), 0, j, valid, values, m_width,
			m_height, m_samplType);
	m_lock.unlock();
}

void CPUImageBuffer::valuesAlongI(int i, bool *valid, double *values) {
	m_lock.lockForRead();
	allocateInternalBufferOnDemandUnsafe();
	TextureHelper::valuesAlongI(m_backingPointer.constData(), i, 0, valid, values, m_width,
			m_height, m_samplType);
	m_lock.unlock();
}

bool CPUImageBuffer::value(double worldX, double worldY, int &i, int &j,
		double &value) {
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
	if (i < 0 || j < 0 || i >= m_width || j >= m_height)
		return false;

	bool val = TextureHelper::valueAt(m_backingPointer.constData(), i, j, m_width,
			m_samplType, value);
	m_lock.unlock();
	return val;
}

QRectF CPUImageBuffer::worldExtent() {
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

