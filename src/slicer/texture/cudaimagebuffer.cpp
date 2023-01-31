#include "cudaimagebuffer.h"
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
#include "colortableregistry.h"
#include "sampletypebinder.h"

CUDAImageBuffer::CUDAImageBuffer(int width, int height,
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

	m_backingPointer=nullptr;
	m_cudaPointer=nullptr;
}

void CUDAImageBuffer::allocateInternalBufferOnDemandUnsafe()
{
	if(m_backingPointer!=nullptr)
		return;

	size_t pointerSize = internalBufferSize();
	checkCudaErrors(cudaMalloc(&m_cudaPointer, pointerSize));

	m_backingPointer = new char[pointerSize];
	memset(m_backingPointer, 0, pointerSize);
}

LookupTable CUDAImageBuffer::lookupTable() {
	m_lock.lockForRead();
	LookupTable internalLut = m_lookupTable;
	m_lock.unlock();
	return internalLut;
}

void CUDAImageBuffer::setLookupTable(const LookupTable &table) {
	m_lock.lockForWrite();
	m_lookupTable = table;
	m_lock.unlock();
}


void CUDAImageBuffer::resetRangeAndHistogramUnsafe() {
	m_dataRangeComputed = false;
	m_cachedHistogram=QHistogram();
}

void CUDAImageBuffer::initRangeUnsafe() {
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

void CUDAImageBuffer::updateRangeRatioUnsafe() {
	m_rangeRatio.setX(m_range.x());
	if (m_range.y() - m_range.x() != 0) {
		m_rangeRatio.setY(1.0f / (m_range.y() - m_range.x()));
	}
}


QVector2D CUDAImageBuffer::dataRange() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_dataRange;
	m_lock.unlock();
	return r;
}

QVector2D CUDAImageBuffer::range() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_range;
	m_lock.unlock();
	return r;
}

QVector2D CUDAImageBuffer::rangeRatio() {
	m_lock.lockForRead();
	initRangeUnsafe();
	QVector2D r = m_rangeRatio;
	m_lock.unlock();
	return r;
}

void CUDAImageBuffer::setRange(const QVector2D &range) {
	m_lock.lockForWrite();
	initRangeUnsafe(); // to init data range
	m_range = range;
	updateRangeRatioUnsafe();
	m_lock.unlock();
}

ImageFormats::QColorFormat CUDAImageBuffer::colorFormat() {
	m_lock.lockForRead();
	ImageFormats::QColorFormat f = m_colorFormat;
	m_lock.unlock();
	return f;
}

ImageFormats::QSampleType CUDAImageBuffer::sampleType() {
	m_lock.lockForRead();
	ImageFormats::QSampleType type = m_samplType;
	m_lock.unlock();
	return type;
}

void CUDAImageBuffer::lock() {
	m_lock.lockForWrite();
}
void* CUDAImageBuffer::cudaPointer() {
	allocateInternalBufferOnDemandUnsafe();
	return m_cudaPointer;
}

void* CUDAImageBuffer::backingPointer() {
	allocateInternalBufferOnDemandUnsafe();
	return m_backingPointer;
}

const void* CUDAImageBuffer::constBackingPointer() {
	allocateInternalBufferOnDemandUnsafe();
	return m_backingPointer;
}

void CUDAImageBuffer::unlock() {
	m_lock.unlock();
}

float CUDAImageBuffer::opacity() {
	m_lock.lockForRead();
	float val = m_opacity;
	m_lock.unlock();
	return val;
}

void CUDAImageBuffer::setOpacity(float value) {
	m_lock.lockForWrite();
	m_opacity = value;
	m_lock.unlock();
}

bool CUDAImageBuffer::hasNoDataValue() {
	m_lock.lockForRead();
	bool val = m_hasNodataValue;
	m_lock.unlock();
	return val;
}
float CUDAImageBuffer::noDataValue() {
	m_lock.lockForRead();
	float val = m_noDataValue;
	m_lock.unlock();
	return val;
}

int CUDAImageBuffer::width() {
	m_lock.lockForRead();
	int val = m_width;
	m_lock.unlock();
	return val;
}
int CUDAImageBuffer::height() {
	m_lock.lockForRead();
	int val = m_height;
	m_lock.unlock();
	return val;
}

CUDAImageBuffer::~CUDAImageBuffer() {
	m_lock.lockForWrite();
	if (m_cudaPointer)
		cudaFree(m_cudaPointer);
	if (m_backingPointer)
		delete[] m_backingPointer;
	m_lock.unlock();
}

size_t CUDAImageBuffer::internalBufferSize() {
	size_t pointerSize = m_width * m_height * m_samplType.byte_size();
	return pointerSize;
}

size_t CUDAImageBuffer::internalPointerSizeSafe() {
	m_lock.lockForRead();
	size_t pointerSize = internalBufferSize();
	m_lock.unlock();
	return pointerSize;
}

void CUDAImageBuffer::swapCudaPointer() {
	m_lock.lockForWrite();
	allocateInternalBufferOnDemandUnsafe();
	size_t pointerSize = internalBufferSize();
	cudaMemcpy(m_backingPointer, m_cudaPointer, pointerSize,
			cudaMemcpyDeviceToHost);

	resetRangeAndHistogramUnsafe();
	m_lock.unlock();
}

template<typename InputType>
void CUDAImageBuffer::ByteSwapAndTransposeImageDataKernel<InputType>::run(
		CUDAImageBuffer* obj, void* swappingPointer) {
	byteSwapAndTransposeImageData((InputType*) obj->m_cudaPointer,
					(InputType*) swappingPointer, obj->m_width, obj->m_height);
}

void CUDAImageBuffer::updateTexture(const void *input, bool byteSwapAndTranspose) {
	m_lock.lockForWrite();
	allocateInternalBufferOnDemandUnsafe();
	size_t pointerSize = internalBufferSize();
	bool doByteSwap = byteSwapAndTranspose;

	memcpy(m_backingPointer, input, pointerSize);
	if (doByteSwap) {
		void *swappingPointer;
		checkCudaErrors(cudaMalloc(&swappingPointer, pointerSize));
		cudaMemcpy(swappingPointer, m_backingPointer, pointerSize,
				cudaMemcpyHostToDevice);
		SampleTypeBinder binder(m_samplType);
		binder.bind<ByteSwapAndTransposeImageDataKernel>(this, swappingPointer);
		//checkCudaErrors(cudaDeviceSynchronize());
		cudaFree(swappingPointer);

		//mirro back the pointer to the backing pointer
		cudaMemcpy(m_backingPointer, m_cudaPointer, pointerSize,
				cudaMemcpyDeviceToHost);
	} else
		cudaMemcpy(m_cudaPointer, m_backingPointer, pointerSize,
				cudaMemcpyHostToDevice);

	resetRangeAndHistogramUnsafe();
	m_lock.unlock();
}

void CUDAImageBuffer::updateTexture(const void *input, bool byteSwapAndTranspose, const QVector2D& cacheRange,
		const QHistogram& cacheHistogram) {
	m_lock.lockForWrite();
	allocateInternalBufferOnDemandUnsafe();
	size_t pointerSize = internalBufferSize();
	bool doByteSwap = byteSwapAndTranspose;

	memcpy(m_backingPointer, input, pointerSize);
	if (doByteSwap) {
		void *swappingPointer;
		checkCudaErrors(cudaMalloc(&swappingPointer, pointerSize));
		cudaMemcpy(swappingPointer, m_backingPointer, pointerSize,
				cudaMemcpyHostToDevice);
		SampleTypeBinder binder(m_samplType);
		binder.bind<ByteSwapAndTransposeImageDataKernel>(this, swappingPointer);
		//checkCudaErrors(cudaDeviceSynchronize());
		cudaFree(swappingPointer);

		//mirro back the pointer to the backing pointer
		cudaMemcpy(m_backingPointer, m_cudaPointer, pointerSize,
				cudaMemcpyDeviceToHost);
	} else
		cudaMemcpy(m_cudaPointer, m_backingPointer, pointerSize,
				cudaMemcpyHostToDevice);

	resetRangeAndHistogramUnsafe();
	m_dataRange = cacheRange;
	m_range = m_dataRange;
	m_cachedHistogram = cacheHistogram;
	updateRangeRatioUnsafe();
	m_dataRangeComputed = true;
	m_lock.unlock();
}

template<typename InputType>
void CUDAImageBuffer::ComputeRangeUnsafeKernel<InputType>::run(CUDAImageBuffer* obj, float& x, float& y) {
	computeMinMaxOptimized((InputType*) obj->m_cudaPointer, obj->m_width * obj->m_height, x, y);
}

QVector2D CUDAImageBuffer::computeRangeUnsafe() {
	allocateInternalBufferOnDemandUnsafe();
	float x, y;
	if(m_width == 0 || m_height == 0)
	{
		x=0;y=0;
	}
	else
	{
		SampleTypeBinder binder(m_samplType);
		binder.bind<ComputeRangeUnsafeKernel>(this, x, y);
	}
	return QVector2D(x, y);
}

template<typename InputType>
void CUDAImageBuffer::ComputeHistogramUnsafeKernel<InputType>::run(CUDAImageBuffer* obj, unsigned int *hist, const QVector2D &range) {
	computeImageHistogram((InputType*) obj->m_cudaPointer, hist,
					obj->m_width, obj->m_height, range.x(), 1 / (range.y() - range.x()));
}

QHistogram CUDAImageBuffer::computeHistogramUnsafe(const QVector2D &range,
		int nBuckets) {
	allocateInternalBufferOnDemandUnsafe();

	if (nBuckets != 256)
		std::cerr
				<< "Histogram length is not compatible with QGLGraphicsItem (expected 256 got "
				<< nBuckets << ")" << std::endl;

	int hSize = 256;
	size_t length = hSize * sizeof(unsigned int);

	void *d_output;
	cudaMalloc(&d_output, length);
	cudaMemset(d_output, 0, length);
	SampleTypeBinder binder(m_samplType);
	binder.bind<ComputeHistogramUnsafeKernel>(this, (unsigned int*) d_output, range);
//	checkCudaErrors(cudaDeviceSynchronize());

	void *h_output = malloc(length);
	cudaMemcpy(h_output, d_output, length, cudaMemcpyDeviceToHost);
	QHistogram histo;
	unsigned int *hh = (unsigned int*) h_output;
	for (int i = 0; i < 256; i++) {
		histo[i] = hh[i];
	}
	histo.setRange(range);

	free(h_output);
	cudaFree(d_output);
	return histo;
}

QHistogram CUDAImageBuffer::computeHistogram(const QVector2D &range,
		int nBuckets) {
	m_lock.lockForRead();
	QHistogram histo;
	if (m_cachedHistogram.range() != range)
		m_cachedHistogram = computeHistogramUnsafe(range, nBuckets);
	histo = m_cachedHistogram;

	m_lock.unlock();
	return histo;
}

void CUDAImageBuffer::worldToImage(double worldX, double worldY, double &imageX,
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

void CUDAImageBuffer::imageToWorld(double imageX, double imageY, double &worldX,
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
QMatrix4x4 CUDAImageBuffer::imageToWorldTransformation() {
	QMatrix4x4 id;
	id.setToIdentity();
	m_lock.lockForRead();
	if (m_externalTransfoProvider != nullptr)
		id = m_externalTransfoProvider->imageToWorldTransformation();
	m_lock.unlock();
	return id;
}

bool CUDAImageBuffer::setValue(int i, int j, double value) {
	m_lock.lockForWrite();
	bool val = TextureHelper::setValue(m_backingPointer, i, j, m_width,
			m_samplType, value);
	m_lock.unlock();
	return true;
}

bool CUDAImageBuffer::valueAt(int i, int j, double &value) {
	m_lock.lockForRead();
	allocateInternalBufferOnDemandUnsafe();
	bool val = TextureHelper::valueAt(m_backingPointer, i, j, m_width,
			m_samplType, value);
	m_lock.unlock();
	return val;
}

void CUDAImageBuffer::valuesAlongJ(int j, bool *valid, double *values) {
	m_lock.lockForRead();
	allocateInternalBufferOnDemandUnsafe();
	TextureHelper::valuesAlongJ(m_backingPointer, 0, j, valid, values, m_width,
			m_height, m_samplType);
	m_lock.unlock();
}

void CUDAImageBuffer::valuesAlongI(int i, bool *valid, double *values) {
	m_lock.lockForRead();
	allocateInternalBufferOnDemandUnsafe();
	TextureHelper::valuesAlongI(m_backingPointer, i, 0, valid, values, m_width,
			m_height, m_samplType);
	m_lock.unlock();
}

bool CUDAImageBuffer::value(double worldX, double worldY, int &i, int &j,
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
	bool val = false;
	if (i >= 0 && j >= 0 && i < m_width && j < m_height) {
		val = TextureHelper::valueAt(m_backingPointer, i, j, m_width,
				m_samplType, value);
	}

	m_lock.unlock();
	return val;
}

QRectF CUDAImageBuffer::worldExtent() {
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

