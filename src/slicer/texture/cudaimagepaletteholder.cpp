#include "cudaimagepaletteholder.h"

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
#include "surfacemeshcacheutils.h"

CUDAImagePaletteHolder::CUDAImagePaletteHolder(int width, int height,
		ImageFormats::QSampleType type, const IGeorefImage * const transfoProvider,
		QObject *parent) :
		IImagePaletteHolder(parent) {
	m_buffer = new CUDAImageBuffer(width, height, type, transfoProvider, this);
}

size_t CUDAImagePaletteHolder::internalPointerSize()
{
	return 	m_buffer->internalPointerSizeSafe();
}

LookupTable CUDAImagePaletteHolder::lookupTable() const {
	return m_buffer->lookupTable();
}

void CUDAImagePaletteHolder::setLookupTable(const LookupTable &table) {
	m_buffer->setLookupTable(table);
	emit lookupTableChanged();
	emit lookupTableChanged(table);
}

float CUDAImagePaletteHolder::opacity() const {
	return m_buffer->opacity();
}

void CUDAImagePaletteHolder::setOpacity(float value) {
	m_buffer->setOpacity(value);
	emit opacityChanged();
	emit opacityChanged(value);
}

bool CUDAImagePaletteHolder::hasNoDataValue() const {
	return m_buffer->hasNoDataValue();
}

float CUDAImagePaletteHolder::noDataValue() const {
	return m_buffer->noDataValue();
}

QVector2D CUDAImagePaletteHolder::dataRange() {
	return m_buffer->dataRange();
}

QVector2D CUDAImagePaletteHolder::rangeRatio() {
	return m_buffer->rangeRatio();
}
QVector2D CUDAImagePaletteHolder::range() {
	return m_buffer->range();
}
void CUDAImagePaletteHolder::setRange(const QVector2D &range) {
	m_buffer->setRange(range);
	emit rangeChanged();
	emit rangeChanged(range);
}

ImageFormats::QColorFormat CUDAImagePaletteHolder::colorFormat() const {
	return m_buffer->colorFormat();
}
ImageFormats::QSampleType CUDAImagePaletteHolder::sampleType() const {
	return m_buffer->sampleType();
}

void CUDAImagePaletteHolder::updateTexture(const void *input,
		bool byteSwapAndTranspose) {
	m_buffer->updateTexture(input, byteSwapAndTranspose);
	emit dataChanged();
}

void CUDAImagePaletteHolder::updateTexture(const void *input,
		bool byteSwapAndTranspose, const QVector2D& cacheRange,
		const QHistogram& cacheHistogram) {

	m_buffer->updateTexture(input, byteSwapAndTranspose, cacheRange,
			cacheHistogram);
	emit dataChanged();
}

CUDAImagePaletteHolder::~CUDAImagePaletteHolder() {
}


int CUDAImagePaletteHolder::width() const {
	return m_buffer->width();
}
int CUDAImagePaletteHolder::height() const {
	return m_buffer->height();
}

QHistogram CUDAImagePaletteHolder::computeHistogram(const QVector2D &range,
		int nBuckets) {
	return m_buffer->computeHistogram(range, nBuckets);
}


void* CUDAImagePaletteHolder::cudaPointer() {
	return m_buffer->cudaPointer();
}

void * CUDAImagePaletteHolder::backingPointer()
{
	return m_buffer->backingPointer();
}

const void * CUDAImagePaletteHolder::constBackingPointer()
{
	return m_buffer->constBackingPointer();
}

QByteArray CUDAImagePaletteHolder::getDataAsByteArray() {
	size_t size = internalPointerSize();
	lockPointer();
	QByteArray out = byteArrayFromRawData(static_cast<char*>(backingPointer()), size);
	unlockPointer();
	return out;
}

void CUDAImagePaletteHolder::swapCudaPointer()
{
	m_buffer->swapCudaPointer();
	emit dataChanged();
}

void CUDAImagePaletteHolder::lockPointer() {
	m_buffer->lock();
}
void CUDAImagePaletteHolder::unlockPointer() {
	m_buffer->unlock();
}

void CUDAImagePaletteHolder::worldToImage(double worldX, double worldY,
		double &imageX, double &imageY) const {
	m_buffer->worldToImage(worldX, worldY, imageX, imageY);
}

void CUDAImagePaletteHolder::imageToWorld(double imageX, double imageY,
		double &worldX, double &worldY) const {
	m_buffer->imageToWorld(imageX, imageY, worldX, worldY);
}
QMatrix4x4 CUDAImagePaletteHolder::imageToWorldTransformation() const {
	return m_buffer->imageToWorldTransformation();
}

bool CUDAImagePaletteHolder::setValue(int i, int j, double value)  {
	return m_buffer->setValue(i, j, value);
}

bool CUDAImagePaletteHolder::valueAt(int i, int j, double &value) const {
	return m_buffer->valueAt(i, j, value);
}
void CUDAImagePaletteHolder::valuesAlongJ(int j, bool *valid,
		double *values) const {
	m_buffer->valuesAlongJ(j, valid, values);
}
void CUDAImagePaletteHolder::valuesAlongI(int i, bool *valid,
		double *values) const {
	m_buffer->valuesAlongI(i, valid, values);
}

bool CUDAImagePaletteHolder::value(double worldX, double worldY, int &i, int &j,
		double &value) const {
	return m_buffer->value(worldX, worldY, i, j, value);
}

QRectF CUDAImagePaletteHolder::worldExtent() const {
	return m_buffer->worldExtent();
}

