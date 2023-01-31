#include "cpuimagepaletteholder.h"

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
#include "cpuimagebuffer.h"
#include "surfacemeshcacheutils.h"

CPUImagePaletteHolder::CPUImagePaletteHolder(int width, int height,
		ImageFormats::QSampleType type, const IGeorefImage * const transfoProvider,
		QObject *parent) :
		IImagePaletteHolder(parent) {
	m_buffer = new CPUImageBuffer(width, height, type, transfoProvider, this);
}

size_t CPUImagePaletteHolder::internalPointerSize()
{
	return 	m_buffer->internalPointerSizeSafe();
}

LookupTable CPUImagePaletteHolder::lookupTable() const {
	return m_buffer->lookupTable();
}

void CPUImagePaletteHolder::setLookupTable(const LookupTable &table) {
	m_buffer->setLookupTable(table);
	emit lookupTableChanged();
	emit lookupTableChanged(table);
}

float CPUImagePaletteHolder::opacity() const {
	return m_buffer->opacity();
}

void CPUImagePaletteHolder::setOpacity(float value) {
	m_buffer->setOpacity(value);
	emit opacityChanged();
	emit opacityChanged(value);
}

bool CPUImagePaletteHolder::hasNoDataValue() const {
	return m_buffer->hasNoDataValue();
}

float CPUImagePaletteHolder::noDataValue() const {
	return m_buffer->noDataValue();
}

QVector2D CPUImagePaletteHolder::dataRange() {
	return m_buffer->dataRange();
}

QVector2D CPUImagePaletteHolder::rangeRatio() {
	return m_buffer->rangeRatio();
}
QVector2D CPUImagePaletteHolder::range() {
	return m_buffer->range();
}
void CPUImagePaletteHolder::setRange(const QVector2D &range) {
	m_buffer->setRange(range);
	emit rangeChanged();
	emit rangeChanged(range);
}

ImageFormats::QColorFormat CPUImagePaletteHolder::colorFormat() const {
	return m_buffer->colorFormat();
}
ImageFormats::QSampleType CPUImagePaletteHolder::sampleType() const {
	return m_buffer->sampleType();
}

void CPUImagePaletteHolder::updateTexture(const QByteArray& input,
		bool byteSwapAndTranspose) {
	m_buffer->updateTexture(input, byteSwapAndTranspose);
	emit dataChanged();
}

void CPUImagePaletteHolder::updateTexture(const QByteArray& input,
		bool byteSwapAndTranspose, const QVector2D& cacheRange,
		const QHistogram& cacheHistogram) {
	m_buffer->updateTexture(input, byteSwapAndTranspose, cacheRange,
			cacheHistogram);
	emit dataChanged();
}

CPUImagePaletteHolder::~CPUImagePaletteHolder() {
}


int CPUImagePaletteHolder::width() const {
	return m_buffer->width();
}
int CPUImagePaletteHolder::height() const {
	return m_buffer->height();
}

QHistogram CPUImagePaletteHolder::computeHistogram(const QVector2D &range,
		int nBuckets) {
	return m_buffer->computeHistogram(range, nBuckets);
}

void * CPUImagePaletteHolder::backingPointer()
{
	return m_buffer->backingPointer();
}

const void * CPUImagePaletteHolder::constBackingPointer()
{
	return m_buffer->constBackingPointer();
}

QByteArray CPUImagePaletteHolder::getDataAsByteArray() {
	lockPointer();
	QByteArray out = m_buffer->byteArray();
	unlockPointer();
	return out;
}

void CPUImagePaletteHolder::lockPointer() {
	m_buffer->lock();
}
void CPUImagePaletteHolder::unlockPointer() {
	m_buffer->unlock();
}

void CPUImagePaletteHolder::worldToImage(double worldX, double worldY,
		double &imageX, double &imageY) const {
	m_buffer->worldToImage(worldX, worldY, imageX, imageY);
}

void CPUImagePaletteHolder::imageToWorld(double imageX, double imageY,
		double &worldX, double &worldY) const {
	m_buffer->imageToWorld(imageX, imageY, worldX, worldY);
}
QMatrix4x4 CPUImagePaletteHolder::imageToWorldTransformation() const {
	return m_buffer->imageToWorldTransformation();
}

bool CPUImagePaletteHolder::setValue(int i, int j, double value)
{
	return m_buffer->setValue(i, j, value);
}

bool CPUImagePaletteHolder::valueAt(int i, int j, double &value) const {
	return m_buffer->valueAt(i, j, value);
}
void CPUImagePaletteHolder::valuesAlongJ(int j, bool *valid,
		double *values) const {
	m_buffer->valuesAlongJ(j, valid, values);
}
void CPUImagePaletteHolder::valuesAlongI(int i, bool *valid,
		double *values) const {
	m_buffer->valuesAlongI(i, valid, values);
}

bool CPUImagePaletteHolder::value(double worldX, double worldY, int &i, int &j,
		double &value) const {
	return m_buffer->value(worldX, worldY, i, j, value);
}

QRectF CPUImagePaletteHolder::worldExtent() const {
	return m_buffer->worldExtent();
}

