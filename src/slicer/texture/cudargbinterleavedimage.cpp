#include "cudargbinterleavedimage.h"
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

CUDARGBInterleavedImageHolder::CUDARGBInterleavedImageHolder(
		CUDARGBInterleavedImage* image, int channelIndex) {
	m_image = image;
	m_channelIndex = channelIndex;
}

CUDARGBInterleavedImageHolder::~CUDARGBInterleavedImageHolder() {

}

QHistogram CUDARGBInterleavedImageHolder::computeHistogram(
		const QVector2D &range, int nBuckets) {
	QHistogram histo;
	/*for (int i = 0; i < 256; i++) {
		histo[i] = 0;
	}
	histo.setRange(range);*/

	std::vector<QHistogram> out = m_image->computeHistogram(m_image->redDataRange(),
			m_image->greenDataRange(), m_image->blueDataRange(), nBuckets);
	histo = out[m_channelIndex];
	return histo;
}

bool CUDARGBInterleavedImageHolder::hasNoDataValue() const {
	return m_image->hasNoDataValue();
}

float CUDARGBInterleavedImageHolder::noDataValue() const {
	return m_image->noDataValue();
}

QVector2D CUDARGBInterleavedImageHolder::dataRange() {
	if (m_channelIndex==0) {
		return m_image->redDataRange();
	} else if (m_channelIndex==1) {
		return m_image->greenDataRange();
	} else if (m_channelIndex==2) {
		return m_image->blueDataRange();
	} else {
		return QVector2D(0, 1);
	}
}

QVector2D CUDARGBInterleavedImageHolder::rangeRatio() {
	if (m_channelIndex==0) {
		return m_image->redRangeRatio();
	} else if (m_channelIndex==1) {
		return m_image->greenRangeRatio();
	} else if (m_channelIndex==2) {
		return m_image->blueRangeRatio();
	} else {
		return QVector2D(0, 1);
	}
}

QVector2D CUDARGBInterleavedImageHolder::range() {
	if (m_channelIndex==0) {
		return m_image->redRange();
	} else if (m_channelIndex==1) {
		return m_image->greenRange();
	} else if (m_channelIndex==2) {
		return m_image->blueRange();
	} else {
		return QVector2D(0, 1);
	}
}

CUDARGBInterleavedImage::CUDARGBInterleavedImage(int width, int height,
		ImageFormats::QSampleType type, const IGeorefImage * const transfoProvider,
		QObject *parent) :
		IGeorefGrid() {
	m_buffer = new CUDARGBInterleavedImageBuffer(width, height, type, transfoProvider, this);

	m_redHolder.reset(new CUDARGBInterleavedImageHolder(this, 0));
	m_greenHolder.reset(new CUDARGBInterleavedImageHolder(this, 1));
	m_blueHolder.reset(new CUDARGBInterleavedImageHolder(this, 2));
}

size_t CUDARGBInterleavedImage::internalPointerSize()
{
	return m_buffer->internalPointerSizeSafe();
}

float CUDARGBInterleavedImage::opacity() const {
	return m_buffer->opacity();
}

void CUDARGBInterleavedImage::setOpacity(float value) {
	m_buffer->setOpacity(value);
	emit opacityChanged(value);
}

bool CUDARGBInterleavedImage::hasNoDataValue() const {
	return m_buffer->hasNoDataValue();
}

float CUDARGBInterleavedImage::noDataValue() const {
	return m_buffer->noDataValue();
}

QVector2D CUDARGBInterleavedImage::redDataRange() {
	return m_buffer->redDataRange();
}

QVector2D CUDARGBInterleavedImage::redRangeRatio() {
	return m_buffer->redRangeRatio();
}

QVector2D CUDARGBInterleavedImage::redRange() {
	return m_buffer->redRange();
}

void CUDARGBInterleavedImage::setRedRange(const QVector2D &range) {
	m_buffer->setRedRange(range);
	emit redRangeChanged(range);
	emit rangeChanged(0, range);
}

QVector2D CUDARGBInterleavedImage::greenDataRange() {
	return m_buffer->greenDataRange();
}

QVector2D CUDARGBInterleavedImage::greenRangeRatio() {
	return m_buffer->greenRangeRatio();
}

QVector2D CUDARGBInterleavedImage::greenRange() {
	return m_buffer->greenRange();
}

void CUDARGBInterleavedImage::setGreenRange(const QVector2D &range) {
	m_buffer->setGreenRange(range);
	emit greenRangeChanged(range);
	emit rangeChanged(1, range);
}

QVector2D CUDARGBInterleavedImage::blueDataRange() {
	return m_buffer->blueDataRange();
}

QVector2D CUDARGBInterleavedImage::blueRangeRatio() {
	return m_buffer->blueRangeRatio();
}

QVector2D CUDARGBInterleavedImage::blueRange() {
	return m_buffer->blueRange();
}

void CUDARGBInterleavedImage::setBlueRange(const QVector2D &range) {
	m_buffer->setBlueRange(range);
	emit blueRangeChanged(range);
	emit rangeChanged(2, range);
}

void CUDARGBInterleavedImage::setRange(unsigned int i, const QVector2D & range) {
	if (i==0) {
		setRedRange(range);
	} else if (i==1) {
		setGreenRange(range);
	} else if (i==2) {
		setBlueRange(range);
	}
}

ImageFormats::QColorFormat CUDARGBInterleavedImage::colorFormat() const {
	return m_buffer->colorFormat();
}

ImageFormats::QSampleType CUDARGBInterleavedImage::sampleType() const {
	return m_buffer->sampleType();
}

void CUDARGBInterleavedImage::updateTexture(const QByteArray& input,
		bool byteSwapAndTranspose) {
	m_buffer->updateTexture(input, byteSwapAndTranspose);
	emit dataChanged();
}

void CUDARGBInterleavedImage::updateTexture(const QByteArray& input,
		bool byteSwapAndTranspose, const QVector2D& redCacheRange,
		const QVector2D& greenCacheRange, const QVector2D& blueCacheRange) {
	m_buffer->updateTexture(input, byteSwapAndTranspose, redCacheRange,
			greenCacheRange, blueCacheRange);
	emit dataChanged();
}

void CUDARGBInterleavedImage::updateTexture(const QByteArray& input,
		bool byteSwapAndTranspose, const QVector2D& redCacheRange,
		const QVector2D& greenCacheRange, const QVector2D& blueCacheRange,
		const QHistogram& redHistogram, const QHistogram& greenHistogram,
		const QHistogram& blueHistogram) {
	m_buffer->updateTexture(input, byteSwapAndTranspose, redCacheRange,
			greenCacheRange, blueCacheRange, redHistogram, greenHistogram,
			blueHistogram);
	emit dataChanged();
}

CUDARGBInterleavedImage::~CUDARGBInterleavedImage() {
}


int CUDARGBInterleavedImage::width() const {
	return m_buffer->width();
}
int CUDARGBInterleavedImage::height() const {
	return m_buffer->height();
}

void * CUDARGBInterleavedImage::backingPointer()
{
	return m_buffer->backingPointer();
}

const void * CUDARGBInterleavedImage::constBackingPointer()
{
	return m_buffer->constBackingPointer();
}

const QByteArray& CUDARGBInterleavedImage::byteArray() {
	return m_buffer->byteArray();
}

void CUDARGBInterleavedImage::lockPointer() {
	m_buffer->lock();
}
void CUDARGBInterleavedImage::unlockPointer() {
	m_buffer->unlock();
}

void CUDARGBInterleavedImage::worldToImage(double worldX, double worldY,
		double &imageX, double &imageY) const {
	m_buffer->worldToImage(worldX, worldY, imageX, imageY);
}

void CUDARGBInterleavedImage::imageToWorld(double imageX, double imageY,
		double &worldX, double &worldY) const {
	m_buffer->imageToWorld(imageX, imageY, worldX, worldY);
}
QMatrix4x4 CUDARGBInterleavedImage::imageToWorldTransformation() const {
	return m_buffer->imageToWorldTransformation();
}

bool CUDARGBInterleavedImage::valueAt(int i, int j, int channel, double &value) const {
	return m_buffer->valueAt(i, j, channel, value);
}

bool CUDARGBInterleavedImage::setValue(int i, int j, double value)
{
	return m_buffer->setValue(i, j, value);
}

bool CUDARGBInterleavedImage::value(double worldX, double worldY, int channel, int &i, int &j,
		double &value) const {
	return m_buffer->value(worldX, worldY, channel, i, j, value);
}

QRectF CUDARGBInterleavedImage::worldExtent() const {
	return m_buffer->worldExtent();
}

QVector<IPaletteHolder*> CUDARGBInterleavedImage::holders() const {
	QVector<IPaletteHolder*> out;
	out.push_back(m_redHolder.get());
	out.push_back(m_greenHolder.get());
	out.push_back(m_blueHolder.get());
	return out;
}

CUDARGBInterleavedImageHolder* CUDARGBInterleavedImage::redHolder() {
	return m_redHolder.get();
}

CUDARGBInterleavedImageHolder* CUDARGBInterleavedImage::greenHolder() {
	return m_greenHolder.get();
}

CUDARGBInterleavedImageHolder* CUDARGBInterleavedImage::blueHolder() {
	return m_blueHolder.get();
}

CUDARGBInterleavedImageHolder* CUDARGBInterleavedImage::holder(int i) {
	CUDARGBInterleavedImageHolder* out = nullptr;
	if (i==0) {
		out = redHolder();
	} else if (i==1) {
		out = greenHolder();
	} else if (i==2) {
		out = blueHolder();
	}
	return out;
}


std::vector<QHistogram> CUDARGBInterleavedImage::computeHistogram(const QVector2D &redRange,
		const QVector2D &greenRange, const QVector2D &blueRange, int nBuckets) {
	return m_buffer->computeHistogram(redRange, greenRange, blueRange, nBuckets);
}

