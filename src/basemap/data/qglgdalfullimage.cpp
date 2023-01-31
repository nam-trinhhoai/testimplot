#include "qglgdalfullimage.h"

#include <QOpenGLTexture>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "texturehelper.h"
#include "gdalimagewrapper.h"

QGLGDALFullImage::QGLGDALFullImage(QObject *parent) :
		QGLAbstractFullImage(parent) {
	m_internalImage = new GDALImageWrapper(this);
	m_texture = nullptr;
}

QGLGDALFullImage::~QGLGDALFullImage() {
	close();
}
void QGLGDALFullImage::close() {
	m_internalImage->close();
}
int QGLGDALFullImage::width() const {
	return m_internalImage->width();
}
int QGLGDALFullImage::height() const {
	return m_internalImage->height();
}

void QGLGDALFullImage::worldToImage(double worldX, double worldY,
		double &imageX, double &imageY) const {
	m_internalImage->worldToImage(worldX, worldY, imageX, imageY);
}
void QGLGDALFullImage::imageToWorld(double imageX, double imageY,
		double &worldX, double &worldY) const {
	m_internalImage->imageToWorld(imageX, imageY, worldX, worldY);
}

QMatrix4x4 QGLGDALFullImage::imageToWorldTransformation() const {
	return m_internalImage->imageToWorldTransformation();
}

bool QGLGDALFullImage::open(const QString &imageFilePath) {
	m_internalImage->close();
	if (!m_internalImage->open(imageFilePath))
		return false;

	m_colorFormat = m_internalImage->colorFormat();
	m_samplType = m_internalImage->sampleType();

	int numBands = m_internalImage->numBands();
	int offset = ImageFormats::byteSize(sampleType());
	size_t size = ((size_t) width()) * height() * offset * numBands;
	m_buffer.resize(size);

	m_internalImage->readData(0, 0, width(), height(), (void*) m_buffer.data(),
			numBands);
	m_noDataValue = m_internalImage->noDataValue(m_hasNodataValue);

	return true;
}

QVector2D QGLGDALFullImage::computeRange() const {
	return m_internalImage->computeRange();
}

QHistogram QGLGDALFullImage::computeHistogram(const QVector2D &range,
		int nBuckets) {
	if (range == m_cachedHisto.range())
		return m_cachedHisto;
	m_cachedHisto = m_internalImage->computeHistogram(range, nBuckets);
}

bool QGLGDALFullImage::valueAt(int i, int j, double &value) const {
	return TextureHelper::valueAt(m_buffer.data(), i, j, width(), sampleType(),
			value);
}

void QGLGDALFullImage::valuesAlongJ(int j, bool *valid, double *values) const {
	return TextureHelper::valuesAlongJ(m_buffer.data(), 0, j, valid, values,
			width(), height(), sampleType());
}
void QGLGDALFullImage::valuesAlongI(int i, bool *valid, double *values) const {
	return TextureHelper::valuesAlongI(m_buffer.data(), i, 0, valid, values,
			width(), height(), sampleType());
}
void QGLGDALFullImage::bindTexture(unsigned int unit) {
	if (m_texture != nullptr) {
		m_texture->bind(unit,
				QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
		return;
	}
	m_texture = TextureHelper::generateTexture(m_buffer.data(), width(),
			height(), sampleType(), colorFormat());
	m_texture->bind(unit, QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
}

void QGLGDALFullImage::releaseTexture(unsigned int unit) {
	m_texture->release(unit,
			QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
}

