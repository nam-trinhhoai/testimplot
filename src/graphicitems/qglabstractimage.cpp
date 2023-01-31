#include "qglabstractimage.h"

#include <QOpenGLTexture>
#include <QDebug>
#include <QFileInfo>
#include <QOpenGLPixelTransferOptions>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "texturehelper.h"
#include "colortableregistry.h"

QGLAbstractImage::QGLAbstractImage(QObject *parent) :
		QObject(parent), IPaletteHolder(), IGeorefImage() {
	m_colorMapTexture = nullptr;

	m_opacity = 1.0;
	m_noDataValue = 0;
	m_hasNodataValue = false;
	m_needColorTableReload = false;
	m_lookupTable = ColorTableRegistry::DEFAULT();

	m_rangeRatio = QVector2D(0, 1);
	m_range=m_dataRange = QVector2D(0, 1);
	m_dataRangeComputed=false;
}

QGLAbstractImage::~QGLAbstractImage() {

}

ImageFormats::QColorFormat QGLAbstractImage::colorFormat() const {
	return m_colorFormat;
}
ImageFormats::QSampleType QGLAbstractImage::sampleType() const {
	return m_samplType;
}

void QGLAbstractImage::initRange()
{
	if(m_dataRangeComputed)
		return;

	m_dataRange=computeRange();
	m_range=IPaletteHolder::smartAdjust(m_dataRange,computeHistogram(m_dataRange, QHistogram::HISTOGRAM_SIZE));
	updateRangeRatio();
	m_dataRangeComputed=true;
}

void QGLAbstractImage::updateRangeRatio()
{
	m_rangeRatio.setX(m_range.x());
	if (m_range.y() - m_range.x() != 0) {
		m_rangeRatio.setY(1.0f / (m_range.y() - m_range.x()));
	}
}

QVector2D QGLAbstractImage::dataRange() {
	initRange();
	return m_dataRange;
}

QVector2D QGLAbstractImage::range() {
	initRange();
	return m_range;
}

void QGLAbstractImage::setRange(const QVector2D &range) {
	m_range = range;
	updateRangeRatio();
	emit rangeChanged();
}
QVector2D QGLAbstractImage::rangeRatio() {
	initRange();
	return m_rangeRatio;
}

LookupTable QGLAbstractImage::lookupTable() const {
	return m_lookupTable;
}

float QGLAbstractImage::opacity() const {
	return m_opacity;
}

void QGLAbstractImage::setOpacity(float value) {
	m_opacity = value;
	emit opacityChanged();
}

void QGLAbstractImage::setLookupTable(const LookupTable &table) {
	m_lookupTable = table;
	m_needColorTableReload = true;
	emit lookupTableChanged();
}

void QGLAbstractImage::bindLUTTexture(unsigned int unit) {
	if (m_needColorTableReload) {
		if (m_colorMapTexture != nullptr)
			m_colorMapTexture->destroy();
		m_colorMapTexture=nullptr;
		m_needColorTableReload = false;
	}

	if (m_colorMapTexture != nullptr) {
		m_colorMapTexture->bind(unit,
				QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
		return;
	}
	m_colorMapTexture = TextureHelper::generateLUTTexture(m_lookupTable);
	m_colorMapTexture->bind(unit,
			QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
}
void QGLAbstractImage::releaseLUTTexture(unsigned int unit) {
	m_colorMapTexture->release(unit,
			QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
}

bool QGLAbstractImage::value(double worldX, double worldY, int &i, int &j,
		double &value) const {
	return IGeorefImage::value(this, worldX, worldY, i, j, value);
}

QRectF QGLAbstractImage::worldExtent() const {
	return IGeorefImage::worldExtent(this);
}

