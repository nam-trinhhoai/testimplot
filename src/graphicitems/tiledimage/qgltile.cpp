#include "qgltile.h"
#include <QOpenGLTexture>
#include <QDebug>
#include "texturehelper.h"
#include <QOpenGLPixelTransferOptions>

QGLTile::QGLTile(const QGLTileCoord &coords,
		ImageFormats::QColorFormat colorFormat,
		ImageFormats::QSampleType sampleType) {

	this->m_coords = coords;
	m_width = coords.imageBoundingRect().width();
	m_height = coords.imageBoundingRect().height();

	m_colorFormat=colorFormat;
	m_sampleType=sampleType;

	int offset = ImageFormats::byteSize(sampleType);
	int numBands = ImageFormats::numBands(colorFormat);
	size_t size = ((size_t) m_width) * m_height * offset * numBands;
	m_buffer.resize(size);

	m_isValid=false;
}

void QGLTile::valid(bool valid)
{
	m_isValid=valid;
}
bool QGLTile::valid() const
{
	return m_isValid;
}

const QGLTileCoord QGLTile::coords() const
{
	return this->m_coords;

}
bool QGLTile::operator==(const QGLTileCoord &rhs) const {
	return this->m_coords == rhs;
}

QOpenGLTexture* QGLTile::getAndBindTexture(unsigned int unit) {
	if (m_texture != nullptr) {
		m_texture->bind(unit,QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
		return m_texture;
	}
	m_texture = TextureHelper::generateTexture(m_buffer.data(), width(), height(),sampleType(),colorFormat());;
	m_texture->bind(unit,QOpenGLTexture::TextureUnitReset::ResetTextureUnit);
	return m_texture;
}

void * QGLTile::data()
{
	return m_buffer.data();
}

