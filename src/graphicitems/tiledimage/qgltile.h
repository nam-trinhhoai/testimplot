#ifndef QGLTile_H
#define QGLTile_H

#include "qgltilecoord.h"

#include <QRect>
#include <QImage>
#include <QObject>
#include "imageformats.h"
class QOpenGLTexture;

class QGLTile {
public:
	QGLTile(const QGLTileCoord & coords,ImageFormats::QColorFormat colorFormat,ImageFormats::QSampleType sampleType);
	QGLTile();

	bool operator ==(const QGLTileCoord &b) const;

	QOpenGLTexture * getAndBindTexture(unsigned int unit);

	inline int width() const{return m_width;}
	inline int height() const{return m_height;}

	ImageFormats::QColorFormat colorFormat() const {
		return m_colorFormat;
	};

	ImageFormats::QSampleType sampleType() const {
		return m_sampleType;
	};

	const QGLTileCoord coords() const;

	void * data();

	void valid(bool valid);
	bool valid() const;
private:
	QGLTileCoord m_coords;
	QOpenGLTexture *m_texture = nullptr;

	int m_width;
	int m_height;

	ImageFormats::QColorFormat m_colorFormat;
	ImageFormats::QSampleType m_sampleType;

	QByteArray m_buffer;

	bool m_isValid;
};

#endif
