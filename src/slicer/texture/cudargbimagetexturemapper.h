#ifndef CUDARGBIMAGETEXTUREMAPPER_H
#define CUDARGBIMAGETEXTUREMAPPER_H

#include <QWidget>
#include "qglabstractfullimage.h"
#include "imageformats.h"

class QOpenGLTexture;
class CUDARGBInterleavedImage;

class CUDARGBImageTextureMapper: public QObject
{
	Q_OBJECT
public:
	CUDARGBImageTextureMapper(CUDARGBInterleavedImage* image, QObject *parent = 0);
	virtual ~CUDARGBImageTextureMapper();

	virtual void bindTexture(unsigned int unit);
	virtual void releaseTexture(unsigned int unit);
	bool textureInitialized();

private slots:
	void updateTexture();
private:
	void createTexture();

private:
	CUDARGBInterleavedImage *m_buffer;
	QOpenGLTexture *m_texture;

	bool m_needInternalBufferUpdate;

	char* m_internalBuffer;
};

#endif
