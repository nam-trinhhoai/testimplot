#ifndef CUDAImageTextureMapper_H
#define CUDAImageTextureMapper_H

#include <QWidget>
#include "qglabstractfullimage.h"
#include "imageformats.h"
#include "lookuptable.h"

class IImagePaletteHolder;
class QOpenGLTexture;



class CUDAImageTextureMapper: public QObject
{
	Q_OBJECT
public:
	CUDAImageTextureMapper(IImagePaletteHolder *image, QObject *parent = 0);
	virtual ~CUDAImageTextureMapper();

	virtual void bindTexture(unsigned int unit);
	virtual void releaseTexture(unsigned int unit);
	bool textureInitialized();

	void bindLUTTexture(unsigned int unit);
	void releaseLUTTexture(unsigned int unit);

private slots:
	void lookupTableChanged(const LookupTable & table);
	void updateTexture();
private:
	void createTexture();

private:
	IImagePaletteHolder *m_buffer;
	QOpenGLTexture *m_colorMapTexture;
	QOpenGLTexture *m_texture;

	bool m_needInternalBufferUpdate;
	bool m_needColorTableReload;

	char* m_internalBuffer;
};

#endif
