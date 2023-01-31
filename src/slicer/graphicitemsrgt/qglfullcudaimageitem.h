#ifndef QGLFullCUDAImageItem_H
#define QGLFullCUDAImageItem_H

#include "qabstractglgraphicsitem.h"
#include "CUDAImageMask.h"

#include <QGraphicsObject>
#include <QTransform>
#include <QOpenGLBuffer>

class CUDAImagePaletteHolder;
class IImagePaletteHolder;
class QOpenGLShaderProgram;
class QOpenGLTexture;
class CUDAImageTextureMapper;

class QGLFullCUDAImageItem: public QAbstractGLGraphicsItem, public CUDAImageMask {
Q_OBJECT
public:
	QGLFullCUDAImageItem(IImagePaletteHolder *image, QGraphicsItem *parent=0, bool applyMask =false);
	~QGLFullCUDAImageItem();

	void setMask();

	void updateImage(IImagePaletteHolder *image);

protected:
	virtual void preInitGL();
	virtual void postInitGL();
private:
	void initializeGL();
	void initializeCornerGL();
	void initializeShaders();

	void setPaletteParameter(QOpenGLShaderProgram *program);

protected:
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;
private:

	bool m_initialized = false;
	bool m_initializedCorner = false;
	IImagePaletteHolder *m_image;

	CUDAImageTextureMapper * m_mapper;


	QOpenGLBuffer m_vertexBuffer;
	QOpenGLShaderProgram *m_program;
};

#endif
