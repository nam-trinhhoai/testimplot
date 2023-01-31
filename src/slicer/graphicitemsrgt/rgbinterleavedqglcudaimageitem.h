#ifndef RGBINTERLEAVEDQGLCUDAIMAGEITEM_H
#define RGBINTERLEAVEDQGLCUDAIMAGEITEM_H

#include "qabstractglgraphicsitem.h"
#include "CUDAImageMask.h"

#include <QGraphicsObject>
#include <QVector>
#include <QTransform>
#include <QOpenGLBuffer>

class QOpenGLTexture;
class QOpenGLShaderProgram;
class QGLContext;
class QOpenGLFunctions;
class CUDARGBInterleavedImage;
class IImagePaletteHolder;
class CUDARGBImageTextureMapper;
class CUDAImageTextureMapper;

class RGBInterleavedQGLCUDAImageItem: public QAbstractGLGraphicsItem, public CUDAImageMask {
Q_OBJECT
public:
	RGBInterleavedQGLCUDAImageItem(IImagePaletteHolder * isoSurfaceHolder,CUDARGBInterleavedImage * holder,int defaultExtractionWindow,QGraphicsItem *parent=0, bool addMask = false);
	~RGBInterleavedQGLCUDAImageItem();


	bool InsidePolygon(QVector<QPointF> polygon, QPointF p);

	// HSV
	bool minimumValueActive() const;
	void setMinimumValueActive(bool activated);
	float minimumValue() const;
	void setMinimumValue(float minValue);
signals:
	void initialized();
private:
	void initializeGL();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
				const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;
	void setPaletteParameter(QOpenGLShaderProgram *program);
protected:
	void initShaders();
protected:
	IImagePaletteHolder *m_isoSurfaceImage;
	CUDARGBInterleavedImage *m_image;

	CUDARGBImageTextureMapper * m_rgbMapper;


	CUDAImageTextureMapper * m_isoSurfaceMapper;

	//QOpenGLTexture* m_maskTex;

	QOpenGLBuffer m_vertexBuffer;
	//Depending on the texture format we must handle different samplers
	QOpenGLShaderProgram *m_program;

	bool m_initialized;

	bool m_minimumValueActive = false;
	float m_minimumValue = 0.0f;

};

#endif /* RGBINTERLEAVEDQGLCUDAIMAGEITEM_H */
