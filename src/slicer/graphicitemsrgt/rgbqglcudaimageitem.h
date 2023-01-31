#ifndef RGBQGLCUDAImageItem_H
#define RGBQGLCUDAImageItem_H

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
class CUDARGBImage;
class CUDAImagePaletteHolder;
class CUDAImageTextureMapper;

class RGBQGLCUDAImageItem: public QAbstractGLGraphicsItem, public CUDAImageMask {
	Q_OBJECT
public:
	RGBQGLCUDAImageItem(CUDAImagePaletteHolder * isoSurfaceHolder,CUDARGBImage * holder,
			int defaultExtractionWindow,QGraphicsItem *parent=0, bool applyMAsk = false);
	~RGBQGLCUDAImageItem();


	void setTextureMask( const QImage& im);

	void updateImage(QGraphicsObject*, QGraphicsItem*);
	CUDAImagePaletteHolder *isoSurface()
	{
		return 	m_isoSurfaceImage;
	}

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
	CUDAImagePaletteHolder *m_isoSurfaceImage;
	CUDARGBImage *m_image;

	CUDAImageTextureMapper * m_redMapper;
	CUDAImageTextureMapper * m_greenMapper;
	CUDAImageTextureMapper * m_blueMapper;

	CUDAImageTextureMapper * m_isoSurfaceMapper;



	QOpenGLBuffer m_vertexBuffer;
	//Depending on the texture format we must handle different samplers
	QOpenGLShaderProgram *m_program;

	bool m_initialized;

	bool m_minimumValueActive = false;
	float m_minimumValue = 0.0f;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */
