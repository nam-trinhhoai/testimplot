#ifndef QGLFullCUDARgbaImageItem_H
#define QGLFullCUDARgbaImageItem_H

#include "qabstractglgraphicsitem.h"
#include "imageformats.h"
#include "rgbdataset.h"

#include <QGraphicsObject>
#include <QTransform>
#include <QOpenGLBuffer>

class CUDAImagePaletteHolder;
class QOpenGLShaderProgram;
class QOpenGLTexture;
class CUDAImageTextureMapper;

class QGLFullCUDARgbaImageItem: public QAbstractGLGraphicsItem {
Q_OBJECT
public:
	QGLFullCUDARgbaImageItem(CUDAImagePaletteHolder *red, CUDAImagePaletteHolder *green,
			CUDAImagePaletteHolder *blue, CUDAImagePaletteHolder *alpha, RgbDataset::Mode mode, QGraphicsItem *parent=0);
	~QGLFullCUDARgbaImageItem();

	void setOpacity(float opacity);
	void setConstantAlpha(float alpha); // taken into account if the correct mode is used
	void setRadiusAlpha(float radius); // taken into account if the correct mode is used

protected:
	virtual void preInitGL();
	virtual void postInitGL();
private:
	void initializeGL();
	void initializeShaders();

	void setPaletteParameter(QOpenGLShaderProgram *program);

protected:
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;
private:
	QString createFragmentShader();
	bool loadProgram(QOpenGLShaderProgram *program,
			const QString &vert, const QString &frag);
	static QString samplerType(ImageFormats::QSampleType type);

	bool m_initialized = false;
	CUDAImagePaletteHolder *m_red;
	CUDAImagePaletteHolder *m_green;
	CUDAImagePaletteHolder *m_blue;
	CUDAImagePaletteHolder *m_alpha;

	CUDAImageTextureMapper * m_mapperRed;
	CUDAImageTextureMapper * m_mapperGreen;
	CUDAImageTextureMapper * m_mapperBlue;
	CUDAImageTextureMapper * m_mapperAlpha;

	RgbDataset::Mode m_alphaMode;
	float m_radiusAlpha;

	QOpenGLBuffer m_vertexBuffer;
	QOpenGLShaderProgram *m_program;
	float m_opacity;
};

#endif
