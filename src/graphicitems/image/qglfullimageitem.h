#ifndef QGLFullImageItem_H
#define QGLFullImageItem_H

#include "qabstractglgraphicsitem.h"

#include <QGraphicsObject>
#include <QTransform>
#include <QOpenGLBuffer>

class QGLAbstractFullImage;
class QOpenGLShaderProgram;
class QOpenGLTexture;

class QGLFullImageItem: public QAbstractGLGraphicsItem {
Q_OBJECT
public:
	QGLFullImageItem(QGLAbstractFullImage *image, QGraphicsItem *parent=0);
	~QGLFullImageItem();

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

	bool m_initialized = false;
	QGLAbstractFullImage *m_image;

	QOpenGLBuffer m_vertexBuffer;
	QOpenGLShaderProgram *m_program;
};

#endif
