#ifndef QGLTiledImageItem_H
#define QGLTiledImageItem_H

#include "qabstractglgraphicsitem.h"
#include <QOpenGLBuffer>

class QGLRenderThread;
class QOpenGLTexture;
class QGLAbstractTiledImage;
class QOpenGLShaderProgram;
class QGLTile;

class QGLTiledImageItem: public QAbstractGLGraphicsItem {

	Q_OBJECT
public:
	QGLTiledImageItem(QGLAbstractTiledImage *image, QGraphicsItem *parent);
	~QGLTiledImageItem();
private:
	void initializeGL();
	void initializeShaders();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;

	void setPaletteParameter();
	void setupShader(const QMatrix4x4 & viewProjectionMatrix);

	void renderTile(const QMatrix4x4 &viewProjectionMatrix,QGLTile * tile);

private slots:
	void cacheImageInserted(QGLTile *queueItem);

private:
	QGLRenderThread* m_thread;
	QOpenGLTexture* m_transparentTile;

	QOpenGLBuffer m_vertexBuffer;
	QGLAbstractTiledImage *m_image;

	QOpenGLShaderProgram *m_program;
	bool m_initialized;

};


#endif /* QTLARGEIMAGEVIEWER_SRC_QTHREADEDGLGRAPHICSSCENE_H_ */
