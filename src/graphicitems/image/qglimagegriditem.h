#ifndef QGLImageGridItem_H_
#define QGLImageGridItem_H_

#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>
#include <qopenglextensions.h>

class QOpenGLFunctions;
class QOpenGLTexture;
class QGLContext;
class QOpenGLShaderProgram;
class IGeorefImage;

class QGLImageGridItem: public QAbstractGLGraphicsItem {
Q_OBJECT
public:
	QGLImageGridItem(const IGeorefImage *const provider,
			QGraphicsItem *parent=0);
	~QGLImageGridItem();
	void setColor(QColor c);

private:
	void initializeGL();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;
	void initShaders();
	void updateInternalBuffer();
	void setupShader(const QMatrix4x4 &viewProjectionMatrix,const QMatrix4x4 &transfo);
	void drawText( GLuint glyphBase,const std::string &text);
protected:
	QOpenGLBuffer m_vertexBuffer;
	const IGeorefImage *  const m_provider;
	bool m_initialized;
	bool m_needUpdate;

	QOpenGLShaderProgram *m_program;
	QVector4D m_color;

	QScopedPointer<QOpenGLExtension_NV_path_rendering> m_nvPathFuncs;

	QMatrix4x4 m_transfoMatrix;
	double angleTextX;
	double angleTextY;

	int m_width;
	int m_height;

	int m_lineCount;

	int m_tickInterval;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */
