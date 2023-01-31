#ifndef QGLGridItem_H_
#define QGLGridItem_H_

#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

class QOpenGLTexture;
class QGLContext;
class QOpenGLShaderProgram;

class QGLGridItem: public QAbstractGLGraphicsItem {
Q_OBJECT
public:
	QGLGridItem(const QRectF &worldExtent, QGraphicsItem *parent=0);
	~QGLGridItem();
	void setColor(QColor c);
public slots:
	void UpdateRatio(float);
private:
	void initializeGL();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
			const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;
	void initShaders();
	int updateInternalBuffer(const QRectF &exposed,int iWidth,int iHeight) ;
	void setupShader(const QMatrix4x4 &viewProjectionMatrix,
			const QMatrix4x4 &transfo);
protected:
	QOpenGLBuffer m_vertexBuffer;
	bool m_initialized;
	QOpenGLShaderProgram *m_program;
	QMatrix4x4 m_transfoMatrix;
	QVector4D m_color;

	float m_tickRatio;
	int m_numMaxPoints;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */
