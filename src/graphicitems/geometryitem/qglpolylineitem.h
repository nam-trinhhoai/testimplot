#ifndef QGLPolylineItem_H_
#define QGLPolylineItem_H_

#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

class QOpenGLShaderProgram;

class QGLPolylineItem: public QAbstractGLGraphicsItem {
	Q_OBJECT

	Q_PROPERTY(float lineWidth READ lineWidth CONSTANT WRITE setLineWidth)
	Q_PROPERTY(float opacity READ opacity CONSTANT WRITE setOpacity)
	Q_PROPERTY(QColor lineColor READ lineColor CONSTANT WRITE setLineColor)

public:
	QGLPolylineItem(const QVector<QVector2D> & vertices,const QRectF &worldExtent,QGraphicsItem *parent=0);
	~QGLPolylineItem();

	float lineWidth() const;
	void setLineWidth(float value);

	float opacity() const;
	void setOpacity(float value);

	QColor lineColor() const;
	void setLineColor(QColor c);

private:
	void initBuffers();
	void initShaders();

	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix, const QRectF &exposed,int width, int height, int dpiX,int dpiY) override;
protected:
	QOpenGLBuffer m_vertexBuffer;

	QOpenGLShaderProgram* m_program;

	float m_lineWidth;
	float m_opacity;
	QColor m_lineColor;

	std::vector<float> m_vertices;

	bool m_initialized;

	QMatrix4x4 m_transfoMatrix;

};


#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */
