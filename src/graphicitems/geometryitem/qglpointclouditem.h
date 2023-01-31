#ifndef QGLPointCloudItem_H_
#define QGLPointCloudItem_H_

#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

class QOpenGLShaderProgram;

class QGLPointCloudItem: public QAbstractGLGraphicsItem {
	Q_OBJECT

	Q_PROPERTY(float pointSize READ pointSize CONSTANT WRITE setPointSize)
	Q_PROPERTY(float opacity READ opacity CONSTANT WRITE setOpacity)
	Q_PROPERTY(QColor pointColor READ pointColor CONSTANT WRITE setPointColor)

public:
	QGLPointCloudItem(const QVector<QVector2D> & vertices,const QRectF &worldExtent,QGraphicsItem *parent=0);
	~QGLPointCloudItem();

	QColor pointColor() const;
	void setPointColor(QColor c);

	float opacity() const;
	void setOpacity(float value);

	void setPointSize(float size);
	float pointSize() const;
private:
	void initBuffers();
	void initShaders();

	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix, const QRectF &exposed,int width, int height, int dpiX,int dpiY) override;
protected:
	QOpenGLBuffer m_vertexBuffer;

	QOpenGLShaderProgram* m_program;
	QColor m_color;

	std::vector<float> m_vertices;

	bool m_initialized;

	QMatrix4x4 m_transfoMatrix;

	float m_pointSize;
	float m_opacity;
};


#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */
