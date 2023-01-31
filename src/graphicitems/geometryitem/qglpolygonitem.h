#ifndef QGLPolygonItem_H_
#define QGLPolygonItem_H_

#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

#include <QScopedPointer>
#include <qopenglextensions.h>

class QOpenGLShaderProgram;

class QGLPolygonItem: public QAbstractGLGraphicsItem {
	Q_OBJECT

	Q_PROPERTY(float lineWidth READ lineWidth CONSTANT WRITE setLineWidth)
	Q_PROPERTY(float opacity READ opacity CONSTANT WRITE setOpacity)
	Q_PROPERTY(QColor outlineColor READ outlineColor CONSTANT WRITE setOutlineColor)
	Q_PROPERTY(QColor interiorColor READ interiorColor CONSTANT WRITE setInteriorColor)
public:
	QGLPolygonItem(const QVector<QVector2D> & vertices,const QRectF &worldExtent,QGraphicsItem *parent=0);
	~QGLPolygonItem();

	float lineWidth() const;
	void setLineWidth(float value);

	float opacity() const;
	void setOpacity(float value);

	QColor outlineColor() const;
	void setOutlineColor(QColor c);

	QColor interiorColor() const;
	void setInteriorColor(QColor c);
private:
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix, const QRectF &exposed,int width, int height, int dpiX,int dpiY) override;
protected:
	GLubyte * m_pathCommands;
	GLfloat *  m_pathCoords;

	int m_numVertices;

	bool m_initialized;

	float m_lineWidth;

	QColor m_outlineColor;
	QColor m_interiorColor;

	float m_opacity;

	QScopedPointer<QOpenGLExtension_NV_path_rendering> m_nvPathFuncs;
};


#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */
