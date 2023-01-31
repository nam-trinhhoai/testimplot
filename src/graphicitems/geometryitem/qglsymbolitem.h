#ifndef QGLSymbolItem_H_
#define QGLSymbolItem_H_

#include "qabstractglgraphicsitem.h"
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

#include <QScopedPointer>
#include <qopenglextensions.h>

class QOpenGLFunctions;
class QOpenGLTexture;
class QGLContext;
class QOpenGLShaderProgram;

class QGLSymbolItem: public QAbstractGLGraphicsItem {
	Q_OBJECT

	Q_PROPERTY(float size READ size CONSTANT WRITE setSize)
	Q_PROPERTY(QColor color READ color CONSTANT WRITE setColor)
public:
	QGLSymbolItem(const QPointF & point, const char symbol,const QRectF &worldExtent,QGraphicsItem *parent=0);
	~QGLSymbolItem();

	float size() const;
	void setSize(float value);

	QColor color() const;
	void setColor(QColor value);
private:
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix, const QRectF &exposed,int width, int height, int dpiX,int dpiY) override;
protected:
	float m_size;
	QColor m_color;
	QPointF  m_point;

	bool m_initialized;
	char m_symbol;

	QScopedPointer<QOpenGLExtension_NV_path_rendering> m_nvPathFuncs;
};


#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */
