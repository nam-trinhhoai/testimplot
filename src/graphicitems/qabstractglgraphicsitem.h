#ifndef QAbstractGLGraphicsItem_h
#define QAbstractGLGraphicsItem_h

#include <QGraphicsObject>
#include <QMatrix4x4>
#include <cmath>

class QStyleOptionGraphicsItem;
class QPainter;
class QOpenGLContext;
class QOpenGLShaderProgram;
class QPaintEngine;

#define NV_ROUND(_VAL)  ((float)(int)((_VAL) + 0.5f))

//Do nothing except calling OpenGL
class QAbstractGLGraphicsItem : public QGraphicsObject {
	Q_OBJECT
public:
	QAbstractGLGraphicsItem(QGraphicsItem *parent);
	virtual ~QAbstractGLGraphicsItem();

	void updateWorldExtent(const QRectF & val);

	virtual QRectF boundingRect() const;
	virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget);
	double NiceNum(double x, bool round);
	QOpenGLContext *getContext();
	template<typename T> static inline T nvMax(T lhs, T rhs)  { return lhs >= rhs ? lhs : rhs; }
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix, const QRectF &exposed,int width, int height, int dpiX,int dpiY) = 0;
protected:
	bool loadProgram(QOpenGLShaderProgram *program, const QString &vert,
			const QString &frag);
	QMatrix4x4 computeOrthoViewScaleMatrix(int w, int h);
protected:
	QRectF m_worldExtent;
};

#endif
