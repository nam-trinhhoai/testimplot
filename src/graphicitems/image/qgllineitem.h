#ifndef QGLLineItem_H
#define QGLLineItem_H

#include "qabstractglgraphicsitem.h"

#include <QGraphicsObject>
#include <QTransform>
#include <QOpenGLBuffer>
#include <QVector4D>

class IGeorefImage;
class QOpenGLShaderProgram;
class QGraphicsSceneMouseEvent;
class SectionNumText;

//Represent a line on a grid represented by extent and on top of which we apply a GeorefImage transfo (could be null in this case identitity is applied)
class QGLLineItem: public QAbstractGLGraphicsItem {
	Q_OBJECT
public:
	enum Direction{HORIZONTAL,VERTICAL};
	QGLLineItem( const QRectF & imageExtent,const IGeorefImage * const imageToWorld,   Direction dir = Direction::HORIZONTAL,QGraphicsItem *parent=0);
	~QGLLineItem();

	void setColor(QColor c);

	void mouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
	void mousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
	void mouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);

signals:
	void positionChanged(int position);

public slots:
	void updatePosition(int position);
private:
	void initializeGL();
	virtual void drawGL(const QMatrix4x4 &viewProjectionMatrix,
				const QRectF &exposed, int width, int height, int dpiX,int dpiY) override;
	void initShaders();

	void computeLinePosition();

	bool computePosition(double worldX, double worldY, int &pos);
protected:
	const IGeorefImage * const m_extentToWorld;
	Direction m_dir;
	QOpenGLBuffer m_vertexBuffer;

	bool m_initialized;
	bool m_needUpate;

	QOpenGLShaderProgram* m_program;
	float m_indexPos;
	QVector4D m_color;
	bool m_startDrag;

	QRectF m_imageExtent;

	QMatrix4x4 m_transfo;
	float m_LineWidth;
	SectionNumText *m_textItem;
};


#endif /* QTCUDAIMAGEVIEWER_SRC_ABSTRACTQGLGRAPHICSITEM_H_ */
