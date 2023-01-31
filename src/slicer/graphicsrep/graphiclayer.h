#ifndef GraphicLayer_H
#define GraphicLayer_H
#include <QObject>

class QGraphicsScene;

class GraphicLayer: public QObject
{
	  Q_OBJECT
public:
	virtual ~GraphicLayer();

	virtual void show()=0;
	virtual void hide()=0;

	virtual QRectF boundingRect() const = 0;
	virtual void mouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys){}
	virtual void mousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys){}
	virtual void mouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys){}
	virtual void mouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys){}
	virtual void refresh()=0;

	signals:
	void boundingRectChanged(QRectF);

protected:
	GraphicLayer(QGraphicsScene *scene, int defaultZDepth);
protected:
	QGraphicsScene *m_scene;
	int m_defaultZDepth;
	bool isShown = true;
};

#endif
