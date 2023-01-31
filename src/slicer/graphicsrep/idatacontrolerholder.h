#ifndef IDataControlerHolder_H
#define IDataControlerHolder_H
#include <QWidget>
class DataControler;
class QGraphicsItem;

class IDataControlerHolder{
public:
	virtual ~IDataControlerHolder(){}

	virtual QGraphicsItem * getOverlayItem(DataControler * controler,QGraphicsItem *parent)=0;

	virtual void notifyDataControlerMouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys)=0;
	virtual void notifyDataControlerMousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys)=0;
	virtual void notifyDataControlerMouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys)=0;
	virtual void notifyDataControlerMouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys)=0;

	virtual QGraphicsItem * releaseOverlayItem(DataControler * controler)=0;
};

#endif
