#ifndef PickingTask_H
#define PickingTask_H
#include <QObject>
#include "pickinginfo.h"

class PickingTask: public QObject
{
Q_OBJECT
public:
	PickingTask(QObject *parent);

	virtual void mouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info){};
	virtual void mousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info){};
	virtual void mouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info){};
	virtual void mouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info){};

	~PickingTask();
};

#endif
