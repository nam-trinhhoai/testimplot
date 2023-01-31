#ifndef PointPickingTask_H
#define PointPickingTask_H


#include "pickingtask.h"

class PointPickingTask: public PickingTask
{
Q_OBJECT
public:
	PointPickingTask(QObject *parent);
	PointPickingTask(Qt::MouseButton mouseButton, QObject *parent);
	~PointPickingTask();

	virtual void mousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info);

signals:
	void pointPicked(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys, const QVector<PickingInfo> & info);

protected:
	Qt::MouseButton m_mouseButton;
};


#endif
