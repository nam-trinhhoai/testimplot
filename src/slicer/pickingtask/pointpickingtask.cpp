#include "pointpickingtask.h"

PointPickingTask::PointPickingTask(QObject *parent) :
PickingTask(parent), m_mouseButton(Qt::AllButtons) {

}
PointPickingTask::PointPickingTask(Qt::MouseButton mouseButton, QObject *parent) :
PickingTask(parent), m_mouseButton(mouseButton) {

}
PointPickingTask::~PointPickingTask() {
}

void PointPickingTask::mousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info){
	if (m_mouseButton & button) {
		emit pointPicked(worldX, worldY, button,keys,info);
	}

}
