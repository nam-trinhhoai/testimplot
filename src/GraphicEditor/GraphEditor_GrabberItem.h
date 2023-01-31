/*
 * GraphEditor_GrabberItem.h
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GRAPHEDITOR_GRABBERITEM_H_
#define SRC_GENERICEDITOR_GRAPHEDITOR_GRABBERITEM_H_

#include "qgraphicsitem.h"
#include <QObject>
#include <QPainter>

class QGraphicsSceneHoverEventPrivate;
class QGraphicsSceneMouseEvent;
class QGraphicsSceneHoverEvent;
class QGraphicsItem;
class QStyleOptionGraphicsItem;

class GraphEditor_GrabberItem : public QObject , public QGraphicsEllipseItem
{
	Q_OBJECT

public:
	GraphEditor_GrabberItem(QGraphicsItem *parentItem,QColor color=Qt::white);
	~GraphEditor_GrabberItem();
	void setPos(const QPointF &pos);
	void setPos(qreal x, qreal y);

	void moveX(qreal dx);
	void moveY(qreal dy);
	void setDetectCollision(bool val)
	{
		m_detectCollision = val;
	}
	bool hasDetectedCollision()
	{
		return m_HasDetectedCollision;
	}
	void setFirstGrabberInList(bool value)
	{
		m_IsFirstGrabberInList = true;
	}
	bool isFirstGrabberIsList()
	{
		return m_IsFirstGrabberInList;
	}
	signals:
	void signalMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void signalDoubleClick(QGraphicsItem *signalOwner);
	void signalClick(QGraphicsItem *signalOwner);
	void signalRelease(QGraphicsItem *signalOwner);

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
	void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

private:
	void detectCollision();
	QPointF m_previousPosition;
	QPointF m_LastScenePos;
	bool m_detectCollision = false;
	bool m_HasDetectedCollision = false;
	bool m_IsFirstGrabberInList= false;
};


#endif /* SRC_GENERICEDITOR_GRAPHEDITOR_GRABBERITEM_H_ */
