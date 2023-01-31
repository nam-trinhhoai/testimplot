/*
 * GraphEditor_GrabberItem.cpp
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#include "GraphEditor_GrabberItem.h"

#include <QBrush>
#include <QColor>
#include <QPen>
#include <QGraphicsSceneHoverEvent>
#include <QStyleOptionGraphicsItem>
#include <QDebug>
#include <QGraphicsScene>
#include <QGraphicsView>

GraphEditor_GrabberItem::GraphEditor_GrabberItem(QGraphicsItem *parentItem, QColor color)
{
	setParentItem(parentItem);
	setAcceptHoverEvents(true);
	setFlags(QGraphicsItem::ItemIgnoresTransformations|ItemIsMovable);

	setBrush(QBrush(color));
	int size =2;
	setRect(-size * 2,-size * 2, size * 4, size * 4);

	QPen pen(QColor(Qt::black),3);
	pen.setCosmetic(true);
	setPen(pen);

}
void GraphEditor_GrabberItem::setPos(qreal x, qreal y)
{
	QGraphicsItem::setPos(x,y);
}

void GraphEditor_GrabberItem::setPos(const QPointF &pos)
{
	QGraphicsItem::setPos(pos);
}

GraphEditor_GrabberItem::~GraphEditor_GrabberItem()
{

}

void GraphEditor_GrabberItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
	painter->setRenderHint(QPainter::Antialiasing,true);
	//painter->setRenderHint(QPainter::HighQualityAntialiasing, true);
	painter->setRenderHint(QPainter::SmoothPixmapTransform, true);


	if (m_detectCollision)
	{
		if (scenePos() != m_LastScenePos)
		{
			m_LastScenePos = scenePos();
			bool collideWithOtherGrabber = false;

			foreach (QGraphicsItem *p, scene()->items(sceneBoundingRect(), Qt::IntersectsItemShape, Qt::AscendingOrder,(scene()->views())[0]->transform()) )
			{
				if (dynamic_cast<GraphEditor_GrabberItem *>(p))
				{
					if (dynamic_cast<GraphEditor_GrabberItem *>(p)->isFirstGrabberIsList())
						collideWithOtherGrabber=true;
				}
			}

			if (collideWithOtherGrabber)
			{
				m_HasDetectedCollision = true;
				setBrush(QBrush(Qt::red));
				int size =4;
				setRect(-size * 2,-size * 2, size * 4, size * 4);

				QPen pen(QColor(Qt::black),3);
				pen.setCosmetic(true);
				setPen(pen);
			}
			else
			{
				m_HasDetectedCollision = false;
				setBrush(QBrush(Qt::white));
				int size =2;
				setRect(-size * 2,-size * 2, size * 4, size * 4);

				QPen pen(QColor(Qt::black),3);
				pen.setCosmetic(true);
				setPen(pen);
			}
		}
	}
	QStyleOptionGraphicsItem myoption = (*option);
	myoption.state &= !QStyle::State_Selected;
	QGraphicsEllipseItem::paint(painter, &myoption, widget);
	//setFlag(QGraphicsItem::ItemIgnoresTransformations,true);
}

void GraphEditor_GrabberItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	auto dx = event->scenePos().x() - m_previousPosition.x();
	auto dy = event->scenePos().y() - m_previousPosition.y();
	moveBy(dx,dy);
	m_previousPosition = event->scenePos();
	emit signalMove(this, dx, dy);
}

void GraphEditor_GrabberItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
	emit signalRelease(this);
}

void GraphEditor_GrabberItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
	emit signalDoubleClick(this);
}

void GraphEditor_GrabberItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	m_previousPosition = event->scenePos();
	emit signalClick(this);
	event->accept();
}

void GraphEditor_GrabberItem::moveX(qreal dx)
{
	QGraphicsItem::setX(x()+ dx);
}

void GraphEditor_GrabberItem::moveY(qreal dy)
{
	QGraphicsItem::setY(y()+ dy);
}

