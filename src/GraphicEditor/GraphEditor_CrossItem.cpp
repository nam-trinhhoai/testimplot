#include "GraphEditor_CrossItem.h"
#include <QPen>
#include <QDebug>

GraphEditor_CrossItem::GraphEditor_CrossItem(QGraphicsItem *parentItem)
{


	int m_sizeCross= 15;
	int posX = 0;
	int posY =0;
	m_lineVItem = new QGraphicsLineItem(posX-m_sizeCross,posY-m_sizeCross,posX+m_sizeCross,posY+m_sizeCross);
	m_lineVItem->setZValue(3000);
	QPen pen(Qt::yellow);
	pen.setWidth(2);
	pen.setCosmetic(true);
	m_lineVItem->setPen(pen);


	m_lineHItem = new QGraphicsLineItem(posX-m_sizeCross,posY+m_sizeCross,posX+m_sizeCross,posY-m_sizeCross);
	m_lineHItem->setZValue(3000);
	m_lineHItem->setPen(pen);


	addToGroup(m_lineHItem);
	addToGroup(m_lineVItem);



}

GraphEditor_CrossItem::~GraphEditor_CrossItem()
{
	if(m_lineVItem != nullptr)
	{
		delete m_lineVItem;
		m_lineVItem = nullptr;
	}
	if(m_lineHItem != nullptr)
	{
		delete m_lineHItem;
		m_lineHItem = nullptr;
	}
}

void GraphEditor_CrossItem::setPos(const QPointF &pos)
{
	QGraphicsItem::setPos(pos);
}

void GraphEditor_CrossItem::setColor(QColor c)
{
	QPen pen(m_lineVItem->pen());
	pen.setColor(c);
	m_lineVItem->setPen(pen);
	m_lineHItem->setPen(pen);

}

void GraphEditor_CrossItem::setGrabberCurrent(int index)
{
	m_indexGrabberCurrent = index;
}


void GraphEditor_CrossItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	auto dx = event->scenePos().x() - m_previousPosition.x();
	auto dy = event->scenePos().y() - m_previousPosition.y();
	moveBy(dx,dy);
	m_previousPosition = event->scenePos();
//	emit signalMoveCross(m_indexGrabberCurrent, dx*50, dy*50);
	emit signalPositionCross(m_indexGrabberCurrent, m_previousPosition);
}

void GraphEditor_CrossItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {

	emit signalMoveCrossFinish();
}


void GraphEditor_CrossItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	m_previousPosition = event->scenePos();

	emit signalAddPoints(m_previousPosition);
	//emit signalClick(this);
	//event->accept();
}
