/*
 * GraphEditor_CrossItem.h
 *
 *  Created on: avril 11, 2022
 *      Author: l1049100 (sylvain)
 */

#ifndef GRAPHEDITOR_CROSSITEM_H
#define GRAPHEDITOR_CROSSITEM_H

#include <QObject>
#include <QGraphicsItemGroup>
#include <QGraphicsLineItem>
#include <QGraphicsSceneMouseEvent>



class GraphEditor_CrossItem : public QObject, public QGraphicsItemGroup
{
	Q_OBJECT

public:
	GraphEditor_CrossItem(QGraphicsItem *parentItem= nullptr);
	~GraphEditor_CrossItem();

	void setPos(const QPointF &pos);
	void setColor(QColor);


signals:
	void signalMoveCross(int,int,int);
	void signalPositionCross(int,QPointF);
	void signalAddPoints(QPointF);
	void signalMoveCrossFinish();

public slots:
	void setGrabberCurrent(int index);


protected:
	void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
	//void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
	//QRectF boundingRect()const override;
private:

	int m_indexGrabberCurrent = -1;


	QGraphicsLineItem * m_lineVItem = nullptr;
	QGraphicsLineItem * m_lineHItem = nullptr;

	QPointF m_previousPosition;

};
#endif
