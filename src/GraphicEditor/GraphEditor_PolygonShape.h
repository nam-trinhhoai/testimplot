/*
 * GraphEditor_PolygonShape.h
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GRAPHEDITOR_POLYGONSHAPE_H_
#define SRC_GENERICEDITOR_GRAPHEDITOR_POLYGONSHAPE_H_

#include <QObject>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QGraphicsSimpleTextItem>
#include "qgraphicsitem.h"
#include "GraphEditor_Item.h"
#include "GraphEditor_ItemInfo.h"

class QGraphicsItem;
class GraphEditor_GrabberItem;
class QGraphicsSceneMouseEvent;

class GraphEditor_PolygonShape : public QObject, public QGraphicsPolygonItem, public GraphEditor_Item, public GraphEditor_ItemInfo
{
	Q_OBJECT

public:
	GraphEditor_PolygonShape(QPolygonF polygon, QPen pen, QBrush brush, QMenu*, bool isEditable = true);
	~GraphEditor_PolygonShape();

	void setPolygon(const QPolygonF &poly);
	GraphEditor_PolygonShape* clone();
	void ContextualMenu(QPoint)override;
	QVector<QPointF> SceneCordinatesPoints() override;
	QVector<QPointF> ImageCordinatesPoints() override;

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
	void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
	void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
	QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

	public slots:
	void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void slotDeleted(QGraphicsItem *signalOwner);

	signals:
	void ItemHasMoved();

	private:

	void clearGrabbers();
	void showGrabbers();
	void setGrabbersVisibility(bool visible);
	void calculateArea();

	//QGraphicsSimpleTextItem *m_textItem ;
	QList<GraphEditor_GrabberItem *> grabberList;
	bool m_IsEditable;

};

#endif /* SRC_GENERICEDITOR_GRAPHEDITOR_POLYGONSHAPE_H_ */
