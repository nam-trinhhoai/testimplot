/*
 * GraphEditor_EllipseShape.h
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GRAPHEDITOR_ELLIPSESHAPE_H_
#define SRC_GENERICEDITOR_GRAPHEDITOR_ELLIPSESHAPE_H_

#include <QObject>
#include <QGraphicsEllipseItem>
#include <QBrush>
#include <QPen>
#include "GraphEditor_Item.h"
#include "GraphEditor_ItemInfo.h"

class GraphEditor_GrabberItem;

class GraphEditor_EllipseShape : public QObject, public QGraphicsEllipseItem, public GraphEditor_Item, public GraphEditor_ItemInfo
{
	Q_OBJECT
public:
	GraphEditor_EllipseShape(QRectF, QPen, QBrush, QMenu*);

	~GraphEditor_EllipseShape();

	void setRect(const QRectF &rect);
	GraphEditor_EllipseShape* clone();
	void ContextualMenu(QPoint)override;
	QVector<QPointF> SceneCordinatesPoints() override;
	QVector<QPointF> ImageCordinatesPoints() override;

	QVector<QPointF> getKeyPoints();

public slots:
	void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void polygonResize(int,int);

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
	void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
	void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

	QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
	void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

	private:

	enum ActionStates {
		ResizeState = 0x01,
		RotationState = 0x02
	};

	enum CornerFlags {
		Top = 0x01,
		Bottom = 0x02,
		Left = 0x04,
		Right = 0x08,
		TopLeft = Top|Left,
		TopRight = Top|Right,
		BottomLeft = Bottom|Left,
		BottomRight = Bottom|Right
	};

	enum CornerGrabbers {
		GrabberTop = 0,
		GrabberBottom,
		GrabberLeft,
		GrabberRight,
		GrabberTopLeft,
		GrabberTopRight,
		GrabberBottomLeft,
		GrabberBottomRight
	};


	GraphEditor_GrabberItem *m_cornerGrabber[8];
	QBrush grabberBrush;


	void resizeLeft( const QPointF &pt);
	void resizeLeftBis(qreal dx);
	void resizeRight( const QPointF &pt);
	void resizeRightBis(qreal dx);
	void resizeBottom(const QPointF &pt);
	void resizeBottomBis(qreal dy);
	void resizeTop(const QPointF &pt);
	void resizeTopBis(qreal heightOffset);

	void createGrabbers();
	void setPositionGrabbers();
	void setGrabbersVisibility(bool);
	void calculateArea();
};

#endif /* SRC_GENERICEDITOR_GRAPHEDITOR_ELLIPSESHAPE_H_ */
