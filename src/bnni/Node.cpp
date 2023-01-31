/*
 * Node.cpp
 *
 *  Created on: 16 févr. 2018
 *      Author: j0483271
 */

#include "Node.h"

#include <cmath>
#include <QDebug>
#include <QPainter>
#include <QPen>
#include <QGraphicsSceneMouseEvent>

namespace murat {
namespace gui {

Node::Node(qreal rx, qreal ry, QObject* parent) : QObject(parent) {
	// TODO Auto-generated constructor stub
	setFlag(ItemIsMovable);
	setFlag(ItemSendsGeometryChanges);
	setCacheMode(DeviceCoordinateCache);
	setZValue(15);

    ratioX = rx;
    ratioY = ry;
}

Node::~Node() {
	// TODO Auto-generated destructor stub
}

/*
 * Public Class Method of Node
 *
 * Update scale ratio
 */
void Node::updateSize(qreal sx, qreal sy) {
    //qDebug() << "Rectangle" << s << ratio << ratio/s;
    ratioX /= sx;
    ratioY /= sy;
}

/*
 * Protected Class Method
 *
 * Notify that geometry has changed
 */
QVariant Node::itemChange(GraphicsItemChange change, const QVariant &value) {
	return QGraphicsItem::itemChange(change, value);
}

/*
 * Protected Class Method
 *
 * Called by QGraphicsScene to paint the object
 * Define how to paint the object
 */
void Node::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
	QPen pen;
	pen.setColor(QColor(255,0,0));
    pen.setWidthF(ratioX*10);
	painter->setPen(pen);

	painter->drawPoint(pos());

}

/*
 * Protected Class Method
 *
 * Call QGraphicsItem::update
 */
void Node::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
	update();
	QGraphicsItem::mouseReleaseEvent(event);
	event->accept();
}

/*
 * Protected method
 *
 * emit rectangleChanged
 */
void Node::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
	QGraphicsItem::mouseMoveEvent(event);
	emit nodeChanged(this, event->pos(), event->lastPos());
	//qDebug() << "RectangleMovable mouseMoveEvent";
	event->accept();
}

/*
 * Public Class Method
 *
 * Return bounding rectangle of object looks
 */
QRectF Node::boundingRect() const {
    qreal adjust = (1+ratioX);
	// create rectangle taking into account the ratio
	return QRectF( -adjust, -adjust,
	             2* adjust,  2*adjust);
}

} /* namespace gui */
} /* namespace murat */
