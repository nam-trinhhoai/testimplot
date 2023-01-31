/*
 * Node.h
 *
 *  Created on: 16 févr. 2018
 *      Author: j0483271
 */

#ifndef MURATAPP_SRC_VIEW_CANVAS2D_NODE_H_
#define MURATAPP_SRC_VIEW_CANVAS2D_NODE_H_

class QGraphicsView;
class QGraphicsSceneMouseEvent;
class QPainter;
class QStyleOption;
#include <QObject>
#include <QGraphicsItem>

namespace murat {
namespace gui {

class Node : public QObject, public QGraphicsItem {
    Q_OBJECT
	Q_INTERFACES(QGraphicsItem)
public:
    Node(qreal rx, qreal ry, QObject* parent=0);
	virtual ~Node();

    /*
     * Public Class Method of Node
     *
     * Update scale ratio
     */
    void updateSize(qreal, qreal);

    /*
     * Public Class Method
     *
     * Return bounding rectangle of object looks
     */
    QRectF boundingRect() const override;

protected:
    /*
     * Protected Class Method
     *
     * Notify that geometry has changed
     */
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;

    /*
     * Protected Class Method
     *
     * Called by QGraphicsScene to paint the object
     * Define how to paint the object
     */
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

    /*
     * Protected Class Method
     *
     * Call QGraphicsItem::update
     */
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

    /*
     * Protected method
     *
     * emit rectangleChanged
     */
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;

    qreal ratioX;
    qreal ratioY;

signals:
	/*
 	 * Class Signal
 	 *
 	 * Is emited when node position has changed
 	 */
	void nodeChanged(Node* node, QPointF pos, QPointF lastPos);
};

} /* namespace gui */
} /* namespace murat */

#endif /* MURATAPP_SRC_VIEW_CANVAS2D_NODE_H_ */
