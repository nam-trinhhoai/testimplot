#include "dotsignal.h"

#include <QBrush>
#include <QColor>
#include <QPen>
#include <QGraphicsSceneHoverEvent>
#include <QGraphicsSceneMouseEvent>

DotSignal::DotSignal(QGraphicsItem *parentItem, int size, QObject *parent) :
    QObject(parent)
{
    setParentItem(parentItem);
    setAcceptHoverEvents(true);
    setBrush(QBrush(Qt::white));
    setRect(-size * 2,-size * 2, size * 4, size * 4);
    setDotFlags(0);

    QPen pen(QColor(Qt::black),4);
    pen.setCosmetic(true);
    setPen(pen);
}

DotSignal::DotSignal(QPointF pos, int size, QGraphicsItem *parentItem, QObject *parent) :
    QObject(parent)
{
    setParentItem(parentItem);
    setAcceptHoverEvents(true);
    setBrush(QBrush(Qt::white));
    setRect(-size * 2,-size * 2, size * 4, size * 4);
    setPos(pos);
    setPreviousPosition(pos);
    setDotFlags(0);
    QPen pen(QColor(Qt::black),4);
    pen.setCosmetic(true);
    setPen(pen);
}

DotSignal::~DotSignal()
{

}

QPointF DotSignal::previousPosition() const
{
    return m_previousPosition;
}

void DotSignal::setPreviousPosition(const QPointF previousPosition)
{
    if (m_previousPosition == previousPosition)
        return;

    m_previousPosition = previousPosition;
    emit previousPositionChanged();
}

void DotSignal::setDotFlags(unsigned int flags)
{
    m_flags = flags;
}

void DotSignal::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    if(m_flags & Movable){
        auto dx = event->scenePos().x() - m_previousPosition.x();
        auto dy = event->scenePos().y() - m_previousPosition.y();
        moveBy(dx,dy);
        setPreviousPosition(event->scenePos());

        emit signalMove(this, dx, dy);
    } else {
        QGraphicsItem::mouseMoveEvent(event);
    }
}

void DotSignal::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
    emit signalDoubleClick(this);
}

void DotSignal::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if(m_flags & Movable){
        setPreviousPosition(event->scenePos());
        event->accept();
    } else {
        QGraphicsItem::mousePressEvent(event);
    }
}

void DotSignal::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    emit signalMouseRelease();
    QGraphicsItem::mouseReleaseEvent(event);
}

//void DotSignal::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
//{
//    Q_UNUSED(event)
//    //setBrush(QBrush(Qt::red));
//}

//void DotSignal::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
//{
//    Q_UNUSED(event)
//   // setBrush(QBrush(Qt::black));
//}
