#ifndef DOTSIGNAL_H
#define DOTSIGNAL_H

#include <QObject>
#include "qgraphicsitem.h"

class QGraphicsSceneHoverEventPrivate;
class QGraphicsSceneMouseEvent;
class QGraphicsItem;

class DotSignal : public QObject, public QGraphicsEllipseItem
{
    Q_OBJECT
    Q_PROPERTY(QPointF previousPosition READ previousPosition WRITE setPreviousPosition NOTIFY previousPositionChanged)

public:
    explicit DotSignal(QGraphicsItem *parentItem, int size, QObject *parent = 0);
    explicit DotSignal(QPointF pos, int size, QGraphicsItem *parentItem = 0, QObject *parent = 0);
    ~DotSignal();

    enum Flags {
        Movable = 0x01
    };

    QPointF previousPosition() const;
    void setPreviousPosition(const QPointF previousPosition);

    void setDotFlags(unsigned int flags);

signals:
    void previousPositionChanged();
    void signalMouseRelease();
    void signalMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
    void signalDoubleClick(QGraphicsItem *signalOwner);

protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
  //  void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
   // void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);

public slots:

private:
    unsigned int m_flags;
    QPointF m_previousPosition;
};

#endif // DOTSIGNAL_H
