/*
 * GraphEditor_TextItem.h
 *
 *  Created on: Oct 27, 2021
 *      Author: l1046262
 */

#ifndef SRC_GRAPHICEDITOR_GRAPHEDITOR_TEXTITEM_H_
#define SRC_GRAPHICEDITOR_GRAPHEDITOR_TEXTITEM_H_

#include <QGraphicsTextItem>
#include <QPen>
#include <QPainter>
#include <QStyleOptionGraphicsItem>

QT_BEGIN_NAMESPACE
class QFocusEvent;
class QGraphicsItem;
class QGraphicsScene;
class QGraphicsSceneMouseEvent;
QT_END_NAMESPACE

class GraphEditor_TextItem : public QGraphicsTextItem
{
    Q_OBJECT

public:

    GraphEditor_TextItem(QGraphicsItem *parent = nullptr);

    bool contentIsUpdated() { return contentHasChanged; }
    bool positionIsUpdated() { return isMoved; }
    void setUpdated() { isMoved = false;  contentHasChanged=false;}
    GraphEditor_TextItem* clone();

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    void focusInEvent(QFocusEvent* event) override;
    void focusOutEvent(QFocusEvent *event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;


private:
    QString contentLastTime;
    QPointF positionLastTime;
    bool isMoved = false;
    bool contentHasChanged = false;
    bool m_AlreadySelected = true;
};

#endif /* SRC_GRAPHICEDITOR_GRAPHEDITOR_TEXTITEM_H_ */
