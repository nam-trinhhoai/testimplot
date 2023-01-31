#ifndef VEPOLYLINE_H
#define VEPOLYLINE_H

#include <QObject>
#include "GeShape.h"
#include "qgraphicsitem.h"
#include "ipopmenu.h"

class QGraphicsItem;
class DotSignal;
class QGraphicsSceneMouseEvent;

class GePolyLine : public QObject, public QGraphicsPathItem, public GeShape, public IPopMenu
{
    Q_OBJECT
    Q_PROPERTY(QPointF previousPosition READ previousPosition WRITE setPreviousPosition NOTIFY previousPositionChanged)

public:
    explicit GePolyLine(/*data::MtData* vaData,*/ GeGlobalParameters& globalParameters,
    		const GeObjectId& objectId, QColor& color,
    		bool pickedHere, QObject * parent = 0);
    ~GePolyLine();

    QPointF previousPosition() const;
    void setPreviousPosition(const QPointF previousPosition);
    const QPolygonF& polygon() const;
    void setPolygon(const QPolygonF &poly);
    void setPolygonFromMove(const QPolygonF &poly);
    void updateGrabbers();

    void setColor(QColor color);

    virtual void fillContextMenu(QPointF scenePos, QContextMenuEvent::Reason, QMenu& mainMenu) override;

signals:
    void previousPositionChanged();
    void clicked(GePolyLine *rect);
    void signalMove(QGraphicsItem *item, qreal dx, qreal dy);
	/**
	 * The boolean flag means this call is supposed to be the last of a move action
	 */
	void polygonChanged(const QPolygonF& polygon, GeObjectId& modifiedObjectId, bool end, GePolyLine* originGraphicsItem);
	void polygonPointAddedSignal(int modifiedObjectId, int index, double factor);
	void polygonPointDeletedSignal(int modifiedObjectId, int index);
    void itemSelected(int objectId);
    void contextMenuSignal(QPointF scenePos, QContextMenuEvent::Reason, QMenu& mainMenu);

protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;

public slots:

private slots:
    void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
    void slotDeleted(QGraphicsItem *signalOwner);
    void updateColor();
    void setGrabbersColor(const QBrush &brush);
    void clearGrabbers();
    void showGrabbers();
    void setGrabbersVisibility(bool visible);

private:
    void notifyObservers();

    QPointF m_previousPosition;
    bool m_leftMouseButtonPressed;
    QList<DotSignal *> grabberList;
    int m_pointForCheck;
    QPolygonF m_polygon;
};

#endif // VEPOLYGON_H
