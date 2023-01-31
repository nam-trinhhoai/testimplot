#ifndef RECTANGLE_H
#define RECTANGLE_H

#include <QObject>
#include <QObject>
#include <QGraphicsRectItem>
#include <QBrush>
#include <QColor>

#include <qgraphicsitem.h>
#include "GeShape.h"
class DotSignal;
class QGraphicsSceneMouseEvent;

class GeRectangle : public QObject, public QGraphicsRectItem, public GeShape
{
    Q_OBJECT
    Q_PROPERTY(QPointF previousPosition READ previousPosition WRITE setPreviousPosition NOTIFY previousPositionChanged)

public:
    explicit GeRectangle(/*data::MtData* data,*/ GeGlobalParameters& globalParameters, const QString objectName,
    		const GeObjectId& objectId, QColor& color,	bool pickedHere, QObject * parent = 0);
    explicit GeRectangle(/*data::MtData* dData,*/ GeGlobalParameters& globalParameters, const QString objectName,
    		const GeObjectId& objectId, QColor& color,
    		bool pickedHere, QRectF& rect, QObject * parent = 0);
    ~GeRectangle();

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

    QPointF previousPosition() const;
    void setPreviousPosition(const QPointF previousPosition);

    void setRect(qreal x, qreal y, qreal w, qreal h);
    void setRect(const QRectF &rect);
    QRectF getRect();
    void setPositionGrabbers();
    void setColColororDots(const QBrush &brush);
	void setColorDots(const QBrush &brush);


signals:
    void rectChanged(GeRectangle *rect);
    void previousPositionChanged();
    void clicked(GeRectangle *rect);
    void signalMove(QGraphicsItem *item, qreal dx, qreal dy);
	/**
	 * The boolean flag means this call is supposed to be the last of a move action
	 */
	void rectangleChanged(const QRectF& newRect, int modifiedObjectId, bool end);
	void itemSelected(int objectId);

protected:
//	virtual void mouseMoved(double worldX,double worldY,
//			Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;
//	virtual void mousePressed(double worldX,double worldY,
//			Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
	void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

private:
    unsigned int m_cornerFlags;
    unsigned int m_actionFlags;
    QPointF m_previousPosition;
    bool m_leftMouseButtonPressed;
    DotSignal *m_cornerGrabber[8];
    bool m_grabberOn = false;
    QBrush grabberBrush;
    QGraphicsSimpleTextItem *textItem;
    QString objectName;

    void updateColor();
    void resizeLeft( const QPointF &pt);
    void resizeLeftBis(qreal dx);
    void resizeRight( const QPointF &pt);
    void resizeRightBis(qreal dx);
    void resizeBottom(const QPointF &pt);
    void resizeBottomBis(qreal dy);
    void resizeTop(const QPointF &pt);
    void resizeTopBis(qreal heightOffset);

    void rotateItem(const QPointF &pt);
    void setVisibilityGrabbers();
    void hideGrabbers();
    void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);

    void notifyObservers(); // emit rectangleChanged signal for "observers"
};


//Q_DECLARE_METATYPE(VERectangle* )

#endif // RECTANGLE_H
