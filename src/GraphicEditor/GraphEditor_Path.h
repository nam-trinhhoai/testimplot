/*
 * GraphEditor_Path.h
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GraphEditor_Path_H_
#define SRC_GENERICEDITOR_GraphEditor_Path_H_

#include <QObject>
#include <QPainter>
#include <QGraphicsPathItem>
#include <QStyleOptionGraphicsItem>

#include "GraphEditor_Item.h"
#include "GraphEditor_ItemInfo.h"

class QGraphicsItem;
class GraphEditor_GrabberItem;
class QGraphicsSceneMouseEvent;

class GraphEditor_Path : public QObject, public QGraphicsPathItem, public GraphEditor_Item, public GraphEditor_ItemInfo
{
	Q_OBJECT

public:
	GraphEditor_Path() {}/*setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	setFlag(ItemSendsGeometryChanges,true);
	setFlag(ItemIsMovable, true );};*/
	GraphEditor_Path(QPolygonF polygon, QPen pen, QBrush brush, QMenu*, bool isClosedPath = false);
	~GraphEditor_Path();

	GraphEditor_Path* clone();
	virtual void setPolygon(const QPolygonF &poly);
	const QPolygonF& polygon() const;

	void ContextualMenu(QPoint)override;
	QVector<QPointF> SceneCordinatesPoints() override;
	QVector<QPointF> ImageCordinatesPoints() override;
	virtual void setDrawFinished(bool value);
	void setClosedPath(bool value);
	bool isClosedPath()
	{
		return m_IsClosedCurved;
	}

	bool isDrawFinished()
	{
		return m_DrawFinished;
	}

	virtual QVector<QPointF> getKeyPoints();
	virtual void insertNewPoints(QPointF);

	QString getNameNurbs();
	void setNameNurbs(QString s);

	QString getNameId();
	void setNameId(QString s);
	bool sceneEvent(QEvent *event);

	void wheelEvent(QGraphicsSceneWheelEvent *event);


	signals:
	void BezierSelected( GraphEditor_Path*);
	void BezierDeleted(QString);
	void polygonChanged(QVector<QPointF>,bool);
	void signalCurrentIndex(int);

	private slots:
	bool eventFilter(QObject* watched, QEvent* ev);

	public slots:
	void FirstPointMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void LastPointMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void moveGrabber(int, int dx, int dy);
	void positionGrabber(int, QPointF);
	void polygonChanged1();
	void polygonResize(int widthO, int width);


	void slotDeleted(QGraphicsItem *signalOwner);
	void GrabberMouseReleased(QGraphicsItem *signalOwner);

	void receiveAddPts(QPointF);

/*
	void setRotation(qreal angle) {
		if ( !m_readOnly ) QGraphicsItem::setRotation(angle);
	}
	*/

	protected:
	void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
	QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
	// void wheelEvent(QGraphicsSceneWheelEvent *event);

	virtual void setPolygonFromMove(const QPolygonF &poly);
	void clearGrabbers();
	virtual void showGrabbers();
	void setGrabbersVisibility(bool visible);
	void calculateArea(QPolygonF);

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

	QPolygonF m_polygon;
	QList<GraphEditor_GrabberItem *> grabberList;
	bool m_IsClosedCurved=false;
	bool m_DrawFinished = false;

	QString m_nameId="";
	QString m_nameNurbs="";

	int m_nbPtsMin = 2;

};


#endif /* SRC_GENERICEDITOR_GraphEditor_Path_H_ */
