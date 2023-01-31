/*
 * GraphEditor_PolyLineShape.h
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GRAPHEDITOR_POLYLINESHAPE_H_
#define SRC_GENERICEDITOR_GRAPHEDITOR_POLYLINESHAPE_H_

#include <QPainterPath>
class QGraphicsSimpleTextItem;
class LineLengthText;
class GraphicSceneEditor;
#include "GraphEditor_Path.h"


class GraphEditor_PolyLineShape : public GraphEditor_Path
{
	Q_OBJECT
public:
	GraphEditor_PolyLineShape(QPolygonF polygon, QPen pen, QBrush brush, QMenu*,
			GraphicSceneEditor* scene=nullptr, bool issClosedCurved=false);
	GraphEditor_PolyLineShape();
    ~GraphEditor_PolyLineShape();

    virtual void setPolygon(const QPolygonF &poly);
    GraphEditor_PolyLineShape* clone();
	QVector<QPointF> SceneCordinatesPoints() override;
	QVector<QPointF> ImageCordinatesPoints() override;
	void setDrawFinished(bool value) override;
	bool checkClosedPath();

	void insertNewPoints(QPointF) override;

	QVector<QPointF> getKeyPoints() override;
	signals:
	void currentIndexChanged(int);
	void polygonChanged(QVector<QPointF>,bool);

	public slots:
	void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void GrabberMouseReleased(QGraphicsItem *signalOwner);
	void moveGrabber(int, int dx, int dy);
	void positionGrabber(int, QPointF);
	void polygonChanged1();
	void polygonResize(int,int);
	void setDisplayPerimetre(bool value) { m_displayPerimetre = value; }

	public:
	void wheelEvent(QGraphicsSceneWheelEvent *event) override;
	void setRotation(qreal angle) { QGraphicsItem::setRotation(0.0); }
	bool sceneEvent(QEvent *event);

	protected slots:
		bool eventFilter(QObject* watched, QEvent* ev) override;

private:
	double m_scale = .1;
	QPainterPath shape() const override;
	bool m_displayPerimetre = true;
protected:
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    virtual void setPolygonFromMove(const QPolygonF &poly);
    virtual void showGrabbers();
	void calculatePerimeter();
};


#endif /* SRC_GENERICEDITOR_GRAPHEDITOR_POLYLINESHAPE_H_ */
