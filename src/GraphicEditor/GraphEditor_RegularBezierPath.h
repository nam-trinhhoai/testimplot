/*
 * GraphEditor_RegularBezierPath.h
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GraphEditor_RegularBezierPath_H_
#define SRC_GENERICEDITOR_GraphEditor_RegularBezierPath_H_

class QPainterPath;
#include "GraphEditor_Path.h"
#include "bezierinterpolator.h"

class GraphEditor_RegularBezierPath : public GraphEditor_Path
{
	Q_OBJECT

public:
	GraphEditor_RegularBezierPath(QPolygonF polygon, QPen pen, QBrush brush, int smoothValue,  QMenu*);
	GraphEditor_RegularBezierPath(QPolygonF polygon, QPen pen, QBrush brush,  QMenu*);
	~GraphEditor_RegularBezierPath();

	void setPolygon(const QPolygonF &poly);

	GraphEditor_RegularBezierPath* clone();
	//void ContextualMenu(QPoint)override;
	void setDrawFinished(bool value) override;
	void setSmoothValue(int);

	QVector<QPointF> SceneCordinatesPoints() override;
	QVector<QPointF> ImageCordinatesPoints() override;

	// void setResizeWidth(int ,int)override;


	void restoreState(QPolygonF poly, bool isSmoothed, bool isClosedCurve, int);

	int smoothValue()
	{
		return m_smoothValue;
	}
	QPolygonF initialPolygon()
	{
		return m_InitialPolygon;
	}

	QPolygonF polygon()
	{
		return m_polygon;
	}

	bool isSmoothed()
	{
		return m_IsSmoothed;
	}

	//int getIndexKey(int );
/*
	bool isClosedCurve()
	{
		return m_IsClosedCurved;
	}*/



	QVector<QPointF> getKeyPoints() override;


	 void bezierCurveByCasteljau(float u);
	 QPointF bezierCurveByCasteljauRec(QVector<QPointF> in_pts, float i);


/*	void setInitialWidth(float w)
	{
		m_initialWidth = w;
	}
*/
	void onPositionChanged(float u);

	void resizeTopBis(qreal dy) ;
	void resizeBottomBis(qreal dy);

	void insertNewPoints(QPointF);

	signals:
	//void BezierDeleted(QString);
	//void polygonChanged(QVector<QPointF>,bool);
	//void signalCurrentIndex(int);


	public slots:

	//void receiveAddPts(QPointF);
	void polygonResize(int,int);
	void polygonChanged1();
	void grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void moveGrabber(int, int dx, int dy);
	void positionGrabber(int, QPointF);

	void FirstPointSmoothedMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void LastPointSmoothedMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);
	void GrabberSmoothedMouseReleased(QGraphicsItem *signalOwner);

protected:
	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
	void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
	//QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
	void showGrabbers();
	void setPolygonFromMove(const QPolygonF &poly);


	private:
	void interpolateCurve();
	void fillKnotVector();
	void smoothCurve();

	void getMinMax(QPointF &min, QPointF &max);
	QPainterPath shape() const;
	int m_searchArea = 2;




	BezierInterpolator bezierInterpolator;
	QVector<qreal> knotVector;
	QPolygonF boorNetPoints;
	QPolygonF interpolatedPoints;
	QVector<QPointF*> controlPoints;

	QPolygonF m_InitialPolygon;
	int m_smoothValue;
	bool m_IsSmoothed = false;

	//float m_initialWidth = 3000.0f;

};


#endif /* SRC_GENERICEDITOR_GraphEditor_RegularBezierPath_H_ */
