/*
 * GraphEditor_CurveShape.h
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GraphEditor_CurveShape_H_
#define SRC_GENERICEDITOR_GraphEditor_CurveShape_H_

#include "bezierinterpolator.h"
#include "GraphEditor_PolyLineShape.h"
#include <vector>

class GraphicSceneEditor;

class GraphEditor_CurveShape : public GraphEditor_PolyLineShape
{
	Q_OBJECT

public:
	GraphEditor_CurveShape(QPolygonF polygon, eShape shape, QPen pen, QBrush brush, QMenu*,
			GraphicSceneEditor* scene=nullptr, bool isClosedCurved = false);
	~GraphEditor_CurveShape();

	void setPolygon(const QPolygonF &poly);
	int type();
	QVector<QPointF> SceneCordinatesPoints() override;
	QVector<QPointF> ImageCordinatesPoints() override;
	GraphEditor_CurveShape* clone();


	void createCornerGrabbers();
	void setPositionCornerGrabbers();
	void setCornerGrabbersVisibility(bool);

	public slots:
		void cornerGrabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy);


protected:
	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

private:
	void setPolygonFromMove(const QPolygonF &poly);
	void interpolateCurve();
	void fillKnotVector();
	void interpolatePoints();
	float B(int i,int n,float u);
	QPointF drawBezier2(QPolygonF p,float u);
	QPointF drawRBezier(std::vector<QPointF> p,float u);
	std::vector<QPointF> OneSubDivide(std::vector<QPointF> p,std::vector<QPointF>* poly1,std::vector<QPointF>* poly2,float u);
	std::vector<QPointF> SubDivide(std::vector<QPointF> p,int m,float u);
	std::vector<int> genKnot(int n,int D);
	float N(int i,int d,float u,std::vector<int> T);
	std::vector<QPointF> BSubDivide(std::vector<QPointF> p,int m);



	void resizeLeftBis(qreal dx);
	void resizeRightBis(qreal dx);
	void resizeBottomBis(qreal dy);
	void resizeTopBis(qreal heightOffset);


	BezierInterpolator bezierInterpolator;
	QVector<qreal> knotVector;
	QPolygonF boorNetPoints;
	QPolygonF interpolatedPoints;
	QVector<QPointF*> controlPoints;
	eShape m_Curve;

	GraphEditor_GrabberItem *m_cornerGrabber[8];
};


#endif /* SRC_GENERICEDITOR_GraphEditor_CurveShape_H_ */
