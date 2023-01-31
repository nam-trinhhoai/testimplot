/*
 * GraphEditor_CurveShape.cpp
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */


#include <QGraphicsSceneMouseEvent>
#include <QPainterPath>
#include <QGraphicsScene>
#include <QGraphicsPathItem>
#include <QDebug>
#include <math.h>
#include <QGraphicsScene>
#include <QGraphicsView>

#include "GraphEditor_GrabberItem.h"
#include "GraphEditor_CurveShape.h"
#include "GraphicSceneEditor.h"

using namespace std;

std::vector<std::vector<int>> pascal;

GraphEditor_CurveShape::GraphEditor_CurveShape(QPolygonF polygon, eShape shape, QPen pen, QBrush brush, QMenu* itemMenu,
		GraphicSceneEditor* scene, bool isClosedCurved) : GraphEditor_PolyLineShape(polygon,pen,brush,itemMenu,scene,isClosedCurved)
{
	vector<int>elements;
	for(int j=0;j<1001;j++){
		for(int i=0;i<=j;i++){
			if(i==0 || i==j)
				elements.push_back(1);
			else
				elements.push_back(pascal[j-1][i-1]+pascal[j-1][i]);
		}
		pascal.push_back(elements);
		elements.clear();
	}
	m_Curve = shape;
	m_scene=scene;
	interpolatePoints();
	m_textItem->setVisible(false);
	m_View = nullptr;

	createCornerGrabbers();
	setPositionCornerGrabbers();
	setCornerGrabbersVisibility(true);



}

GraphEditor_CurveShape::~GraphEditor_CurveShape()
{

}


void GraphEditor_CurveShape::createCornerGrabbers()
{
	for(int i = 0; i < 8; i++){
		m_cornerGrabber[i] = new GraphEditor_GrabberItem(this,Qt::cyan);
		QObject::connect(m_cornerGrabber[i], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_CurveShape::cornerGrabberMove);
	}
}

void GraphEditor_CurveShape::setPositionCornerGrabbers()
{
	QRectF tmpRect =m_polygon.boundingRect();// rect();

	m_cornerGrabber[GrabberTop]->setPos(tmpRect.left() + tmpRect.width()/2, tmpRect.top());
	m_cornerGrabber[GrabberBottom]->setPos(tmpRect.left() + tmpRect.width()/2, tmpRect.bottom());
	m_cornerGrabber[GrabberLeft]->setPos(tmpRect.left(), tmpRect.top() + tmpRect.height()/2);
	m_cornerGrabber[GrabberRight]->setPos(tmpRect.right(), tmpRect.top() + tmpRect.height()/2);
	m_cornerGrabber[GrabberTopLeft]->setPos(tmpRect.topLeft().x(), tmpRect.topLeft().y());
	m_cornerGrabber[GrabberTopRight]->setPos(tmpRect.topRight().x(), tmpRect.topRight().y());
	m_cornerGrabber[GrabberBottomLeft]->setPos(tmpRect.bottomLeft().x(), tmpRect.bottomLeft().y());
	m_cornerGrabber[GrabberBottomRight]->setPos(tmpRect.bottomRight().x(), tmpRect.bottomRight().y());

	for(int i = 0; i < 8; i++){
		m_cornerGrabber[i]->setFlags(QGraphicsItem::ItemIgnoresTransformations);
	}
}

void GraphEditor_CurveShape::setCornerGrabbersVisibility(bool visible)
{
	for(int i = 0; i < 8; i++){
		m_cornerGrabber[i]->setVisible(visible);
	}
}
void GraphEditor_CurveShape::cornerGrabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy){
	m_ItemGeometryChanged = true;
	const QRectF rect1 (m_polygon.boundingRect()); // rect() );
	for(int i = 0; i < 8; i++){
		if(m_cornerGrabber[i] == signalOwner){
			switch (i)
			{
				case GrabberTop:{
					resizeTopBis(dy);
					break;
				}
				case GrabberBottom:{
					resizeBottomBis(dy);
					break;
				}
				case GrabberLeft:{
					resizeLeftBis(dx);
					break;
				}
				case GrabberRight:{
					resizeRightBis(dx);
					break;
				}
				case GrabberTopLeft: {
					resizeTopBis(dy);
					resizeLeftBis(dx);
					break;
				}
				case GrabberTopRight:{
					resizeTopBis(dy);
					resizeRightBis(dx);
					break;
				}
				case GrabberBottomLeft:{
					resizeBottomBis(dy);
					resizeLeftBis(dx);
					break;
				}
				case GrabberBottomRight:{
					resizeBottomBis(dy);
					resizeRightBis(dx);
					break;
				}
				default:
					break;
			}
			break;
		}
	}
	setPositionCornerGrabbers();
	setPolygon(m_polygon);
}


void GraphEditor_CurveShape::resizeLeftBis(qreal dx) {

	QRectF tmpRect =m_polygon.boundingRect();


	for(int i=0;i<grabberList.size();i++)
	{
		float posA = tmpRect.x();
		float posB = tmpRect.x()+tmpRect.width();
		float coef = (grabberList[i]->pos().x()- posB)/(posA - posB);
		grabberList[i]->moveX(dx*coef);
	}
	for(int i=0;i<m_polygon.size();i++)
	{
		float posA = tmpRect.x();
		float posB = tmpRect.x()+tmpRect.width();
		float coef = (grabberList[i]->pos().x()- posB)/(posA - posB);
		m_polygon[i].setX(m_polygon[i].x()+ dx*coef);
	}
}



void GraphEditor_CurveShape::resizeRightBis(qreal dx){
	QRectF tmpRect =m_polygon.boundingRect();
	for(int i=0;i<grabberList.size();i++)
	{
		float posA = tmpRect.x();
		float posB = tmpRect.x()+tmpRect.width();
		float coef = (grabberList[i]->pos().x()- posA)/(posB - posA);
		grabberList[i]->moveX(dx*coef);
	}

	for(int i=0;i<m_polygon.size();i++)
	{
		float posA = tmpRect.x();
		float posB = tmpRect.x()+tmpRect.width();
		float coef = (grabberList[i]->pos().x()- posA)/(posB - posA);
		m_polygon[i].setX(m_polygon[i].x()+ dx*coef);
	}

}



void GraphEditor_CurveShape::resizeBottomBis(qreal dy) {

	QRectF tmpRect =m_polygon.boundingRect();
	for(int i=0;i<grabberList.size();i++)
	{
		float posA = tmpRect.y();
		float posB = tmpRect.y()+tmpRect.height();
		float coef = (grabberList[i]->pos().y()- posA)/(posB - posA);
		grabberList[i]->moveY(dy*coef);
	}

	for(int i=0;i<m_polygon.size();i++)
	{
		float posA = tmpRect.y();
		float posB = tmpRect.y()+tmpRect.height();
		float coef = (grabberList[i]->pos().y()- posA)/(posB - posA);
		m_polygon[i].setY(m_polygon[i].y()+ dy*coef);
	}
}



void GraphEditor_CurveShape::resizeTopBis(qreal dy) {

	QRectF tmpRect =m_polygon.boundingRect();
	for(int i=0;i<grabberList.size();i++)
	{
		float posA = tmpRect.y();
		float posB = tmpRect.y()+tmpRect.height();
		float coef = (grabberList[i]->pos().y()- posB)/(posA - posB);
		grabberList[i]->moveY(dy*coef);
	}

	for(int i=0;i<m_polygon.size();i++)
	{
		float posA = tmpRect.y();
		float posB = tmpRect.y()+tmpRect.height();
		float coef = (grabberList[i]->pos().y()- posB)/(posA - posB);
		m_polygon[i].setY(m_polygon[i].y()+ dy*coef);
	}

}

void GraphEditor_CurveShape::setPolygon(const QPolygonF &polygon)
{


	GraphEditor_PolyLineShape::setPolygon(polygon);
	interpolatePoints();
}

void GraphEditor_CurveShape::setPolygonFromMove(const QPolygonF &polygon)
{


	GraphEditor_PolyLineShape::setPolygonFromMove(polygon);
	interpolatePoints();

	setPositionCornerGrabbers();
}

void GraphEditor_CurveShape::fillKnotVector()
{
	if (polygon().size()>2)
	{
		int middleKnotNumber = polygon().size() - 4;
		knotVector.clear();
		for (int counter = 0; counter < 4; ++counter)
			knotVector.push_back(0.0);
		for (int counter = 1; counter <= middleKnotNumber; ++counter)
			knotVector.push_back(1.0 / (middleKnotNumber + 1) * counter);
		for (int counter = 0; counter < 4; ++counter)
			knotVector.push_back(1.0);
	}
}

void GraphEditor_CurveShape::interpolateCurve()
{
	if (polygon().size()>2)
	{
		interpolatedPoints.clear();
		controlPoints.clear();

		foreach(QPointF p, polygon())
		{
			controlPoints.push_back(new QPointF(p));
		}

		bezierInterpolator.CalculateBoorNet(controlPoints, knotVector, boorNetPoints);
		interpolatedPoints.push_back(*(controlPoints.first()));
		for (int counter = 0; counter < boorNetPoints.size() - 3; counter += 3)
			bezierInterpolator.InterpolateBezier(boorNetPoints[counter],
					boorNetPoints[counter + 1],
					boorNetPoints[counter + 2],
					boorNetPoints[counter + 3],
					interpolatedPoints);
		interpolatedPoints.push_back(*(controlPoints.last()));
	}
}

void GraphEditor_CurveShape::interpolatePoints()
{
	int D = 4;
	int max = 100;
	int level =1;
	float u=0.0;
	float delta=(float)(1.0/max);
	QPointF l1,l2,r1,r2;
	vector<QPointF> curvepoints;
	//vector<QPointF> pointsVec = polygon().toStdVector();
	QPolygonF poly(polygon());
	if(m_IsClosedCurved)
	{

		poly<<poly[0];
		//pointsVec.push_back(pointsVec[0]);
	}
	vector<int> T;

	if(m_Curve==eShape_Bezier_Curve)
	{
		l1=poly[0];
		l2=poly[0];
		interpolatedPoints.clear();
		interpolatedPoints.push_back(l1);
		for(int i=0;i<=max;i++)
		{
			l1=l2;
			l2=drawBezier2(poly,u);
			interpolatedPoints.push_back(l2);
			u=u+delta;
		}
	}
	else if(m_Curve==eShape_CubicBSpline)
	{
		QPointF p;
		l1=poly[0];
		l2=poly[0];
		interpolatedPoints.clear();
		interpolatedPoints.push_back(l1);
		int n= (int) poly.size()-1;

		T=genKnot(n,D);
		u=0.0;
		max = 10;
		delta=(float)(1.0/max);
		int ul=(int) T.back()/delta;
		for(int j=0;j<ul;j++)
		{
			p.setX(0.0);
			p.setY(0.0);
			for(int i=0;i<=n;i++)
			{
				float z=N(i,D,u,T);
				p.setX(((p.x())+(z)*(poly[i].x())));
				p.setY(((p.y())+(z)*(poly[i].y())));
			}
			l1=l2;
			l2=p;
			interpolatedPoints.push_back(l2);
			u=u+delta;
		}
	}
	//	else if(m_Curve==eShape_SubDivideBezier)
	//	{
	//		l1=pointsVec[0];
	//		l2=pointsVec[0];
	//		u=0.5;
	//		curvepoints=SubDivide(pointsVec,level,u);
	//		//u=0.0;
	//		for(int i=1;i<curvepoints.size();i++)
	//		{
	//			l1=l2;
	//			l2=curvepoints[i];
	//			//painter->drawLine(l1,l2);
	//		}
	//	}
	//	else if(m_Curve==eShape_SubdivideBSpline)
	//	{
	//		l1=pointsVec[0];
	//		l2=pointsVec[0];
	//		curvepoints.clear();
	//		curvepoints=BSubDivide(pointsVec,level);
	//		for(int i=1;i<curvepoints.size();i++)
	//		{
	//			l1=l2;
	//			l2=curvepoints[i];
	//			//painter->drawLine(l1,l2);
	//		}
	//		//painter->drawLine(l2,pointsVec.back());
	//	}
	//	else if(m_Curve==eShape_RationalBezier)
	//	{
	//		u=0.0;
	//		l1=pointsVec[0];
	//		l2=pointsVec[0];
	//		r1=pointsVec[0];
	//		r2=r1;
	//		for(int i=0;i<=max;i++)
	//		{
	//			l1=l2;
	//			r1=r2;
	//			//l2=drawBezier2(pointsVec,u);
	//			r2=drawRBezier(pointsVec,u);
	//			//painter->drawLine(l1,l2);
	//			//painter->drawLine(r1,r2);
	//			u=u+delta;
	//		}
	//	}
	//	else if(m_Curve==eShape_NURBS)
	//	{
	//		QPointF p(0.0,0.0);
	//		l1=pointsVec[0];
	//		l2=pointsVec[0];
	//		int n= (int) pointsVec.size()-1;
	//		T.clear();
	//		int D=4;
	//		T=genKnot(n,D);
	//		u=0.0;
	//		float d;
	//		int ul=(int) T.back()/delta;
	//		float w[]={1.0f,1.0f,5.0f,1.0f};
	//		for(int j=0;j<ul;j++)
	//		{
	//			d=0.0;
	//			p.setX(0.0);
	//			p.setY(0.0);
	//			for(int i=0;i<=n;i++)
	//			{
	//				float z=N(i,D,u,T);
	//				p+=z*pointsVec[i]*w[i%4];
	//				d+=z*w[i%4];
	//			}
	//			l1=l2;
	//			l2=p/d;
	//			//painter->drawLine(l1,l2);
	//			u=u+delta;
	//		}
	//	}
}


float GraphEditor_CurveShape::B(int i,int n,float u)
{
	float a;
	a=(float) (pascal[n][i]*pow((1-u),n-i)*pow(u,i));
	return a;
}

QPointF GraphEditor_CurveShape::drawBezier2(QPolygonF p,float u)
{
	int n=(int)p.size()-1;
	QPointF pp(0.0,0.0);
	for(int i=0;i<=n;i++){
		pp+=B(i,n,u)*p[i];
	}
	return pp;
}

QPointF GraphEditor_CurveShape::drawRBezier(vector<QPointF> p,float u)
{
	QPointF z(0.0,0.0);
	float d=0.0;
	int n=(int)p.size()-1;
	float w[]={1.0,0.5,1.0};
	for(int i=0;i<p.size();i++){
		z+=B(i,n,u)*p[i]*w[i%3];
		d+=B(i,n,u)*w[i%3];
	}
	z=z/d;
	return z;
}

vector<QPointF> GraphEditor_CurveShape::OneSubDivide(vector<QPointF> p,vector<QPointF>* poly1,vector<QPointF>* poly2,float u)
{
	size_t n=p.size();
	vector<QPointF> pprime;
	if(n==1){
		poly1->push_back(p[0]);
		poly1->insert(poly1->end(),poly2->begin(),poly2->end());
		return *poly1;
	}
	poly1->insert(poly1->end(),p[0]);
	poly2->insert(poly2->begin(),p[n-1]);
	for(int i=0;i<n-1;i++){
		pprime.push_back(p[i]+(u*(p[i+1]-p[i])));
	}

	return OneSubDivide(pprime,poly1,poly2,u);
}

vector<QPointF> GraphEditor_CurveShape::SubDivide(vector<QPointF> p,int m,float u)
{
	vector<QPointF>* poly1=new vector<QPointF>();
	vector<QPointF>* poly2=new vector<QPointF>();
	vector<QPointF> pprime;
	if(m==1)
		return OneSubDivide(p,poly1,poly2,u);
	pprime=OneSubDivide(p,poly1,poly2,u);
	vector<QPointF> s1,s2;

	size_t mid=pprime.size()/2;
	vector<QPointF>::iterator middleIter(pprime.begin());
	std::advance(middleIter, mid);

	vector<QPointF> lhalf;
	lhalf.insert(lhalf.begin(),pprime.begin(), middleIter);
	lhalf.push_back(pprime[mid]);
	vector<QPointF> rhalf;
	rhalf.insert(rhalf.begin(),middleIter, pprime.end());

	s1=SubDivide(lhalf,m-1,u);
	s2=SubDivide(rhalf,m-1,u);
	s1.insert(s1.end(),s2.begin(),s2.end());
	return s1;
}

vector<int> GraphEditor_CurveShape::genKnot(int n,int D)
{
	vector<int> T;
	for(int j=0;j<=(n+D);j++){
		if(j<D)
			T.push_back(0);
		else if(D<=j && j<=n)
			T.push_back(j-D+1);
		else
			T.push_back(n-D+2);
	}
	return T;
}

float GraphEditor_CurveShape::N(int i,int d,float u,vector<int> T)
{
	float z;
	if(d==1)
	{
		if(T[i]<=u && u<T[i+1])
			z= 1.0;
		else
			z=0.0;
	}
	else
	{
		if(T[i+d-1]==T[i] && T[i+d]==T[i+1])
			z= 0.0;
		else if(T[i+d-1]==T[i])
			z= (((T[i+d]-u)*N(i+1,d-1,u,T))/(T[i+d]-T[i+1]));
		else if(T[i+d]==T[i+1])
			z= (((u-T[i])*N(i,d-1,u,T))/(T[i+d-1]-T[i]));
		else
			z= (((T[i+d]-u)*N(i+1,d-1,u,T))/(T[i+d]-T[i+1]))+(((u-T[i])*N(i,d-1,u,T))/(T[i+d-1]-T[i]));

	}

	return z;
}

vector<QPointF> GraphEditor_CurveShape::BSubDivide(vector<QPointF> p,int m)
{
	size_t n=(int) p.size();
	vector<QPointF> q;
	for(int i=0;i<n-1;i++){
		q.push_back((3*(p.at(i))+p.at(i+1))/4);
		q.push_back(((p.at(i))+3*p.at(i+1))/4);
	}

	if(m==1)
		return q;

	return BSubDivide(q,m-1);
}

void GraphEditor_CurveShape::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

	if (polygon().size()>2)
	{
		painter->setRenderHint(QPainter::Antialiasing,true);
	//	painter->setRenderHint(QPainter::HighQualityAntialiasing, true);

		QBrush brsh = brush();
		brsh.setTransform(QTransform((scene()->views())[0]->transform().inverted()));
		this->setBrush(brsh);

		if ((option->state & QStyle::State_Selected) || m_IsHighlighted)
		{
			QPen newPen = pen();
			newPen.setWidth(newPen.width()+3);
			painter->setPen(newPen);
		}
		else
		{
			painter->setPen(pen());
		}

		if(m_IsClosedCurved)
		{
			painter->setBrush(brush());
		}
		else
		{
			painter->setBrush(QBrush(Qt::NoBrush));
		}

		QPainterPath curvePath;
		curvePath.addPolygon(interpolatedPoints);
		painter->drawPath(curvePath);
	}

	if (option->state & QStyle::State_Selected)
	{
		QPen newPen = pen();
		newPen.setWidth(1);
		painter->setPen(newPen);
		painter->setBrush(QBrush(Qt::NoBrush));
		painter->drawPath(path());
	}
}

int GraphEditor_CurveShape::type()
{
	return m_Curve;
}

GraphEditor_CurveShape* GraphEditor_CurveShape::clone()
{
	GraphEditor_CurveShape* cloned = new GraphEditor_CurveShape(polygon(),m_Curve, pen(), brush(), m_Menu, m_scene,m_IsClosedCurved);
	cloned->setPos(scenePos());
	cloned->setZValue(zValue());
	cloned->setRotation(rotation());
	cloned->setID(m_LayerID);
	cloned->setGrabbersVisibility(false);
	return cloned;
}

QVector<QPointF>  GraphEditor_CurveShape::SceneCordinatesPoints()
{
	return mapToScene(interpolatedPoints);
}

QVector<QPointF>  GraphEditor_CurveShape::ImageCordinatesPoints()
{
	QVector<QPointF> vec;
	QPolygonF polygon_ = mapToScene(interpolatedPoints);
	GraphicSceneEditor *scene_ = dynamic_cast<GraphicSceneEditor*>(scene());
	if (scene_)
	{
		foreach(QPointF p, polygon_) {
			vec.push_back(scene_->innerView()->ConvertToImage(p));
		}
	}
	return vec;
}

