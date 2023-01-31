/*
 * GraphEditor_ListBezierPath.cpp
 *
 *  Created on: mai 18, 2022
 *      Author: l1049100
 */


#include <QGraphicsSceneMouseEvent>
#include <QPainterPath>
#include <QGraphicsScene>
#include <QDebug>
#include <math.h>
#include <QGraphicsScene>
#include <QGraphicsView>

#include "GraphEditor_GrabberItem.h"
#include "GraphEditor_ListBezierPath.h"
#include "GraphicSceneEditor.h"
#include "LineLengthText.h"
#include "singlesectionview.h"

GraphEditor_ListBezierPath::GraphEditor_ListBezierPath(QPolygonF polygon, QPen pen, QBrush brush, QMenu* itemMenu,
		GraphicSceneEditor* scene, bool isClosedCurved,QColor color)
{
	m_scene = scene;
	m_View = nullptr;
	m_IsClosedCurved = isClosedCurved;
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	m_Menu = itemMenu;
	setPen(pen);
	m_Pen=pen;
	setBrush(brush);
	m_textItem = new LineLengthText(this);
	setPolygon(polygon);
	m_textItem->setVisible(false);

	m_penRed.setColor(QColor(255, 0, 0));
	m_penRed.setCosmetic(true);

	m_penGreen.setColor(QColor(0, 255, 0));
	m_penGreen.setCosmetic(true);

	m_penYellow.setColor(color);
	m_penYellow.setCosmetic(true);
	m_penYellow.setWidth(3);

	pathCurrent.setFillRule(Qt::WindingFill);

	createCornerGrabbers();
	setPositionCornerGrabbers();
	setCornerGrabbersVisibility(true);




}


GraphEditor_ListBezierPath::GraphEditor_ListBezierPath(QVector<PointCtrl> listeCtrl, QPen pen, QBrush brush, QMenu* itemMenu,
		GraphicSceneEditor* scene, bool isClosedCurved,QColor color )
{
	m_scene = scene;
	m_View = nullptr;
	m_IsClosedCurved = isClosedCurved;
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	m_Menu = itemMenu;
	setPen(pen);
	m_Pen=pen;
	setBrush(brush);
	m_textItem = new LineLengthText(this);


	//setPolygon(polygon);
	m_textItem->setVisible(false);

	m_penRed.setColor(QColor(255, 0, 0));
	m_penRed.setCosmetic(true);

	m_penGreen.setColor(QColor(0, 255, 0));
	m_penGreen.setCosmetic(true);

//	m_penYellow.setColor(QColor(255, 255, 0));
//	m_penYellow.setCosmetic(true);

	m_penYellow = pen;
	m_penYellow.setColor(color);

	pathCurrent.setFillRule(Qt::WindingFill);

	m_listePtCtrls = listeCtrl;


	QPolygonF polygon;

	for(int i=0;i<m_listePtCtrls.count();i++)
	{
	//	qDebug()<<i<<" , positon :  "<<m_listePtCtrls[i]->m_position;
		polygon.push_back(m_listePtCtrls[i].m_position);
	}

	//setPolygon(polygon);

//	m_polygon = polygon;

	setPolygone(polygon);
	//setPolygon(polygon);

	//showGrabbersCtrl();

	createCornerGrabbers();
	setPositionCornerGrabbers();
	setCornerGrabbersVisibility(true);

	if(m_listePtCtrls.size()>= 2)
		{
		//	QPainterPath path2;
			pathCurrent.clear();
			pathCurrent.moveTo(m_listePtCtrls[0].m_position.x(), m_listePtCtrls[0].m_position.y());


			for(int i=0;i<m_listePtCtrls.count()-1;i++)
			{

				QPointF ctrl1 = m_listePtCtrls[i].m_ctrl2;
				QPointF ctrl2 = m_listePtCtrls[i+1].m_ctrl1;
				if( i== 0)
				{
					ctrl1 = m_listePtCtrls[i].m_ctrl1;
					ctrl2 = m_listePtCtrls[i+1].m_ctrl1;
				}


				pathCurrent.cubicTo(ctrl1.x(),ctrl1.y(),  ctrl2.x(),ctrl2.y(),  m_listePtCtrls[i+1].m_position.x(),m_listePtCtrls[i+1].m_position.y());


			}
		}


}

void GraphEditor_ListBezierPath::setColor(QColor c)
{

	m_penYellow.setColor(c);
	update();
}


void GraphEditor_ListBezierPath::restoreState(QPolygonF poly,  bool isClosedCurve)
{

	m_IsClosedCurved = isClosedCurve;
	m_polygon = poly;

	setPolygone(m_polygon);

	/*clearGrabbers();
	if (!m_IsClosedCurved || m_IsSmoothed)
	{
		showGrabbers();
	}*/
}

float GraphEditor_ListBezierPath::getPrecision()
{
	return m_precision;
}

void GraphEditor_ListBezierPath::setPrecision(float value)
{
	m_precision = value;
}

void GraphEditor_ListBezierPath::createCornerGrabbers()
{
	for(int i = 0; i < 8; i++){
		m_cornerGrabber[i] = new GraphEditor_GrabberItem(this,Qt::cyan);
		QObject::connect(m_cornerGrabber[i], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_ListBezierPath::cornerGrabberMove);
		connect(m_cornerGrabber[i], &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_ListBezierPath::GrabberMouseReleased);
	}
}

void GraphEditor_ListBezierPath::setPositionCornerGrabbers()
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

void GraphEditor_ListBezierPath::setCornerGrabbersVisibility(bool visible)
{
	for(int i = 0; i < 8; i++){
		m_cornerGrabber[i]->setVisible(visible);
	}
}
void GraphEditor_ListBezierPath::cornerGrabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy){
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
	if(m_indexCurrentCrtl>=0)
	{
		m_indexCurrentCrtl = -1;

	}
	setPositionCornerGrabbers();
	setPolygonTangent(m_polygon);
}


void GraphEditor_ListBezierPath::resizeLeftBis(qreal dx) {

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
		int index =i;
		if( index >=grabberList.count()) index=0;
		float posA = tmpRect.x();
		float posB = tmpRect.x()+tmpRect.width();

		float coef = (grabberList[index]->pos().x()- posB)/(posA - posB);
		m_polygon[i].setX(m_polygon[i].x()+ dx*coef);
	}
}



void GraphEditor_ListBezierPath::resizeRightBis(qreal dx){
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
		int index =i;
		if( index >=grabberList.count()) index=0;
		float posA = tmpRect.x();
		float posB = tmpRect.x()+tmpRect.width();

		float coef = (grabberList[index]->pos().x()- posA)/(posB - posA);
		m_polygon[i].setX(m_polygon[i].x()+ dx*coef);
	}

}



void GraphEditor_ListBezierPath::resizeBottomBis(qreal dy) {

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
		int index =i;
		if( index >=grabberList.count()) index=0;
		float posA = tmpRect.y();
		float posB = tmpRect.y()+tmpRect.height();

		float coef = (grabberList[index]->pos().y()- posA)/(posB - posA);
		m_polygon[i].setY(m_polygon[i].y()+ dy*coef);
	}
}



void GraphEditor_ListBezierPath::resizeTopBis(qreal dy) {

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
		int index =i;
		if( index >=grabberList.count()) index=0;
		float posA = tmpRect.y();
		float posB = tmpRect.y()+tmpRect.height();
		float coef = (grabberList[index]->pos().y()- posB)/(posA - posB);
		m_polygon[i].setY(m_polygon[i].y()+ dy*coef);
	}

}


GraphEditor_ListBezierPath::~GraphEditor_ListBezierPath() {
	if(m_nameId !="") emit BezierDeleted(m_nameId);
}

bool GraphEditor_ListBezierPath::checkClosedPath()
{
	if (grabberList.size()>2)
	{
		if (grabberList[grabberList.size()-1]->hasDetectedCollision())
		{
			return true;
		}
	}
	return false;
}

QVector<QPointF> GraphEditor_ListBezierPath::getKeyPoints()
{
	//return mapToScene(pathCurrent.toFillPolygon());


	return mapToScene(getPointInterpolated());

}

QVector<QPointF> GraphEditor_ListBezierPath::getPointInterpolated()
{
	QVector<QPointF>  points;

	for(int i=0;i<=m_precision;i++)
	{
		points.push_back(getPosition((float)(i/m_precision)));
	}
	//qDebug()<<"getPointInterpolated size : "<<points.count();

	return points;
}

void GraphEditor_ListBezierPath::insertNewPoints(QPointF pos)
{

	if (m_DrawFinished)
	{
		QPointF clickPos = pos;
		QPolygonF polygonPath = m_polygon;
		QPolygonF newPath( polygonPath );

		bool found = false;
		double distanceMin = 100000;
		QPointF newPointBest;
		double factor =1;
		float epsilon = 0.1f;
		int currentindex = -1;
		int indexAdded;
		for(int i = 0; i < polygonPath.size()-1; i++){ // i go from 0 to N-1 because we do not want to pick on the last line of the polygon
			QPointF p1 = polygonPath.at(i);
			QPointF p2 = (i < polygonPath.size()-1) ? polygonPath.at(i+1) : polygonPath.at(0);
			double APx = clickPos.x() - p1.x();
			double APy = clickPos.y() - p1.y();
			double ABx = p2.x() - p1.x();
			double ABy = p2.y() - p1.y();
			double magAB2 = ABx*ABx + ABy*ABy;
			double ABdotAP = ABx*APx + ABy*APy;
			double t = ABdotAP / magAB2;

			//qDebug()<<" T : "<<t;
			if(t > -epsilon && t < epsilon)
			{
				//qDebug()<<" insertNewPoints  signalCurrentIndex"<<i;
				emit signalCurrentIndex(i);
				return;
			}
			if(t > 1.0-epsilon && t < 1.0+epsilon)
			{
				//qDebug()<<" insertNewPoints  signalCurrentIndex  1+"<<i;
				emit signalCurrentIndex(i+1);
				return;
			}


			QPointF newPoint;

			if ( t < 0) {
				//newPoint = trackLine.p1();
			}else if (t > 1){
				//newPoint = trackLine.p2();
			}else{
				newPoint.setX(p1.x() + ABx*t);
				newPoint.setY(p1.y() + ABy*t);
				double d = sqrt( pow( (newPoint.x() - clickPos.x()), 2) + pow( (newPoint.y() - clickPos.y()), 2));
				if ( d < distanceMin) {
					distanceMin = d;
					newPointBest = newPoint;
					found = true;
					indexAdded = i + 1;
					factor = t;
				}
			}
		}
		if (found) {
			if(currentindex<0) currentindex = indexAdded;
			QPointF nextPoint = (indexAdded - 1 < polygonPath.size()-1) ? polygonPath.at(indexAdded) : polygonPath.at(0);
			int x = polygonPath.at(indexAdded - 1).x() + factor * (nextPoint.x() - polygonPath.at(indexAdded - 1).x());
			int y = polygonPath.at(indexAdded - 1).y() + factor * (nextPoint.y() - polygonPath.at(indexAdded - 1).y());
			QPointF newPoint(x, y );
			newPath.insert(indexAdded, newPoint);
			setPolygon(newPath);
			setGrabbersVisibility(true);
			m_ItemGeometryChanged=true;
		}
	//	qDebug()<<found<<" current index  = "<<currentindex;
		emit signalCurrentIndex(currentindex);
	}


}


void GraphEditor_ListBezierPath::polygonResize(int widthO, int width)
{


	float decal = (width- widthO)/2.0f;

	QPolygonF polygon = m_polygon;
	for(int i=0;i<polygon.size();i++)
	{
		polygon[i].setX(decal+ polygon[i].x());
	}


	for(int i=0;i<grabberList.count();i++)
	{
		grabberList[i]->moveX(decal );
	}

	if(m_polygon.size()>=2)
	{
		for(int i=0;i<m_polygon.size();i++)
		{
			m_listePtCtrls[i].m_position.setX(decal+ m_listePtCtrls[i].m_position.x());
			m_listePtCtrls[i].m_ctrl1.setX(decal+ m_listePtCtrls[i].m_ctrl1.x());
			m_listePtCtrls[i].m_ctrl2.setX(decal+ m_listePtCtrls[i].m_ctrl2.x());
		}
	}

	//setPolygonFromMove(polygon);

	m_polygon = polygon;
	setPositionCornerGrabbers();
	//emit polygonChanged(m_polygon);
}


void GraphEditor_ListBezierPath::setPolygone(const QPolygonF &polygon)
{
	if (polygon.isEmpty())
			return;
		m_polygon = polygon;
		QPainterPath newPath;
		newPath.addPolygon(m_polygon);
		if (m_IsClosedCurved)
		{
			newPath.closeSubpath();
		}
		setPath(newPath);
		clearGrabbers();
		showGrabbers();

		//refreshCurve();
		showGrabbersCtrl( false);//true

}
void GraphEditor_ListBezierPath::refreshCurve()
{
//	m_listePtCtrls.clear();
	//qDebug()<<m_polygon.size()  <<" == " <<m_listePtCtrls.count();
	if(m_polygon.size()>=2)
	{
		for(int i=0;i<m_polygon.size();i++)
		{
			m_listePtCtrls[i].m_position = m_polygon[i];
		}
	}
}


void GraphEditor_ListBezierPath::setPolygonTangent(const QPolygonF &polygon)
{
	if (polygon.isEmpty())
		return;


	m_polygon = polygon;
	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	if (m_IsClosedCurved)
	{
		newPath.closeSubpath();
	}
	setPath(newPath);
	clearGrabbers();
	showGrabbers();

	showtang=true;


	/*if(m_IsClosedCurved)
	{
		m_polygon.push_back(m_polygon[0]);
	}*/



	QVector<QPointF>  listeTangentesDir;


	//qDebug()<<"setPolygonTangent  m_listePtCtrls : "<<m_listePtCtrls.count();
	if(m_listePtCtrls.count()> 0)
	{
		for(int i=0;i<m_listePtCtrls.count();i++)
		{

		//	qDebug()<<i<<" ctrl1 :"<< m_listePtCtrls[i].m_ctrl1;
		//	qDebug()<<i<<" ctrl2 :"<< m_listePtCtrls[i].m_ctrl2;

			listeTangentesDir.push_back( m_listePtCtrls[i].m_ctrl1 - m_listePtCtrls[i].m_position);
			listeTangentesDir.push_back( m_listePtCtrls[i].m_ctrl2 - m_listePtCtrls[i].m_position);
		}
	}


	/*if(m_IsClosedCurved)
	{
		m_polygon.push_back(m_polygon[0]);
		//listeTangentesDir.push_back( m_listePtCtrls[0].m_ctrl1 - m_listePtCtrls[0].m_position);
		//listeTangentesDir.push_back( m_listePtCtrls[0].m_ctrl2 - m_listePtCtrls[0].m_position);
	}
*/
	m_listePtCtrls.clear();

	int index=0;

	if(m_polygon.size()>=2)
	{
		for(int i=0;i<m_polygon.size();i++)
		{

			if(i== 0 || i==m_polygon.size() -1)
			{
				QPointF ctrl;
				if(i == m_polygon.size()-1 )
				{
					ctrl = m_polygon[i] + listeTangentesDir[index];
					int incr= 2;
					//if(m_IsClosedCurved ) incr=1;
					index+=incr;//m_polygon[i] + (m_polygon[i-1]-m_polygon[i])*m_coefTangent;

				}
				else
				{
					ctrl = m_polygon[i] + listeTangentesDir[index]; //m_polygon[i] + (m_polygon[i+1] -m_polygon[i])*m_coefTangent;
					int incr= 2;
					//if(m_IsClosedCurved ) incr=1;
					index+=incr;
				}
				m_listePtCtrls.push_back(PointCtrl(m_polygon[i], ctrl));
			}
			else
			{
				QPointF ctrl1 = m_polygon[i] + listeTangentesDir[index];//m_polygon[i] + (m_polygon[i-1]-m_polygon[i])*m_coefTangent;
				index++;
				QPointF ctrl2 = m_polygon[i] + listeTangentesDir[index];//m_polygon[i] + (m_polygon[i+1]-m_polygon[i])*m_coefTangent;
				index++;

				m_listePtCtrls.push_back(PointCtrl(m_polygon[i], ctrl1, ctrl2));
			}

		}
		showGrabbersCtrl(false);
	}


}

void GraphEditor_ListBezierPath::setPolygon(const QPolygonF &polygon) {
	if (polygon.isEmpty())
		return;
	m_polygon = polygon;
	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	if (m_IsClosedCurved)
	{
		newPath.closeSubpath();
	}
	setPath(newPath);
	clearGrabbers();
	showGrabbers();
	//calculatePerimeter();

	if(m_IsClosedCurved)
	{
		m_polygon.push_back(m_polygon[0]);
	}

	QVector<QPointF> pol =this->polygon();

//	m_listeNoeuds.clear();
//	m_listeControls.clear();
	m_listePtCtrls.clear();

	if(m_polygon.size()>=2)
	{
		for(int i=0;i<m_polygon.size();i++)
		{

			if(i== 0 || i==m_polygon.size() -1)
			{
				QPointF ctrl;
				if(i == m_polygon.size()-1 )
					ctrl = m_polygon[i] + (m_polygon[i-1]-m_polygon[i])*m_coefTangent;
				else
					ctrl = m_polygon[i] + (m_polygon[i+1] -m_polygon[i])*m_coefTangent;

				m_listePtCtrls.push_back(PointCtrl(m_polygon[i], ctrl));
			}
			else
			{
				QPointF ctrl1 = m_polygon[i] + (m_polygon[i-1]-m_polygon[i])*m_coefTangent;
				QPointF ctrl2 = m_polygon[i] + (m_polygon[i+1]-m_polygon[i])*m_coefTangent;

				m_listePtCtrls.push_back(PointCtrl(m_polygon[i], ctrl1, ctrl2));
			}

		}
		showGrabbersCtrl();
	}
}

void GraphEditor_ListBezierPath::setPolygonFromMove(const QPolygonF &polygon) {
	m_polygon = polygon;
	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	if (m_IsClosedCurved)
	{
		newPath.closeSubpath();
	}
	setPath(newPath);
	//calculatePerimeter();

	QVector<QPointF> pol =this->polygon();// this->SceneCordinatesPoints();



//	emit polygonChanged(pol,!m_IsClosedCurved);
}

void GraphEditor_ListBezierPath::setDrawFinished(bool value)
{
	m_DrawFinished = value;
	if (grabberList.size()>1)
	{
		if (grabberList[grabberList.size()-1]->hasDetectedCollision())
		{
			m_polygon.removeAt(grabberList.size()-1);
			m_IsClosedCurved=true;
			setPolygon(m_polygon);
		}
	}


	//showGrabbersCtrl();

}

void GraphEditor_ListBezierPath::polygonChanged1()
{
	setPositionCornerGrabbers();
	emit polygonChanged(m_listePtCtrls, getPointInterpolated(),!m_IsClosedCurved);
	emit polygonChanged2(this);
}

void GraphEditor_ListBezierPath::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

	painter->setRenderHint(QPainter::Antialiasing,true);
	//painter->setRenderHint(QPainter::HighQualityAntialiasing, true);
	painter->setRenderHint(QPainter::SmoothPixmapTransform, true);

/*	QBrush brsh = brush();
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

	if (m_IsClosedCurved)
	{
		painter->setBrush(brush());
	}
	else
	{
		painter->setBrush(QBrush(Qt::NoBrush));
	}
	painter->drawPath(path());

*/



	//===================================
	if(m_listePtCtrls.size()>= 2)
	{
	//	QPainterPath path2;
		pathCurrent.clear();
		pathCurrent.moveTo(m_listePtCtrls[0].m_position.x(), m_listePtCtrls[0].m_position.y());


		for(int i=0;i<m_listePtCtrls.count()-1;i++)
		{


			QPointF ctrl1 = m_listePtCtrls[i].m_ctrl2;
			QPointF ctrl2 = m_listePtCtrls[i+1].m_ctrl1;
			if( i== 0)
			{
				ctrl1 = m_listePtCtrls[i].m_ctrl1;
				ctrl2 = m_listePtCtrls[i+1].m_ctrl1;
			}


			pathCurrent.cubicTo(ctrl1.x(),ctrl1.y(),  ctrl2.x(),ctrl2.y(),  m_listePtCtrls[i+1].m_position.x(),m_listePtCtrls[i+1].m_position.y());



		/*	painter->setPen(m_penBlue);
			painter->drawLine(m_listePtCtrls[i]->m_position,m_listePtCtrls[i]->m_ctrl1);
			if(m_listePtCtrls[i]->m_nbPoints == 2)
			{
			//	painter->drawPoint(ctrl2);
				painter->setPen(m_penGreen);
				painter->drawLine(m_listePtCtrls[i]->m_position,m_listePtCtrls[i]->m_ctrl2);
			}*/
		}


	//	painter->drawLine(m_listePtCtrls[m_listePtCtrls.count()-1]->m_position,m_listePtCtrls[m_listePtCtrls.count()-1]->m_ctrl1);


		if( (option->state & QStyle::State_Selected))
		{
			if(m_indexCurrentCrtl != -1)
			{
				painter->setPen(m_penRed);



				if (m_IsClosedCurved)
				{
					if(m_indexCurrentCrtl ==0)
					{
						painter->drawLine(m_listePtCtrls[m_listePtCtrls.count()-1].m_position,m_listePtCtrls[m_listePtCtrls.count()-1].m_ctrl1);
						painter->drawLine(m_listePtCtrls[0].m_position,m_listePtCtrls[0].m_ctrl1);
					}
					else if(m_indexCurrentCrtl ==m_listePtCtrls.count()-1)
					{
						painter->drawLine(m_listePtCtrls[0].m_position,m_listePtCtrls[0].m_ctrl1);
						painter->drawLine(m_listePtCtrls[m_listePtCtrls.count()-1].m_position,m_listePtCtrls[m_listePtCtrls.count()-1].m_ctrl1);
					}
					else
					{
						painter->drawLine(m_listePtCtrls[m_indexCurrentCrtl].m_position,m_listePtCtrls[m_indexCurrentCrtl].m_ctrl1);
						if(m_listePtCtrls[m_indexCurrentCrtl].m_nbPoints == 2)
						{
							painter->setPen(m_penGreen);
							painter->drawLine(m_listePtCtrls[m_indexCurrentCrtl].m_position,m_listePtCtrls[m_indexCurrentCrtl].m_ctrl2);
						}
					}
				}
				else
				{
					painter->drawLine(m_listePtCtrls[m_indexCurrentCrtl].m_position,m_listePtCtrls[m_indexCurrentCrtl].m_ctrl1);
					if(m_listePtCtrls[m_indexCurrentCrtl].m_nbPoints == 2)
					{
						painter->setPen(m_penGreen);
						painter->drawLine(m_listePtCtrls[m_indexCurrentCrtl].m_position,m_listePtCtrls[m_indexCurrentCrtl].m_ctrl2);
					}
				}

			}
		}
		else
		{
			if(m_lastIndexCurrent>= 0 )
			{
				showGrabberCurrent();
			}
			m_indexCurrentCrtl = -1;
			m_lastIndexCurrent = -2;

		}

		if(m_IsClosedCurved)
		{
			pathCurrent.closeSubpath();
		}

		if ((option->state & QStyle::State_Selected) || m_IsHighlighted)
		{
			m_penYellow.setWidth(5);
		}
		else
		{
			m_penYellow.setWidth(2);
		}


	//	painter->setPen(m_penYellow);

		painter->strokePath(pathCurrent,m_penYellow);


	}
	/*else
	{
		qDebug()<< "****** m_listePtCtrls : "<<m_listePtCtrls.count();
	}*/

}

void GraphEditor_ListBezierPath::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {


	//qDebug()<<" mouseDoubleClickEvent ....";
	if (m_DrawFinished)
	{
		QPointF clickPos = event->pos();
		QPolygonF polygonPath = m_polygon;
		QPolygonF newPath( polygonPath );
		bool found = false;
		double distanceMin = 10000;
		QPointF newPointBest;
		double factor =1;
		int indexAdded;
		for(int i = 0; i < m_listePtCtrls.size()-1; i++){ // i go from 0 to N-1 because we do not want to pick on the last line of the polygon
			QPointF p1 =m_listePtCtrls[i].m_position;
			QPointF p2 = (i < m_listePtCtrls.size()-1) ? m_listePtCtrls[i+1].m_position : m_listePtCtrls[0].m_position;
			double APx = clickPos.x() - p1.x();
			double APy = clickPos.y() - p1.y();
			double ABx = p2.x() - p1.x();
			double ABy = p2.y() - p1.y();
			double magAB2 = ABx*ABx + ABy*ABy;
			double ABdotAP = ABx*APx + ABy*APy;
			double t = ABdotAP / magAB2;

			QPointF newPoint;

			if ( t < 0) {
				//newPoint = trackLine.p1();
			}else if (t > 1){
				//newPoint = trackLine.p2();
			}else{
				newPoint.setX(p1.x() + ABx*t);
				newPoint.setY(p1.y() + ABy*t);
				double d = sqrt( pow( (newPoint.x() - clickPos.x()), 2) + pow( (newPoint.y() - clickPos.y()), 2));
				if ( d < distanceMin) {
					distanceMin = d;
					newPointBest = newPoint;
					found = true;
					indexAdded = i + 1;
					factor = t;
				}
			}
		}
		if (found) {

			QPointF nextPoint = (indexAdded - 1 < m_listePtCtrls.size()-1) ? m_listePtCtrls.at(indexAdded).m_position : m_listePtCtrls.at(0).m_position;
			int x = m_listePtCtrls.at(indexAdded - 1).m_position.x() + factor * (nextPoint.x() - m_listePtCtrls.at(indexAdded - 1).m_position.x());
			int y = m_listePtCtrls.at(indexAdded - 1).m_position.y() + factor * (nextPoint.y() - m_listePtCtrls.at(indexAdded - 1).m_position.y());
			QPointF newPoint(x, y );
			QPointF ctrl1 = newPoint +(  m_listePtCtrls.at(indexAdded-1 ).m_position -  newPoint)*m_coefTangent;
			QPointF ctrl2 = newPoint +(  m_listePtCtrls.at(indexAdded ).m_position -  newPoint)*m_coefTangent;
			PointCtrl pts(newPoint,ctrl1, ctrl2);



			m_polygon.insert(indexAdded, newPoint);

			QPainterPath newPath;
				newPath.addPolygon(m_polygon);
			setPath(newPath);

				clearGrabbers();
				showGrabbers();
			//	calculatePerimeter();

			//setPolygon(newPath);
			addGrabbersCtrl(indexAdded,pts);

			setGrabbersVisibility(true);
			m_ItemGeometryChanged=true;


			//showGrabbersCtrl();
			//addGrabbersCtrl(indexAdded,pts);


		}

	}

	QGraphicsItem::mouseDoubleClickEvent(event);
}

void GraphEditor_ListBezierPath::addGrabbersCtrl(int i,PointCtrl pts)
{
	GraphEditor_GrabberItem *dot = new GraphEditor_GrabberItem(this,QColor(Qt::red));
	connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_ListBezierPath::grabberCtrlMove);

	dot->setPos(pts.m_ctrl1);
	dot->setVisible(true);
	pts.m_grab1 = dot;
//	m_listePtCtrls[i]->m_grab2= dot;

	m_grabbersCtrl.append(dot);

	if(pts.m_nbPoints == 2)
	{
		GraphEditor_GrabberItem *dot2 = new GraphEditor_GrabberItem(this,QColor(Qt::green));
		connect(dot2, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_ListBezierPath::grabberCtrlMove);

		dot2->setPos(pts.m_ctrl2);
		dot2->setVisible(true);
		pts.m_grab2 = dot2;

		m_grabbersCtrl.append(dot2);
	}
	else
	{
		m_grabbersCtrl.append(dot);
		pts.m_grab2 = dot;
	}


	m_listePtCtrls.insert(i, pts);
}



void GraphEditor_ListBezierPath::showGrabbersCtrl(bool recomputeTangente)
{


	for(int i =0;i< m_grabbersCtrl.size();i++)
	{
		if(m_grabbersCtrl[i] != nullptr)
		{
			m_grabbersCtrl[i]->deleteLater();
			//delete m_grabbersCtrl[i];
		}
	}
	m_grabbersCtrl.clear();


	//qDebug()<<" showGrabbersCtrl "<<m_listePtCtrls.size();
	if( recomputeTangente)
	{
		for(int i=1;i<m_listePtCtrls.size()-1;i++)
		{
			QPointF dir1 = m_listePtCtrls[i].m_position -  m_listePtCtrls[i-1].m_position;
			QPointF dir2 = m_listePtCtrls[i+1].m_position -  m_listePtCtrls[i].m_position;
			QPointF decal =  dir1*m_coefTangent + dir2*m_coefTangent;

			m_listePtCtrls[i].m_ctrl1 = m_listePtCtrls[i].m_position -decal;
			m_listePtCtrls[i].m_ctrl2 = m_listePtCtrls[i].m_position +decal;

		}
	}

	for(int i=0;i<m_listePtCtrls.size();i++)
	{

		GraphEditor_GrabberItem *dot = new GraphEditor_GrabberItem(this,QColor(Qt::red));
		connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_ListBezierPath::grabberCtrlMove);
		connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_ListBezierPath::GrabberMouseReleased);

		dot->setPos(m_listePtCtrls[i].m_ctrl1);
		dot->setVisible(false);
		m_listePtCtrls[i].m_grab1 = dot;
	//	m_listePtCtrls[i]->m_grab2= dot;

		m_grabbersCtrl.append(dot);

		if(m_listePtCtrls[i].m_nbPoints == 2)
		{
			GraphEditor_GrabberItem *dot2 = new GraphEditor_GrabberItem(this,QColor(Qt::green));
			connect(dot2, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_ListBezierPath::grabberCtrlMove);
			connect(dot2, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_ListBezierPath::GrabberMouseReleased);

			dot2->setPos(m_listePtCtrls[i].m_ctrl2);
			dot2->setVisible(false);
			m_listePtCtrls[i].m_grab2 = dot2;

			m_grabbersCtrl.append(dot2);
		}
		else
		{
			m_grabbersCtrl.append(dot);
			m_listePtCtrls[i].m_grab2 = dot;
		}
	}
}


void GraphEditor_ListBezierPath::selectGrabber(QGraphicsItem *signalOwner)
{
	int indexSelected;
	bool found = false;
	for(int i = 0; i < grabberList.size(); i++){
		if(grabberList.at(i) == signalOwner){
			found = true;
			indexSelected = i;
			//break;
		} else {

		}
	}

	if(found)
	{
		m_indexCurrentCrtl = indexSelected;
		showGrabberCurrent();
	}

}

void GraphEditor_ListBezierPath::hideGrabberCurrent()
{
	if(m_lastIndexCurrent >=0)
	{
		m_listePtCtrls[m_lastIndexCurrent].m_grab1->setVisible(false);
		if(m_listePtCtrls[m_lastIndexCurrent].m_nbPoints == 2) m_listePtCtrls[m_lastIndexCurrent].m_grab2->setVisible(false);
		m_lastIndexCurrent =-2;
	}
}


void GraphEditor_ListBezierPath::showGrabberCurrent()
{

	if(m_lastIndexCurrent >=0 && m_lastIndexCurrent !=m_indexCurrentCrtl)
	{
		m_listePtCtrls[m_lastIndexCurrent].m_grab1->setVisible(false);
		if(m_listePtCtrls[m_lastIndexCurrent].m_nbPoints == 2) m_listePtCtrls[m_lastIndexCurrent].m_grab2->setVisible(false);


		if (m_IsClosedCurved)
		{
			if(m_lastIndexCurrent ==0)
			{
				m_listePtCtrls[m_listePtCtrls.count()-1].m_grab1->setVisible(false);
				m_listePtCtrls[m_listePtCtrls.count()-1].m_grab2->setVisible(false);
				m_listePtCtrls[0].m_grab1->setVisible(false);
				m_listePtCtrls[0].m_grab2->setVisible(false);
			}
			if(m_lastIndexCurrent ==m_listePtCtrls.count()-1)
			{
				m_listePtCtrls[0].m_grab1->setVisible(false);
				m_listePtCtrls[0].m_grab2->setVisible(false);
			}
		}
	}
	if( m_indexCurrentCrtl>= 0 )
	{
		m_listePtCtrls[m_indexCurrentCrtl].m_grab1->setVisible(true);
		if(m_listePtCtrls[m_indexCurrentCrtl].m_nbPoints == 2) m_listePtCtrls[m_indexCurrentCrtl].m_grab2->setVisible(true);

		if (m_IsClosedCurved)
		{
			if(m_indexCurrentCrtl ==0)
			{
				//qDebug()<<"m_indexCurrentCrtl :"<<m_indexCurrentCrtl;
				m_listePtCtrls[m_listePtCtrls.count()-1].m_grab1->setVisible(showtang);
				m_listePtCtrls[m_listePtCtrls.count()-1].m_grab2->setVisible(!showtang);//false;
				m_listePtCtrls[0].m_grab1->setVisible(showtang);
				m_listePtCtrls[0].m_grab2->setVisible(!showtang);//false
			}
			if(m_indexCurrentCrtl ==m_listePtCtrls.count()-1)
			{
				//qDebug()<<"m_indexCurrentCrtl :"<<m_indexCurrentCrtl;
				m_listePtCtrls[0].m_grab1->setVisible(!showtang);
				m_listePtCtrls[0].m_grab2->setVisible(showtang);//false
				m_listePtCtrls[m_listePtCtrls.count()-1].m_grab1->setVisible(!showtang);
				m_listePtCtrls[m_listePtCtrls.count()-1].m_grab2->setVisible(showtang);//false
			}
		}
		m_lastIndexCurrent = m_indexCurrentCrtl;
	}
}

void GraphEditor_ListBezierPath::showGrabbers() {
	QPolygonF polygonPath = m_polygon;
	for(int i = 0; i < polygonPath.size(); i++){
		QPointF point = polygonPath.at(i);

		GraphEditor_GrabberItem *dot = new GraphEditor_GrabberItem(this);

		connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_ListBezierPath::grabberMove);
		connect(dot, &GraphEditor_GrabberItem::signalDoubleClick, this, &GraphEditor_ListBezierPath::GrabberDeleted);
		connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_ListBezierPath::polygonChanged1);
		connect(dot, &GraphEditor_GrabberItem::signalClick, this, &GraphEditor_ListBezierPath::selectGrabber);

		dot->setVisible(true);
		grabberList.append(dot);
		dot->setPos(point);
		if (i==0)
		{
			dot->setFirstGrabberInList(true);
		}
		else if (i == polygonPath.size() -1)
		{
			if (m_IsClosedCurved)
				dot->setDetectCollision(false);
			else
			{
				dot->setDetectCollision(true);
				connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_ListBezierPath::GrabberMouseReleased);
			}
		}
	}

}

void GraphEditor_ListBezierPath::GrabberMouseReleased(QGraphicsItem *signalOwner)
{
	setDrawFinished(true);

	//qDebug()<<"GrabberMouseReleased ";

	emit polygonChanged(m_listePtCtrls,getPointInterpolated(),!m_IsClosedCurved);
	emit polygonChanged2(this);
}

void GraphEditor_ListBezierPath::calculatePerimeter()
{
	qreal TotalLength =0;
	if (m_polygon.size()>1)
	{
		for(int i = 1; i < m_polygon.size(); i++)
		{
			QLineF line(m_polygon.at(i-1),m_polygon.at(i));
			TotalLength +=line.length();
		}
		if (m_IsClosedCurved)
		{
			QLineF line(m_polygon.last(),m_polygon.at(0));
			TotalLength +=line.length();
		}
		int polygon_middle;
		if (m_IsClosedCurved)
		{
			polygon_middle=(m_polygon.size()+1)/2-1;
		}
		else
		{
			polygon_middle=m_polygon.size()/2-1;
		}

		QLineF line(m_polygon.at(polygon_middle),m_polygon.at(polygon_middle+1));
		float lineAngle = line.angle();
		if (m_scene)
		{
			if (m_scene->innerView())
			{
				if ((m_scene->innerView()->viewType() == InlineView) ||
						(m_scene->innerView()->viewType() == XLineView) ||
						(m_scene->innerView()->viewType() == RandomView) )
				{
					lineAngle = 360 - lineAngle;
				}
			}
		}
		QPointF textPos;

		if ( lineAngle > 90 && lineAngle < 260)
		{  // Right to left line
			lineAngle -= 180;
		}
		textPos = line.center();
		m_textItem->setText(QString::number(TotalLength)+ " m");
		m_textItem->setPos(textPos);
		m_textItem->setRotation(lineAngle);
		m_textItem->setVisible(true);
	}
}

void GraphEditor_ListBezierPath::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {
	m_ItemGeometryChanged=true;
	if ( grabberList.isEmpty() )
		return;
	//QPolygonF polygonPath = m_polygon;
	for(int i = 0; i < grabberList.size(); i++){
		if(grabberList.at(i) == signalOwner)
		{
			QPointF pathPoint = m_listePtCtrls.at(i).m_position;
			QPointF grab1= m_listePtCtrls.at(i).m_grab1->pos();
			QPointF grab2;
			if(  m_listePtCtrls.at(i).m_grab2 != nullptr)  grab2= m_listePtCtrls.at(i).m_grab2->pos();

			m_listePtCtrls[i].m_position  = QPointF(pathPoint.x() + dx, pathPoint.y() + dy);
			m_listePtCtrls.at(i).m_grab1->setPos( QPointF(grab1.x() + dx, grab1.y() + dy));

			m_listePtCtrls[i].m_ctrl1 = grab1;
			if(  m_listePtCtrls.at(i).m_grab2 != nullptr)
			{
				m_listePtCtrls[i].m_ctrl2 = grab2;
				m_listePtCtrls.at(i).m_grab2->setPos( QPointF(grab2.x() + dx, grab2.y() + dy));
			}



			//QPointF pathPoint = polygonPath.at(i);
			m_polygon.replace(i, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));


			if(m_IsClosedCurved)
			{
				if( i == 0 )
				{
					QPointF pathPoint = m_listePtCtrls.at(m_listePtCtrls.count()-1).m_position;
					QPointF grab1= m_listePtCtrls.at(m_listePtCtrls.count()-1).m_grab1->pos();
					QPointF grab2;
					if(  m_listePtCtrls.at(m_listePtCtrls.count()-1).m_grab2 != nullptr)  grab2= m_listePtCtrls.at(m_listePtCtrls.count()-1).m_grab2->pos();

					m_listePtCtrls[m_listePtCtrls.count()-1].m_position  = QPointF(pathPoint.x() + dx, pathPoint.y() + dy);
					m_listePtCtrls.at(m_listePtCtrls.count()-1).m_grab1->setPos( QPointF(grab1.x() + dx, grab1.y() + dy));

					m_listePtCtrls[m_listePtCtrls.count()-1].m_ctrl1 = grab1;
					if(  m_listePtCtrls.at(m_listePtCtrls.count()-1).m_grab2 != nullptr)
					{
						m_listePtCtrls[m_listePtCtrls.count()-1].m_ctrl2 = grab2;
						m_listePtCtrls.at(m_listePtCtrls.count()-1).m_grab2->setPos( QPointF(grab2.x() + dx, grab2.y() + dy));
					}



					//QPointF pathPoint = polygonPath.at(i);
					m_polygon.replace(i, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
					m_polygon.replace(m_polygon.count()-1, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));

				}
				if(i == m_polygon.count()-1)
				{
					QPointF pathPoint = m_listePtCtrls.at(0).m_position;
					QPointF grab1= m_listePtCtrls.at(0).m_grab1->pos();
					QPointF grab2;
					if(  m_listePtCtrls.at(0).m_grab2 != nullptr)  grab2= m_listePtCtrls.at(0).m_grab2->pos();

					m_listePtCtrls[0].m_position  = QPointF(pathPoint.x() + dx, pathPoint.y() + dy);
					m_listePtCtrls.at(0).m_grab1->setPos( QPointF(grab1.x() + dx, grab1.y() + dy));

					m_listePtCtrls[0].m_ctrl1 = grab1;
					if(  m_listePtCtrls.at(0).m_grab2 != nullptr)
					{
						m_listePtCtrls[0].m_ctrl2 = grab2;
						m_listePtCtrls.at(0).m_grab2->setPos( QPointF(grab2.x() + dx, grab2.y() + dy));
					}

					m_listePtCtrls[m_listePtCtrls.count()-1].m_position  = QPointF(pathPoint.x() + dx, pathPoint.y() + dy);

					grabberList[0]->setPos(QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
					//QPointF pathPoint = polygonPath.at(i);
					m_polygon.replace(m_polygon.count()-1, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
					m_polygon.replace(0, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));

				}

			}

		}
	}

}




void GraphEditor_ListBezierPath::grabberCtrlMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {

	m_ItemGeometryChanged=true;
	if ( m_grabbersCtrl.isEmpty() )
		return;


	for(int i = 0; i < m_grabbersCtrl.size(); i++)
	{
		if(m_grabbersCtrl.at(i) == signalOwner)
		{


			for(int j=0;j< m_listePtCtrls.size();j++)
			{
				if(m_listePtCtrls[j].m_grab1 == m_grabbersCtrl.at(i))
				{

					QPointF pathPoint = m_grabbersCtrl.at(i)->pos() ;//m_listePtCtrls[j]->m_ctrl1;

				 m_listePtCtrls[j].m_ctrl1 =  QPointF(pathPoint.x() + dx, pathPoint.y() + dy);

				}

				else if(m_listePtCtrls[j].m_grab2 == m_grabbersCtrl.at(i))
				{
					QPointF pathPoint =m_grabbersCtrl.at(i)->pos();// m_listePtCtrls[j]->m_ctrl2;
					//m_listeControls.replace(j, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
					m_listePtCtrls[j].m_ctrl2 =  QPointF(pathPoint.x() + dx, pathPoint.y() + dy);

				}
			}


		}
	}



/*  for(int i = 0; i < m_grabbersCtrl.size(); i++){
		if(m_grabbersCtrl.at(i) == signalOwner){

			for(int j=0;j< m_listePtCtrls.size();j++)
			{


				float man1  =(m_grabbersCtrl.at(i)->pos() - m_listePtCtrls[j]->m_ctrl1 ).manhattanLength();
				float man2  =(m_grabbersCtrl.at(i)->pos() - m_listePtCtrls[j]->m_ctrl2 ).manhattanLength();
				//qDebug()<<i<<"  ,  " <<j <<"  man1: "<<man1<<"  ;  man2: "<<man2;
				float tolerance = 50.0f;
				if( man1 < tolerance)
				{
					qDebug()<<i<<" , "<<j  <<"    m_ctrl1 selected ";
					QPointF pathPoint = m_grabbersCtrl.at(i)->pos() ;//m_listePtCtrls[j]->m_ctrl1;
					//m_listePtCtrls.replace(j, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
					m_listePtCtrls.at(j)->m_ctrl1 =  QPointF(pathPoint.x() + dx, pathPoint.y() + dy);

				}
				else if( man2 < tolerance)
				{
					qDebug()<<i<<" , "<<j  <<"    m_ctrl2 selected ";
					QPointF pathPoint =m_grabbersCtrl.at(i)->pos();// m_listePtCtrls[j]->m_ctrl2;
					//m_listeControls.replace(j, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
					m_listePtCtrls.at(j)->m_ctrl2 =  QPointF(pathPoint.x() + dx, pathPoint.y() + dy);

				}
			}


		}
	}
*/


}


void GraphEditor_ListBezierPath::moveGrabber(int index, int dx, int dy) {


	m_ItemGeometryChanged=true;
		if ( grabberList.isEmpty() )
			return;
	QPolygonF polygonPath = m_polygon;
	QPointF pathPoint = polygonPath.at(index);
	polygonPath.replace(index, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));


	emit currentIndexChanged(index);




	setPolygonFromMove(polygonPath);
}

void GraphEditor_ListBezierPath::positionGrabber(int index, QPointF pos)
{
	m_ItemGeometryChanged=true;
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = m_polygon;

	if(index >= 0 && index<grabberList.size())
	{

		grabberList[index]->setPos(pos);
		QPointF pathPoint = polygonPath.at(index);
		polygonPath.replace(index, pos);
		emit currentIndexChanged(index);

		setPolygonFromMove(polygonPath);

		refreshCurve();


		emit polygonChanged(m_listePtCtrls,getPointInterpolated(),!m_IsClosedCurved);
		emit polygonChanged2(this);
	}
	else
		qDebug()<<"GraphEditor_PolyLineShape::positionGrabber:"<<index;

}

void GraphEditor_ListBezierPath::GrabberDeleted(QGraphicsItem *signalOwner)
{
	if ( grabberList.isEmpty() )
			return;
		QPolygonF polygonPath = m_polygon;
		QPolygonF newPolygonPath;
		bool found = false;
		int indexDeleted;
		for(int i = 0; i < grabberList.size(); i++){
			if(grabberList.at(i) == signalOwner){
				found = true;
				indexDeleted = i;
				//break;
			} else {
				newPolygonPath << polygonPath[i];
			}
		}
		if (found) {
			//m_listePtCtrls.removeAt(indexDeleted);
			GraphEditor_GrabberItem* dot = grabberList.at(indexDeleted);
			grabberList.removeAt(indexDeleted);
			dot->deleteLater();
			setPolygon(newPolygonPath);

			m_ItemGeometryChanged=true;

			showGrabbersCtrl();

		}
}

void GraphEditor_ListBezierPath::setPathCurrent(QPainterPath painter)
{
	this->pathCurrent = painter;
}


GraphEditor_ListBezierPath* GraphEditor_ListBezierPath::clone()
{

	//GraphEditor_ListBezierPath* cloned = new GraphEditor_ListBezierPath(polygon(), pen(), brush(), m_Menu, m_scene,m_IsClosedCurved);
	GraphEditor_ListBezierPath* cloned = new GraphEditor_ListBezierPath(m_listePtCtrls, m_penYellow, brush(), m_Menu, m_scene,m_IsClosedCurved,m_penYellow.color());

	cloned->setPos(scenePos());
	cloned->setZValue(zValue());
	cloned->setRotation(rotation());
	cloned->setID(m_LayerID);
	cloned->setGrabbersVisibility(false);
	cloned->setPathCurrent(this->pathCurrent);
	cloned->setNameNurbs(getNameNurbs());

	return cloned;
}


QVariant GraphEditor_ListBezierPath::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
	if(m_DrawFinished)
	{
	switch (change) {
	case QGraphicsItem::ItemSelectedChange:
	{
		if(!value.toBool()) {
			if(!m_selectStart) m_AlreadySelected=false;

			m_selectStart =false;
			setGrabbersVisibility(false);
			setCornerGrabbersVisibility(false);
			hideGrabberCurrent();
			m_indexCurrentCrtl=-1;

			if (m_textItem)
				m_textItem->setVisible(false);
		}
		else
		{
			setGrabbersVisibility(true);
			setCornerGrabbersVisibility(true);
			if (m_textItem)
				m_textItem->setVisible(true);
		}
		break;
	}
	case QGraphicsItem::ItemPositionHasChanged:
	{
		m_ItemMoved = true;
		break;
	}
	case QGraphicsItem::ItemSceneHasChanged:
	{
		m_scene = dynamic_cast<GraphicSceneEditor *> (scene());
	}
	default: break;
	}
	}
	return QGraphicsItem::itemChange(change, value);
}

QVector<QPointF>  GraphEditor_ListBezierPath::SceneCordinatesPoints()
{
	//QPolygonF pathFill = pathCurrent.toFillPolygon();
	//return mapToScene(polygon());

	return mapToScene(m_polygon);
}

QVector<QPointF>  GraphEditor_ListBezierPath::ImageCordinatesPoints()
{
	QVector<QPointF> vec;
	QPolygonF polygon_ = mapToScene(polygon());
	GraphicSceneEditor *scene_ = dynamic_cast<GraphicSceneEditor*>(scene());
	if (scene_)
	{
		foreach(QPointF p, polygon_) {
			vec.push_back(scene_->innerView()->ConvertToImage(p));
		}
	}
	return vec;
}


QPointF GraphEditor_ListBezierPath::getPosition(float t)
{
	if( t> 1.0f ) t = 1.0f;
		if( t<0.0f ) t = 0.0f;

	//	if(pathCurrent.length() == 0.0f)qDebug()<<" pathcurrent length vaut 0";
	return pathCurrent.pointAtPercent(t);
}

QPointF GraphEditor_ListBezierPath::getNormal(float t)
{
	float eps = 0.001f;

	if( t> 1.0f ) t = 1.0f;
	if( t<0.0f ) t = 0.0f;
	QPointF normal;
	if( t< 0.995f)
	{
		float t1 = t+eps;
		if( t1 >1.0f) t1 = 1.0f;
		QPointF pts1 = pathCurrent.pointAtPercent(t);
		QPointF pts2 = pathCurrent.pointAtPercent(t1);
		normal = pts2 - pts1;
	}
	else
	{
		float t1 = t-eps;
		if( t1 <0.0f) t1 = 0.0f;
		QPointF pts1 = pathCurrent.pointAtPercent(t1);
		QPointF pts2 = pathCurrent.pointAtPercent(t);
		normal = pts2 - pts1;

	}
	return normal;
}



QPainterPath GraphEditor_ListBezierPath::shape() const
{
	QPainterPath path;
	QPolygonF polygon0 = m_polygon;
	if ( polygon0.size() == 0 ) return path;
	double xg = 0.0;
	double yg = 0.0;
	for (int n=0; n<polygon0.size(); n++)
	{
		xg += polygon0[n].x();
		yg += polygon0[n].y();
	}
	xg /= polygon0.size();
	yg /= polygon0.size();

	QPolygonF polygonShape;
	for (int i=0; i<polygon0.size(); i++)
	{
		QPointF valF;
		valF = polygon0[i].toPoint();
		valF.setX((valF.x()-xg)*(1.0-m_scale)+xg);
		valF.setY((valF.y()-yg)*(1.0-m_scale)+yg);
		polygonShape << valF.toPoint();
	}
	for (int i=polygon0.size()-1; i>=0; i--)
	{
		QPointF valF;
		valF = polygon0[i].toPoint();
		valF.setX((valF.x()-xg)*(1.0+m_scale)+xg);
		valF.setY((valF.y()-yg)*(1.0+m_scale)+yg);
		polygonShape << valF.toPoint();
	}
	path.addPolygon(polygonShape);
	return path;
}
