/*
 * GraphEditor_RegularBezierPath.cpp
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

#include "GraphEditor_GrabberItem.h"
#include "GraphEditor_RegularBezierPath.h"
#include "GraphicSceneEditor.h"
#include "LineLengthText.h"

GraphEditor_RegularBezierPath::GraphEditor_RegularBezierPath(QPolygonF polygon, QPen pen, QBrush brush,
		int smoothValue, QMenu* itemMenu) //: GraphEditor_Path(polygon,pen,brush,itemMenu)
{
	m_nbPtsMin = 3;
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	m_Menu = itemMenu;


	setPen(pen);
	m_Pen=pen;
	setBrush(brush);
	m_InitialPolygon = polygon;
	m_smoothValue = smoothValue;
	m_textItem = new LineLengthText(this);
	setPolygon(polygon);
	m_textItem->setVisible(false);
	m_View = nullptr;


	//qDebug()<<" constructeur1 GraphEditor_RegularBezierPath";


}

GraphEditor_RegularBezierPath::GraphEditor_RegularBezierPath(QPolygonF polygon, QPen pen, QBrush brush,QMenu* itemMenu)
{
	m_nbPtsMin = 3;
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	m_Menu = itemMenu;

	setPen(pen);
	m_Pen=pen;
	setBrush(brush);
	m_InitialPolygon = polygon;
	m_textItem = new LineLengthText(this);
	m_textItem->setVisible(false);
	QPainterPath newPath;
	newPath.addPolygon(m_InitialPolygon);
	setPath(newPath);

	//qDebug()<<" constructeur2 GraphEditor_RegularBezierPath";


}


GraphEditor_RegularBezierPath::~GraphEditor_RegularBezierPath() {

	if(m_nameId !="") emit BezierDeleted(m_nameId);
}

QVector<QPointF> GraphEditor_RegularBezierPath::getKeyPoints()
{
	if (m_IsSmoothed)
	{
		QPolygonF poly;
		foreach(GraphEditor_GrabberItem* p,grabberList)
			poly.push_back(p->scenePos());

			return mapToScene(poly);
	}
		else
			return mapToScene(m_polygon);
}


void GraphEditor_RegularBezierPath::showGrabbers()
{
	if (m_IsSmoothed)
	{
		QPolygonF polygonPath = m_polygon;
		for(int i = 0; i < polygonPath.size(); i++)
		{
			if (m_IsClosedCurved && (i > polygonPath.size()-3))
			{
				continue;
			}

			QPointF point = polygonPath.at(i);

			GraphEditor_GrabberItem *dot = new GraphEditor_GrabberItem(this);
			dot->setPos(point);
			if (!grabberList.empty())
			{
				if ((dot->collidesWithItem(grabberList.last()))
						|| (dot->collidesWithItem(grabberList.first())) )
				{
					if (scene())
						scene()->removeItem(dot);
					delete dot;
					continue;
				}
			}

			dot->setVisible(false);
			grabberList.append(dot);
		}

		QPolygonF polyPath;

		for (int i=0; i< grabberList.size(); i++)
		{
			if (!m_IsClosedCurved)
			{
				if (i == 0)
				{
					connect(grabberList[i], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_RegularBezierPath::FirstPointSmoothedMove);
					connect(grabberList[i], &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_RegularBezierPath::GrabberSmoothedMouseReleased);
					grabberList[i]->setFirstGrabberInList(true);
				}
				else if (i == grabberList.size()-1)
				{
					connect(grabberList[i], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_RegularBezierPath::LastPointSmoothedMove);
					connect(grabberList[i], &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_RegularBezierPath::GrabberSmoothedMouseReleased);
					grabberList[i]->setDetectCollision(true);
				}
				else
				{
					connect(grabberList[i], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_RegularBezierPath::grabberMove);
					connect(grabberList[i], &GraphEditor_GrabberItem::signalDoubleClick, this, &GraphEditor_Path::slotDeleted);

				}

			}
			else
			{
				connect(grabberList[i], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_RegularBezierPath::grabberMove);
				connect(grabberList[i], &GraphEditor_GrabberItem::signalDoubleClick, this, &GraphEditor_Path::slotDeleted);
			}
			connect(grabberList[i], &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_RegularBezierPath::polygonChanged1);
			polyPath.push_back(grabberList[i]->scenePos());
		}
		if (m_IsClosedCurved)
		{
			polyPath.push_back(m_polygon[0]);
			polyPath.push_back(m_polygon[1]);
			polyPath.push_back(m_polygon[2]);
		}
		m_polygon.clear();
		m_polygon=polyPath;
	}
	else
	{
		if (!m_IsClosedCurved)
		{
			QPointF point = m_polygon.first();
			GraphEditor_GrabberItem *dot = new GraphEditor_GrabberItem(this);
			connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_RegularBezierPath::FirstPointMove);
			connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_RegularBezierPath::GrabberMouseReleased);
			dot->setVisible(true);
			dot->setFirstGrabberInList(true);
			grabberList.append(dot);
			dot->setPos(point);

			point = m_polygon.last();
			dot = new GraphEditor_GrabberItem(this);
			connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_RegularBezierPath::LastPointMove);
			connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_RegularBezierPath::GrabberMouseReleased);
			dot->setVisible(true);
			grabberList.append(dot);
			dot->setPos(point);
			if (!m_DrawFinished)
				dot->setDetectCollision(true);
		}
		else
			clearGrabbers();
	}
}

void GraphEditor_RegularBezierPath::FirstPointSmoothedMove(QGraphicsItem *signalOwner, qreal dx, qreal dy)
{
	m_InitialPolygon.push_front(QPointF(m_polygon.first().x() + dx, m_polygon.first().y() + dy));

	QPolygonF polygon = m_polygon;
	polygon.push_front(QPointF(m_polygon.first().x() + dx, m_polygon.first().y() + dy));

	setPolygonFromMove(polygon);
	//smoothCurve();
}

void GraphEditor_RegularBezierPath::LastPointSmoothedMove(QGraphicsItem *signalOwner, qreal dx, qreal dy)
{
	m_InitialPolygon.push_back(QPointF(m_polygon.last().x() + dx, m_polygon.last().y() + dy));
	QPolygonF polygon = m_polygon;
	polygon.push_back(QPointF(m_polygon.last().x() + dx, m_polygon.last().y() + dy));

	setPolygonFromMove(polygon);
	//smoothCurve();

}

void GraphEditor_RegularBezierPath::setPolygonFromMove(const QPolygonF &poly)
{
	m_ItemGeometryChanged=true;
	m_polygon=poly;
	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	setPath(newPath);
	if (m_IsSmoothed)
	{
		fillKnotVector();
		interpolateCurve();
	}
	if (m_IsClosedCurved)
	{
		calculateArea(m_IsSmoothed? interpolatedPoints:m_polygon);
	}

//	QVector<QPointF> pol = this->SceneCordinatesPoints();
//	qDebug()<<" GraphEditor_RegularBezierPath : "<<m_polygon.size();
//	qDebug()<<" GraphEditor_RegularBezierPath controlPoints : "<<controlPoints.size();
	//emit polygonChanged(m_polygon);
}

/*
int GraphEditor_RegularBezierPath::getIndexKey(int indice )
{
	qDebug()<<" m_polygon size:"<<m_polygon.size();


}
*/
void GraphEditor_RegularBezierPath::polygonResize(int widthO, int width)
{

	float decal = (width- widthO)/2;

	QPolygonF polygon = m_polygon;
	for(int i=0;i<polygon.size();i++)
	{
		polygon[i].setX(decal+ polygon[i].x());
	}


	for(int i=0;i<grabberList.count();i++)
	{
		grabberList[i]->moveX(decal );
	}

	setPolygonFromMove(polygon);

	//emit polygonChanged(m_polygon);
}

void GraphEditor_RegularBezierPath::polygonChanged1()
{
//	QVector<QPointF> pol = this->SceneCordinatesPoints();
	//qDebug()<<"GraphEditor_RegularBezierPath : "<<m_polygon[0];
	emit polygonChanged(m_polygon,!m_IsClosedCurved);
}

void GraphEditor_RegularBezierPath::setDrawFinished(bool value)
{

	GraphEditor_Path::setDrawFinished(value);
	//qDebug()<<"setDrawFinished "<<value;




	if (value == true)
	{
		smoothCurve();
		//emit polygonChanged(m_polygon);
	}
}

void GraphEditor_RegularBezierPath::GrabberSmoothedMouseReleased(QGraphicsItem *signalOwner)
{
	setDrawFinished(true);


}

void GraphEditor_RegularBezierPath::smoothCurve()
{
	m_ItemGeometryChanged=true;
	if (!m_IsSmoothed)
	{
		m_InitialPolygon = m_polygon;
	}
	if (m_smoothValue == 0)
	{
		return;
	}

	m_polygon.clear();
	if (m_smoothValue >= m_InitialPolygon.size())
	{
		m_polygon.append(m_InitialPolygon.first());
		m_polygon.append(m_InitialPolygon.last());
	}
	else
	{
		QPolygonF polygon;
		polygon=m_InitialPolygon;

		qreal diffX =0;
		qreal diffY =0;
		for (int i=1; i< polygon.size(); i++)
		{
			diffX += std::abs(polygon[i].x()- polygon[i-1].x());
			diffY += std::abs(polygon[i].y()- polygon[i-1].y());
		}
		qreal moyDiffX = diffX/ (polygon.size()-1);
		qreal moyDiffY = diffY/ (polygon.size()-1);

		QPolygonF FilterPolygon;
		QPointF refPoint = polygon[0];
		if (polygon.size()>50)
		{
			do
			{
				FilterPolygon.clear();
				//FilterPolygon.push_back(polygon[0]);
				for (int i=1; i< polygon.size(); i++)
				{
					if ((std::abs(polygon[i].x()- refPoint.x())>(moyDiffX)) ||
							(std::abs(polygon[i].y() - refPoint.y())>(moyDiffY)) )
					{
						refPoint = polygon[i];
						FilterPolygon.push_back(polygon[i]);
					}
				}
				//FilterPolygon.push_back(polygon[polygon.size()-1]);
				moyDiffX += 1;
				moyDiffY += 1;
			} while ((FilterPolygon.size()>50));
		}
		else
			FilterPolygon = polygon;

		m_polygon.clear();
		for (int i=0; i< FilterPolygon.size(); i+=m_smoothValue)
		{
			m_polygon.append(FilterPolygon[i]);
		}
		if (m_IsClosedCurved)
		{
			m_polygon.push_back(m_polygon[0]);
			m_polygon.push_back(m_polygon[1]);
			m_polygon.push_back(m_polygon[2]);
		}
	}

	clearGrabbers();
	m_IsSmoothed = true;
	showGrabbers();
	setGrabbersVisibility(true);
	fillKnotVector();
	interpolateCurve();


	emit polygonChanged(m_polygon,!m_IsClosedCurved);

}

void GraphEditor_RegularBezierPath::setPolygon(const QPolygonF &polygon) {
	if (polygon.isEmpty())
		return;
	m_polygon=polygon;
	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	setPath(newPath);
	clearGrabbers();
	showGrabbers();
	if (m_IsSmoothed)
	{
		fillKnotVector();
		interpolateCurve();
	}
	if (m_IsClosedCurved)
	{
		calculateArea(m_IsSmoothed? interpolatedPoints:m_polygon);
	}
}
/*
void GraphEditor_RegularBezierPath::receiveAddPts(QPointF pos)
{
	insertNewPoints(pos);
}*/


void GraphEditor_RegularBezierPath::insertNewPoints(QPointF pos)
{
		QPolygonF polygonPath = m_polygon;
		QPolygonF newPath( polygonPath );
		float epsilon = 0.1f;
		bool found = false;
		double distanceMin = 10000;
		QPointF newPointBest;
		double factor =1;
		int indexAdded;
		int currentindex= -1;

		for(int i = 0; i < polygonPath.size()-1; i++){ // i go from 0 to N-1 because we do not want to pick on the last line of the polygon
			QPointF p1 = polygonPath.at(i);
			QPointF p2 = (i < polygonPath.size()-1) ? polygonPath.at(i+1) : polygonPath.at(0);
			double APx = pos.x() - p1.x();
			double APy = pos.y() - p1.y();
			double ABx = p2.x() - p1.x();
			double ABy = p2.y() - p1.y();
			double magAB2 = ABx*ABx + ABy*ABy;
			double ABdotAP = ABx*APx + ABy*APy;
			double t = ABdotAP / magAB2;


			//double distance = sqrt( pow( (p1.x() - pos.x()), 2) + pow( (p1.y() - pos.y()), 2));
			if(t > -epsilon && t < epsilon)
			{
				emit signalCurrentIndex(i);
				return;
			}
			if(t > 1.0-epsilon && t < 1.0+epsilon)
			{
				emit signalCurrentIndex(i+1);
				return;
			}

			QPointF newPoint;

			if ( t < 0.0f + epsilon) {
				//if(currentindex<0) currentindex = i;
				//newPoint = trackLine.p1();
			}else if (t > 1.0f - epsilon){
				//if(currentindex<0) currentindex = i+1;
				//newPoint = trackLine.p2();
			}else{
				newPoint.setX(p1.x() + ABx*t);
				newPoint.setY(p1.y() + ABy*t);
				double d = sqrt( pow( (newPoint.x() - pos.x()), 2) + pow( (newPoint.y() - pos.y()), 2));
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


		emit signalCurrentIndex(currentindex);
}

void GraphEditor_RegularBezierPath::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
	QPointF clickPos = event->pos();
	//insertNewPoints(clickPos);
	QPolygonF polygonPath = m_polygon;
	QPolygonF newPath( polygonPath );

	bool found = false;
	double distanceMin = 10000;
	QPointF newPointBest;
	double factor =1;

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
		QPointF nextPoint = (indexAdded - 1 < polygonPath.size()-1) ? polygonPath.at(indexAdded) : polygonPath.at(0);
		int x = polygonPath.at(indexAdded - 1).x() + factor * (nextPoint.x() - polygonPath.at(indexAdded - 1).x());
		int y = polygonPath.at(indexAdded - 1).y() + factor * (nextPoint.y() - polygonPath.at(indexAdded - 1).y());
		QPointF newPoint(x, y );
		newPath.insert(indexAdded, newPoint);
		setPolygon(newPath);
		setGrabbersVisibility(true);
		m_ItemGeometryChanged=true;
	}

	QGraphicsItem::mouseDoubleClickEvent(event);
}

void GraphEditor_RegularBezierPath::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {
	m_ItemGeometryChanged=true;

	if ( grabberList.isEmpty() || m_polygon.isEmpty())
	{
		return;
	}
	QPolygonF polygonPath = m_polygon;
	for(int i = 0; i < grabberList.size(); i++)
	{
		if (m_IsClosedCurved && (i >= (polygonPath.size()-3)))
		{
			continue;
		}
		if(grabberList.at(i) == signalOwner){
			QPointF pathPoint = polygonPath.at(i);
			polygonPath.replace(i, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
			if (m_IsClosedCurved && signalOwner == grabberList.at(0))
			{
				polygonPath.replace(polygonPath.size()-3, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
			}
			if (m_IsClosedCurved && signalOwner == grabberList.at(1))
			{
				polygonPath.replace(polygonPath.size()-2, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
			}
			if (m_IsClosedCurved && signalOwner == grabberList.at(2))
			{
				polygonPath.replace(polygonPath.size()-1, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
			}
		}
	}

	//emit polygonChanged(polygonPath);
	setPolygonFromMove(polygonPath);
}


void GraphEditor_RegularBezierPath::moveGrabber(int index, int dx, int dy) {
	m_ItemGeometryChanged=true;

	if ( grabberList.isEmpty() || m_polygon.isEmpty())
	{
		return;
	}
	QPolygonF polygonPath = m_polygon;
	if(index >= 0 && index<grabberList.size())
	{

		grabberList[index]->moveX(dx);
		grabberList[index]->moveY(dy);

		QPointF pathPoint = polygonPath.at(index);
		polygonPath.replace(index, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));

		if (m_IsClosedCurved && index == 0)
		{
			polygonPath.replace(polygonPath.size()-3, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
		}
		if (m_IsClosedCurved && index == 1)
		{
			polygonPath.replace(polygonPath.size()-2, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
		}
		if (m_IsClosedCurved && index == 2)
		{
			polygonPath.replace(polygonPath.size()-1, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
		}
	}

	//qDebug()<<" moveGrabber "<<index<< "  , dx : "<<dx<<"  , dy :"<<dy;
	setPolygonFromMove(polygonPath);

}

void GraphEditor_RegularBezierPath::positionGrabber(int index,QPointF pos) {
	m_ItemGeometryChanged=true;

	if ( grabberList.isEmpty() || m_polygon.isEmpty())
	{
		return;
	}
	QPolygonF polygonPath = m_polygon;
	if(index >= 0 && index<grabberList.size())
	{

		grabberList[index]->setPos(pos);

		QPointF pathPoint = polygonPath.at(index);
		polygonPath.replace(index, pos);

		if (m_IsClosedCurved && index == 0)
		{
			polygonPath.replace(polygonPath.size()-3, pos);
		}
		if (m_IsClosedCurved && index == 1)
		{
			polygonPath.replace(polygonPath.size()-2, pos);
		}
		if (m_IsClosedCurved && index == 2)
		{
			polygonPath.replace(polygonPath.size()-1, pos);
		}
	}

	//qDebug()<<" moveGrabber "<<index<< "  , dx : "<<dx<<"  , dy :"<<dy;
	setPolygonFromMove(polygonPath);

}



void GraphEditor_RegularBezierPath::fillKnotVector()
{
	if (m_polygon.size()>2)
	{
		int middleKnotNumber = m_polygon.size() - 4;
		knotVector.clear();
		for (int counter = 0; counter < 4; ++counter)
			knotVector.push_back(0.0);
		for (int counter = 1; counter <= middleKnotNumber; ++counter)
			knotVector.push_back(1.0 / (middleKnotNumber + 1) * counter);
		for (int counter = 0; counter < 4; ++counter)
			knotVector.push_back(1.0);
	}
}

void GraphEditor_RegularBezierPath::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

	if (m_polygon.size()>2)
	{
		painter->setRenderHint(QPainter::Antialiasing,true);
		//painter->setRenderHint(QPainter::HighQualityAntialiasing, true);
		painter->setRenderHint(QPainter::SmoothPixmapTransform, true);

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

		QPainterPath Path;
		if (m_IsSmoothed)
		{
			Path.addPolygon(interpolatedPoints);
			Path.setFillRule(Qt::WindingFill);
		}
		else
		{
			Path.addPolygon(m_polygon);
		}
		if (m_IsClosedCurved)
		{
			Path.closeSubpath();
		}
		painter->drawPath(Path);


	}
}

void GraphEditor_RegularBezierPath::setSmoothValue(int newValue)
{
	m_smoothValue=newValue;
	smoothCurve();
}

void GraphEditor_RegularBezierPath::restoreState(QPolygonF poly, bool isSmoothed, bool isClosedCurve, int smoothValue)
{
	m_smoothValue = smoothValue;
	m_IsSmoothed = isSmoothed;
	m_IsClosedCurved = isClosedCurve;
	m_polygon = poly;

	if (m_IsSmoothed)
	{
		fillKnotVector();
		interpolateCurve();
	}
	clearGrabbers();
	if (!m_IsClosedCurved || m_IsSmoothed)
	{
		showGrabbers();
	}
	if (m_IsClosedCurved)
	{
		calculateArea(m_IsSmoothed? interpolatedPoints:m_polygon);
	}
}

GraphEditor_RegularBezierPath* GraphEditor_RegularBezierPath::clone()
{
	//qDebug()<<"GraphEditor_RegularBezierPath::clone()";
	GraphEditor_RegularBezierPath* cloned = new GraphEditor_RegularBezierPath(m_InitialPolygon, pen(), brush(), m_Menu);
	cloned->restoreState(m_polygon,m_IsSmoothed,m_IsClosedCurved, m_smoothValue);
	cloned->setGrabbersVisibility(false);
	cloned->setPos(scenePos());
	cloned->setZValue(zValue());
	cloned->setID(m_LayerID);
	cloned->setRotation(rotation());
	return cloned;
}


qreal N1( int  i, qreal u)
{
	qreal t = u - i;
	if  ( 0  <= t && t <  1 ){
		return  t;
	}
	if  ( 1  <= t && t <  2 ){
		return   2  - t;
	}
	return   0 ;
}

qreal N2( int  i, qreal u)
{
	qreal t = u - i;
	if  ( 0  <= t && t <  1 ){
		return   0.5  * t * t;
	}
	if  ( 1  <= t && t <  2 ){
		return   3  * t - t * t - 1.5 ;
	}
	if  ( 2  <= t && t <  3 ){
		return   0.5  * pow( 3  - t,  2 );
	}
	return   0 ;
}

qreal N3( int  i, qreal u)
{
	qreal t = u - i;
	qreal a =  1.0  /  6.0 ;
	if  ( 0  <= t && t <  1 ){
		return  a * t * t * t;
	}
	if  ( 1  <= t && t <  2 ){
		return  a * (- 3  * pow(t -  1 ,  3 ) +  3  * pow(t -  1 ,  2 ) +  3  * (t -  1 ) +  1 );
	}
	if  ( 2  <= t && t <  3 ){
		return  a * ( 3  * pow(t -  2 ,  3 ) -  6  * pow(t -  2 ,  2 ) +  4 );
	}
	if  ( 3  <= t && t <  4 ){
		return  a * pow( 4  - t,  3 );
	}
	return   0 ;
}

qreal N( int  k,  int  i, qreal u)
{
	switch  (k) {
	case   1 :
		return  N1(i, u);
	case   2 :
		return   N2(i, u);
	case   3 :
		return   N3(i, u);
	default :
		break ;
	}
}

void  GraphEditor_RegularBezierPath::interpolateCurve()
{
	if (m_polygon.size()>2)
	{
		interpolatedPoints.clear();

		if (m_IsClosedCurved)
		{
			QPolygonF m_ctrlPoints = m_polygon;
			int  currentK =  3 ;

			for  (qreal u = currentK; u < m_ctrlPoints.size(); u +=  0.01 )
			{
				QPointF pt( 0.0 ,  0.0 );
				for  ( int  i = 0 ; i < m_ctrlPoints.size(); ++i){
					QPointF pts = m_ctrlPoints[i];
					pts *= N(currentK, i, u);
					pt += pts;
				}
				interpolatedPoints.push_back(pt);
			}

		}
		else
		{
			controlPoints.clear();
			foreach(QPointF p, m_polygon)
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
}


/*



 std::vector<glm::vec3> & Curve::bezierCurveByCasteljau(const std::vector<glm::vec3>& controlPoints){
    _controlPoints = controlPoints;
    _vertices.clear();

    auto etape = (float)_iterations;
    if(etape < 0) etape = 0;
    if(etape > (float) NB_POINTS*_controlPoints.size()) etape = (float) NB_POINTS;
    long int nbU = NB_POINTS;

    float i = 0.0f;
    while(i <=  etape/(float)nbU){
        glm::vec3 v = bezierCurveByCasteljauRec(controlPoints, (float) i);
        //std::cout << v.x << " " << v.y << " " << v.z << std::endl;
        _vertices.push_back(v);
        i+=1.0f/(float)nbU;
    }
    if( etape/(float)nbU == 1) _vertices.push_back(controlPoints.at(controlPoints.size()-1));
    return _vertices;
}

glm::vec3 Curve::bezierCurveByCasteljauRec(std::vector<glm::vec3> in_pts, float i){
    if(in_pts.size() == 1) return in_pts.at(0);
    std::vector<glm::vec3> pts ;
    for(unsigned int it = 0 ; it < in_pts.size() - 1; it++){
        glm::vec3 vecteur = in_pts.at(it + 1 ) - in_pts.at(it);
        vecteur = vecteur * i;
        pts.push_back(in_pts.at(it) + vecteur);
    }

    return bezierCurveByCasteljauRec(pts, i);
}

 */


void GraphEditor_RegularBezierPath::bezierCurveByCasteljau(float u)
{

	QVector<QPointF> ctrlPts;
	for (int i=0;i< controlPoints.size();i++)
		ctrlPts.push_back(*controlPoints[i]);
//	qDebug()<<controlPoints.size() <<" ,  bezierCurveByCasteljau :"<<u;

//	 _controlPoints = controlPoints;
//	    _vertices.clear();

	  /*  auto etape = (float)_iterations;
	    if(etape < 0) etape = 0;
	    if(etape > (float) NB_POINTS*_controlPoints.size()) etape = (float) NB_POINTS;
	    long int nbU = NB_POINTS;

	    float i = 0.0f;
	    while(i <=  etape/(float)nbU){*/
	        QPointF  pos= bezierCurveByCasteljauRec(ctrlPts, u);
	        //std::cout << v.x << " " << v.y << " " << v.z << std::endl;
	/*        _vertices.push_back(v);
	        i+=1.0f/(float)nbU;
	    }
	    if( etape/(float)nbU == 1) _vertices.push_back(controlPoints.at(controlPoints.size()-1));*/
	  // qDebug()<<u<<" , position :"<<pos;
}


QPointF GraphEditor_RegularBezierPath::bezierCurveByCasteljauRec(QVector<QPointF> in_pts, float i){
    if(in_pts.size() == 1) return in_pts[0];
    QVector<QPointF> pts ;
    for(unsigned int it = 0 ; it < in_pts.size() - 1; it++){
    	QPointF vecteur = in_pts[it + 1] - in_pts[it];
        vecteur = vecteur * i;
        pts.push_back(in_pts[it] + vecteur);
    }

    return bezierCurveByCasteljauRec(pts, i);
}

void GraphEditor_RegularBezierPath::onPositionChanged( float u)
{
//	bezierCurveByCasteljau(u);
}

QVector<QPointF>  GraphEditor_RegularBezierPath::SceneCordinatesPoints()
{
	if (m_IsSmoothed)
		return mapToScene(interpolatedPoints);
	else
		return mapToScene(m_polygon);
}

QVector<QPointF>  GraphEditor_RegularBezierPath::ImageCordinatesPoints()
{
	QVector<QPointF> vec;
	QPolygonF polygon_ = m_IsSmoothed? mapToScene(interpolatedPoints) : mapToScene(m_polygon);
	GraphicSceneEditor *scene_ = dynamic_cast<GraphicSceneEditor*>(scene());
	if (scene_)
	{
		foreach(QPointF p, polygon_) {
			vec.push_back(scene_->innerView()->ConvertToImage(p));
		}
	}
	return vec;
}



QPainterPath GraphEditor_RegularBezierPath::shape() const
{
    QPainterPath path;
    QPolygonF poly;

    for (int i=0; i<m_polygon.size(); i++)
    {
    	QPointF valF;
    	valF = m_polygon[i].toPoint();
    	valF.setY(valF.y()-m_searchArea);
		poly << valF.toPoint();
    }

    for (int i=m_polygon.size()-1; i>0; i--)
    {
    	QPointF valF;
    	valF = m_polygon[i].toPoint();
    	valF.setY(valF.y()+m_searchArea);
		poly << valF.toPoint();
    }
    path.addPolygon(poly);
    return path;
}
