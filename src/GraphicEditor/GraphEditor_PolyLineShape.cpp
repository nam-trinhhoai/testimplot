/*
 * GraphEditor_PolyLineShape.cpp
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */


#include <QGraphicsSceneMouseEvent>
#include <QPainterPath>
#include <QGraphicsScene>
#include <QDebug>
#include <math.h>
#include <QGraphicsScene>
#include <QGraphicsView>

#include "GraphEditor_GrabberItem.h"
#include "GraphEditor_PolyLineShape.h"
#include "GraphicSceneEditor.h"
#include "LineLengthText.h"
#include "singlesectionview.h"

#include <malloc.h>
GraphEditor_PolyLineShape::GraphEditor_PolyLineShape()
{
	m_textItem = new LineLengthText(this);
	m_textItem->setVisible(false);
	setAcceptHoverEvents(true);
	m_View = nullptr;
	m_IsClosedCurved = false;
	setFlags(ItemIsSelectable|ItemIsFocusable);
	m_Menu = nullptr;
	setData(0, "noRotation");
}

void GraphEditor_PolyLineShape::wheelEvent(QGraphicsSceneWheelEvent *event)
{
	qDebug() << "wheel";
}



GraphEditor_PolyLineShape::GraphEditor_PolyLineShape(QPolygonF polygon, QPen pen, QBrush brush, QMenu* itemMenu,
		GraphicSceneEditor* scene, bool isClosedCurved)
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
}

GraphEditor_PolyLineShape::~GraphEditor_PolyLineShape() {
	if(m_nameId !="") emit BezierDeleted(m_nameId);
}

bool GraphEditor_PolyLineShape::checkClosedPath()
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

QVector<QPointF> GraphEditor_PolyLineShape::getKeyPoints()
{
	return mapToScene(m_polygon);
}

void GraphEditor_PolyLineShape::insertNewPoints(QPointF pos)
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


void GraphEditor_PolyLineShape::polygonResize(int widthO, int width)
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


void GraphEditor_PolyLineShape::setPolygon(const QPolygonF &polygon) {
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
	calculatePerimeter();

	QVector<QPointF> pol =this->polygon();

	emit polygonChanged(pol,!m_IsClosedCurved);
}

void GraphEditor_PolyLineShape::setPolygonFromMove(const QPolygonF &polygon) {
	m_polygon = polygon;
	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	if (m_IsClosedCurved)
	{
		newPath.closeSubpath();
	}
	setPath(newPath);
	calculatePerimeter();

	QVector<QPointF> pol =this->polygon();// this->SceneCordinatesPoints();


//	emit polygonChanged(pol,!m_IsClosedCurved);
}

void GraphEditor_PolyLineShape::setDrawFinished(bool value)
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

}

void GraphEditor_PolyLineShape::polygonChanged1()
{
	//qDebug()<<"GraphEditor_PolyLineShape::polygonChanged1 ";
	emit polygonChanged(m_polygon,!m_IsClosedCurved);
}

void GraphEditor_PolyLineShape::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

	painter->setRenderHint(QPainter::Antialiasing,true);
//	painter->setRenderHint(QPainter::HighQualityAntialiasing, true);
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

	if (m_IsClosedCurved)
	{
		painter->setBrush(brush());
	}
	else
	{
		painter->setBrush(QBrush(Qt::NoBrush));
	}
	painter->drawPath(path());
}

void GraphEditor_PolyLineShape::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
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
	}
	QGraphicsItem::mouseDoubleClickEvent(event);
}

void GraphEditor_PolyLineShape::showGrabbers() {
	if ( m_readOnly ) return;
	QPolygonF polygonPath = m_polygon;
	for(int i = 0; i < polygonPath.size(); i++){
		QPointF point = polygonPath.at(i);

		GraphEditor_GrabberItem *dot = new GraphEditor_GrabberItem(this);

		connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_PolyLineShape::grabberMove);
		connect(dot, &GraphEditor_GrabberItem::signalDoubleClick, this, &GraphEditor_PolyLineShape::slotDeleted);
		connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_PolyLineShape::polygonChanged1);

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
				connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_PolyLineShape::GrabberMouseReleased);
			}
		}
	}

}

void GraphEditor_PolyLineShape::GrabberMouseReleased(QGraphicsItem *signalOwner)
{
	if ( m_readOnly ) return;
	setDrawFinished(true);
}

void GraphEditor_PolyLineShape::calculatePerimeter()
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
		if ( m_textItem && m_displayPerimetre )
		{
			m_textItem->setText(QString::number(TotalLength)+ " m");
			m_textItem->setPos(textPos);
			m_textItem->setRotation(lineAngle);
			m_textItem->setVisible(true);
		}
	}
}

void GraphEditor_PolyLineShape::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {
	if ( m_readOnly ) return;
	m_ItemGeometryChanged=true;
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = m_polygon;
	for(int i = 0; i < grabberList.size(); i++){
		if(grabberList.at(i) == signalOwner){
			QPointF pathPoint = polygonPath.at(i);
			polygonPath.replace(i, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
			emit currentIndexChanged(i);
		}
	}


	setPolygonFromMove(polygonPath);
}



void GraphEditor_PolyLineShape::moveGrabber(int index, int dx, int dy) {
	if ( m_readOnly ) return;
	m_ItemGeometryChanged=true;
		if ( grabberList.isEmpty() )
			return;
	QPolygonF polygonPath = m_polygon;
	QPointF pathPoint = polygonPath.at(index);
	polygonPath.replace(index, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
	emit currentIndexChanged(index);



	setPolygonFromMove(polygonPath);
}

void GraphEditor_PolyLineShape::positionGrabber(int index, QPointF pos)
{
	if ( m_readOnly ) return;
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
	}
	else
		qDebug()<<"GraphEditor_PolyLineShape::positionGrabber:"<<index;

}



GraphEditor_PolyLineShape* GraphEditor_PolyLineShape::clone()
{
	GraphEditor_PolyLineShape* cloned = new GraphEditor_PolyLineShape(polygon(), pen(), brush(), m_Menu, m_scene,m_IsClosedCurved);
	cloned->setPos(scenePos());
	cloned->setZValue(zValue());
	cloned->setRotation(rotation());
	cloned->setID(m_LayerID);
	cloned->setGrabbersVisibility(false);
	return cloned;
}

QVector<QPointF>  GraphEditor_PolyLineShape::SceneCordinatesPoints()
{
	return mapToScene(polygon());
}

QVector<QPointF>  GraphEditor_PolyLineShape::ImageCordinatesPoints()
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

bool GraphEditor_PolyLineShape::eventFilter(QObject* watched, QEvent* ev) {
	if (ev->type() == QEvent::Wheel) {
		//transformItemZoomScale(m_item,m_ctrl,m_ctrl->position());
	}
	fprintf(stderr, "event %d [ %d ]\n", ev->type(), QEvent::Wheel);
	return false;
}

bool GraphEditor_PolyLineShape::sceneEvent(QEvent *event)
{
	// fprintf(stderr, "event %d [ %d ]\n", event->type(), QEvent::Wheel);
	/*
	if ( event->type() == QEvent::Wheel )
	{
		return true;
	}
	else
	{
		return GraphEditor_Path::sceneEvent(event);
	}
	fprintf(stderr, "event %d\n", event->type());
	*/
	return GraphEditor_Path::sceneEvent(event);
}


QPainterPath GraphEditor_PolyLineShape::shape() const
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



