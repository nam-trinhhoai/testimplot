/*
 * GraphEditor_PolygonShape.cpp
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
//#include <QGLWidget>
//#include <QGLFormat>
//#include <QGLContext>

#include "GraphEditor_PolygonShape.h"
#include "GraphEditor_GrabberItem.h"
#include "GraphicSceneEditor.h"
#include "LineLengthText.h"

GraphEditor_PolygonShape::GraphEditor_PolygonShape(QPolygonF polygon, QPen pen, QBrush brush, QMenu* itemMenu, bool isEditable ) : m_IsEditable(isEditable)
{
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	setPen(pen);
	m_Pen=pen;
	m_Menu = itemMenu;
	setBrush(brush);
	m_textItem = new LineLengthText(this);
	setPolygon(polygon);
	m_textItem->setVisible(false);
	m_View = nullptr;
}

GraphEditor_PolygonShape::~GraphEditor_PolygonShape()
{

}

void GraphEditor_PolygonShape::calculateArea()
{
	long long d , area = 0; // AS removed unsigned because abs of an unsigned does not make sense and gcc reject it on some systems

	QPolygonF poly = polygon();
	if (poly.size()>2)
	{
		for (int i=0 ; i<poly.size(); i++)
		{
			QPointF p1 = poly[i];
			QPointF p2 = poly[(i + 1) % poly.size()];
			d = p1.x() * p2.y() - p2.x() * p1.y();
			area += d;
		}
		QRectF bbox = boundingRect().normalized();
		QPointF center = bbox.center();
		m_textItem->setText(QString::number(abs(area)/2000000) + " km²");
		m_textItem->setPos(center);
	}
}

void GraphEditor_PolygonShape::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

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

	painter->setBrush(brush());
	painter->drawPolygon(polygon());
}

void GraphEditor_PolygonShape::setPolygon(const QPolygonF &polygon) {
	if (polygon.isEmpty())
		return;
	QGraphicsPolygonItem::setPolygon(polygon);
	clearGrabbers();
	showGrabbers();
	calculateArea();
}


void GraphEditor_PolygonShape::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (m_AlreadySelected && (event->button() != Qt::LeftButton))
		return;
	m_AlreadySelected = true;
	QGraphicsItem::mousePressEvent(event);
}

void GraphEditor_PolygonShape::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsItem::mouseReleaseEvent(event);
}

void GraphEditor_PolygonShape::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if( event->modifiers() & Qt::ShiftModifier )
	{
		QGraphicsItem::mouseMoveEvent(event);
	}
}

void GraphEditor_PolygonShape::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
	if (m_IsEditable)
	{
		QPointF clickPos = event->pos();
		QPolygonF polygonPath = polygon();
		QPolygonF newPath( polygonPath );

		bool found = false;
		double distanceMin = 10000;
		QPointF newPointBest;
		double factor =1;
		int indexAdded;
		for(int i = 0; i < polygonPath.size(); i++){
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


void GraphEditor_PolygonShape::hoverEnterEvent(QGraphicsSceneHoverEvent *event) {
	if (scene()->selectedItems().empty())
	{
		m_IsHighlighted = true;
		setSelected(true);
	}
	QGraphicsItem::hoverEnterEvent(event);
}

void GraphEditor_PolygonShape::hoverLeaveEvent(QGraphicsSceneHoverEvent *event) {
	m_IsHighlighted = false;
	if (!m_AlreadySelected)
		setSelected(false);
	QGraphicsItem::hoverLeaveEvent(event);
}

void GraphEditor_PolygonShape::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {
	m_ItemGeometryChanged=true;
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = polygon();
	for(int i = 0; i < grabberList.size(); i++){
		if(grabberList.at(i) == signalOwner){
			QPointF pathPoint = polygonPath.at(i);
			polygonPath.replace(i, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
		}
	}
	QGraphicsPolygonItem::setPolygon(polygonPath);
	calculateArea();
}

void GraphEditor_PolygonShape::slotDeleted(QGraphicsItem *signalOwner) {
	m_ItemGeometryChanged=true;
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = polygon();
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
	if ( found) {
		GraphEditor_GrabberItem* dot = grabberList.at(indexDeleted);
		grabberList.removeAt(indexDeleted);
		dot->deleteLater();
		setPolygon(newPolygonPath);
	}
}

void GraphEditor_PolygonShape::clearGrabbers() {
	if(!grabberList.isEmpty()){
		foreach (GraphEditor_GrabberItem *dot, grabberList) {
			scene()->removeItem(dot);
			dot->deleteLater();
		}
		grabberList.clear();
	}
}

void GraphEditor_PolygonShape::showGrabbers() {
	QPolygonF polygonPath = polygon();
	for(int i = 0; i < polygonPath.size(); i++){
		QPointF point = polygonPath.at(i);

		GraphEditor_GrabberItem *dot = new GraphEditor_GrabberItem(this);

		connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_PolygonShape::grabberMove);
		connect(dot, &GraphEditor_GrabberItem::signalDoubleClick, this, &GraphEditor_PolygonShape::slotDeleted);

		dot->setVisible(true);
		grabberList.append(dot);
		dot->setPos(point);
	}
}

void GraphEditor_PolygonShape::setGrabbersVisibility(bool visible) {
	if(!grabberList.isEmpty()){
		foreach (GraphEditor_GrabberItem *grabber, grabberList) {
			grabber->setVisible(visible);
		}
	}
}

QVariant GraphEditor_PolygonShape::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
	switch (change) {
	case QGraphicsItem::ItemSelectedChange:
	{
		if(!value.toBool()) {
			setGrabbersVisibility(false);
			m_textItem->setVisible(false);
			m_AlreadySelected=false;
		} else {
			setGrabbersVisibility(true);
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
	return QGraphicsItem::itemChange(change, value);
}

GraphEditor_PolygonShape* GraphEditor_PolygonShape::clone()
{
	GraphEditor_PolygonShape* cloned = new GraphEditor_PolygonShape(polygon(), pen() ,brush(), m_Menu);
	cloned->setPos(scenePos());
	cloned->setZValue(zValue());
	cloned->setRotation(rotation());
	cloned->setID(m_LayerID);
	cloned->setGrabbersVisibility(false);
	return cloned;
}

void GraphEditor_PolygonShape::ContextualMenu(QPoint Mousepos)
{
	setSelected(true);
	m_Menu->exec(Mousepos);
}

QVector<QPointF>  GraphEditor_PolygonShape::SceneCordinatesPoints()
{
	return mapToScene(polygon());
}

QVector<QPointF>  GraphEditor_PolygonShape::ImageCordinatesPoints()
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


