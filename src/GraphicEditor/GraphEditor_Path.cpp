/*
 * GraphEditor_Path.cpp
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
#include <mutex>

#include "GraphEditor_GrabberItem.h"
#include "GraphEditor_Path.h"
#include "GraphicSceneEditor.h"
#include "LineLengthText.h"

GraphEditor_Path::GraphEditor_Path(QPolygonF polygon, QPen pen, QBrush brush,
		QMenu* itemMenu, bool isClosed)
{
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	m_Menu = itemMenu;
	setPen(pen);
	m_Pen=pen;
	setBrush(brush);
	m_IsClosedCurved = isClosed;
	m_textItem = new LineLengthText(this);
	setPolygon(polygon);
	m_textItem->setVisible(false);
	m_View = nullptr;
}

GraphEditor_Path::~GraphEditor_Path() {

}


QVector<QPointF> GraphEditor_Path::getKeyPoints()
{
	return mapToScene(m_polygon);
}

void GraphEditor_Path::insertNewPoints(QPointF pt)
{
	//qDebug()<<"GraphEditor_Path::insertNewPoints";
}

void GraphEditor_Path::receiveAddPts(QPointF pos)
{
	insertNewPoints(pos);
}

void GraphEditor_Path::FirstPointMove(QGraphicsItem *signalOwner, qreal dx, qreal dy)
{
	QPolygonF polygon = m_polygon;
	polygon.push_front(QPointF(m_polygon.first().x() + dx, m_polygon.first().y() + dy));

	setPolygonFromMove(polygon);
}

void GraphEditor_Path::LastPointMove(QGraphicsItem *signalOwner, qreal dx, qreal dy)
{
	QPolygonF polygon = m_polygon;
	polygon.push_back(QPointF(m_polygon.last().x() + dx, m_polygon.last().y() + dy));

	setPolygonFromMove(polygon);
}

void GraphEditor_Path::showGrabbers()
{
	if (!m_IsClosedCurved)
	{
		QPointF point = m_polygon.first();
		GraphEditor_GrabberItem *dot = new GraphEditor_GrabberItem(this);
		connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_Path::FirstPointMove);
		connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_Path::GrabberMouseReleased);

		dot->setVisible(true);
		dot->setFirstGrabberInList(true);
		grabberList.append(dot);
		dot->setPos(point);

		point = m_polygon.last();
		dot = new GraphEditor_GrabberItem(this);
		connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_Path::LastPointMove);
		connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_Path::GrabberMouseReleased);

		dot->setVisible(true);
		grabberList.append(dot);
		if (!m_DrawFinished)
			dot->setDetectCollision(true);
		dot->setPos(point);
	}
	else
		clearGrabbers();
}

void GraphEditor_Path::setDrawFinished(bool value)
{
	m_DrawFinished = value;
	if (polygon().size()>=m_nbPtsMin  && grabberList.size()>=2)
	{
		if (grabberList[grabberList.size()-1]->hasDetectedCollision())
		{

			m_polygon.push_back(m_polygon[0]);
			m_IsClosedCurved=true;
			dynamic_cast<GraphicSceneEditor*>(scene())->backupUndostack();
			setPolygon(m_polygon);
		}
	}
}

void GraphEditor_Path::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsItem::mouseDoubleClickEvent(event);
}

void GraphEditor_Path::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

	if (polygon().size()>2)
	{
		painter->setRenderHint(QPainter::Antialiasing,true);
		//painter->setRenderHint(QPainter::HighQualityAntialiasing, true);

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

		painter->drawPath(path());
	}
}

void GraphEditor_Path::setClosedPath(bool value)
{
	m_IsClosedCurved = value;
}

GraphEditor_Path* GraphEditor_Path::clone()
{
	GraphEditor_Path* cloned = new GraphEditor_Path(polygon(), pen(), brush(), m_Menu, m_IsClosedCurved);
	cloned->setPos(scenePos());
	cloned->setZValue(zValue());
	cloned->setRotation(rotation());
	cloned->setID(m_LayerID);
	cloned->setGrabbersVisibility(false);
	return cloned;
}

const QPolygonF& GraphEditor_Path::polygon() const {
	return m_polygon;
}

void GraphEditor_Path::setPolygon(const QPolygonF &polygon) {
	if (polygon.isEmpty())
		return;
	m_polygon = polygon;
	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	setPath(newPath);
	clearGrabbers();
	showGrabbers();
	calculateArea(m_polygon);
}

void GraphEditor_Path::setPolygonFromMove(const QPolygonF &polygon) {
	m_ItemGeometryChanged=true;
	m_polygon = polygon;
	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	setPath(newPath);
	calculateArea(m_polygon);
}



void GraphEditor_Path::mousePressEvent(QGraphicsSceneMouseEvent *event)
{

	if (m_AlreadySelected && (event->button() != Qt::LeftButton))
		return;
	m_AlreadySelected = true;
	emit BezierSelected(this);
	QGraphicsItem::mousePressEvent(event);

}

void GraphEditor_Path::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsItem::mouseReleaseEvent(event);
}

void GraphEditor_Path::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if( event->modifiers() & Qt::ShiftModifier )
	{
		QGraphicsItem::mouseMoveEvent(event);
	}
}

void GraphEditor_Path::hoverEnterEvent(QGraphicsSceneHoverEvent *event) {
	if (scene()->selectedItems().empty())
	{
		m_IsHighlighted = true;
		setSelected(true);

	}
	QGraphicsItem::hoverEnterEvent(event);
}

void GraphEditor_Path::hoverLeaveEvent(QGraphicsSceneHoverEvent *event) {
	m_IsHighlighted = false;

	if (!m_AlreadySelected)
		setSelected(false);
	QGraphicsItem::hoverLeaveEvent(event);
}

void GraphEditor_Path::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {
	m_ItemGeometryChanged=true;

	if ( grabberList.isEmpty() || m_polygon.isEmpty())
	{
		return;
	}
	QPolygonF polygonPath = m_polygon;
	for(int i = 0; i < polygonPath.size(); i++)
	{
		if(grabberList.at(i) == signalOwner){
			QPointF pathPoint = polygonPath.at(i);
			polygonPath.replace(i, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
		}
	}
	setPolygonFromMove(polygonPath);
}

void GraphEditor_Path::moveGrabber(int index,int dx,int dy)
{

}

void GraphEditor_Path::positionGrabber(int, QPointF)
{

}
void GraphEditor_Path::polygonChanged1()
{

}

void GraphEditor_Path::polygonResize(int widthO, int width)
{

}

void GraphEditor_Path::GrabberMouseReleased(QGraphicsItem *signalOwner)
{

	setDrawFinished(true);
}

void GraphEditor_Path::slotDeleted(QGraphicsItem *signalOwner) {
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
		GraphEditor_GrabberItem* dot = grabberList.at(indexDeleted);
		grabberList.removeAt(indexDeleted);
		dot->deleteLater();
		setPolygon(newPolygonPath);
		m_ItemGeometryChanged=true;
	}
}

void GraphEditor_Path::clearGrabbers() {
	if(!grabberList.isEmpty()){
		foreach (GraphEditor_GrabberItem *dot, grabberList) {
			if (scene())
			{
				scene()->removeItem(dot);
			}
			dot->deleteLater();
		}
		grabberList.clear();
	}
}

void GraphEditor_Path::setGrabbersVisibility(bool visible) {
	if(!grabberList.isEmpty()){
		foreach (GraphEditor_GrabberItem *grabber, grabberList) {
			grabber->setVisible(visible);
		}
	}
}

QVariant GraphEditor_Path::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
	switch (change) {
	case QGraphicsItem::ItemSelectedChange:
	{
		if(!value.toBool()) {
			// qDebug()<<"setselected false";
			setGrabbersVisibility(false);
			m_AlreadySelected=false;
			if (m_textItem)
				m_textItem->setVisible(false);
		}
		else
		{
			// qDebug()<<"setselected true";
			setGrabbersVisibility(true);
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
		break;
	}
	default: break;
	}
	return QGraphicsItem::itemChange(change, value);
}

void GraphEditor_Path::calculateArea(QPolygonF poly)
{
	if (m_IsClosedCurved)
	{
		long long d , area = 0; // AS removed unsigned because abs of an unsigned does not make sense and gcc reject it on some systems
		if (poly.size()>2)
		{
			for (int i=0 ; i<poly.size()-1; i++)
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
}


void GraphEditor_Path::ContextualMenu(QPoint Mousepos)
{
	setSelected(true);
	if ( m_Menu )
		m_Menu->exec(Mousepos);
}

QVector<QPointF>  GraphEditor_Path::SceneCordinatesPoints()
{
	return mapToScene(polygon());
}

QVector<QPointF>  GraphEditor_Path::ImageCordinatesPoints()
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

QString GraphEditor_Path::getNameId()
{
	return m_nameId;
}

void GraphEditor_Path::setNameId(QString s)
{
	m_nameId = s;
}


QString GraphEditor_Path::getNameNurbs()
{
	return m_nameNurbs;
}

void GraphEditor_Path::setNameNurbs(QString s)
{
	m_nameNurbs = s;
	setToolTip(m_nameNurbs);
}

bool GraphEditor_Path::sceneEvent(QEvent *event)
{
	/*
	fprintf(stderr, "event %d [ %d ]\n", event->type(), QEvent::Wheel);

	if ( event->type() == QEvent::Wheel )
	{
		return true;
	}
	else
	{
		return QGraphicsPathItem::sceneEvent(event);
	}
	// return GraphEditor_Path::sceneEvent(event);
	 * */
	return QGraphicsPathItem::sceneEvent(event);
}


bool GraphEditor_Path::eventFilter(QObject* watched, QEvent* ev) {
	if (ev->type() == QEvent::Wheel) {
		//transformItemZoomScale(m_item,m_ctrl,m_ctrl->position());
	}
	fprintf(stderr, "event %d [ %d ]\n", ev->type(), QEvent::Wheel);
	return false;
}


void GraphEditor_Path::wheelEvent(QGraphicsSceneWheelEvent *event)
{
	fprintf(stderr, "event %d [ %d ]\n", event->type(), QEvent::Wheel);
}
