/*
 * GraphEditorEllipse.cpp
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#include "GraphEditor_EllipseShape.h"

#include <QObject>
#include <QPainter>
#include <QDebug>
#include <QGraphicsSceneMouseEvent>
#include <QStyleOptionGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>

#include "GraphEditor_GrabberItem.h"
#include "GraphicSceneEditor.h"
#include "LineLengthText.h"

GraphEditor_EllipseShape::GraphEditor_EllipseShape(QRectF rect, QPen pen, QBrush brush, QMenu* itemMenu)
{
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	setPen(pen);
	m_Pen=pen;
	m_Menu = itemMenu;
	setBrush(brush);
	m_textItem = new LineLengthText(this);
	m_textItem->setVisible(false);
	createGrabbers();
	setPositionGrabbers();
	setGrabbersVisibility(true);
	m_View = nullptr;
}


GraphEditor_EllipseShape::~GraphEditor_EllipseShape(){
	for(int i = 0; i < 8; i++){
		delete m_cornerGrabber[i];
	}
}

void GraphEditor_EllipseShape::setRect(const QRectF &rect)
{
	QGraphicsEllipseItem::setRect(rect);
	setPositionGrabbers();
	calculateArea();
}

void GraphEditor_EllipseShape::calculateArea()
{
	qreal area = M_PI * rect().width()/2 * rect().height()/2;
	QRectF bbox = boundingRect().normalized();
	QPointF center = bbox.center();

	m_textItem->setText(QString::number(area/1000000) + " km²");
	m_textItem->setPos(center);
}

void GraphEditor_EllipseShape::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
	if (scene()->selectedItems().empty())
	{
		m_IsHighlighted = true;
		setSelected(true);
	}
	QGraphicsItem::hoverEnterEvent(event);
}

void GraphEditor_EllipseShape::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (m_AlreadySelected && (event->button() != Qt::LeftButton))
		return;
	m_AlreadySelected = true;
	QGraphicsItem::mousePressEvent(event);
}

void GraphEditor_EllipseShape::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsItem::mouseReleaseEvent(event);
}

void GraphEditor_EllipseShape::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if( event->modifiers() & Qt::ShiftModifier )
	{
		QGraphicsItem::mouseMoveEvent(event);
	}
}

void GraphEditor_EllipseShape::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
	m_IsHighlighted = false;
	if (!m_AlreadySelected)
		setSelected(false);
	QGraphicsItem::hoverLeaveEvent(event);
}

void GraphEditor_EllipseShape::polygonResize(int widthO, int width)
{
	float decal = (width- widthO)/2;

	QRectF tmpRect = rect();

	tmpRect.translate( decal , 0 );

		setRect( tmpRect );
		// Update to see the result
		update();

		setPositionGrabbers();
}



void GraphEditor_EllipseShape::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy){
	m_ItemGeometryChanged = true;
	const QRectF rect1 (rect() );
	for(int i = 0; i < 8; i++){
		if(m_cornerGrabber[i] == signalOwner){
			switch (i) {
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
}

QVariant GraphEditor_EllipseShape::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
	switch (change) {
	case QGraphicsItem::ItemSelectedHasChanged:
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
	}
	return QGraphicsEllipseItem::itemChange(change, value);
}

void GraphEditor_EllipseShape::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

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

	painter->setBrush(brush());
	painter->drawEllipse(rect());
}

void GraphEditor_EllipseShape::resizeLeft(const QPointF &pt){
	QRectF tmpRect = rect();

	if( pt.x() > tmpRect.right() )
		return;
	qreal widthOffset =  ( pt.x() - tmpRect.right() );
	// limit the minimum width
	if( widthOffset > -10 )
		return;
	// if it's negative we set it to a positive width value
	if( widthOffset < 0 )
		tmpRect.setWidth( -widthOffset );
	else
		tmpRect.setWidth( widthOffset );
	// Since it's a left side , the rectange will increase in size
	// but keeps the topLeft as it was
	tmpRect.translate( rect().width() - tmpRect.width() , 0 );
	prepareGeometryChange();
	// Set the ne geometry
	setRect( tmpRect );
	// Update to see the result
	update();

	setPositionGrabbers();
}

void GraphEditor_EllipseShape::resizeLeftBis(qreal dx) {
	QRectF tmpRect = rect();
	// if the mouse is on the right side we return
	qreal newWidth = tmpRect.width() - dx;
	if( newWidth <  11 )
		return;

	tmpRect.setWidth( newWidth );
	// Since it's a left side , the rectange will increase in size
	// but keeps the topLeft as it was
	tmpRect.translate( dx , 0 );
	prepareGeometryChange();
	// Set the ne geometry
	setRect( tmpRect );
	// Update to see the result
	update();

	setPositionGrabbers();
}

void GraphEditor_EllipseShape::resizeRight(const QPointF &pt){
	QRectF tmpRect = rect();
	if( pt.x() < tmpRect.left() )
		return;
	qreal widthOffset =  ( pt.x() - tmpRect.left() );
	if( widthOffset < 10 ) /// limit
		return;
	if( widthOffset < 10)
		tmpRect.setWidth( -widthOffset );
	else
		tmpRect.setWidth( widthOffset );
	prepareGeometryChange();
	setRect( tmpRect );
	update();
	setPositionGrabbers();
}

void GraphEditor_EllipseShape::resizeRightBis(qreal dx){
	QRectF tmpRect = rect();
	qreal newWidth =  tmpRect.width() + dx;
	if( newWidth < 10 ) /// limit
		return;

	tmpRect.setWidth( newWidth );
	prepareGeometryChange();
	setRect( tmpRect );
	update();
	setPositionGrabbers();
}

void GraphEditor_EllipseShape::resizeBottom(const QPointF &pt)
{
	QRectF tmpRect = rect();
	if( pt.y() < tmpRect.top() )
		return;
	qreal heightOffset =  ( pt.y() - tmpRect.top() );
	if( heightOffset < 11 ) /// limit
		return;
	if( heightOffset < 0)
		tmpRect.setHeight( -heightOffset );
	else
		tmpRect.setHeight( heightOffset );
	prepareGeometryChange();
	setRect( tmpRect );
	update();
	setPositionGrabbers();
}

void GraphEditor_EllipseShape::resizeBottomBis(qreal dy) {
	QRectF tmpRect = rect();
	qreal heightOffset = tmpRect.height() + dy;
	if( heightOffset < 11) /// limit
		return;
	else
		tmpRect.setHeight( heightOffset );
	prepareGeometryChange();
	setRect( tmpRect );
	update();
	setPositionGrabbers();
}

void GraphEditor_EllipseShape::resizeTop(const QPointF &pt){
	QRectF tmpRect = rect();
	if( pt.y() > tmpRect.bottom() )
		return;
	qreal heightOffset =  ( pt.y() - tmpRect.bottom() );
	if( heightOffset > -11 ) /// limit
		return;
	if( heightOffset < 0)
		tmpRect.setHeight( -heightOffset );
	else
		tmpRect.setHeight( heightOffset );
	tmpRect.translate( 0 , rect().height() - tmpRect.height() );
	prepareGeometryChange();
	setRect( tmpRect );
	update();
	setPositionGrabbers();
}

void GraphEditor_EllipseShape::resizeTopBis(qreal dy) {
	QRectF tmpRect = rect();
	qreal newHeight = tmpRect.height() - dy;
	if( newHeight < 11)
		return;

	tmpRect.setHeight( newHeight );
	tmpRect.translate( 0 , dy );
	prepareGeometryChange();
	setRect( tmpRect );
	update();
	setPositionGrabbers();
}

void GraphEditor_EllipseShape::createGrabbers()
{
	for(int i = 0; i < 8; i++){
		m_cornerGrabber[i] = new GraphEditor_GrabberItem(this);
		QObject::connect(m_cornerGrabber[i], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_EllipseShape::grabberMove);
	}
}

void GraphEditor_EllipseShape::setPositionGrabbers()
{
	QRectF tmpRect = rect();
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

void GraphEditor_EllipseShape::setGrabbersVisibility(bool visible)
{
	for(int i = 0; i < 8; i++){
		m_cornerGrabber[i]->setVisible(visible);
	}
}

GraphEditor_EllipseShape* GraphEditor_EllipseShape::clone()
{
	GraphEditor_EllipseShape* cloned = new GraphEditor_EllipseShape(rect(), pen(), brush(), m_Menu);
	cloned->setPos(scenePos());
	cloned->setRect(rect());
	cloned->setZValue(zValue());
	cloned->setRotation(rotation());
	cloned->setID(m_LayerID);
	cloned->setGrabbersVisibility(false);
	return cloned;
}

void GraphEditor_EllipseShape::ContextualMenu(QPoint Mousepos)
{
	setSelected(true);
	m_Menu->exec(Mousepos);
}

QVector<QPointF>  GraphEditor_EllipseShape::getKeyPoints()
{
	QVector<QPointF>  listePts;
//	qDebug()<<"getKeyPoints SceneCordinatesPoints :"<<shape().elementCount();

	int size =13;// shape().elementCount();
	 for (int idx = 0; idx < size; ++idx)
	 {
		 // DO NOT consider moveTo() elements
		 // we only need lineTo() elements
		 if (!shape().elementAt(idx).isMoveTo())
		 {
			 // push into the container
			 listePts.push_back(mapToScene(shape().elementAt(idx)));
		 }
	 }
//	qDebug()<<"getKeyPoints listePts :"<<listePts.count();
	return listePts;
}

QVector<QPointF>  GraphEditor_EllipseShape::SceneCordinatesPoints()
{
	return mapToScene(shape().toFillPolygon());
}

QVector<QPointF>  GraphEditor_EllipseShape::ImageCordinatesPoints()
{
	QVector<QPointF> vec;
	QPolygonF polygon_ = mapToScene(shape().toFillPolygon());
	polygon_.pop_back();
	GraphicSceneEditor *scene_ = dynamic_cast<GraphicSceneEditor*>(scene());
	if (scene_)
	{
		foreach(QPointF p, polygon_)
				{
			vec.push_back(scene_->innerView()->ConvertToImage(p));
				}
	}
	return vec;
}
