/*
 * GraphEditor_RectShape.cpp
 *
 *  Created on: Oct 1, 2021
 *      Author: l1046262
 */

#include <QObject>
#include <QPainter>
#include <QDebug>
#include <QGraphicsSceneMouseEvent>
#include <QStyleOptionGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>

#include "GraphEditor_GrabberItem.h"
#include "GraphEditor_RectShape.h"
#include "GraphicSceneEditor.h"
#include "LineLengthText.h"

GraphEditor_RectShape::GraphEditor_RectShape(QRectF rect, QPen pen, QBrush brush, QMenu* itemMenu, bool is_Rounded) : m_isRounded(is_Rounded)
{
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	setFlag(QGraphicsItem::ItemSendsScenePositionChanges);
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
	setToolTip("rectangle");
}

GraphEditor_RectShape::~GraphEditor_RectShape(){
	for(int i = 0; i < 8; i++){
		delete m_cornerGrabber[i];
	}
}

void GraphEditor_RectShape::setRect(const QRectF &rect)
{
	QGraphicsRectItem::setRect(rect);
	setPositionGrabbers();
	calculateArea();
}


QRectF GraphEditor_RectShape::rect() const
{
	return QGraphicsRectItem::rect();
}



void GraphEditor_RectShape::calculateArea()
{
	qreal area = rect().width()*rect().height();

	QRectF bbox = boundingRect().normalized();
	QPointF center = bbox.center();

	m_textItem->setText(QString::number(area/1000000) + " km²");
	m_textItem->setPos(center);
}


void GraphEditor_RectShape::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (m_AlreadySelected && (event->button() != Qt::LeftButton))
		return;
	m_AlreadySelected = true;
	QGraphicsItem::mousePressEvent(event);
}

void GraphEditor_RectShape::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsItem::mouseReleaseEvent(event);
}

void GraphEditor_RectShape::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if( event->modifiers() & Qt::ShiftModifier )
	{
		QGraphicsItem::mouseMoveEvent(event);
	}
}

void GraphEditor_RectShape::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
	if (scene()->selectedItems().empty())
	{
		m_IsHighlighted = true;
		setSelected(true);
	}
	QGraphicsItem::hoverEnterEvent(event);
}

void GraphEditor_RectShape::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
	m_IsHighlighted = false;
	if (!m_AlreadySelected)
		setSelected(false);
	QGraphicsItem::hoverLeaveEvent( event );
}

void GraphEditor_RectShape::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy){
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

QVariant GraphEditor_RectShape::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
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
	return QGraphicsRectItem::itemChange(change, value);
}

void GraphEditor_RectShape::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

	painter->setRenderHint(QPainter::Antialiasing,true);
	//painter->setRenderHint(QPainter::HighQualityAntialiasing, true);
	painter->setRenderHint(QPainter::SmoothPixmapTransform, true);

	QBrush brsh = brush();
	brsh.setTransform(QTransform((scene()->views())[0]->transform().inverted()));
	this->setBrush(brsh);

	painter->setBrush(brush());

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

	if (m_isRounded)
	{
		painter->drawRoundedRect(this->rect(),10,10);
	}
	else
	{
		painter->drawRect(this->rect());
	}
}

void GraphEditor_RectShape::resizeLeft(const QPointF &pt){
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

void GraphEditor_RectShape::resizeLeftBis(qreal dx) {
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

void GraphEditor_RectShape::resizeRight(const QPointF &pt){
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

void GraphEditor_RectShape::resizeRightBis(qreal dx){
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

void GraphEditor_RectShape::resizeBottom(const QPointF &pt)
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

void GraphEditor_RectShape::resizeBottomBis(qreal dy) {
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

void GraphEditor_RectShape::resizeTop(const QPointF &pt){
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

void GraphEditor_RectShape::resizeTopBis(qreal dy) {
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

void GraphEditor_RectShape::createGrabbers()
{
	for(int i = 0; i < 8; i++){
		m_cornerGrabber[i] = new GraphEditor_GrabberItem(this);
		QObject::connect(m_cornerGrabber[i], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_RectShape::grabberMove);
	}
}

void GraphEditor_RectShape::setPositionGrabbers()
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

void GraphEditor_RectShape::setGrabbersVisibility(bool visible)
{
	for(int i = 0; i < 8; i++)
	{
		m_cornerGrabber[i]->setVisible(visible);
	}
}

GraphEditor_RectShape* GraphEditor_RectShape::clone()
{
	GraphEditor_RectShape* cloned = new GraphEditor_RectShape(rect(), pen(), brush(), m_Menu, m_isRounded);
	cloned->setPos(scenePos());
	cloned->setRect(rect());
	cloned->setZValue(zValue());
	cloned->setRotation(rotation());
	cloned->setID(m_LayerID);
	cloned->setGrabbersVisibility(false);
	return cloned;
}


void GraphEditor_RectShape::ContextualMenu(QPoint Mousepos)
{
	setSelected(true);
	m_Menu->exec(Mousepos);
}

QVector<QPointF>  GraphEditor_RectShape::SceneCordinatesPoints()
{
	QPolygonF poly = mapToScene(rect());
	//poly.pop_back();
	return poly;
}

QVector<QPointF>  GraphEditor_RectShape::ImageCordinatesPoints()
{
	QVector<QPointF> vec;
	QPolygonF polygon_ = mapToScene(rect());
//	polygon_.pop_back();
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
