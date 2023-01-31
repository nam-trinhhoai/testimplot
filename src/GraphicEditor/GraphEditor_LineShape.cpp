/*
 * GraphEditor_LineShape.cpp
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
#include <QGraphicsSimpleTextItem>

#include "GraphEditor_GrabberItem.h"
#include "GraphEditor_LineShape.h"
#include "GraphicSceneEditor.h"
#include "LineLengthText.h"
#include "singlesectionview.h"
#include "randomlineview.h"

GraphEditor_LineShape::GraphEditor_LineShape(QLineF lineItem, QPen pen, QMenu* itemMenu,
		GraphicSceneEditor* scene)
{
	m_scene = scene;
	//m_OrthogonalLineshape = nullptr;
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemSendsGeometryChanges|ItemIsMovable|ItemIsFocusable);
	m_Menu = itemMenu;
	setPen(pen);
	m_Pen=pen;
	m_textItem = new LineLengthText(this);
	m_textItem->setVisible(false);
	createGrabbers();
	setLine(lineItem);
	m_OrthoItem = nullptr;
	m_View = nullptr;

}

void GraphEditor_LineShape::createGrabbers()
{
	grabberList[0] = new GraphEditor_GrabberItem(this);
	QObject::connect(grabberList[0], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_LineShape::FirstPointMove);

	grabberList[1] = new GraphEditor_GrabberItem(this);
	QObject::connect(grabberList[1], &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_LineShape::LastPointMove);

}

GraphEditor_LineShape::~GraphEditor_LineShape()
{
	grabberList[0]->deleteLater();
	grabberList[1]->deleteLater();
}


void GraphEditor_LineShape::setRandomView(RandomLineView *pRandView)
{
	if(m_View != nullptr)
	{
		disconnect(m_View,SIGNAL(destroyed()),this,SLOT(resetRandomLineView()));
	}
	m_View = pRandView;
	if(m_View != nullptr)
	{
		connect(m_View,SIGNAL(destroyed()),this,SLOT(resetRandomLineView()));
	}
}

void GraphEditor_LineShape::resetRandomLineView()
{
	m_View = nullptr;
}

void GraphEditor_LineShape::calculateLineWidth()
{
	float lineAngle = line().angle();
	if (m_scene)
	{
		if ((m_scene->innerView()->viewType() == InlineView) ||
				(m_scene->innerView()->viewType() == XLineView) ||
				(m_scene->innerView()->viewType() == RandomView) )
		{
			lineAngle = 360 - lineAngle;
		}
	}
	QPointF textPos;

	QPointF p1 = line().p1();
	QPointF p2 = line().p2();
	if ( lineAngle > 90 && lineAngle < 260)
	{  // Right to left line
		lineAngle -= 180;
	}
	else
	{  // Left to right line
		textPos = line().center();
	}
	textPos = line().center();
	m_textItem->setPos(textPos);
	m_textItem->setRotation(lineAngle);
	m_textItem->setText(QString::number(line().length())+ " m");
}

void GraphEditor_LineShape::FirstPointMove(QGraphicsItem *signalOwner, qreal dx, qreal dy)
{
	QLineF lineItem = line();
	lineItem.setP1(QPointF(lineItem.p1().x() + dx, lineItem.p1().y() + dy));

	setLineFromMove(lineItem);
}

void GraphEditor_LineShape::LastPointMove(QGraphicsItem *signalOwner, qreal dx, qreal dy)
{
	QLineF lineItem = line();
	lineItem.setP2(QPointF(lineItem.p2().x() + dx, lineItem.p2().y() + dy));

	setLineFromMove(lineItem);
}

void GraphEditor_LineShape::updateGrabbersPosition()
{
	QPointF point = line().p1();
	grabberList[0]->setPos(point);

	point = line().p2();
	grabberList[1]->setPos(point);
}


void GraphEditor_LineShape::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsItem::mouseDoubleClickEvent(event);
}


GraphEditor_LineShape* GraphEditor_LineShape::clone()
{
	//qDebug()<<" GraphEditor_LineShape clone ";
	GraphEditor_LineShape* cloned = new GraphEditor_LineShape(line(), pen(), m_Menu,m_scene);
	cloned->setPos(scenePos());
	cloned->setZValue(zValue());
	cloned->setRotation(rotation());
	cloned->setID(m_LayerID);
	cloned->setGrabbersVisibility(false);

	if(getRandomView() != nullptr){
	    if(getRandomView()->getRandomType() == eTypeOrthogonal){
	        cloned->setRandomView(getRandomView());
	        connect(getRandomView(),SIGNAL(newPointPosition(QPointF)),cloned, SLOT(updateOrtholine(QPointF)));
	        connect(getRandomView(),SIGNAL(newWidthOrthogonal(double)),cloned,SLOT(updateOrthoWidthline(double)));

	        connect(getRandomView(), SIGNAL(updateOrthoFrom3D(QVector3D,QPointF )),cloned,SLOT(refreshOrtholine(QVector3D,QPointF)));
	        getRandomView()->addOrthogonalLine(cloned);
	        cloned->setOrthogonal(m_OrthoItem);
	    }
	}

	return cloned;
}

void GraphEditor_LineShape::setLine(const QLineF &lineItem) {
	QGraphicsLineItem::setLine(lineItem);
	updateGrabbersPosition();
	calculateLineWidth();
	//showGrabbers();
}

void GraphEditor_LineShape::setLineFromMove(const QLineF &lineItem) {
	m_ItemGeometryChanged=true;
	QGraphicsLineItem::setLine(lineItem);
	updateGrabbersPosition();
	calculateLineWidth();
}

void GraphEditor_LineShape::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (m_AlreadySelected && (event->button() != Qt::LeftButton))
		return;
	m_AlreadySelected = true;
	QGraphicsItem::mousePressEvent(event);
}

void GraphEditor_LineShape::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsItem::mouseReleaseEvent(event);
}

void GraphEditor_LineShape::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if( event->modifiers() & Qt::ShiftModifier )
	{
		QGraphicsItem::mouseMoveEvent(event);
	}
}

void GraphEditor_LineShape::hoverEnterEvent(QGraphicsSceneHoverEvent *event) {
	if (scene()->selectedItems().empty())
	{
		m_IsHighlighted = true;
		setSelected(true);
	}
	QGraphicsItem::hoverEnterEvent(event);
}

void GraphEditor_LineShape::hoverLeaveEvent(QGraphicsSceneHoverEvent *event) {
	m_IsHighlighted = false;
	if (!m_AlreadySelected)
		setSelected(false);
	QGraphicsItem::hoverLeaveEvent(event);
}

void GraphEditor_LineShape::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

	painter->setRenderHint(QPainter::Antialiasing,true);
//	painter->setRenderHint(QPainter::HighQualityAntialiasing, true);
	painter->setRenderHint(QPainter::SmoothPixmapTransform, true);

	//calculateLineWidth();

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

	painter->drawLine(line());
}

void GraphEditor_LineShape::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {
	//	m_ItemGeometryChanged=true;
	//
	//	QLineF lineItem = line();
	//	if(grabberList.at(0) == signalOwner)
	//	{
	//		lineItem.setP1(QPointF(lineItem.p1().x() + dx, lineItem.p1().y() + dy));
	//	}
	//	else
	//	{
	//		lineItem.setP2(QPointF(lineItem.p2().x() + dx, lineItem.p2().y() + dy));
	//	}
	//
	//	setLineFromMove(lineItem);
}

void GraphEditor_LineShape::setGrabbersVisibility(bool visible) {
	grabberList[0]->setVisible(visible);
	grabberList[1]->setVisible(visible);
}

QVariant GraphEditor_LineShape::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
	switch (change) {
	case QGraphicsItem::ItemSelectedChange:
	{
		if(!value.toBool()) {
			setGrabbersVisibility(false);
			m_AlreadySelected=false;
			m_textItem->setVisible(false);
		} else {
			setGrabbersVisibility(true);
			m_textItem->setVisible(true);
		}
		break;
	}
	case QGraphicsItem::ItemPositionHasChanged:
	{
		//emit moved();
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


void GraphEditor_LineShape::ContextualMenu(QPoint Mousepos)
{
	setSelected(true);
	m_Menu->exec(Mousepos);
}

QVector<QPointF>  GraphEditor_LineShape::SceneCordinatesPoints()
{
	QPolygonF points;
	points.push_back(line().p1());
	points.push_back(line().p2());
	return mapToScene(points);
}

QVector<QPointF>  GraphEditor_LineShape::ImageCordinatesPoints()
{
	QVector<QPointF> vec;
	QPolygonF points;
	points.push_back(line().p1());
	points.push_back(line().p2());
	QPolygonF polygon_ = mapToScene(points);
	GraphicSceneEditor *scene_ = dynamic_cast<GraphicSceneEditor*>(scene());
	if (scene_)
	{
		foreach(QPointF p, polygon_){
			vec.push_back(scene_->innerView()->ConvertToImage(p));
		}
	}
	return vec;
}

void GraphEditor_LineShape::setOrthogonal(GraphEditor_ItemInfo* pItem){
	m_OrthoItem = pItem;
}

GraphEditor_ItemInfo *GraphEditor_LineShape::getOrthogonal(){
	return m_OrthoItem ;
}

void GraphEditor_LineShape::updateOrthoWidthline(double iLength){


    QVector<QPointF> vectPointScene =  this->SceneCordinatesPoints();
    QLineF line = QLineF(vectPointScene[0],vectPointScene[1]);

    QPointF pt = line.center();

    QLineF half1 = QLineF(pt,line.p1());
    QLineF half2 = QLineF(pt,line.p2());
    half1.setLength(iLength/2);
    half2.setLength(iLength/2);
    QLineF newLine = QLineF(half1.p2(),half2.p2());
    this->setLine(newLine);
    QPolygonF polygone = this->SceneCordinatesPoints();
    emit orthogonalUpdated(polygone);
}

/*! updateOrtholine
 * @brief : updateOrtholine
 *
 */

void GraphEditor_LineShape::refreshOrtholine(QVector3D n,QPointF pos)
{

	if(m_OrthoItem != nullptr)
	{

		n = n.normalized();
		//qDebug()<<" refreshOrtholine :" <<n;

		QVector3D up(0.0f,-1.0f,0.0f);
		QVector3D right = QVector3D::crossProduct(n,up);
		right = right.normalized();

		QVector<QPointF> vectPointScene =  this->SceneCordinatesPoints();
		QLineF line = QLineF(vectPointScene[0],vectPointScene[1]);
		qreal length = line.length();

		//QPointF posC = (vectPointScene[0]+vectPointScene[1])*0.5;

		QPointF right3d(right.x(),right.z());

		QPointF pos1 = pos - right3d* length*0.5f;
		QPointF pos2 = pos + right3d* length*0.5f;

		QVector<QPointF> listepts;
		listepts.push_back(pos1);
		listepts.push_back(pos2);
		QLineF lineortho(pos1,pos2);




		this->setLine(lineortho);
	//	QPolygonF polygone = this->SceneCordinatesPoints();

		emit orthogonalUpdated(listepts);


	}
}

void GraphEditor_LineShape::updateOrtholine(QPointF point){

	return;


    if(first == true && m_OrthoItem != nullptr){

    	//m_pts = point;
    	float eps = 1.5f;//0.005
        QVector<QPointF> directrice = m_OrthoItem->SceneCordinatesPoints();

        QVector<QPointF> vectPointScene =  this->SceneCordinatesPoints();
        QLineF line = QLineF(vectPointScene[0],vectPointScene[1]);
        qreal length = line.length();

        int index = 0;
        qreal minDistance = 10000 ;
        for(int i = 0;i < directrice.size();i++)
        {
            if(i <= directrice.size()- 2)
            {
                if(directrice[i] != directrice[i+1]){
                    QLineF smalLine = QLineF(directrice[i],directrice[i+1]);
                    QPointF intersectionPoint;
                    qreal line1Length = smalLine.length();
                    qreal line2Length = QLineF(point,directrice[i]).length();
                    qreal line3Length = QLineF(point,directrice[i+1]).length();

                    qreal tmpDistance = line2Length +line3Length-line1Length;
                    if(tmpDistance < eps && tmpDistance>=0){
                        //qDebug() << "index " <<  i << " bc " << line1Length <<  "ba " <<line2Length << " ab " <<line3Length;
                        if (tmpDistance < minDistance){
                            minDistance = tmpDistance;
                            index = i;
                        }
                    }
                }
            }
        }
        if(minDistance < eps){
          //  qDebug() << minDistance << index;
            QLineF LineHalf1Line = QLineF(point,directrice[index]);
            LineHalf1Line = LineHalf1Line.normalVector();
            LineHalf1Line.setLength(length/2);

            QLineF LineHalf2Line = LineHalf1Line;
            LineHalf2Line.setAngle(LineHalf1Line.angle()+ 180);
            LineHalf2Line.setLength(length/2);

            QLineF newLine = QLineF(LineHalf1Line.p2(),LineHalf2Line.p2());
            this->setLine(newLine);
            QPolygonF polygone = this->SceneCordinatesPoints();
            emit orthogonalUpdated(polygone);

            first = false;
        }

    }
}
