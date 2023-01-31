#include "GePolygon.h"

#include <QGraphicsSceneMouseEvent>
#include <QPainterPath>
#include <QGraphicsScene>
#include <QGraphicsPathItem>
#include <QDebug>
#include "dotsignal.h"
#include <math.h>
#include "GeGlobalParameters.h"

GePolygon::GePolygon(/*data::MtData* vaData,*/ GeGlobalParameters& globalParameters, const GeObjectId& objectId, QColor& color, bool pickedHere, QObject *parent) :
QObject(parent), GeShape( /*vaData,*/ globalParameters, objectId, color, pickedHere)
{
    setAcceptHoverEvents(true);
    setFlags(ItemIsSelectable|ItemSendsGeometryChanges);

	updateColor();

    QGraphicsSimpleTextItem *textItem = new QGraphicsSimpleTextItem("A FAIRE", this);
    textItem->setVisible(true);
}

GePolygon::~GePolygon() {

}

void GePolygon::setColor(QColor color) {
	m_color = color;
	updateColor();
}

void GePolygon::updateColor() {
    QPen p;
    m_color.setAlpha(255);
    p.setColor(m_color);
    int a;
    if (m_pickedHere) {
    	p.setStyle(Qt::SolidLine);
    	a = 127;
    }
    else {
    	p.setStyle(Qt::DashDotLine);
    	a = 40;
    }
    p.setCosmetic(true);
    this->setPen(p);
    QColor c1 = m_color;
	c1.setAlpha(m_alphaForFill);
	QBrush brushR (c1, Qt::SolidPattern);
	this->setBrush(brushR);
	QBrush brush (m_color, Qt::SolidPattern);
	this->setGrabbersColor( brush);
}

QPointF GePolygon::previousPosition() const {
    return m_previousPosition;
}

void GePolygon::setPreviousPosition(const QPointF previousPosition) {
    if (m_previousPosition == previousPosition)
        return;

    m_previousPosition = previousPosition;
    emit previousPositionChanged();
}

void GePolygon::setPolygon(const QPolygonF &polygon) {
	if (polygon.isEmpty())
		return;
    QGraphicsPolygonItem::setPolygon(polygon);
	clearGrabbers();
	showGrabbers();
}

void GePolygon::setPolygonFromMove(const QPolygonF &polygon) {
	qDebug() << "setPolygonFromMove AVANT SET POLYGON SIZE:" << polygon.size();
	qDebug() << "setPolygonFromMove AVANT SET POLYGON " << polygon;
    QGraphicsPolygonItem::setPolygon(polygon);
	qDebug() << "setPolygonFromMove APRES SET POLYGON SIZE:" << polygon.size();
	qDebug() << "setPolygonFromMove APRES SET POLYGON " << polygon;
}

void GePolygon::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() & Qt::LeftButton) {
        m_leftMouseButtonPressed = true;
        setPreviousPosition(event->scenePos());
        emit clicked(this);
    }
    QGraphicsItem::mousePressEvent(event);
}

void GePolygon::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    if (m_leftMouseButtonPressed) {
        auto dx = event->scenePos().x() - m_previousPosition.x();
        auto dy = event->scenePos().y() - m_previousPosition.y();
//        moveBy(dx,dy); // should not be use because polygon is unaffected by a change of the item position
        QPolygonF polyOrigin = polygon();
        QPolygonF newPolygon;
        for (QPointF pt : polyOrigin) {
        	newPolygon << QPointF(pt.x()+dx, pt.y()+dy);
        }
        setPolygon(newPolygon);
        setPreviousPosition(event->scenePos());
        emit signalMove(this, dx, dy);
    }
    QGraphicsItem::mouseMoveEvent(event);
}

void GePolygon::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() & Qt::LeftButton) {
        m_leftMouseButtonPressed = false;
    }
    notifyObservers();
    QGraphicsItem::mouseReleaseEvent(event);
}

void GePolygon::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
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
		updateGrabbers();

		emit polygonPointAddedSignal(m_objectId.getObjectId(), indexAdded, factor);
	}
	QGraphicsItem::mouseDoubleClickEvent(event);
}


void GePolygon::hoverEnterEvent(QGraphicsSceneHoverEvent *event) {

    updateGrabbers();
    setSelected(true);
    emit GePolygon::itemSelected(m_objectId.getObjectId());
    QGraphicsItem::hoverEnterEvent(event);
}

void GePolygon::hoverMoveEvent(QGraphicsSceneHoverEvent *event) {
    QGraphicsItem::hoverMoveEvent(event);
}

void GePolygon::hoverLeaveEvent(QGraphicsSceneHoverEvent *event) {
    /*if(!listDotes.isEmpty()){
        foreach (DotSignal *dot, listDotes) {
            dot->deleteLater();
        }
        listDotes.clear();
    }*/
    QGraphicsItem::hoverLeaveEvent(event);
}

void GePolygon::notifyObservers() {
	QPointF newPosShift = pos();
	qDebug() << "ITEM CHANGED " << polygon() << "New Pos Shift " << newPosShift;
	QPolygonF newPoly( polygon() );
	newPoly.translate(newPosShift);
	qDebug() << "AFTER TRANSLATE " << newPoly;
	emit polygonChanged(newPoly, m_objectId, false, this);
}

void GePolygon::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = polygon();
    for(int i = 0; i < polygonPath.size(); i++){
        if(grabberList.at(i) == signalOwner){
            QPointF pathPoint = polygonPath.at(i);
            polygonPath.replace(i, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
            m_pointForCheck = i;
        }
    }
    setPolygonFromMove(polygonPath);
    notifyObservers();
}

void GePolygon::slotDeleted(QGraphicsItem *signalOwner) {
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = polygon();
	QPolygonF newPolygonPath;
	bool found = false;
	int indexDeleted;
    for(int i = 0; i < polygonPath.size(); i++){
        if(grabberList.at(i) == signalOwner){
        	found = true;
        	indexDeleted = i;
        	DotSignal* dot = grabberList.at(i);
        	grabberList.removeAt(i);
        	dot->deleteLater();
        	break;
        } else {
        	newPolygonPath << polygonPath[i];
        }
    }
    if ( found) {
    	setPolygon(newPolygonPath);
    	emit polygonPointDeletedSignal(m_objectId.getObjectId(), indexDeleted);
    }
}

void GePolygon::clearGrabbers() {
    if(!grabberList.isEmpty()){
        foreach (DotSignal *dot, grabberList) {
            dot->deleteLater();
        }
        grabberList.clear();
    }
}

void GePolygon::showGrabbers() {
    QBrush dotsBrush (m_color, Qt::SolidPattern);
    QPolygonF polygonPath = polygon();
    for(int i = 0; i < polygonPath.size(); i++){
        QPointF point = polygonPath.at(i);

        DotSignal *dot = new DotSignal( this, m_globalParameters.getGrabberThickness());

        connect(dot, &DotSignal::signalMove, this, &GePolygon::grabberMove);
        connect(dot, &DotSignal::signalDoubleClick, this, &GePolygon::slotDeleted);

        dot->setDotFlags(DotSignal::Movable);
        dot->setFlags(QGraphicsItem::ItemIgnoresTransformations);
        dot->setVisible(true);
        dot->setBrush(dotsBrush);
        grabberList.append(dot);
        dot->setPos(point);
    }
}

void GePolygon::setGrabbersVisibility(bool visible) {
    if(!grabberList.isEmpty()){
        foreach (DotSignal *grabber, grabberList) {
            grabber->setVisible(visible);
        }
    }
}

void GePolygon::updateGrabbers(){
    //clearGrabbers();
    //displayDots();
	setGrabbersVisibility(true);
}

void GePolygon::setGrabbersColor(const QBrush &brush) {
    for(int i = 0; i < grabberList.size(); i++){
    	grabberList[i]->setBrush(brush);
    }
}

QVariant GePolygon::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
    switch (change) {
    case QGraphicsItem::ItemSelectedChange:
    	if(!value.toBool()) {
    	   // clearGrabbers();
    		setGrabbersVisibility(false);
    	} else {
    	    //displayDots();
    		setGrabbersVisibility(true);
    	}
    	//notifyObservers();
        break;
    case QGraphicsItem::ItemSelectedHasChanged:
        break;
    case QGraphicsItem::ItemPositionChange:
        break;
    default:
        break;
    }
    return QGraphicsItem::itemChange(change, value);
}
