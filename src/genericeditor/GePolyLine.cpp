#include "GePolyLine.h"

#include <QGraphicsSceneMouseEvent>
#include <QPainterPath>
#include <QGraphicsScene>
#include <QGraphicsPathItem>
#include <QDebug>
#include "dotsignal.h"
#include <math.h>
#include "GeGlobalParameters.h"

GePolyLine::GePolyLine(/*data::MtData* vaData,*/ GeGlobalParameters& globalParameters, const GeObjectId& objectId, QColor& color, bool pickedHere, QObject *parent) :
QObject(parent), GeShape( /*vaData,*/ globalParameters, objectId, color, pickedHere)
{
    setAcceptHoverEvents(true);
    setFlags(ItemIsSelectable|ItemSendsGeometryChanges);

	updateColor();

    QGraphicsSimpleTextItem *textItem = new QGraphicsSimpleTextItem("A FAIRE", this);
    textItem->setVisible(true);
}

GePolyLine::~GePolyLine() {

}

void GePolyLine::setColor(QColor color) {
	m_color = color;
	updateColor();
}

void GePolyLine::updateColor() {
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
//    QColor c1 = m_color;
//	c1.setAlpha(m_alphaForFill);
//	QBrush brushR (c1, Qt::SolidPattern);
//	this->setBrush(brushR);
	QBrush brush (m_color, Qt::SolidPattern);
	this->setGrabbersColor( brush);
}

QPointF GePolyLine::previousPosition() const {
    return m_previousPosition;
}

void GePolyLine::setPreviousPosition(const QPointF previousPosition) {
    if (m_previousPosition == previousPosition)
        return;

    m_previousPosition = previousPosition;
    emit previousPositionChanged();
}

const QPolygonF& GePolyLine::polygon() const {
	return m_polygon;
}

void GePolyLine::setPolygon(const QPolygonF &polygon) {
	if (polygon.isEmpty())
		return;
    m_polygon = polygon;
    QPainterPath newPath;
    newPath.addPolygon(m_polygon);
    setPath(newPath);
	clearGrabbers();
	showGrabbers();
}

void GePolyLine::setPolygonFromMove(const QPolygonF &polygon) {
	qDebug() << "setPolygonFromMove AVANT SET POLYGON SIZE:" << polygon.size();
	qDebug() << "setPolygonFromMove AVANT SET POLYGON " << polygon;
//    QGraphicsPolygonItem::setPolygon(polygon);
    m_polygon = polygon;
    QPainterPath newPath;
    newPath.addPolygon(m_polygon);
    setPath(newPath);
	qDebug() << "setPolygonFromMove APRES SET POLYGON SIZE:" << polygon.size();
	qDebug() << "setPolygonFromMove APRES SET POLYGON " << polygon;
}

void GePolyLine::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() & Qt::LeftButton) {
        m_leftMouseButtonPressed = true;
        setPreviousPosition(event->scenePos());
        emit clicked(this);
    }
    QGraphicsItem::mousePressEvent(event);
}

void GePolyLine::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    if (m_leftMouseButtonPressed) {
        auto dx = event->scenePos().x() - m_previousPosition.x();
        auto dy = event->scenePos().y() - m_previousPosition.y();
//        moveBy(dx,dy); // should not be use because polygon is unaffected by a change of the item position
        QPolygonF polyOrigin = m_polygon;
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

void GePolyLine::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() & Qt::LeftButton) {
        m_leftMouseButtonPressed = false;
    }
    notifyObservers();
    QGraphicsItem::mouseReleaseEvent(event);
}

void GePolyLine::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
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
		updateGrabbers();

		emit polygonPointAddedSignal(m_objectId.getObjectId(), indexAdded, factor);
	}
	QGraphicsItem::mouseDoubleClickEvent(event);
}


void GePolyLine::hoverEnterEvent(QGraphicsSceneHoverEvent *event) {

    updateGrabbers();
    setSelected(true);
    emit GePolyLine::itemSelected(m_objectId.getObjectId());
    QGraphicsItem::hoverEnterEvent(event);
}

void GePolyLine::hoverMoveEvent(QGraphicsSceneHoverEvent *event) {
    QGraphicsItem::hoverMoveEvent(event);
}

void GePolyLine::hoverLeaveEvent(QGraphicsSceneHoverEvent *event) {
    /*if(!listDotes.isEmpty()){
        foreach (DotSignal *dot, listDotes) {
            dot->deleteLater();
        }
        listDotes.clear();
    }*/
    QGraphicsItem::hoverLeaveEvent(event);
}

void GePolyLine::notifyObservers() {
	QPointF newPosShift = pos();
	qDebug() << "ITEM CHANGED " << m_polygon << "New Pos Shift " << newPosShift;
	QPolygonF newPoly( m_polygon );
	newPoly.translate(newPosShift);
	qDebug() << "AFTER TRANSLATE " << newPoly;
	emit polygonChanged(newPoly, m_objectId, false, this);
}

void GePolyLine::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = m_polygon;
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

void GePolyLine::slotDeleted(QGraphicsItem *signalOwner) {
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = m_polygon;
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

void GePolyLine::clearGrabbers() {
    if(!grabberList.isEmpty()){
        foreach (DotSignal *dot, grabberList) {
            dot->deleteLater();
        }
        grabberList.clear();
    }
}

void GePolyLine::showGrabbers() {
    QBrush dotsBrush (m_color, Qt::SolidPattern);
    QPolygonF polygonPath = m_polygon;
    for(int i = 0; i < polygonPath.size(); i++){
        QPointF point = polygonPath.at(i);

        DotSignal *dot = new DotSignal( this, m_globalParameters.getGrabberThickness());

        connect(dot, &DotSignal::signalMove, this, &GePolyLine::grabberMove);
        connect(dot, &DotSignal::signalDoubleClick, this, &GePolyLine::slotDeleted);

        dot->setDotFlags(DotSignal::Movable);
        dot->setFlags(QGraphicsItem::ItemIgnoresTransformations);
        dot->setVisible(true);
        dot->setBrush(dotsBrush);
        grabberList.append(dot);
        dot->setPos(point);
    }
}

void GePolyLine::setGrabbersVisibility(bool visible) {
    if(!grabberList.isEmpty()){
        foreach (DotSignal *grabber, grabberList) {
            grabber->setVisible(visible);
        }
    }
}

void GePolyLine::updateGrabbers(){
    //clearGrabbers();
    //displayDots();
	setGrabbersVisibility(true);
}

void GePolyLine::setGrabbersColor(const QBrush &brush) {
    for(int i = 0; i < grabberList.size(); i++){
    	grabberList[i]->setBrush(brush);
    }
}

QVariant GePolyLine::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
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

void GePolyLine::fillContextMenu(QPointF scenePos, QContextMenuEvent::Reason reason, QMenu& mainMenu) {
	emit contextMenuSignal(scenePos, reason, mainMenu);
}

