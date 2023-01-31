#include "GeEllipse.h"

#include <math.h>

#include <QPainter>
#include <QMenu>
#include <QDebug>
#include <QCursor>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsEllipseItem>
#include <QStyleOptionGraphicsItem>

#include "dotsignal.h"
#include "GeObjectId.h"
#include "GeGlobalParameters.h"

static double TwoPi = 2.0 * M_PI;

static qreal normalizeAngle(qreal angle)
{
    while (angle < 0)
        angle += TwoPi;
    while (angle > TwoPi)
        angle -= TwoPi;
    return angle;
}

/**
 *
 * pickedHere false means dash line
 */
GeEllipse::GeEllipse(/*data::MtData* vaData,*/ GeGlobalParameters& globalParameters, const GeObjectId& objectId,
		QColor& color, bool pickedHere, QObject *parent) :
    QObject(parent), GeShape( /*vaData,*/ globalParameters, objectId, color, pickedHere), m_cornerFlags(0), m_actionFlags(ResizeState)
{
    setAcceptHoverEvents(true);
    setFlags(ItemIsSelectable|ItemSendsGeometryChanges);
    for(int i = 0; i < 8; i++){
        cornerGrabber[i] = new DotSignal(this, globalParameters.getGrabberThickness());
        cornerGrabber[i]->setFlags(QGraphicsItem::ItemIgnoresTransformations);
        cornerGrabber[i]->setDotFlags(DotSignal::Movable);
        connect(cornerGrabber[i], &DotSignal::signalMove, this, &GeEllipse::grabberMove);
    }
    updateColor();
    setPositionGrabbers();
}

/**
 *
 * pickedHere false means dash line
 */
GeEllipse::GeEllipse(/*data::MtData* vaData,*/ GeGlobalParameters& globalParameters, const GeObjectId& objectId, QColor& color, bool pickedHere, QRectF& rect, QObject *parent) :
    QObject(parent), GeShape( /*vaData,*/ globalParameters, objectId, color, pickedHere),
    m_cornerFlags(0),  m_actionFlags(ResizeState)
{
    setAcceptHoverEvents(true);
    setFlags(ItemIsSelectable|ItemSendsGeometryChanges);
    for(int i = 0; i < 8; i++){
        cornerGrabber[i] = new DotSignal(this, globalParameters.getGrabberThickness());
        cornerGrabber[i]->setFlags(QGraphicsItem::ItemIgnoresTransformations);
        cornerGrabber[i]->setDotFlags(DotSignal::Movable);
        connect(cornerGrabber[i], &DotSignal::signalMove, this, &GeEllipse::grabberMove);
    }
	updateColor();
	setRect( rect );
    setPositionGrabbers();
}

GeEllipse::~GeEllipse()
{
    for(int i = 0; i < 8; i++){
        delete cornerGrabber[i];
    }
}

void GeEllipse::updateColor() {
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
	c1.setAlpha(a);
	QBrush brushR (c1, Qt::SolidPattern);
	this->setBrush(brushR);
	QBrush brush (m_color, Qt::SolidPattern);
	this->setColorDots( brush);
}

QPointF GeEllipse::previousPosition() const
{
    return m_previousPosition;
}

void GeEllipse::setPreviousPosition(const QPointF previousPosition)
{
    if (m_previousPosition == previousPosition)
        return;

    m_previousPosition = previousPosition;
    emit previousPositionChanged();
}

void GeEllipse::setRect(qreal x, qreal y, qreal w, qreal h)
{
    setRect(QRectF(x,y,w,h));
}

void GeEllipse::setRect(const QRectF &rect)
{
    QGraphicsEllipseItem::setRect(rect);

    setPositionGrabbers();
}

QRectF GeEllipse::getRect() {
	return rect();
}

void GeEllipse::mouseMoveEvent(QGraphicsSceneMouseEvent *event){
	QPointF pt = event->pos();

    qDebug() << "MOUSE MOVED " << pt;

    if(m_actionFlags == ResizeState){
        switch (m_cornerFlags) {
//        case Top:
//            resizeTop(pt);
//            break;
//        case Bottom:
//            resizeBottom(pt);
//            break;
//        case Left:
//            resizeLeft(pt);
//            break;
//        case Right:
//            resizeRight(pt);
//            break;
//        case TopLeft:
//            resizeTop(pt);
//            resizeLeft(pt);
//            break;
//        case TopRight:
//            resizeTop(pt);
//            resizeRight(pt);
//            break;
//        case BottomLeft:
//            resizeBottom(pt);
//            resizeLeft(pt);
//            break;
//        case BottomRight:
//            resizeBottom(pt);
//            resizeRight(pt);
//            break;
        default:
            if (m_leftMouseButtonPressed) {
                setCursor(Qt::ClosedHandCursor);
                auto dx = event->scenePos().x() - m_previousPosition.x();
                auto dy = event->scenePos().y() - m_previousPosition.y();
                moveBy(dx,dy);
                setPreviousPosition(event->scenePos());
                //emit signalMove(this, dx, dy);
            }
            break;
        }
    } else {
        switch (m_cornerFlags) {
        case TopLeft:
        case TopRight:
        case BottomLeft:
        case BottomRight: {
            rotateItem(pt);
            break;
        }
        default:
            if (m_leftMouseButtonPressed) {
                setCursor(Qt::ClosedHandCursor);
                auto dx = event->scenePos().x() - m_previousPosition.x();
                auto dy = event->scenePos().y() - m_previousPosition.y();
                moveBy(dx,dy);
                setPreviousPosition(event->scenePos());
                //emit signalMove(this, dx, dy);
            }
            break;
        }
    }
    QGraphicsItem::mouseMoveEvent(event);
}

void GeEllipse::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() & Qt::LeftButton) {
        m_leftMouseButtonPressed = true;
        setPreviousPosition(event->scenePos());
        emit clicked(this);
    }
    QGraphicsItem::mousePressEvent(event);
}

void GeEllipse::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() & Qt::LeftButton) {
        m_leftMouseButtonPressed = false;
    }

    notifyObservers();
    QGraphicsItem::mouseReleaseEvent(event);
}

void GeEllipse::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
    m_actionFlags = (m_actionFlags == ResizeState)?RotationState:ResizeState;
    setVisibilityGrabbers();
    QGraphicsItem::mouseDoubleClickEvent(event);
}

void GeEllipse::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setPositionGrabbers();
    setVisibilityGrabbers();
    setSelected(true);
    emit GeEllipse::itemSelected(m_objectId.getObjectId());
    QGraphicsItem::hoverEnterEvent(event);
}

void GeEllipse::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    m_cornerFlags = 0;
    //hideGrabbers();
    setCursor(Qt::CrossCursor);
    QGraphicsItem::hoverLeaveEvent( event );
}

void GeEllipse::hoverMoveEvent(QGraphicsSceneHoverEvent *event){
//    QPointF pt = event->pos();              // The current position of the mouse
//    qreal drx = pt.x() - rect().right();    // Distance between the mouse and the right
//    qreal dlx = pt.x() - rect().left();     // Distance between the mouse and the left
//
//    qreal dby = pt.y() - rect().top();      // Distance between the mouse and the top
//    qreal dty = pt.y() - rect().bottom();   // Distance between the mouse and the bottom
//
//    // If the mouse position is within a radius of 7
//    // to a certain side( top, left, bottom or right)
//    // we set the Flag in the Corner Flags Register
//
//    m_cornerFlags = 0;
//    if( dby < 14 && dby > -14 ) m_cornerFlags |= Top;       // Top side
//    if( dty < 14 && dty > -14 ) m_cornerFlags |= Bottom;    // Bottom side
//    if( drx < 14 && drx > -14 ) m_cornerFlags |= Right;     // Right side
//    if( dlx < 14 && dlx > -14 ) m_cornerFlags |= Left;      // Left side
//
//    if(m_actionFlags == ResizeState){
//        QPixmap p(":/icons/arrow-up-down.png");
//        QPixmap pResult;
//        QTransform trans = transform();
//        switch (m_cornerFlags) {
//        case Top:
//        case Bottom:
//            pResult = p.transformed(trans);
//            //setCursor(pResult.scaled(24,24,Qt::KeepAspectRatio));
//            setCursor(Qt::SizeVerCursor);
//            break;
//        case Left:
//        case Right:
//            trans.rotate(90);
//            pResult = p.transformed(trans);
//           // setCursor(pResult.scaled(24,24,Qt::KeepAspectRatio));
//            setCursor(Qt::SizeHorCursor);
//            break;
//        case TopRight:
//        case BottomLeft:
//            trans.rotate(45);
//            pResult = p.transformed(trans);
//            //setCursor(pResult.scaled(24,24,Qt::KeepAspectRatio));
//            setCursor(Qt::SizeBDiagCursor);
//            break;
//        case TopLeft:
//        case BottomRight:
//            trans.rotate(135);
//            pResult = p.transformed(trans);
//            //setCursor(pResult.scaled(24,24,Qt::KeepAspectRatio));
//            setCursor(Qt::SizeFDiagCursor);
//            break;
//        default:
//            setCursor(Qt::CrossCursor);
//            break;
//        }
//    } else {
//        switch (m_cornerFlags) {
//        case TopLeft:
//        case TopRight:
//        case BottomLeft:
//        case BottomRight: {
//            QPixmap p(":/icons/rotate-right.png");
//            setCursor(QCursor(p.scaled(24,24,Qt::KeepAspectRatio)));
//            break;
//        }
//        default:
//            setCursor(Qt::CrossCursor);
//            break;
//        }
//    }
    QGraphicsItem::hoverMoveEvent( event );
}

void GeEllipse::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy){
	 qDebug() << "--- grabberMove Point: " << dx << " /  " << dy;
	 const QRectF rect1 (rect() );
   for(int i = 0; i < 8; i++){
       if(cornerGrabber[i] == signalOwner){
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
   notifyObservers();
}

QVariant GeEllipse::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
    switch (change) {
    case QGraphicsItem::ItemSelectedChange: {
    	qDebug() << "ITEM CHANGE " << value.toBool();
        if(!value.toBool()) {
        	grabberOn = false;
            hideGrabbers();
        } else {
        	grabberOn = true;
            setVisibilityGrabbers();
        }
        m_actionFlags = ResizeState;
    	}
        break;
    case QGraphicsItem::ItemPositionChange: {
//    	QPointF newPos = value.toPointF();
//    	return newPos;
    }
    break;
    case QGraphicsItem::ItemPositionHasChanged: {
    	break;
   }
    default:
        break;
    }
    return QGraphicsEllipseItem::itemChange(change, value);
}

void GeEllipse::notifyObservers() {
	QPointF newPosShift = pos();
	QPointF initialPos = rect().topLeft();
	QPointF newTopLeft = QPointF( initialPos.x() + newPosShift.x(), initialPos.y() + newPosShift.y());
	QRectF rect1 (rect() );
	rect1.moveTo(newTopLeft);
	qDebug() << "ITEM CHANGED " << rect() << " NEW POS " << rect1;
	emit ellipseChanged(rect1, m_objectId, false);
}

void GeEllipse::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){
    QStyleOptionGraphicsItem myoption = (*option);
    myoption.state &= !QStyle::State_Selected;
    QGraphicsEllipseItem::paint(painter, &myoption, widget);


	QPointF p0InScene = widget->mapToGlobal(QPoint(0, 0));
	QPointF p1InScene = widget->mapToGlobal(QPoint(5, 5));
	double scalSizeInScene = p1InScene.x() - p0InScene.x();
	//QGraphicsItem::setFlags(QGraphicsItem::ItemIgnoresTransformations);

//    for(int i = 0; i < 8; i++){
//        QPointF testP = cornerGrabber[i]->pos();
//        int posX = testP.x() - scalSizeInScene/2, posY = testP.y() -scalSizeInScene/2 ;
//        painter->setBrush(grabberBrush);
//    	painter->drawEllipse( posX, posY, scalSizeInScene, scalSizeInScene);
//    }
}

void GeEllipse::setColorDots(const QBrush &brush){
    for(int i = 0; i < 8; i++){
        cornerGrabber[i]->setBrush(brush);
    }
}

void GeEllipse::resizeLeft(const QPointF &pt){
    QRectF tmpRect = rect();
    // if the mouse is on the right side we return
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

   // notifyObservers();
}

void GeEllipse::resizeLeftBis(qreal dx) {
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

void GeEllipse::resizeRight(const QPointF &pt){
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

    //notifyObservers();
}

void GeEllipse::resizeRightBis(qreal dx){
    QRectF tmpRect = rect();
    qreal newWidth =  tmpRect.width() + dx;
    if( newWidth < 10 ) /// limit
        return;

    tmpRect.setWidth( newWidth );
    prepareGeometryChange();
    setRect( tmpRect );
    update();
    setPositionGrabbers();

    //notifyObservers();
}

void GeEllipse::resizeBottom(const QPointF &pt)
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

    //notifyObservers();
}

void GeEllipse::resizeBottomBis(qreal dy) {
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

    //notifyObservers();
}

void GeEllipse::resizeTop(const QPointF &pt)
{
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

    //notifyObservers();
}

void GeEllipse::resizeTopBis(qreal dy) {
    QRectF tmpRect = rect();
//    if( tmp.y() > tmpRect.bottom() )
//        return;
//    if( heightOffset > -11 ) /// limit
//        return;
    qreal newHeight = tmpRect.height() - dy;
    if( newHeight < 11)
    	return;

    tmpRect.setHeight( newHeight );
    tmpRect.translate( 0 , dy );
    prepareGeometryChange();
    setRect( tmpRect );
    update();
    setPositionGrabbers();

    //notifyObservers();
}

void GeEllipse::rotateItem(const QPointF &pt)
{
    QRectF tmpRect = rect();
    QPointF center = boundingRect().center();
    QPointF corner;
    switch (m_cornerFlags) {
    case TopLeft:
        corner = tmpRect.topLeft();
        break;
    case TopRight:
        corner = tmpRect.topRight();
        break;
    case BottomLeft:
        corner = tmpRect.bottomLeft();
        break;
    case BottomRight:
        corner = tmpRect.bottomRight();
        break;
    default:
        break;
    }

    QLineF lineToTarget(center,corner);
    QLineF lineToCursor(center, pt);
    // Angle to Cursor and Corner Target points
    qreal angleToTarget = ::acos(lineToTarget.dx() / lineToTarget.length());
    qreal angleToCursor = ::acos(lineToCursor.dx() / lineToCursor.length());

    if (lineToTarget.dy() < 0)
        angleToTarget = TwoPi - angleToTarget;
    angleToTarget = normalizeAngle((M_PI - angleToTarget) + M_PI / 2);

    if (lineToCursor.dy() < 0)
        angleToCursor = TwoPi - angleToCursor;
    angleToCursor = normalizeAngle((M_PI - angleToCursor) + M_PI / 2);

    // Result difference angle between Corner Target point and Cursor Point
    auto resultAngle = angleToTarget - angleToCursor;

    QTransform trans = transform();
    trans.translate( center.x(), center.y());
    trans.rotateRadians(rotation() + resultAngle, Qt::ZAxis);
    trans.translate( -center.x(),  -center.y());
    setTransform(trans);
}

void GeEllipse::setPositionGrabbers()
{
    QRectF tmpRect = rect();
    cornerGrabber[GrabberTop]->setPos(tmpRect.left() + tmpRect.width()/2, tmpRect.top());
    cornerGrabber[GrabberBottom]->setPos(tmpRect.left() + tmpRect.width()/2, tmpRect.bottom());
    cornerGrabber[GrabberLeft]->setPos(tmpRect.left(), tmpRect.top() + tmpRect.height()/2);
    cornerGrabber[GrabberRight]->setPos(tmpRect.right(), tmpRect.top() + tmpRect.height()/2);
    cornerGrabber[GrabberTopLeft]->setPos(tmpRect.topLeft().x(), tmpRect.topLeft().y());
    cornerGrabber[GrabberTopRight]->setPos(tmpRect.topRight().x(), tmpRect.topRight().y());
    cornerGrabber[GrabberBottomLeft]->setPos(tmpRect.bottomLeft().x(), tmpRect.bottomLeft().y());
    cornerGrabber[GrabberBottomRight]->setPos(tmpRect.bottomRight().x(), tmpRect.bottomRight().y());

    for(int i = 0; i < 8; i++){
        cornerGrabber[i]->setFlags(QGraphicsItem::ItemIgnoresTransformations);
    }
}

void GeEllipse::setVisibilityGrabbers()
{
    cornerGrabber[GrabberTopLeft]->setVisible(true);
    cornerGrabber[GrabberTopRight]->setVisible(true);
    cornerGrabber[GrabberBottomLeft]->setVisible(true);
    cornerGrabber[GrabberBottomRight]->setVisible(true);

    if(m_actionFlags == ResizeState){
        cornerGrabber[GrabberTop]->setVisible(true);
        cornerGrabber[GrabberBottom]->setVisible(true);
        cornerGrabber[GrabberLeft]->setVisible(true);
        cornerGrabber[GrabberRight]->setVisible(true);
    } else {
        cornerGrabber[GrabberTop]->setVisible(false);
        cornerGrabber[GrabberBottom]->setVisible(false);
        cornerGrabber[GrabberLeft]->setVisible(false);
        cornerGrabber[GrabberRight]->setVisible(false);
    }
    qDebug() << "ELLIPSE VISIBLE " << cornerGrabber[GrabberTopLeft]->pos();

}

void GeEllipse::hideGrabbers()
{
    for(int i = 0; i < 8; i++){
        cornerGrabber[i]->setVisible(false);
    }
    qDebug() << "ELLIPSE HIDE ";
}
