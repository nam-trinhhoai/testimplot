#include "GeRectangle.h"

#include <QPainter>
#include <QMenu>
#include <QDebug>
#include <QCursor>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsRectItem>
#include <QStyleOptionGraphicsItem>
#include <math.h>
#include "dotsignal.h"
#include "GeObjectId.h"
#include "GeGlobalParameters.h"
//#include "data/MtVaData.h"

static const double Pi = 3.14159265358979323846264338327950288419717;
static double TwoPi = 2.0 * Pi;

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
GeRectangle::GeRectangle(/*data::MtData* data,*/ GeGlobalParameters& globalParameters,
		const QString objectName,
		const GeObjectId& objectId, QColor& color, bool pickedHere, QObject *parent) :
    QObject(parent), GeShape( /*data,*/ globalParameters, objectId, color, pickedHere),
    m_cornerFlags(0), m_actionFlags(ResizeState), objectName(objectName)
{
    //setAcceptHoverEvents(true);
    //setFlags(ItemIsSelectable|ItemSendsGeometryChanges);
    for(int i = 0; i < 8; i++){
        m_cornerGrabber[i] = new DotSignal(this, m_globalParameters.getGrabberThickness());
        m_cornerGrabber[i]->setFlags(QGraphicsItem::ItemIgnoresTransformations);
        m_cornerGrabber[i]->setDotFlags(DotSignal::Movable);
        connect(m_cornerGrabber[i], &DotSignal::signalMove, this, &GeRectangle::grabberMove);
    }
    updateColor();
    setPositionGrabbers();
    hideGrabbers();

    textItem = new QGraphicsSimpleTextItem(objectName);
    textItem->setVisible(globalParameters.isDisplayText());
}

/**
 *
 * pickedHere false means dash line
 */
GeRectangle::GeRectangle(/*data::MtData* data,*/ GeGlobalParameters& globalParameters,
		const QString objectName, const GeObjectId& objectId, QColor& color,
		bool pickedHere, QRectF& rect, QObject *parent) :
    QObject(parent), GeShape( /*data,*/ globalParameters, objectId, color, pickedHere),
    m_cornerFlags(0), m_actionFlags(ResizeState), objectName(objectName)
{
    setAcceptHoverEvents(true);
    setFlags(ItemIsSelectable|ItemSendsGeometryChanges);
    for(int i = 0; i < 8; i++){
        m_cornerGrabber[i] = new DotSignal(this, globalParameters.getGrabberThickness());
        m_cornerGrabber[i]->setFlags(QGraphicsItem::ItemIgnoresTransformations);
        m_cornerGrabber[i]->setDotFlags(DotSignal::Movable);
        connect(m_cornerGrabber[i], &DotSignal::signalMove, this, &GeRectangle::grabberMove);
    }
	updateColor();
	setRect( rect );
    setPositionGrabbers();
    //TODO ???????????????hideGrabbers();

    textItem = new QGraphicsSimpleTextItem(objectName);

    textItem->setVisible(false/*globalParameters.isDisplayText()*/);
    textItem->setPos(rect.x()+rect.width(), rect.y());
    //textItem->setFont(globalParameters.getFont());
    textItem->setFlag(QGraphicsTextItem::ItemIgnoresTransformations, true);
    QString text1 = objectName;
    text1.append( "\n Size: " + QString::number( rect.width()) + "/" + QString::number( rect.height()));
    textItem->setText(text1);
}

GeRectangle::~GeRectangle(){
    for(int i = 0; i < 8; i++){
        delete m_cornerGrabber[i];
    }
}

void GeRectangle::updateColor() {
    QPen p;
    this->m_color.setAlpha(255);
    p.setColor(this->m_color);
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

QPointF GeRectangle::previousPosition() const
{
    return m_previousPosition;
}

void GeRectangle::setPreviousPosition(const QPointF previousPosition)
{
    if (m_previousPosition == previousPosition)
        return;

    m_previousPosition = previousPosition;
    emit previousPositionChanged();
}

void GeRectangle::setRect(qreal x, qreal y, qreal w, qreal h)
{
    setRect(QRectF(x,y,w,h));

    textItem->setPos(x+w, y);
    textItem->setFlag(QGraphicsTextItem::ItemIgnoresTransformations, true);
    //textItem->setFont(globalParameters.getFont());
    QString text1 = objectName;
    text1.append( "\n Size: "+ QString::number( w ) + "/" + QString::number( h ));
    textItem->setText(text1);
    //textItem->setVisible(globalParameters.isDisplayText());
}

void GeRectangle::setRect(const QRectF &rect)
{
    QGraphicsRectItem::setRect(rect);

    setPositionGrabbers();
}

QRectF GeRectangle::getRect() {
	return rect();
}

//void GeRectangle::mouseMoved(double worldX,double worldY,Qt::MouseButton button,
//		Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info){
////	QPointF pt(worldX, worldY);
////
////    qDebug() << "MOUSE MOVED PICKING " << pt << " Action: " << m_actionFlags << " Corner: " << m_cornerFlags;
////
////    if(m_actionFlags == ResizeState){
////        switch (m_cornerFlags) {
//////        case Top:
//////            resizeTop(pt);
//////            break;
//////        case Bottom:
//////            resizeBottom(pt);
//////            break;
//////        case Left:
//////            resizeLeft(pt);
//////            break;
//////        case Right:
//////            resizeRight(pt);
//////            break;
//////        case TopLeft:
//////            resizeTop(pt);
//////            resizeLeft(pt);
//////            break;
//////        case TopRight:
//////            resizeTop(pt);
//////            resizeRight(pt);
//////            break;
//////        case BottomLeft:
//////            resizeBottom(pt);
//////            resizeLeft(pt);
//////            break;
//////        case BottomRight:
//////            resizeBottom(pt);
//////            resizeRight(pt);
//////            break;
////        default:
////            if (m_leftMouseButtonPressed) {
////                setCursor(Qt::ClosedHandCursor);
////                auto dx = worldX - m_previousPosition.x();
////                auto dy = worldY - m_previousPosition.y();
////                moveBy(dx,dy);
////                setPreviousPosition(pt);
////                //emit signalMove(this, dx, dy);
////            }
////            break;
////        }
////    } else {
////        switch (m_cornerFlags) {
////        case TopLeft:
////        case TopRight:
////        case BottomLeft:
////        case BottomRight: {
////            rotateItem(pt);
////            break;
////        }
////        default:
////            if (m_leftMouseButtonPressed) {
////                setCursor(Qt::ClosedHandCursor);
////                auto dx = worldX - m_previousPosition.x();
////                auto dy = worldY - m_previousPosition.y();
////                moveBy(dx,dy);
////                setPreviousPosition(pt);
////                //emit signalMove(this, dx, dy);
////            }
////            break;
////        }
////    }
//}
//
//void GeRectangle::mousePressed(double worldX,double worldY,
//		Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) {
////    if (button == Qt::LeftButton) {
////        m_leftMouseButtonPressed = true;
////        QPointF pos(worldX, worldY);
////        setPreviousPosition(pos);
////        emit clicked(this);
////    }
//}

void GeRectangle::mouseMoveEvent(QGraphicsSceneMouseEvent *event){
	QPointF pt = event->pos();

    qDebug() << "MOUSE MOVED EVENT " << pt << " Action: " << m_actionFlags << " Corner: " << m_cornerFlags;

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

void GeRectangle::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() & Qt::LeftButton) {
        m_leftMouseButtonPressed = true;
        setPreviousPosition(event->scenePos());
        emit clicked(this);
    }
    QGraphicsItem::mousePressEvent(event);
}

void GeRectangle::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() & Qt::LeftButton) {
        m_leftMouseButtonPressed = false;
    }

    //notifyObservers();
    QGraphicsItem::mouseReleaseEvent(event);
}

void GeRectangle::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
    //m_actionFlags = (m_actionFlags == ResizeState)?RotationState:ResizeState;
    setVisibilityGrabbers();
    QGraphicsItem::mouseDoubleClickEvent(event);
}

void GeRectangle::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    setPositionGrabbers();
    setVisibilityGrabbers();
    setSelected(true);
    emit GeRectangle::itemSelected(m_objectId.getObjectId());
    QGraphicsItem::hoverEnterEvent(event);
}

void GeRectangle::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    m_cornerFlags = 0;
    qDebug() << "--- hoverLeaveEvent " << " Action: " << m_actionFlags << " Corner: " << m_cornerFlags;
    //hideGrabbers();
    setCursor(Qt::CrossCursor);
    QGraphicsItem::hoverLeaveEvent( event );
}

void GeRectangle::hoverMoveEvent(QGraphicsSceneHoverEvent *event){
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

void GeRectangle::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy){
	 qDebug() << "--- grabberMove Point: " << dx << " /  " << dy;
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
    notifyObservers();
}

QVariant GeRectangle::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
    switch (change) {
    case QGraphicsItem::ItemSelectedChange: {
    	qDebug() << "ITEM CHANGE " << value.toBool();
        if(!value.toBool()) {
        	m_grabberOn = false;
            hideGrabbers();
        } else {
        	m_grabberOn = true;
            setVisibilityGrabbers();
        }
        //m_actionFlags = ResizeState;
    	}
        break;
    case QGraphicsItem::ItemPositionChange: {
//    	QPointF newPos = value.toPointF();
//    	return newPos;
    }
    break;
    case QGraphicsItem::ItemPositionHasChanged: {
    	notifyObservers();

//    	QPointF newPosShift = value.toPointF();
//    	QPointF initialPos = rect().topLeft();
//    	QPointF newTopLeft = QPointF( initialPos.x() + newPosShift.x(), initialPos.y() + newPosShift.y());
//    	//TODO emit avec le bon rectangle
//    	QRectF rect1 (rect() );
//    	rect1.moveTo(newTopLeft);
//		qDebug() << "ITEM CHANGED " << rect() << " NEW POS " << rect1;
//		emit rectangleChanged(rect1, objectId, false);
    	break;
   }
    default:
        break;
    }
    return QGraphicsRectItem::itemChange(change, value);
}

void GeRectangle::notifyObservers() {
	QPointF newPosShift = pos();
	QPointF initialPos = rect().topLeft();
	QPointF newTopLeft = QPointF( initialPos.x() + newPosShift.x(), initialPos.y() + newPosShift.y());
	QRectF rect1 (rect() );
	rect1.moveTo(newTopLeft);
	qDebug() << "ITEM CHANGED " << rect() << " NEW POS " << rect1;
	emit rectangleChanged(rect1, m_objectId.getObjectId(), false);
}

void GeRectangle::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){
    QStyleOptionGraphicsItem myoption = (*option);
    myoption.state &= !QStyle::State_Selected;
    QGraphicsRectItem::paint(painter, &myoption, widget);


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
	QGraphicsRectItem::paint(painter, option, widget);
}

void GeRectangle::setColorDots(const QBrush &brush){
    for(int i = 0; i < 8; i++){
        m_cornerGrabber[i]->setBrush(brush);
    }
}

void GeRectangle::resizeLeft(const QPointF &pt){
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

void GeRectangle::resizeLeftBis(qreal dx) {
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

void GeRectangle::resizeRight(const QPointF &pt){
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

void GeRectangle::resizeRightBis(qreal dx){
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

void GeRectangle::resizeBottom(const QPointF &pt)
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

void GeRectangle::resizeBottomBis(qreal dy) {
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

void GeRectangle::resizeTop(const QPointF &pt){
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

void GeRectangle::resizeTopBis(qreal dy) {
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

void GeRectangle::rotateItem(const QPointF &pt)
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
    angleToTarget = normalizeAngle((Pi - angleToTarget) + Pi / 2);

    if (lineToCursor.dy() < 0)
        angleToCursor = TwoPi - angleToCursor;
    angleToCursor = normalizeAngle((Pi - angleToCursor) + Pi / 2);

    // Result difference angle between Corner Target point and Cursor Point
    auto resultAngle = angleToTarget - angleToCursor;

    QTransform trans = transform();
    trans.translate( center.x(), center.y());
    trans.rotateRadians(rotation() + resultAngle, Qt::ZAxis);
    trans.translate( -center.x(),  -center.y());
    setTransform(trans);
}

void GeRectangle::setPositionGrabbers()
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

void GeRectangle::setVisibilityGrabbers()
{
    m_cornerGrabber[GrabberTopLeft]->setVisible(true);
    m_cornerGrabber[GrabberTopRight]->setVisible(true);
    m_cornerGrabber[GrabberBottomLeft]->setVisible(true);
    m_cornerGrabber[GrabberBottomRight]->setVisible(true);

    if(m_actionFlags == ResizeState){
        m_cornerGrabber[GrabberTop]->setVisible(true);
        m_cornerGrabber[GrabberBottom]->setVisible(true);
        m_cornerGrabber[GrabberLeft]->setVisible(true);
        m_cornerGrabber[GrabberRight]->setVisible(true);
    } else {
        m_cornerGrabber[GrabberTop]->setVisible(false);
        m_cornerGrabber[GrabberBottom]->setVisible(false);
        m_cornerGrabber[GrabberLeft]->setVisible(false);
        m_cornerGrabber[GrabberRight]->setVisible(false);
    }
    qDebug() << "RECTANGLE VISIBLE " << m_cornerGrabber[GrabberTopLeft]->pos();
}

void GeRectangle::hideGrabbers()
{
    for(int i = 0; i < 8; i++){
        m_cornerGrabber[i]->setVisible(false);
    }
    qDebug() << "RECTANGLE HIDE ";
}
