#include "rectanglemovable.h"
#include <cmath>
#include <QDebug>
#include <QPainter>
#include <QPen>
#include <QGraphicsSceneMouseEvent>

/*
 * Class that originally represented either the moving rectangle on preview or the rectangle  acting as the background image
 * Now it also represent the lines and the rectangle on the images
 *
 * May be rewriten in two separate class or another way.
 */

/*
 * Constructor manage the moving rectangle (movable=true)
 * And the lines and the rectangle on the preview view
 */
RectangleMovable::RectangleMovable(bool movable, qreal rx, qreal ry, QObject* parent) : QObject(parent)
{
    if (movable) {
        setFlag(ItemIsMovable);
        setFlag(ItemSendsGeometryChanges);
        setCacheMode(DeviceCoordinateCache);
        setZValue(10);
        //((QGraphicsView*) parent)->viewport()->installEventFilter(this);
    } else {
        setZValue(0);
    }
    ratioX = rx;
    ratioY = ry;
    newImage = false;
    leftLine = 0;
    rightLine = 0;
    activeLine = false;
    activePoints = false;
    conversionBetweenCanvas = 1;

    if (flags() & ItemIsMovable) {
        color = QColor(255,102,0);
    } else {
        color = QColor(255,200,0);
    }
}

/*
 * Destructor does nothing
 */
RectangleMovable::~RectangleMovable() {}

/*
 * Public Class Method of Node
 *
 * Update scale ratio
 */
void RectangleMovable::updateSize(qreal sx, qreal sy) {
    //qDebug() << "Rectangle" << s << ratio << ratio/s;
    ratioX /= sx;
    ratioY /= sy;
}

int RectangleMovable::getRectHeight() const
{
    return rectHeight;
}

void RectangleMovable::setRectHeight(int value)
{
    rectHeight = value;
}

int RectangleMovable::getRectWidth() const
{
    return rectWidth;
}

void RectangleMovable::setRectWidth(int value)
{
    rectWidth = value;
}

qreal RectangleMovable::getConversionBetweenCanvas() const {
    return conversionBetweenCanvas;
}

void RectangleMovable::setConversionBetweenCanvas(const qreal r) {
    if (activeLine) {
        leftLine = (int) std::floor(leftLine * r / conversionBetweenCanvas);
        rightLine = (int) std::floor(rightLine * r / conversionBetweenCanvas);
    }
    if (activePoints) {
        for (int i=0; i<points.length(); i++) {
            points[i] = points[i] * r / conversionBetweenCanvas;
        }
    }
    conversionBetweenCanvas = r;
    update();
}

void RectangleMovable::setImage(const QImage& img) {
    image = QImage(img);
    newImage = true;
    update();
}

const QImage& RectangleMovable::getImage() {
    return image;
}

void RectangleMovable::setLines(int left, int right) {
    leftLine = (int) std::floor(((double) left) * (double) conversionBetweenCanvas);
    rightLine = (int) std::floor(((double) right) * (double) conversionBetweenCanvas);
    activeLine = true;
    update();
}

void RectangleMovable::getLines(int& left, int& right) {
    left = (int) std::floor((qreal) leftLine / conversionBetweenCanvas);
    right = (int) std::floor((qreal) rightLine / conversionBetweenCanvas);
}

void RectangleMovable::setPoints(const QList<QPointF>& pts) {
    points.clear();
    for (int i=0; i<pts.length(); i++) {
        points.append(QPointF(pts[i]) * conversionBetweenCanvas);
    }
    if (pts.length()>0) {
        activePoints = true;
    } else {
        activePoints = false;
    }
    update();
}

const QList<QPointF> RectangleMovable::getPoints() {
    QList<QPointF> copy;
    for (int i=0; i<points.length(); i++) {
        copy.append(QPointF(points[i]) / conversionBetweenCanvas);
    }
    return copy;
}


/*
 * Protected Class Method
 *
 * Notify that geometry has changed
 */
QVariant RectangleMovable::itemChange(GraphicsItemChange change, const QVariant &value)
{
    switch (change) {
    case ItemPositionHasChanged:
        break;
    default:
        break;
    };

    // call father methods
    return QGraphicsItem::itemChange(change, value);
}

/*
 * Protected Class Method
 *
 * Return bounding rectangle of object looks
 */
QRectF RectangleMovable::boundingRect() const {
    qreal adjust = (1+getRatio());
    // create rectangle taking into account the ratio
    return QRectF( -adjust, -adjust,
                   2* adjust + rectWidth, rectHeight + 2*adjust);
}

/*
 * Protected Class Method
 *
 * Called by QGraphicsScene to paint the object
 * Define how to paint the object
 */
void RectangleMovable::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
{
    //qDebug() << "Rectangle Movable paint start, ratio =" << activeLine << activePoints << leftLine << rightLine << conversionBetweenCanvas;
    QPen pen;
    pen.setColor(QColor(255,0,0));
    pen.setWidthF(getRatio()*2);
    painter->setPen(pen);

    // Build a rectangle
    if ((!image.isNull() && (imageRender.width()!=rectWidth || imageRender.height()!=rectHeight)) || newImage) {
        imageRender = image.scaled(rectWidth, rectHeight);
        newImage = false;
    }
    if (!image.isNull()) {
        painter->drawImage(0,0, imageRender);
    }
    if(activeLine) {
        pen.setColor(QColor(255,0,0));
        pen.setWidthF(getRatio());
        painter->setPen(pen);
        painter->drawLine(leftLine, 0, leftLine, rectHeight);
        painter->drawLine(rightLine, 0, rightLine, rectHeight);
    }
    if(activePoints) {
        pen.setColor(QColor(255,255,255));
        pen.setWidthF(getRatio());
        painter->setPen(pen);
        for (int i=0; i<points.length()-1; i++) {
            painter->drawLine(points[i], points[i+1]);
        }
        if (points.length()>2) {
            painter->drawLine(points[0], points[points.length()-1]);
        }

        pen.setColor(QColor(255,0,0));
        pen.setWidthF(getRatio());
        painter->setPen(pen);
        for (int i=0; i<points.length(); i++) {
            int x = std::floor(points[i].x());
            int y = std::floor(points[i].y());
            painter->drawLine(x - 3, y, x + 3, y);
            painter->drawLine(x, y - 3, x, y + 3);
        }
    }

    if (flags() & ItemIsMovable) {
    	pen.setColor(color);
    } else {
        pen.setColor(color);
    }
    painter->setPen(pen);
    painter->drawRect(0,0, rectWidth, rectHeight);
}

/*
 * Protected Class Method
 *
 * Call QGraphicsItem::update
 */
void RectangleMovable::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    update();
    QGraphicsItem::mouseReleaseEvent(event);
    event->accept();
}

void RectangleMovable::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
    QGraphicsItem::mouseMoveEvent(event);
    emit rectangleChanged(this, event->pos(), event->lastPos());
    //qDebug() << "RectangleMovable mouseMoveEvent";
    event->accept();
}

void RectangleMovable::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
    QGraphicsItem::mouseDoubleClickEvent(event);
    if (activeLine) {
        if(2*event->pos().x()< leftLine+rightLine) {
            emit changeFocusPreviewLine(leftLine/ conversionBetweenCanvas);
        } else {
            emit changeFocusPreviewLine(rightLine/ conversionBetweenCanvas);
        }
        event->accept();
    } else if(activePoints) {
        int index;
        if (2*event->pos().x()< points[0].x()+points[1].x()) {
            index = 0;
        } else {
            index = 1;
        }
        if (2*event->pos().y()> points[0].y()+points[3].y()) {
            index = 3 - index;
        }
        emit changeFocusPreviewRectangle(points[index].x()/ conversionBetweenCanvas, points[index].y()/ conversionBetweenCanvas);
        event->accept();
    }
}

QColor RectangleMovable::getColor() {
	return color;
}

void RectangleMovable::setColor(QColor color) {
	this->color = color;
}

qreal RectangleMovable::getRatio() const {
    return ratioX;
}
