#include "cutline.h"

#include <QDebug>
#include <QPen>
#include <QPainter>
#include <QGraphicsSceneMouseEvent>

/*
 * Class Managing Line used in CutScene to cut images before blending
 */

/*
 * Constructor of class
 *
 * int: size of line
 * double: scaling ratio to apply in order to maintain line visibility
 * QObject*: parent of object, needed by QObject
 *
 * set flags to allow the user to grab the item with left mouse button
 * setZvalue to have it on top of the image
 */
CutLine::CutLine(int line=1, double rx=1.0, double ry=1.0, QObject* parent = Q_NULLPTR) : QObject(parent)
{
    setFlag(ItemIsMovable);
    setFlag(ItemSendsGeometryChanges);
    setCacheMode(DeviceCoordinateCache);
    setZValue(10);
    lineSize = line;
    ratioX = rx;
    ratioY = ry;
    min = 0;
    max = 100;
    allow_emit = true;
}

/*
 * Destructor of class, do nothing
 */
CutLine::~CutLine() {

}

/*
 * Public Class Method of Node
 *
 * Update scale ratio
 */
void CutLine::updateSize(qreal sx, qreal sy) {
    //qDebug() << "CutLine" << s << ratio << ratio/s;
    ratioX /= sx;
    ratioY /= sy;
}

/*
 * Public method of class
 *
 * setter of int min to allow movement of line only in image
 * modify position if line out of image
 */
void CutLine::setMin(int const m) {
    min = m;
    if (this->pos().x()< min) {
        setPos(min,0);
    }
}

/*
 * Public method of class
 *
 * getter of int min to allow movement of line only in image
 */
int CutLine::getMin() const {
    return min;
}

/*
 * Public method of class
 *
 * setter of int max to allow movement of line only in image
 * modify position if line out of image
 */
void CutLine::setMax(int const m) {
    max = m;
    if (this->pos().x()> max) {
        setPos(max,0);
    }
}

/*
 * Public method of class
 *
 * getter of int max to allow movement of line only in image
 */
int CutLine::getMax() const {
    return max;
}

/*
 * Public method of class
 *
 * setter of int linesize to have the line cover image in one direction
 */
void CutLine::setLineSize(int const i) {
    lineSize = i;
}

/*
 * Public method of class
 *
 * getter of int linesize to have the line cover image in one direction
 */
int CutLine::getLineSize() const {
    return lineSize;
}

// redefine setpos
void CutLine::setPos(const QPointF & pos) {
    allow_emit = false;
    QGraphicsItem::setPos(pos);
}

void CutLine::setPos(qreal x, qreal y) {
    allow_emit = false;
    QGraphicsItem::setPos(x,y);
}


/*
 * Protected Class Method
 *
 * Notify that geometry has changed
 */
QVariant CutLine::itemChange(GraphicsItemChange change, const QVariant &value)
{
    switch (change) {
    case ItemPositionChange:
        if (scene()){
            // value is the new position.
            QPointF newPosComp = value.toPointF();

            // Put line inside min and max
            newPosComp.setX(0);
            if (newPosComp.y()< min) {
                newPosComp.setY(min);
            } else if(newPosComp.y() > max) {
                newPosComp.setY(max);
            }
            //qDebug() << "Change newPos" << newPosComp;
            return newPosComp;
        }
        break;
    case ItemPositionHasChanged:
        if (allow_emit) {
            emit lineChanged(this);
        } else {
            allow_emit=true;
        }
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
QRectF CutLine::boundingRect() const {
    qreal adjustX = 5;
    qreal adjustY = 5*(1+ratioY);
    // create rectangle taking into account the ratio
    return QRectF( -adjustX, -adjustY,
                   lineSize + 2* adjustX, 2*adjustY);
}

/*
 * Protected Class Method
 *
 * Called by QGraphicsScene to paint the object
 * Define how to paint the object
 */
void CutLine::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
{
    QPen pen = QPen(QColor(255,0,0));
    pen.setWidthF(std::max(1.0,getRatio()));
    painter->setPen(pen);

    // Build a cross
    painter->drawLine(0,0, lineSize, 0);
}

/*
 * Protected Class Method
 *
 * Put Node on cursor and call QGraphicsItem::update
 */
void CutLine::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->buttons() == Qt::LeftButton) {
        int xCursor = event->pos().x();
        int yCursor = event->pos().y();
        //qDebug() << "Line cursor position : x y : " << xCursor << yCursor;
        int xLine = this->pos().x();
        int yLine = this->pos().y();

        // Center Line on cursor
        this->moveBy(yCursor, 0);
        event->setPos(QPointF(event->pos().y(),0));
        //qDebug() << "Line line position : x y : " << xLine << yLine;

        update();
        QGraphicsItem::mousePressEvent(event);
        event->accept();
    }
}

/*
 * Protected Class Method
 *
 * Call QGraphicsItem::update
 */
void CutLine::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    update();
    QGraphicsItem::mouseReleaseEvent(event);
    event->accept();
    emit lineFreed(this);
}


/*
 * Public Class Method
 *
 * Get the ratio to use for drawing.
 * It is processed using ratioX and ratioY
 */
qreal CutLine::getRatio() {
    return ratioX;
}
