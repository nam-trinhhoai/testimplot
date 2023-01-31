#ifndef CUTLINE_H
#define CUTLINE_H


class QGraphicsView;
class QGraphicsSceneMouseEvent;
class QPainter;
class QStyleOption;

// Default mother class to put line in qgraphicsScene
#include <QGraphicsItem>

// Needed as we use slot and signals
#include <QObject>

/*
 * Class Managing Line used in CutScene to cut images before blending
 */
class CutLine : public QObject, public QGraphicsItem
{
    // Needed as we use slot and signals
    Q_OBJECT

public:
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
    CutLine(int, double, double, QObject*);

    /*
     * Destructor of class, do nothing
     */
    ~CutLine();

    /*
     * Public Class Method of Node
     *
     * Update scale ratio
     */
    void updateSize(qreal, qreal);

    /*
     * Public method of class
     *
     * setter of int min to allow movement of line only in image
     * modify position if line out of image
     */
    void setMin(int const);

    /*
     * Public method of class
     *
     * getter of int min to allow movement of line only in image
     */
    int getMin() const;

    /*
     * Public method of class
     *
     * setter of int max to allow movement of line only in image
     * modify position if line out of image
     */
    void setMax(int const);

    /*
     * Public method of class
     *
     * getter of int max to allow movement of line only in image
     */
    int getMax() const;

    /*
     * Public method of class
     *
     * setter of int linesize to have the line cover image in one direction
     */
    void setLineSize(int const);

    /*
     * Public method of class
     *
     * getter of int linesize to have the line cover image in one direction
     */
    int getLineSize() const;

    // redefine setpos
    void setPos(const QPointF & pos);
    void setPos(qreal x, qreal y);

protected:
    /*
     * Protected Class Method
     *
     * Notify that geometry has changed
     */
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;

    /*
     * Protected Class Method
     *
     * Return bounding rectangle of object looks
     */
    QRectF boundingRect() const override;

    /*
     * Protected Class Method
     *
     * Called by QGraphicsScene to paint the object
     * Define how to paint the object
     */
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

    /*
     * Protected Class Method
     *
     * Put Node on cursor and call QGraphicsItem::update
     */
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

    /*
     * Protected Class Method
     *
     * Call QGraphicsItem::update
     */
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

    /*
     * Public Class Method
     *
     * Get the ratio to use for drawing.
     * It is processed using ratioX and ratioY
     */
    qreal getRatio();

signals:
    /*
     * Class Signal
     *
     * Is emited when node position has changed
     */
    void lineChanged(CutLine* line);

    void lineFreed(CutLine*);

private:
    QGraphicsView* view;

    // Scale ratio
    double ratioX;
    double ratioY;
    int lineSize;
    int min;
    int max;
    bool allow_emit;
};

#endif // CUTLINE_H
