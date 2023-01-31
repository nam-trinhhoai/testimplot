#ifndef MURATAPP_SRC_VIEW_CANVAS2D_RECTANGLEMOVABLE_H
#define MURATAPP_SRC_VIEW_CANVAS2D_RECTANGLEMOVABLE_H

class QGraphicsView;
class QGraphicsSceneMouseEvent;
class QPainter;
class QStyleOption;
#include <QGraphicsItem>
#include <QObject>

/*
 * Class that originally represented either the moving rectangle on preview or the rectangle  acting as the background image
 * Now it also represent the lines and the rectangle on the images
 *
 * May be rewriten in two separate class or another way.
 */
class RectangleMovable : public QObject, public QGraphicsItem
{
    // Use signals and slots
    Q_OBJECT
	Q_INTERFACES(QGraphicsItem)

public:
    /*
     * Constructor manage the moving rectangle (movable=true)
     * And the lines and the rectangle on the preview view
     */
    RectangleMovable(bool movable=false, qreal rx=1, qreal ry=1, QObject* parent=0);

    /*
     * Destructor does nothing
     */
    ~RectangleMovable();

    /*
     * Public Class Method of Node
     *
     * Update scale ratio
     */
    void updateSize(qreal, qreal);

    /*
     * Public method
     *
     * Getter of rectangle width
     */
    int getRectWidth() const;

    /*
     * Public method
     *
     * Setter of rectangle width
     */
    void setRectWidth(int value);

    /*
     * Public method
     *
     * Getter of rectangle height
     */
    int getRectHeight() const;

    /*
     * Public method
     *
     * Setter of rectangle height
     */
    void setRectHeight(int value);

    /*
     * Public method
     *
     * Send the conversion ratio from real canvas to preview canvas
     * preview = real * conversion
     */
    qreal getConversionBetweenCanvas() const;

    /*
     * Public method
     *
     * Setter of Conversion ratio
     * Update values of the object to match new ratio
     */
    void setConversionBetweenCanvas(const qreal);

    /*
     * Public method
     *
     * Setter of backgroung image
     */
    void setImage(const QImage&);

    /*
     * Public method
     *
     * Getter of background image
     */
    const QImage& getImage();

    /*
     * Public method
     *
     * Setter of lines values, activate them on first call
     */
    void setLines(int left, int right);

    /*
     * Public method
     *
     * Getter of lines value, if not activated, values are 0
     */
    void getLines(int& left, int& right);

    /*
     * Public method
     *
     * Setter of points, activate them if array not empty
     */
    void setPoints(const QList<QPointF>& pts);

    /*
     * Public method
     *
     * Getter of points, send an empty array if not activated
     */
    const QList<QPointF> getPoints();

    /*
     * Public Class Method
     *
     * Return bounding rectangle of object looks
     */
    QRectF boundingRect() const override;

    /*
     * Public Class Method
     *
     * Getter and Setter of color
     * Default is (255,102,0) for ItemIsMovable
     * Else color is (255,200,0)
     */
    QColor getColor();
    void setColor(QColor);

    /*
     * Public Class Method
     *
     * Compute drawing ratio thanks to ratioX and ratioY
     */
    qreal getRatio() const;

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
     * Called by QGraphicsScene to paint the object
     * Define how to paint the object
     */
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

    /*
     * Protected Class Method
     *
     * Call QGraphicsItem::update
     */
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

    /*
     * Protected method
     *
     * emit rectangleChanged
     */
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;

    /*
     * Protected method
     *
     * if lines activated
     * Go to the leftline if mouse on left side of image
     * Go to the rightline if mouse is on the right if image
     *
     * if lines not activated but points are
     * Same principle for lines but with corners
     */
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;

signals:
    /*
     * Class Signal
     *
     * Is emited when node position has changed
     */
    void rectangleChanged(RectangleMovable* rectangle, QPointF pos, QPointF lastPos);

    /*
     * Signal
     *
     * Is emited to request a focus on point x, y to make focus on node
     */
    void changeFocusPreviewRectangle(int x, int y);

    /*
     * Signal
     *
     * Is emited to request focus on absicse x to make focus on line
     */
    void changeFocusPreviewLine(int x);

protected:
    // Scale ratio
    qreal ratioX;
    qreal ratioY;
    qreal conversionBetweenCanvas;
    int rectWidth;
    int rectHeight;
    QImage image;
    QImage imageRender;
    bool newImage;

    // Values are store in preview value and converted to real values
    int leftLine;
    int rightLine;
    bool activeLine;

    QList<QPointF> points;
    bool activePoints;

    QColor color;
};

#endif // MURATAPP_SRC_VIEW_CANVAS2D_RECTANGLEMOVABLE_H
