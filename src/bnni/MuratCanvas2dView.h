#ifndef MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DVIEW_H
#define MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DVIEW_H


#include <QGraphicsView>
#include <QImage>
#include <QLabel>

#include "Matrix2D.h"
#include <vector>
#include <utility>

class QMouseEvent;
class QWheelEvent;
class QResizeEvent;
class QShowEvent;
class QString;


class RectangleMovable;


class MuratCanvas2dScene;
class GraphicsViewZoom;

class MuratCanvas2dView : public QGraphicsView
{
    Q_OBJECT

public:
    /*
     * Public Constructor of VisualView class
     *
     * Only call constructor of QGraphicsView
     */
    MuratCanvas2dView(QWidget* parent=0);

    /*
     * Public Destructor of class
     *
     * Stop timer if one is active and destroy it
     */
    virtual ~MuratCanvas2dView();

    /*
     * Public Class method to initialize view and scene
     * QPixmap image : main image
     *
     * Create the canvas (VisualScene*) with the image(QString filename)
     * Initialize ratio (scaling)
     * Initialize variables that take care of moving the scene when mouse is close to edges
     *      -> xRatio, yRatio, xDelta, yDelta, timer
     * Set view options : CacheBackground, BoundingRectViewPortUpdate, Antialiasing
     */
    void initScene();

    /*
     * Public virtual method
     *
     * Define canvas variable
     * Only Access canvas through getter
     */
    virtual MuratCanvas2dScene* initCanvas();

    /*
     * Public Class Method override default scale method
     * qreal sx : scale factor along x axis
     * qreal sy : scale factor along y axis
     *
     * if sx = sy -> check future ratio to have global scale between min and max then scale
     * else scale as QGraphicsView
     */
    virtual void scale(qreal, qreal);

    /*
     * Public Class Method
     *
     * Setter of image
     */
    void setImage(QImage img, Matrix2DInterface* mat = nullptr);
    /*
     * Public Class Method
     *
     * Getter of image rectangle
     */
    QRect getImageRect();

    /*
     * Public method
     *
     * Getter of ratioX
     */
    qreal getRatioX();

    /*
     * Public method
     *
     * Getter of ratioY
     */
    qreal getRatioY();

    /*
     * Public method
     *
     * Setter of ratioX
     */
    void setRatioX(qreal);

    /*
     * Public method
     *
     * Setter of ratioY
     */
    void setRatioY(qreal);

    int getId();


    void setPatchSize(int ps);
    int getPatchSize();

    /*
     * Public Class method
     *
     * Return image Pixmap
     */
    const QPixmap getImage();

    /*
     * Public Class method
     *
     * Return image matrix
     */
    const Matrix2DInterface* getMatrix();

    bool infitImage();

    void setInFitImage(bool val);

    int getCurveOffset();
    void setCurveOffset(int);
    int getCurveSize();
    void setCurveSize(int);

    void setMinMat(qreal);
    qreal getMinMat();
    void setMaxMat(qreal);
    qreal getMaxMat();

    void triggerOverlay(bool);
    bool getOverlayBool();

    QImage getQImage();

    void drawRectangles(std::vector<std::pair<QRect, QColor> > rects);
    std::vector<std::pair<QRect, QColor> > getOverlayRectangles();
    void clearOverlayRectangles();

    void setImageItemVerticalScale(float value);

public slots:
    /*
     * Slot that scale * 2.0
     */
    void zoomIn();

    /*
     * Slot that scale / 2.0
     */
    void zoomOut();

    /*
     * Slot that scale the image to have the image fit in the view
     */
    void fitImage();

    void takePatchSlot(int x, int y);

    void simulate_mouse_press_event(QPointF pos);

private slots:
    /*
     * Slot to scroll in the direction given by xDelta and yDelta
     * Check if we can still scroll
     * Yes : scroll using scrollbars
     * No : stop timer and reset xDelta and yDelta
     */
    virtual void adjustRectPosition();

    /*
     * Private slot
     *
     * Update view size and position
     */
    void overlayChanged();

    /*
     * Private slot
     *
     * Apply changes on preview to canvas
     */
    void previewMoved(RectangleMovable* rectangle, QPointF pos, QPointF lastPos);

    /*
     * Private Slot
     *
     * Apply focus to point in canvas
     */
    void changeFocusPreviewRectangle(int x, int y);

    /*
     * Private Slot
     *
     * Apply focus to absisse in canvas
     */
    void changeFocusPreviewLine(int);


protected:
    /*
     * Protected Class Method override QGraphicsView method
     *
     * Called when mouse move
     * Catch mouse mosition and launch updatePaddingArea
     */
    virtual void mouseMoveEvent(QMouseEvent*);

    /*
     * Public method
     *
     * Do zoom taking into account previous wheel event, centering on mouse position
     */
    virtual void mouseMoveEventWheelManager(QMouseEvent *event);

    /*
     * Protected Class Method override QWidget method
     *
     * Called when mouse leave the widget
     * Stop the timer if active
     */
    virtual void leaveEvent(QEvent*);

    /*
     * Protected Class Method override QWidget method
     *
     * Do scaling according to wheel action
     */
    virtual void wheelEvent(QWheelEvent*);

    /*
     * Protected Class Method override QWidget method
     *
     * Do a fitImage if image just got fit
     * Anyway update overlay
     */
    virtual void resizeEvent(QResizeEvent*);

    /*
     * Protected Class Method override QWidget method
     *
     * Do a fitImage
     */
    virtual void showEvent(QShowEvent*);

    /*
     * Protected Class Method
     * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Deactivated
     *
     * Detect in which area of the view the cursor is
     *      -> (top, bottom, left, right, middle)
     *      -> corner are just like being top and left.
     * Launch timer if needed and stop if needed
     */
    virtual void updatePaddingArea(qreal, qreal);

    /*
     * Protected Method
     *
     * Manage acceleration of scrolling when near edges
     */
    virtual void updatePaddingAcceleration();

    // Variable to store VisualScene object
    MuratCanvas2dScene* canvas;
    QGraphicsView* previewView;
    QGraphicsScene* previewScene;

    RectangleMovable* rectImage;
    RectangleMovable* rectView;

    // Variable to store scaling ratio only updated when sx = sy in scale(sx, sy)
    qreal ratioX;
    qreal ratioY;
    int id;
    static int nextId;

    qreal xDelta;
    qreal yDelta;
    QTimer* timer;

    bool overlayHidden;

    GraphicsViewZoom* zoomWheel;

private:
    /*
     *  Variables to manage scrolling when mouse is near edges
     *
     * xRatio, yRatio : act like an acceleration parameter
     * xRatioStep, yRatioStep : step of increment in acceleration
     * timer : QTimer object
     */
    qreal xRatio;
    qreal yRatio;
    const qreal xRatioStep = 0.01;
    const qreal yRatioStep = 0.01;

    QPoint lastPos;
    bool initLastPos;

public:
    /*
     * Private method
     *
     * Getter of canvas should be use to access canvas
     */
    virtual MuratCanvas2dScene* getCanvas();


    /*
     * Public method
     *
     * Do zoom taking into account previous wheel event, centering on mouse position
     */
    virtual void gentle_zoom(double factorX, double factorY);

    /*
     * Public method
     *
     * Do zoom taking into account previous wheel event, centering on position
     */
    virtual void simulate_gentle_zoom(double factorX, double factorY, QPointF target_scene_pos);

    /*
     * Public method
     *
     * Move arrow to position
     */
    virtual void simulate_gentle_move(QPointF pos);

    /*
     * Public method
     *
     * Move arrow to position
     */
    virtual void simulate_gentle_y_move(qreal yPos);

    /*
     * Public method
     *
     * Simulate gentle zoom but without centering on mouse position
     */
    virtual void simulate_gentle_zoom(double factorX, double factorY);

    /*
     * Public method
     *
     * Define modifiers for wheel zoom
     */
    void set_modifiers(Qt::KeyboardModifiers modifiers);

    /*
     * Public method
     *
     * Define scale base factor for wheel zoom
     */
    void set_zoom_factor_base(double value);

    void toggleCurve(bool val);

    void setWindowName(QString);
    QString getWindowName();

signals:
    /*
     * Class Signal that should be use to stop timer
     */
    void stopTimer();

    /*
     * Signal that should be use to start timer
     */
    void startTimer();

    /*
     * Signal notify that the view zoomed
     */
    void zoomed(double factorX, double factorY, QPointF target_scene_pos );

    void signalFitImage();

    void takePatch(int x, int y);

    void mousePosition(QPointF pos);

private:

  Qt::KeyboardModifiers _modifiers;
  double _zoom_factor_base;
  QPointF target_scene_pos, target_viewport_pos;
  bool inFitImage;
  QLabel* windowLabel = nullptr;
};


#endif // MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DVIEW
