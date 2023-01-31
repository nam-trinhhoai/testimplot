#ifndef MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DSCENE_H
#define MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DSCENE_H

class QGraphicsSceneMouseEvent;
class QGraphicsLineItem;
class QGraphicsPathItem;
class QGraphicsPixmapItem;
class GraphicsTextItem;
class RectangleMovable;
class CutLine;

#include <QGraphicsScene>
#include <QString>

#include "Matrix2D.h"
#include <vector>
#include <utility>

#include "Canvas2DExtension.h"


class SelectionRectangle;

/*
 * Class acting as canvas in a Viewer of an Image
 */
class MuratCanvas2dScene: public QGraphicsScene {
Q_OBJECT

public:
	/*
	 * Public Constructor of VisualScene class
	 *
	 * QString filename : Path of Main Image
	 *
	 * Create Image
	 * Create Cursor and Curves (empty)
	 * Init ratio
	 */
	MuratCanvas2dScene(QObject* parent = 0);

	/**
	 * Add an extension to the canvas.
	 * Notice that the canvas do not take care of cleaning up the extension
	 */
	void addExtension(Canvas2DExtension* extension);
	void removeExtension(Canvas2DExtension* extension);

	/*
	 * Destructor
	 *
	 * Destroy graphicsItems
	 */
	virtual ~MuratCanvas2dScene();

    /*
     * Public Class Method
     *
     * Get the ratio to use for drawing.
     * It is processed using ratioX and ratioY
     */
    qreal getRatio();

	/*
	 * Public Class Method
	 *
	 * Update scale of all items to make them visible after scaling
	 */
    virtual void updateCursor(qreal factorX, qreal factorY);

	/*
	 * Public Class method
	 *
	 * Return Image size
	 */
	QSize imageSize();

	/*
	 * Public Class method
	 *
	 * Return Image Rectangle
	 */
	QRect imageRect() const;

	/*
	 * Public Class method
	 *
	 * Set main image
	 */
	void setImage(const QImage, Matrix2DInterface*);

	/*
	 * Public Class method
	 *
	 * Return image Pixmap
	 */
	const QPixmap getImage();

	/*
	 * Public Class method
	 *
	 * Return image Pixmap
	 */
	const Matrix2DInterface* getMatrix();

	/*
	 * Public method
	 *
	 * Setter of activation of cursor vertical
	 */
	void setVerticalCursor(bool);

	/*
	 * Public method
	 *
	 * Getter of activation of cursor vertical
	 */
	bool getVerticalCursor();

	/*
	 * Public method
	 *
	 * Setter of activation of cursor horizontal
	 */
	void setHorizontalCursor(bool);

	/*
	 * Public method
	 *
	 * Getter of activation of cursor horizontal
	 */
	bool getHorizontalCursor();

	/*
	 * Public method
	 *
	 * Setter of activation of cursor vertical
	 */
	void setVerticalCurve(bool);

	/*
	 * Public method
	 *
	 * Getter of activation of cursor vertical
	 */
	bool getVerticalCurve();

	/*
	 * Public method
	 *
	 * Setter of activation of cursor horizontal
	 */
	void setHorizontalCurve(bool);

	/*
	 * Public method
	 *
	 * Getter of activation of cursor horizontal
	 */
	bool getHorizontalCurve();

	void setPatchSize(int ps);
	int getPatchSize();

	int getCurveOffset();
	void setCurveOffset(int);
	int getCurveSize();
	void setCurveSize(int);

	void setMinMat(qreal);
	qreal getMinMat();
	void setMaxMat(qreal);
	qreal getMaxMat();

	bool getIsImageMode();
	void setIsImageMode(bool);

    std::vector<double> getLines();
    void setLines(const std::vector<double>& lines);

	virtual void simulateMousePressEvent(QPointF pos);

	/*
	 * Protected Class Method
	 *
	 * Update lines from cursor and curves of gray signal
	 */
	virtual void simulateUpdateCurvesAndCursor(QPointF scenePos);

	QImage getQImage();

	void drawRectangles(std::vector<std::pair<QRect, QColor>> rects);

	std::vector<std::pair<QRect, QColor>> getOverlayRectangles();

	/*
	 * Public Class Method
	 *
	 * Delete objects specific to background image
	 */
	void clearOverlayRectangles();

	/**
	 * Need to be public we want to avoid default scrool on wheel
	 */
	virtual void wheelEvent(QGraphicsSceneWheelEvent *wheelEvent) override;

    void toggleCurve(bool);

    /**
     * Set image item scale to allow the display of an image with a different sample rate
     * This is useful to allow synchronization between image of varying
     */
    void setImageItemVerticalScale(float value);

signals:
	void takePatch(int x, int y);
    void linesChanged(const std::vector<double>& lines);

protected:
	/*
	 * Protected Class Method
	 *
	 * Trigger update Cursor position and Curves
	 */
	/**
	 * Mouse binding
	 */
	virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *mouseEvent)
			override;
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent) override;
	virtual void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent) override;
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent)
			override;


	/**
	 * Keyboard binding
	 */
	virtual void keyPressEvent(QKeyEvent *keyEvent) override;
	virtual void keyReleaseEvent(QKeyEvent *keyEvent) override;

    QImage cache_image;
	QGraphicsPixmapItem* picture = nullptr;
	Matrix2DInterface* mat = nullptr;
	qreal minMat = 0;
	qreal maxMat = 1;
	QGraphicsLineItem* hCursor = nullptr;
	QGraphicsLineItem* vCursor = nullptr;
	QGraphicsPathItem* hCurve = nullptr;
	QGraphicsPathItem* vCurve = nullptr;
	QGraphicsTextItem* displayValue = nullptr;
	QPen* pen = nullptr;
	QColor* hColor = nullptr;
	QColor* vColor = nullptr;
    qreal ratioX;
    qreal ratioY;
	RectangleMovable* patch = nullptr;
	int patchSize;
	QImage img;
    std::vector<CutLine*> lines;


	/*
	 * Protected Class Method
	 *
	 * Update lines from cursor and curves of gray signal
	 */
public:
    virtual void updateCurvesAndCursor(QPointF scenePos);
protected:
	virtual void updateCurvesAndCursorImage(QPointF scenePos);
	virtual void updateCurvesAndCursorLog(QPointF scenePos);

	/*
	 * Private Class Method
	 *
	 * Convenience method to get correct QPen with color and width (indexed on ratio)
	 */
	QPen buildQPenWithColor(QPen*, QColor*);

	/*
	 * Protected Class Method
	 *
	 * Convenience method to get correct QPen for horizontal items
	 */
	QPen hQPen();

	/*
	 * Protected Class Method
	 *
	 * Convenience method to get correct QPen for horizontal items
	 */
	QPen vQPen();

	int curveOffset;
	int curveSize;
	bool isImageMode;

	std::vector<RectangleMovable*> overlayRectangles;

	/**
	 * Contains the list of canvas extensions
	 */
	std::vector<Canvas2DExtension*> extensions;

private:
	bool horizontalCursor;
	bool verticalCursor;
	bool horizontalCurve;
    bool verticalCurve;
    bool showCurve = true;

private slots:
    void lineChanged(CutLine* line);

};

#endif // MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DSCENE_H
