#include "MuratCanvas2dView.h"

#include "graphicsviewzoom.h"
#include "rectanglemovable.h"

#include <QScrollBar>
#include <QTimer>
#include <QString>
#include <QDebug>
#include <cmath>
#include <QApplication>
#include <qmath.h>
#include <QPushButton>
#include <QGridLayout>
#include <QSpacerItem>
#include <QMouseEvent>
#include "MuratCanvas2dScene.h"


/*
 * Public Constructor of VisualView class
 *
 * Only call constructor of QGraphicsView
 */
MuratCanvas2dView::MuratCanvas2dView(QWidget* parent) :
		QGraphicsView(parent) {
	timer = nullptr;
	previewView = nullptr;
	inFitImage = false;
	id = nextId;
	nextId++;
	overlayHidden = false;
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	setAcceptDrops(false);
}

/*
 * Public Destructor of class
 *
 * Stop timer if one is active and destroy it
 */
MuratCanvas2dView::~MuratCanvas2dView() {
	if (timer) {
		if (timer->isActive()) {
			emit stopTimer();
		}
		delete timer;
	}
}

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
void MuratCanvas2dView::initScene() {
	canvas = initCanvas();
	setScene(getCanvas());

	initLastPos = false;

	// set QGraphicsView options
	setCacheMode(CacheBackground);
	setViewportUpdateMode(BoundingRectViewportUpdate);
	setRenderHint(QPainter::Antialiasing);

	// Create GraphicsViewZoom object to manage wheel event
	/*zoomWheel = new GraphicsViewZoom(this);
	 zoomWheel->set_modifiers(Qt::NoModifier);*/
	this->setMouseTracking(true);
	_modifiers = Qt::ControlModifier;
	_zoom_factor_base = 1.0015;

	// Init variables of scrolling when mouse is near edges
	xRatio = 0;
	yRatio = 0;
	xDelta = 0;
	yDelta = 0;

	// Init timer
	timer = new QTimer(this);
	timer->setSingleShot(false);
	timer->setInterval(1000.0 / 30.0);

	// adjustRectPosition is called regurlaly to scroll accordiong to xDelta and yDelta
	connect(timer, SIGNAL(timeout()), this, SLOT(adjustRectPosition()));
	connect(this, SIGNAL(startTimer()), timer, SLOT(start()));
	connect(this, SIGNAL(stopTimer()), timer, SLOT(stop()));

	QGridLayout *layout = new QGridLayout(this);

	this->setLayout(layout);
	layout->setContentsMargins(20, 20, 20, 20);
    if (previewView == nullptr) {
        windowLabel = new QLabel("");
        QFont font = windowLabel->font();
        font.setPointSize(12);
        font.setBold(true);
        windowLabel->setFont(font);
        layout->addWidget(windowLabel, 0, 0);


		previewView = new QGraphicsView(this);
		previewScene = new QGraphicsScene(previewView);
		previewView->setScene(previewScene);

        layout->addWidget(previewView, 2, 0);
		layout->addItem(
				new QSpacerItem(1, 60, QSizePolicy::Minimum,
                        QSizePolicy::Expanding), 1, 0);
		layout->addItem(
				new QSpacerItem(60, 1, QSizePolicy::Expanding,
                        QSizePolicy::Minimum), 2, 1);
		previewView->setFixedSize(150, 150);
        previewView->hide();

		//connect(getCanvas(), SIGNAL(changed(QList<QRectF>)), SLOT(overlayChanged()));
		connect(horizontalScrollBar(), SIGNAL(valueChanged(int)),
				SLOT(overlayChanged()));
		connect(verticalScrollBar(), SIGNAL(valueChanged(int)),
				SLOT(overlayChanged()));

		// Init scale ratio to 1;
        ratioX = 1;
        ratioY = 1;
	} else {
		previewScene->clear();
    }

    rectImage = new RectangleMovable(false, ratioX, ratioY, this);
    rectView = new RectangleMovable(true, ratioX, ratioY, this);

    previewScene->addItem(rectImage);
    previewScene->addItem(rectView);

	rectImage->setZValue(0);
    connect(rectView,
			SIGNAL(rectangleChanged(RectangleMovable*,QPointF, QPointF)), this,
            SLOT(previewMoved(RectangleMovable*,QPointF, QPointF)));

	connect(rectImage, SIGNAL(changeFocusPreviewLine(int)),
			SLOT(changeFocusPreviewLine(int)));
	connect(rectImage, SIGNAL(changeFocusPreviewRectangle(int,int)),
            SLOT(changeFocusPreviewRectangle(int,int)));

	connect(getCanvas(), SIGNAL(takePatch(int,int)), this,
			SLOT(takePatchSlot(int,int)));

	getCanvas()->setVerticalCursor(true);
	getCanvas()->setVerticalCurve(true);
	getCanvas()->setHorizontalCursor(true);
	getCanvas()->setHorizontalCurve(true);
}

/*
 * Public virtual method
 *
 * Define canvas variable
 * Only Access canvas through getter
 */
MuratCanvas2dScene* MuratCanvas2dView::initCanvas() {
	return new MuratCanvas2dScene(this);
}

/*
 * Public Class Method override default scale method
 * qreal sx : scale factor along x axis
 * qreal sy : scale factor along y axis
 *
 * if sx = sy -> check future ratio to have global scale between min and max then scale
 * else scale as QGraphicsView
 */
void MuratCanvas2dView::scale(qreal sx, qreal sy) {
	inFitImage = false;
	if (sx == sy) {
		// case keep image ratio
		qreal s = sx;

		/*
		 * print Zoom In if s > 1
		 * print Zoom Out if s < 1
		 * and print view width and height
		 */
		//qDebug() << ((s>1.0)?"Zoom In ":"Zoom Out ") << width() << height();
		// compute min size according to image size and view size
		QSize imgSize = getCanvas()->imageSize();
		double minH_W = std::min(imgSize.height(), imgSize.width());
		qreal sizeMinAllowed = std::min(1.0, 20 / minH_W);
		//qDebug() << "Ratio : " << ratio << sizeMinAllowed;
		// max = 1000
		// min is 20 pixel for the image or 1 if image have less than 20 pixels in length
        if ((s > 1.0 && s * std::max(ratioX, ratioY) < 1000)
                || (s < 1.0 && std::min(ratioX, ratioY) * s > sizeMinAllowed)) {
			// scale view
            ratioX *= s;
            ratioY *= s;
			QGraphicsView::scale(s, s);

			// updateCursor to make items in canvas still look good
            getCanvas()->updateCursor(s, s);
		}
	} else {
        // do not keep image ratio then
        ratioX *= sx;
        ratioY *= sy;
        QGraphicsView::scale(sx, sy);
        getCanvas()->updateCursor(sx, sy);
	}

	// correct scene size to have scene size corresponding to needed size
	//this->setSceneRect(getCanvas()->itemsBoundingRect());
}

/*
 * Public Class Method
 *
 * Setter of image
 */
void MuratCanvas2dView::setImage(QImage img, Matrix2DInterface*mat) {
	getCanvas()->setImage(img, mat);
	rectImage->setImage(img.scaledToHeight(500));
	overlayChanged();
}

/*
 * Public Class Method
 *
 * Getter of image rectangle
 */
QRect MuratCanvas2dView::getImageRect() {
	return this->getCanvas()->imageRect();
}

/*
 * Public Class method
 *
 * Return image matrix
 */
const Matrix2DInterface* MuratCanvas2dView::getMatrix() {
	return this->getCanvas()->getMatrix();
}

/*
 * Public method
 *
 * Getter of ratioX
 */
qreal MuratCanvas2dView::getRatioX() {
    return ratioX;
}

/*
 * Public method
 *
 * Getter of ratioY
 */
qreal MuratCanvas2dView::getRatioY() {
    return ratioY;
}

/*
 * Public method
 *
 * Setter of ratioX
 */
void MuratCanvas2dView::setRatioX(qreal r) {
    scale(r / ratioX, r / ratioX);
}

/*
 * Public method
 *
 * Setter of ratioY
 */
void MuratCanvas2dView::setRatioY(qreal r) {
    scale(r / ratioY, r / ratioY);
}

/*
 * Private method
 *
 * Getter of canvas should be use to access canvas
 */
MuratCanvas2dScene* MuratCanvas2dView::getCanvas() {
	return canvas;
}

/*
 * Slot that scale * 2.0
 */
void MuratCanvas2dView::zoomIn() {
	scale(2.0, 2.0);
}

/*
 * Slot tha scale / 2.0
 */
void MuratCanvas2dView::zoomOut() {
	scale(0.5, 0.5);
}

/*
 * Slot that scale the image to have the image fit in the view
 */
void MuratCanvas2dView::fitImage() {
	QRect imgRect = getCanvas()->imageRect();
	//QRect viewRect = this->rect();
	// Get view left, right, top, bottom
	/*int left = this->viewport()->x();
	 int right = left + this->viewport()->width();
	 int top = viewport()->y();
	 int bottom = top + viewport()->height();*/
	int viewWidth = width();
	int viewHeight = height();

	qreal sx = ((qreal) viewWidth) / ((qreal) imgRect.width());
	qreal sy = ((qreal) viewHeight) / ((qreal) imgRect.height());
    sx = sx / ratioX;
    sy = sy / ratioY;
	//qDebug() << "fit image : " << imgRect << viewWidth << viewHeight << sx << sy << s;
    scale(sx, sy);
	centerOn((qreal) imgRect.width() / 2.0, (qreal) imgRect.height() / 2.0);
	inFitImage = true;

	emit signalFitImage();
}

/*
 * Slot to scroll in the direction given by xDelta and yDelta
 * Check if we can still scroll
 * Yes : scroll using scrollbars
 * No : stop timer and reset xDelta and yDelta
 */
void MuratCanvas2dView::adjustRectPosition() {

	QScrollBar *barH = this->horizontalScrollBar();
	QScrollBar *barV = this->verticalScrollBar();

	// if end reached then stop timer
	if ((xDelta == 0 || (xDelta > 0 && barH->maximum() == barH->value())
			|| (xDelta < 0 && barH->minimum() == barH->value()))
			&& (yDelta == 0 || (yDelta > 0 && barV->maximum() == barV->value())
					|| (yDelta < 0 && barV->minimum() == barV->value()))) {
		xDelta = 0;
		yDelta = 0;
		emit stopTimer();
	} else {
		updatePaddingAcceleration();
		// move scene using scrollbars
		barH->setValue(barH->value() + xDelta * xRatio);
		barV->setValue(barV->value() + yDelta * yRatio);
	}
}

/*
 * Protected Class Method override QGraphicsView method
 *
 * Called when mouse move
 * Catch mouse mosition and launch updatePaddingArea
 */
void MuratCanvas2dView::mouseMoveEvent(QMouseEvent *event) {
	QGraphicsView::mouseMoveEvent(event);
	int x = event->pos().x();
	int y = event->pos().y();

	updatePaddingArea(x, y);
	mouseMoveEventWheelManager(event);

	//qDebug() << "EnhancedGraphicsView mouseMoveEvent" << event->button() << Qt::LeftButton << initLastPos;
	if (event->buttons() == Qt::MiddleButton && initLastPos) {
		QScrollBar* barH = horizontalScrollBar();
		QScrollBar* barV = verticalScrollBar();

		barH->setValue(barH->value() - (event->pos().x() - lastPos.x()));
		barV->setValue(barV->value() - (event->pos().y() - lastPos.y()));
	} else {
		initLastPos = true;
	}
	lastPos = QPoint(event->pos());

	emit mousePosition(mapToScene(event->pos()));
}

void MuratCanvas2dView::mouseMoveEventWheelManager(QMouseEvent *event) {
	QPointF delta = target_viewport_pos - event->pos();
	if (qAbs(delta.x()) > 5 || qAbs(delta.y()) > 5) {
		target_viewport_pos = event->pos();
		target_scene_pos = this->mapToScene(event->pos());
	}
}

/*
 * Protected Class Method override QWidget method
 *
 * Called when mouse leave the widget
 * Stop the timer if active
 */
void MuratCanvas2dView::leaveEvent(QEvent *) {
	if (timer->isActive()) {
		emit stopTimer();
	}
}

/*
 * Protected Class Method override QWidget method
 *
 * Do scaling according to wheel action
 */
void MuratCanvas2dView::wheelEvent(QWheelEvent * event) {
    /*if((event->modifiers()&Qt::ControlModifier )>0){
		QGraphicsView::wheelEvent(event);
		return;
    }*/

	//if (event->orientation() == Qt::Vertical) {
	if (event->angleDelta().y() != 0) {
		double angle = event->angleDelta().y();
		double factor = qPow(_zoom_factor_base, -angle);

        if (QApplication::keyboardModifiers()==Qt::ShiftModifier) {
            gentle_zoom(factor, 1);
        } else if (QApplication::keyboardModifiers()==Qt::ControlModifier) {
            gentle_zoom(1, factor);
        } else {
            gentle_zoom(factor, factor);
        }
	}
}

/*
 * Protected Class Method override QWidget method
 *
 * Do a fitImage if image just got fit
 * Anyway update overlay
 */
void MuratCanvas2dView::resizeEvent(QResizeEvent *event) {
	QGraphicsView::resizeEvent(event);
    if (inFitImage) {
		fitImage();
    }
	overlayChanged();
}

/*
 * Protected Class Method override QWidget method
 *
 * Do a fitImage
 */
void MuratCanvas2dView::showEvent(QShowEvent * event) {
	QGraphicsView::showEvent(event);
	//fitImage();
}

/*
 * Protected Class Method
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Deactivated
 *
 * Detect in which area of the view the cursor is
 *      -> (top, bottom, left, right, middle)
 *      -> corner are just like being top and left.
 * Launch timer if needed and stop if needed
 */
void MuratCanvas2dView::updatePaddingArea(qreal x, qreal y) {
	/*// Define band ratio and band offset
	 const qreal bandRatio = 0.1;
	 const qreal bandOffset = 200;
	 // Define increment
	 const qreal moveDelta = ratio * 50;

	 // Get view left, right, top, bottom
	 int left = this->viewport()->x();
	 int right = left + this->viewport()->width();
	 int top = viewport()->y();
	 int bottom = top + viewport()->height();

	 // Get limits to defines area where we initiate scrolling
	 int xLimitLeft = left + std::min(bandRatio * (right-left), bandOffset);
	 int xLimitRight = right - std::min(bandRatio * (right-left), bandOffset);
	 int yLimitTop = top + std::min(bandRatio * (bottom-top), bandOffset);
	 int yLimitBottom = bottom - std::min(bandRatio * (bottom-top), bandOffset);

	 // Debug values defined above
	 qDebug() << "View Mode Event Mouse " << x << y;
	 qDebug() << "View Move Event Rect " << left << right << top << bottom;
	 qDebug() << "View Move Event Limits " << xLimitLeft << xLimitRight << yLimitTop << yLimitBottom;

	 // Check x axis
	 if (x < xLimitLeft) {
	 qDebug() << "View Move Event Translate x left " << -moveDelta;
	 xDelta = -moveDelta;
	 } else if (x > xLimitRight) {
	 qDebug() << "View Move Event Translate x right " << moveDelta;
	 xDelta = moveDelta;
	 } else {
	 xRatio = 0;
	 xDelta = 0;
	 }

	 // Define y axis
	 if (y < yLimitTop) {
	 qDebug() << "View Move Event Translate y top " << -moveDelta;
	 yDelta = -moveDelta;
	 } else if (y > yLimitBottom) {
	 qDebug() << "View Move Event Translate y bottom " << moveDelta;
	 yDelta = moveDelta;
	 } else {
	 yRatio = 0;
	 yDelta = 0;
	 }

	 // Check if there is scrolling to be done
	 if (xDelta==0 && yDelta==0 && timer->isActive()) {
	 emit this->stopTimer();
	 } else if ( (xDelta!=0 || yDelta!=0) && !timer->isActive()) {
	 emit this->startTimer();
	 }*/
}

/*
 * Protected Method
 *
 * Manage acceleration of scrolling when near edges
 */
void MuratCanvas2dView::updatePaddingAcceleration() {
	if (timer->isActive()) {
		int x = target_viewport_pos.x();
		int y = target_viewport_pos.y();

		// Define band ratio and band offset
		const qreal bandRatio = 0.1;
		const qreal bandOffset = 200;

		// Get view left, right, top, bottom
		int left = this->viewport()->x();
		int right = left + this->viewport()->width();
		int top = viewport()->y();
		int bottom = top + viewport()->height();

		// Get limits to defines area where we initiate scrolling
		int xLimitLeft = left
				+ std::min(bandRatio * (right - left), bandOffset);
		int xLimitRight = right
				- std::min(bandRatio * (right - left), bandOffset);
		int yLimitTop = top + std::min(bandRatio * (bottom - top), bandOffset);
		int yLimitBottom = bottom
				- std::min(bandRatio * (bottom - top), bandOffset);

		// Check x axis
		if (x < xLimitLeft || x > xLimitRight) {
			xRatio += xRatioStep;
			if (xRatio > 1.0) {
				xRatio = 1.0;
			}
		} else {
			xRatio = 0;
		}

		// Define y axis
		if (y < yLimitTop || y > yLimitBottom) {
			yRatio += yRatioStep;
			if (yRatio > 1.0) {
				yRatio = 1.0;
			}
		} else {
			yRatio = 0;
		}
	} else {
		xRatio = 0;
		yRatio = 0;
	}
}

/*
 * Public method
 *
 * Do zoom taking into account previous wheel event, centering on mouse position
 */
void MuratCanvas2dView::gentle_zoom(double factorX, double factorY) {
	//qDebug() << "EnhancedGraphicsView gentle_zoom";
	this->centerOn(target_scene_pos);
	QPointF delta_viewport_pos = target_viewport_pos
			- QPointF(this->viewport()->width() / 2.0,
					this->viewport()->height() / 2.0);
	QPointF viewport_center = this->mapFromScene(target_scene_pos)
			- delta_viewport_pos;
	QPointF pos = this->mapToScene(viewport_center.toPoint());
	this->centerOn(pos);

    this->scale(factorX, factorY);
	//simulate_gentle_zoom(factor, target_scene_pos, target_viewport_pos);
    emit zoomed(factorX, factorY, pos);
	QPointF tmp = mapToScene(mapFromGlobal(QCursor::pos()));
	emit mousePosition(mapToScene(mapFromGlobal(QCursor::pos())));
}

/*
 * Public method
 *
 * Simulate gentle zoom but without centering on mouse position
 */
void MuratCanvas2dView::simulate_gentle_zoom(double factorX, double factorY) {
    this->scale(factorX, factorY);
}

/*
 * Public method
 *
 * Do zoom taking into account previous wheel event, centering on position
 */
void MuratCanvas2dView::simulate_gentle_zoom(double factorX, double factorY,
		QPointF target_scene_pos) {
	//qDebug() << "EnhancedGraphicsView gentle_zoom";
	this->centerOn(target_scene_pos);
    this->scale(factorX, factorY);
}

void MuratCanvas2dView::simulate_gentle_move(QPointF pos) {
	getCanvas()->simulateUpdateCurvesAndCursor(pos);
}

void MuratCanvas2dView::simulate_gentle_y_move(qreal yPos) {

	getCanvas()->simulateUpdateCurvesAndCursor(QPointF(0, yPos));
}

/*
 * Public method
 *
 * Define modifiers for wheel zoom
 */
void MuratCanvas2dView::set_modifiers(Qt::KeyboardModifiers modifiers) {
	_modifiers = modifiers;

}

/*
 * Public method
 *
 * Define scale base factor for wheel zoom
 */
void MuratCanvas2dView::set_zoom_factor_base(double value) {
	_zoom_factor_base = value;
}

/*
 * Private slot
 *
 * Update view size and position
 */
void MuratCanvas2dView::overlayChanged() {
    int wImage = getCanvas()->imageSize().width();
	int hImage = getCanvas()->imageSize().height();

	// Image has no rows or columns
	if (!wImage || !hImage) {
		return;
	}

	int hViewPortImage = 100;
	//qDebug() << hViewPortImage << wImage << hImage;
	int wViewPortImage = (hViewPortImage * wImage) / hImage;

	// Use itemsBoundingRect rather than sceneRect as sceneRect include all the reached area even after
	// a setSceneRect(itemsBoundingRect)
	// itemsBoundingRect only include area with current item location and size.
	QRectF miniRectscene = getCanvas()->itemsBoundingRect();
	int hViewPortGlobal = miniRectscene.height() / hImage * hViewPortImage;
	int wViewPortGlobal = (hViewPortGlobal * miniRectscene.width())
			/ miniRectscene.height();

	int securityOffset = 20;

	previewView->setFixedSize(wViewPortGlobal + securityOffset,
			hViewPortGlobal + securityOffset);
    rectImage->setConversionBetweenCanvas((qreal) hViewPortImage / hImage);

	int wViewInScene = (int) std::floor(
            std::max(1, viewport()->size().width()) / ratioX);
	int hViewInScene = (int) std::floor(
            std::max(1, viewport()->size().height()) / ratioX);
	int wViewPaint, hViewPaint, wImagePaint, hImagePaint;

	QPoint offset = mapFromScene(0, 0);
    int xViewOffset = -offset.x() / ratioX;
    int yViewOffset = -offset.y() / ratioX;

	//qDebug() << "EnhancedGraphicsView overlayChanged" << xViewOffset << yViewOffset;

	int xImageOffsetPaint, yImageOffsetPaint, xViewOffsetPaint,
			yViewOffsetPaint;

	{
		wImagePaint = (int) std::floor(std::max(1, wViewPortImage));
		wViewPaint = (int) std::floor(
				std::max(1.0, (double) wViewPortImage * wViewInScene / wImage));
		xImageOffsetPaint = (getCanvas()->sceneRect().x()
				- getCanvas()->imageRect().x()) * wViewPortImage / wImage;
		xViewOffsetPaint = xImageOffsetPaint
				+ (int) std::floor(
						std::max(1.0,
								(double) xViewOffset / wImage
										* wViewPortImage));
	}

	{
		hImagePaint = (int) std::floor(std::max(1, hViewPortImage));
		hViewPaint = (int) std::floor(
				std::max(1.0, (double) hViewPortImage * hViewInScene / hImage));
		yImageOffsetPaint = (getCanvas()->sceneRect().y()
				- getCanvas()->imageRect().y()) * hViewPortImage / hImage;
		yViewOffsetPaint = yImageOffsetPaint
				+ (int) std::floor(
						std::max(1.0,
								(double) yViewOffset / hImage
										* hViewPortImage));
	}

	rectImage->setRectHeight(hImagePaint);
	rectImage->setRectWidth(wImagePaint);
	rectView->setRectHeight(hViewPaint);
	rectView->setRectWidth(wViewPaint);

	rectImage->setPos(xImageOffsetPaint, yImageOffsetPaint);
	rectView->setPos(xViewOffsetPaint, yViewOffsetPaint);

	QRect maxRect = QRect(
			getCanvas()->sceneRect().x() / wImage * wViewPortImage
					- securityOffset / 4,
			getCanvas()->sceneRect().y() / hImage * hViewPortImage
					- securityOffset / 4, wViewPortGlobal + securityOffset / 2,
            hViewPortGlobal + securityOffset / 2);
	/*QRect maxRect = QRect(QPoint( std::min(rectImage->boundingRect().x(), rectView->boundingRect().x()),
	 std::min(rectImage->boundingRect().y(), rectView->boundingRect().y()) ),
	 QPoint( std::max(rectImage->boundingRect().x()+rectImage->boundingRect().width(), rectView->boundingRect().x()+rectView->boundingRect().width()),
	 std::max(rectImage->boundingRect().y()+rectImage->boundingRect().height(), rectView->boundingRect().y()+rectView->boundingRect().height()) ) );*/
    previewView->centerOn(rectImage);
	previewScene->setSceneRect(maxRect);
	if (!getCanvas()->getImage().isNull()) {
		this->setSceneRect(getCanvas()->getImage().rect());
	}

	if (hViewInScene >= hImage && wViewInScene >= wImage || overlayHidden) {
		previewView->hide();
	} else {
        //previewView->show();
        previewView->hide();
	}
    update();
}

/*
 * Private slot
 *
 * Apply changes on preview to canvas
 */
void MuratCanvas2dView::previewMoved(RectangleMovable *rectangle, QPointF pos,
		QPointF lastPos) {
	// We do not need the sending rectangle for now
    int wViewPort = rectImage->getRectWidth();
	int hViewPort = rectImage->getRectHeight();
	int wImage = getCanvas()->imageSize().width();
	int hImage = getCanvas()->imageSize().height();
    qreal xDelta = (pos.x() - lastPos.x()) * wImage * ratioX / wViewPort;
    qreal yDelta = (pos.y() - lastPos.y()) * hImage * ratioY / hViewPort;
	QScrollBar* barH = horizontalScrollBar();
	QScrollBar* barV = verticalScrollBar();

	disconnect(horizontalScrollBar(), SIGNAL(valueChanged(int)), this,
			SLOT(overlayChanged()));
	disconnect(verticalScrollBar(), SIGNAL(valueChanged(int)), this,
			SLOT(overlayChanged()));
	if (!(xDelta == 0 || (xDelta > 0 && barH->maximum() == barH->value())
			|| (xDelta < 0 && barH->minimum() == barH->value()))) {
		barH->setValue(barH->value() + xDelta);
	}
	if (!(yDelta == 0 || (yDelta > 0 && barV->maximum() == barV->value())
			|| (yDelta < 0 && barV->minimum() == barV->value()))) {
		barV->setValue(barV->value() + yDelta);
	}
	connect(horizontalScrollBar(), SIGNAL(valueChanged(int)),
			SLOT(overlayChanged()));
	connect(verticalScrollBar(), SIGNAL(valueChanged(int)),
            SLOT(overlayChanged()));
}

/*
 * Private Slot
 *
 * Apply focus to point in canvas
 */
void MuratCanvas2dView::changeFocusPreviewRectangle(int x, int y) {
    centerOn(x, y);
}

/*
 * Private Slot
 *
 * Apply focus to absisse in canvas
 */
void MuratCanvas2dView::changeFocusPreviewLine(int x) {
    QPointF center = mapToScene(viewport()->rect().center());
    centerOn(x, (int) std::floor(center.y()));
}

int MuratCanvas2dView::getId() {
	return id;
}

int MuratCanvas2dView::nextId = 0;

void MuratCanvas2dView::setPatchSize(int ps) {
	getCanvas()->setPatchSize(ps);
}

int MuratCanvas2dView::getPatchSize() {
	return getCanvas()->getPatchSize();
}

void MuratCanvas2dView::takePatchSlot(int x, int y) {
	emit takePatch(x, y);
}

/*
 * Public Class method
 *
 * Return image Pixmap
 */
const QPixmap MuratCanvas2dView::getImage() {
	return getCanvas()->getImage();
}

bool MuratCanvas2dView::infitImage() {
	return inFitImage;
}

void MuratCanvas2dView::setInFitImage(bool val) {
	inFitImage = val;
}

int MuratCanvas2dView::getCurveOffset() {
	return getCanvas()->getCurveOffset();
}

void MuratCanvas2dView::setCurveOffset(int value) {
	getCanvas()->setCurveOffset(value);
}

int MuratCanvas2dView::getCurveSize() {
	return getCanvas()->getCurveSize();
}

void MuratCanvas2dView::setCurveSize(int value) {
	getCanvas()->setCurveSize(value);
}

void MuratCanvas2dView::setMinMat(qreal val) {
	getCanvas()->setMinMat(val);
}

qreal MuratCanvas2dView::getMinMat() {
	return getCanvas()->getMinMat();
}

void MuratCanvas2dView::setMaxMat(qreal val) {
	getCanvas()->setMaxMat(val);
}

qreal MuratCanvas2dView::getMaxMat() {
	return getCanvas()->getMaxMat();
}

void MuratCanvas2dView::triggerOverlay(bool val) {
	overlayHidden = !val;
}

bool MuratCanvas2dView::getOverlayBool() {
	return !overlayHidden;
}

QImage MuratCanvas2dView::getQImage() {
	return this->getCanvas()->getQImage();
}

void MuratCanvas2dView::simulate_mouse_press_event(QPointF pos) {
	getCanvas()->simulateMousePressEvent(pos);
}

void MuratCanvas2dView::drawRectangles(
		std::vector<std::pair<QRect, QColor>> rects) {
	getCanvas()->drawRectangles(rects);
}

void MuratCanvas2dView::clearOverlayRectangles() {
	getCanvas()->clearOverlayRectangles();
}

std::vector<std::pair<QRect, QColor>> MuratCanvas2dView::getOverlayRectangles() {
	return getCanvas()->getOverlayRectangles();
}

void MuratCanvas2dView::toggleCurve(bool val) {
    getCanvas()->toggleCurve(val);
}

void MuratCanvas2dView::setWindowName(QString str) {
    windowLabel->setText(str);
}

QString MuratCanvas2dView::getWindowName() {
    return windowLabel->text();
}

void MuratCanvas2dView::setImageItemVerticalScale(float value) {
	getCanvas()->setImageItemVerticalScale(value);
}

