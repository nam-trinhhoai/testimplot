/*
 * MuratCanvas2dVerticalView.cpp
 *
 *  Created on: 16 janv. 2018
 *      Author: j0483271
 */

#include "MuratCanvas2dVerticalView.h"
#include "MuratCanvas2dScene.h"

#include <cmath>
#include <QMouseEvent>
#include <QScrollBar>
#include <qmath.h>
#include <QDebug>

MuratCanvas2dVerticalView::MuratCanvas2dVerticalView(QWidget* parent) : MuratCanvas2dView(parent) {
	// TODO Auto-generated constructor stub
	xRatio = 1;
	yRatio = 1;

}

MuratCanvas2dVerticalView::~MuratCanvas2dVerticalView() {
	// TODO Auto-generated destructor stub
}

void MuratCanvas2dVerticalView::initScene() {
	MuratCanvas2dView::initScene();
	initImageMode();
	triggerOverlay(false);
	getCanvas()->setSceneRect(0, 0, 256, 256);
}

void MuratCanvas2dVerticalView::initImageMode() {
	getCanvas()->setVerticalCursor(false);
	getCanvas()->setVerticalCurve(false);
	getCanvas()->setHorizontalCursor(true);
	getCanvas()->setHorizontalCurve(false);
}

void MuratCanvas2dVerticalView::initLogMode() {
	getCanvas()->setVerticalCursor(false);
	getCanvas()->setVerticalCurve(true);
	getCanvas()->setHorizontalCursor(true);
	getCanvas()->setHorizontalCurve(false);
}

MuratCanvas2dScene* MuratCanvas2dVerticalView::getCanvas() {
	return canvas;
}

void MuratCanvas2dVerticalView::scale(qreal sx, qreal sy) {
	if (sx!=sy) {

		qreal limitSup = 1000;

		QSize imgSize = getCanvas()->imageSize();
		double minH_W = std::min(imgSize.height(), imgSize.width());
        qreal limitInf = std::min(0.0, 20/minH_W);
        double lastXRatio = xRatio;
		if (xRatio*sx<limitSup && xRatio*sx>limitInf) {
			xRatio = xRatio * sx;
		}

        double lastYRatio = yRatio;
		if (yRatio*sy<limitSup && yRatio*sy>limitInf) {
			yRatio = yRatio * sy;
            getCanvas()->updateCursor(1, sy);
		}

        QGraphicsView::scale(xRatio/lastXRatio, yRatio/lastYRatio);
	} else {
        MuratCanvas2dView::scale(sx,sy);
	}
}

/*
 * Slot that scale * 2.0 in y axis
 */
void MuratCanvas2dVerticalView::verticalZoomIn() {
	scale(1.0,2.0);
}

/*
 * Slot that scale / 2.0 in y axis
 */
void MuratCanvas2dVerticalView::verticalZoomOut() {
	scale(1.0,0.5);
}

void MuratCanvas2dVerticalView::setYRatio(qreal s) {
    scale(1, s / (yRatio*ratioY));
}

void MuratCanvas2dVerticalView::setXRatio(qreal s) {
    scale(s / (xRatio*ratioX), 1);
}

qreal MuratCanvas2dVerticalView::getYRatio() {
    return yRatio*ratioY;
}

void MuratCanvas2dVerticalView::simulate_gentle_y_move(qreal yPos) {
	MuratCanvas2dView::simulate_gentle_move(QPointF(0, yPos));
}

/*
 * Public method
 *
 * Do zoom taking into account previous wheel event, centering on mouse position
 */
void MuratCanvas2dVerticalView::gentle_zoom(double factor) {
	QPointF _center = mapToScene(viewport()->width(), viewport()->height());
	this->centerOn(_center.x(), y_target_scene_pos);
	QPointF delta_viewport_pos = QPointF(this->viewport()->width() / 2.0, y_target_viewport_pos) - QPointF(this->viewport()->width() / 2.0,
	                                                             this->viewport()->height() / 2.0);
	QPointF viewport_center = this->mapFromScene(_center.x(), y_target_scene_pos) - delta_viewport_pos;
	QPointF pos = this->mapToScene(viewport_center.toPoint());
    this->centerOn(pos);

    this->scale(1, factor);
    //simulate_gentle_zoom(factor, target_scene_pos, target_viewport_pos);
    emit zoomed(1, factor, pos);
    emit mousePosition(mapToScene(mapFromGlobal(QCursor::pos())));
}

/*
 * Public method
 *
 * Do zoom taking into account previous wheel event, centering on position
 */
void MuratCanvas2dVerticalView::simulate_gentle_zoom(double factor, qreal y_target_scene_pos) {
	QPointF _center = mapToScene(viewport()->width() / 2.0, viewport()->height() / 2.0);
	this->centerOn(_center.x(), y_target_scene_pos);
	this->scale(1, factor);
}

/*
 * Protected Class Method override QGraphicsView method
 *
 * Redefine behavior of mother class
 *
 * Called when mouse move
 * Catch mouse mosition and launch updatePaddingArea
 */
void MuratCanvas2dVerticalView::mouseMoveEventWheelManager(QMouseEvent *event) {
    int y = event->pos().y();

    qreal delta = y_target_viewport_pos - y;
    if (qAbs(delta) > 5 || delta!=delta) {
      y_target_viewport_pos = y;
      y_target_scene_pos = this->mapToScene(event->pos()).y();
    }
}

void MuratCanvas2dVerticalView::setImage(QImage img, Matrix2DInterface* mat) {
	MuratCanvas2dView::setImage(img, mat);
	if (img.width()==1) {
		initLogMode();
	} else {
		initImageMode();
	}

}

