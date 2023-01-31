/*
 * MuratCanvas2dFullView.cpp
 *
 *  Created on: 16 janv. 2018
 *      Author: j0483271
 */

#include "MuratCanvas2dFullView.h"
#include "MuratCanvas2dScene.h"
#include <QDebug>
#include <QMouseEvent>
#include <Qt>
#include <QMenu>
#include <QAction>

MuratCanvas2dFullView::MuratCanvas2dFullView(QWidget* parent) : MuratCanvas2dView(parent) {
	// TODO Auto-generated constructor stub
	displayCurves = false;
	displayCursors = true;

}

MuratCanvas2dFullView::~MuratCanvas2dFullView() {
	// TODO Auto-generated destructor stub
}

void MuratCanvas2dFullView::initScene() {
	MuratCanvas2dView::initScene();
	getCanvas()->setVerticalCursor(displayCursors);
	getCanvas()->setVerticalCurve(displayCurves);
	getCanvas()->setHorizontalCurve(displayCurves);
	getCanvas()->setHorizontalCursor(displayCursors);
}

MuratCanvas2dScene* MuratCanvas2dFullView::getCanvas() {
	return canvas;
}

/*
 * Protected Class Method override QGraphicsView method
 *
 * Called when mouse move
 * Catch mouse mosition and launch updatePaddingArea
 */
void MuratCanvas2dFullView::mouseMoveEvent(QMouseEvent* event) {
	getCanvas()->setVerticalCursor(displayCursors);

	getCanvas()->setVerticalCurve(displayCurves);
	MuratCanvas2dView::mouseMoveEvent(event);
}

void MuratCanvas2dFullView::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::RightButton) {
        /*QMenu menu;
        //add default menu

        QAction* actionX = menu.addAction("Adjust palette");

        QObject::connect(actionX, &QAction::triggered, [=](bool checked) {
            emit paletteRequestedSignal(this);
        });

        //show the menu
        QPoint pt = event->screenPos().toPoint();
        menu.exec(pt);*/
        QPoint pt = event->screenPos().toPoint();
        requestMenu(pt, this);

    }
}

/*
 * Public method
 *
 * Move arrow to position
 */
void MuratCanvas2dFullView::simulate_gentle_move(QPointF pos) {
	getCanvas()->setVerticalCursor(displayCursors);
	getCanvas()->setVerticalCurve(displayCurves);
	MuratCanvas2dView::simulate_gentle_move(pos);
}

/*
 * Public method
 *
 * Move arrow to position
 */
void MuratCanvas2dFullView::simulate_gentle_y_move(qreal yPos) {
	getCanvas()->setVerticalCursor(false);
	getCanvas()->setVerticalCurve(false);
	MuratCanvas2dView::simulate_gentle_y_move(yPos);
	int x  = mapToScene(viewport()->width()/2, viewport()->height()/2).x();
	if (qAbs(y_move_center - yPos) > 5) {
		x_move_center = x;
		qDebug() << "New x move center : " << x_move_center;
		y_move_center = yPos;
	}
	qDebug() << "y move center : " << y_move_center;
}


void MuratCanvas2dFullView::simulate_gentle_y_zoom(double factorY, qreal y_target_scene_pos) {
	QPointF pos;
	QPointF _center = mapToScene(viewport()->width()/2, viewport()->height()/2);
	pos.setY(y_target_scene_pos);
	//int x = _center.x();
	int x = x_move_center;
	if (x<getImageRect().left()) {
		x = getImageRect().left();
	} else if (x>getImageRect().right()) {
		x = getImageRect().right();
	}
	pos.setX(x);
    simulate_gentle_zoom(1, factorY, pos);
}

bool MuratCanvas2dFullView::getDisplayCurves() {
	return displayCurves;
}

void MuratCanvas2dFullView::setDisplayCurves(bool val) {
	displayCurves = val;

	getCanvas()->setHorizontalCurve(displayCurves);
}

bool MuratCanvas2dFullView::getDisplayCursors() {
	return displayCursors;
}

void MuratCanvas2dFullView::setDisplayCursors(bool val) {
	displayCursors = val;
	getCanvas()->setHorizontalCursor(displayCursors);
}

