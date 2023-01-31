/*
 * Canvas2dFullSync.cpp
 *
 *  Created on: 15 janv. 2018
 *      Author: j0483271
 */

#include "Canvas2dFullSync.h"
#include "MuratCanvas2dFullView.h"
#include "MuratCanvas2dVerticalView.h"
#include "Canvas2dVerticalSync.h"

#include <QDebug>
#include <QScrollBar>

#include <vector>
#include <utility>

Canvas2dFullSync::Canvas2dFullSync(QWidget* parent) : QObject(parent), canvases() {
	// TODO Auto-generated constructor stub
	winSize = 0;
}

Canvas2dFullSync::~Canvas2dFullSync() {
	// TODO Auto-generated destructor stub
}

void Canvas2dFullSync::addCanvas2d(MuratCanvas2dFullView* canvas) {
	canvases.append(canvas);

	if (canvases.length() > 1 ) {
		// before enabling interaction do setup
		MuratCanvas2dFullView* example = canvases[0];

		// Set graphicsScene
        qreal ratioX = example->getRatioX();
        qreal ratioY = example->getRatioY();
        canvas->scale(ratioX, ratioY);

		QPointF _center = example->mapToScene(example->viewport()->rect().center());
		canvas->centerOn(_center);

		// Set viewport
		int hBarValue = example->horizontalScrollBar()->value();
		int vBarValue = example->verticalScrollBar()->value();
		canvas->horizontalScrollBar()->setValue(hBarValue);
		canvas->verticalScrollBar()->setValue(vBarValue);

		// set cursor
		QPointF cursorPos = example->mapToScene(example->mapFromGlobal(QCursor::pos()));
		canvas->simulate_gentle_move(cursorPos);

		canvas->setInFitImage(example->infitImage());

		canvas->setCurveOffset(example->getCurveOffset());
		canvas->setCurveSize(example->getCurveSize());

		canvas->setDisplayCurves(example->getDisplayCurves());
		canvas->setDisplayCursors(example->getDisplayCursors());

		canvas->setPatchSize(example->getPatchSize());

		canvas->drawRectangles(example->getOverlayRectangles());

	} else {
		MuratCanvas2dVerticalView* example=nullptr;
		if (otherSync) {
			example = otherSync->getExample();
		}
		if (otherSync && example) {
			MuratCanvas2dVerticalView* example = otherSync->getExample();
            qreal ratioX = example->getRatioX();
            qreal ratioY = example->getRatioY();
            canvas->scale(ratioX, ratioY);

			QPointF _center = example->mapToScene(canvas->viewport()->rect().center().x(), example->viewport()->rect().center().y());
			canvas->centerOn(_center);


			// Set viewport
			int vBarValue = example->verticalScrollBar()->value();
			canvas->verticalScrollBar()->setValue(vBarValue);

			// set cursor
			QPointF cursorPos = example->mapToScene(example->mapFromGlobal(QCursor::pos()));
			canvas->simulate_gentle_y_move(cursorPos.y());

			canvas->setInFitImage(example->infitImage());

			canvas->setCurveOffset(example->getCurveOffset());
			canvas->setCurveSize(example->getCurveSize());


		} else {
			canvas->fitImage();
		}

		canvas->setPatchSize(winSize);

		connect(canvas, &MuratCanvas2dFullView::signalFitImage, this, [this, canvas]() {
            qreal ratioX = canvas->getRatioX();
            qreal ratioY = canvas->getRatioY();
            emit signalFitImage(ratioX, ratioY);
		});

	}

	// Connect canvas to enable interaction
	connect(canvas, &MuratCanvas2dFullView::zoomed,
            this, [this, canvas](double factorX, double factorY, QPointF target_scene_pos){
        simulate_gentle_zoom(factorX, factorY, target_scene_pos, canvas);
	});
	connect(canvas, &MuratCanvas2dFullView::mousePosition, this, [this, canvas](QPointF pos) {
		simulate_gentle_move(pos, canvas);
	});

	connect(canvas->horizontalScrollBar(), &QScrollBar::valueChanged, this, [this, canvas](int value) {
		simulate_scrollbar_value_changed(value, Qt::Horizontal, canvas);
	});

	connect(canvas->verticalScrollBar(), &QScrollBar::valueChanged, this, [this, canvas](int value) {
		simulate_scrollbar_value_changed(value, Qt::Vertical, canvas);
	});

	connect(canvas, &MuratCanvas2dFullView::takePatch, this, [this, canvas](int x, int y) {
		QPointF pos(x,y);
		simulate_mouse_press_event(pos, canvas);
	});

	connect(canvas, &MuratCanvas2dFullView::takePatch, this, [this, canvas](int x, int y) {
		emit takePatch(x, y);
	});
}

void Canvas2dFullSync::remove(MuratCanvas2dFullView* canvas) {
	canvases.removeOne(canvas);
}

MuratCanvas2dFullView* Canvas2dFullSync::getExample() {
	if (canvases.length()>0) {
		return canvases[0];
	} else {
		return nullptr;
	}
}

void Canvas2dFullSync::setOther(Canvas2dVerticalSync* otherSync) {
	this->otherSync = otherSync;
}

void Canvas2dFullSync::fitImage() {
	for (MuratCanvas2dFullView* e : canvases) {
		e->fitImage();
	}
    qreal ratioX;
    qreal ratioY;
	if (canvases.length()>0) {
        ratioX = canvases[0]->getRatioX();
        ratioY = canvases[0]->getRatioY();
	} else {
        ratioX = 1;
        ratioY = 1;
	}
    emit signalFitImage(ratioX, ratioY);
}

void Canvas2dFullSync::zoomIn() {
	for (MuratCanvas2dFullView* e : canvases) {
		e->zoomIn();
	}
	emit signalZoomIn();
}

void Canvas2dFullSync::zoomOut() {
	for( MuratCanvas2dFullView* e : canvases) {
		e->zoomOut();
	}
	emit signalZoomOut();
}

void Canvas2dFullSync::simulate_gentle_zoom(double factorX, double factorY, QPointF target_scene_pos, MuratCanvas2dFullView* canvas) {
	for (MuratCanvas2dFullView* e : canvases) {
		if (e != canvas) {
            e->simulate_gentle_zoom(factorX, factorY, target_scene_pos);
		}
	}
    emit gentle_zoom(factorX, factorY, target_scene_pos);
}

void Canvas2dFullSync::simulate_gentle_zoom(double factorY, qreal y_target_scene_pos, MuratCanvas2dFullView* canvas) {
	for (MuratCanvas2dFullView* e : canvases) {
		if (e != canvas) {
            e->simulate_gentle_y_zoom(factorY, y_target_scene_pos);
		}
	}
}

void Canvas2dFullSync::simulate_gentle_move(QPointF pos, MuratCanvas2dFullView* canvas) {
	for (MuratCanvas2dFullView* e : canvases) {
		if (e != canvas) {
			e->simulate_gentle_move(pos);
		}
	}
	emit gentle_move(pos);
}

void Canvas2dFullSync::simulate_gentle_move(qreal y_target_scene_pos) {
	for (MuratCanvas2dFullView* e : canvases) {
		e->simulate_gentle_y_move(y_target_scene_pos);
	}
}

void Canvas2dFullSync::simulate_scrollbar_value_changed(int value, Qt::Orientation orientation, MuratCanvas2dFullView* canvas) {
	for (MuratCanvas2dFullView* e : canvases) {
		if (e != canvas) {
			if (orientation == Qt::Horizontal) {
				e->horizontalScrollBar()->setValue(value);
			} else {
				e->verticalScrollBar()->setValue(value);
			}
		}
	}
	if (canvas) {
		emit scrollbar_value_changed(value, orientation);
	}
}

void Canvas2dFullSync::simulate_mouse_press_event(QPointF pos, MuratCanvas2dFullView* canvas) {
	for (MuratCanvas2dFullView* e : canvases) {
		if (e != canvas) {
			e->simulate_mouse_press_event(pos);
		}
	}
}

void Canvas2dFullSync::setCurveOffset(int value) {
	for (MuratCanvas2dFullView* e : canvases) {
		e->setCurveOffset(value);
	}
}

void Canvas2dFullSync::setCurveSize(int value) {
	for (MuratCanvas2dFullView* e : canvases) {
		e->setCurveSize(value);
	}
}

int Canvas2dFullSync::getCurveOffset() {
	if (canvases.length()>0) {
		return canvases[0]->getCurveOffset();
	} else {
		return 0;
	}
}

int Canvas2dFullSync::getCurveSize() {
	if (canvases.length()>0) {
		return canvases[0]->getCurveSize();
	} else {
		return 0;
	}
}
bool Canvas2dFullSync::getDisplayCurves() {
	if (canvases.length()>0) {
		return canvases[0]->getDisplayCurves();
	} else {
		return 0;
	}
}

void Canvas2dFullSync::setDisplayCurves(bool value) {
	for (MuratCanvas2dFullView* e : canvases) {
		e->setDisplayCurves(value);
	}
}

bool Canvas2dFullSync::getDisplayCursors(){
	if (canvases.length()>0) {
		return canvases[0]->getDisplayCursors();
	} else {
		return 0;
	}
}
void Canvas2dFullSync::setDisplayCursors(bool value) {
	for (MuratCanvas2dFullView* e : canvases) {
		e->setDisplayCursors(value);
	}
}

int Canvas2dFullSync::getPatchSize() {
	if (canvases.length()>0) {
		return canvases[0]->getPatchSize();
	} else {
		return winSize;
	}
}

void Canvas2dFullSync::setPatchSize(int winSize) {
	this->winSize = winSize;
	for (MuratCanvas2dFullView* e : canvases) {
		e->setPatchSize(winSize);
	}
}

void Canvas2dFullSync::drawRectangles(std::vector<std::pair<QRect, QColor>> rects) {
	for (MuratCanvas2dFullView* e : canvases) {
		e->drawRectangles(rects);
	}
}

void Canvas2dFullSync::clearOverlayRectangles() {
	for (MuratCanvas2dFullView* e : canvases) {
		e->clearOverlayRectangles();
	}
}

void Canvas2dFullSync::centerOn(QPointF pos) {
	for (MuratCanvas2dFullView* e : canvases) {
		e->centerOn(pos);
	}
}
