#include "Canvas2dVerticalSync.h"

#include "Canvas2dVerticalSync.h"
#include "MuratCanvas2dVerticalView.h"

#include <QDebug>
#include <QScrollBar>


Canvas2dVerticalSync::Canvas2dVerticalSync(QWidget* parent) : QObject(parent), canvases() {
    // TODO Auto-generated constructor stub
}

Canvas2dVerticalSync::~Canvas2dVerticalSync() {
    // TODO Auto-generated destructor stub
}

void Canvas2dVerticalSync::addCanvas2d(MuratCanvas2dVerticalView* canvas) {
    canvases.append(canvas);

    if (canvases.length() > 1 ) {
        // before enabling interaction do setup
        MuratCanvas2dView* example = canvases[0];

        // Set graphicsScene
        qreal ratioX = example->getRatioX();
        qreal ratioY = example->getRatioY();
        canvas->scale(ratioX, ratioY);

        QPointF _center = example->mapToScene(example->viewport()->rect().center());
        canvas->centerOn(_center);

        // Set viewport
        int vBarValue = example->verticalScrollBar()->value();
        canvas->verticalScrollBar()->setValue(vBarValue);

        // set cursor
        QPointF cursorPos = example->mapToScene(example->mapFromGlobal(QCursor::pos()));
        canvas->simulate_gentle_move(cursorPos);

        canvas->setInFitImage(example->infitImage());

        canvas->setCurveOffset(example->getCurveOffset());
        canvas->setCurveSize(example->getCurveSize());


    } else {
        canvas->fitImage();
    }

    // Connect canvas to enable interaction
    connect(canvas, &MuratCanvas2dView::zoomed,
            this, [this, canvas](double factorX, double factorY, QPointF target_scene_pos){
        simulate_gentle_zoom(factorY, target_scene_pos.y(), canvas);
    });

    connect(canvas->verticalScrollBar(), &QScrollBar::valueChanged, this, [this, canvas](int value) {
        simulate_scrollbar_value_changed(value, Qt::Vertical, canvas);
    });

    connect(canvas, &MuratCanvas2dView::mousePosition, this, [this, canvas](QPointF pos) {
        simulate_gentle_move(pos.y(), canvas);
    });


}

void Canvas2dVerticalSync::forceUpdate() {
    if (canvases.length()>0) {
        MuratCanvas2dVerticalView* canvas = canvases[0];


        canvas->fitImage();
        for (int i=1; i<canvases.length(); i++) {
            MuratCanvas2dVerticalView* examplebis = canvases[0];
            canvas = canvases[i];
            // Set graphicsScene
            qreal ratioX = examplebis->getRatioX();
            qreal ratioY = examplebis->getRatioY();

            canvas->setRatioX(ratioX);
            canvas->setRatioY(ratioY);

            qreal yRatio = examplebis->getYRatio();
            canvas->setYRatio(yRatio);

            canvas->setXRatio(1);


            QPointF _center = examplebis->mapToScene(examplebis->viewport()->rect().center());
            canvas->centerOn(_center);

            // Set viewport
            int vBarValue = examplebis->verticalScrollBar()->value();
            canvas->verticalScrollBar()->setValue(vBarValue);

            // set cursor
            QPointF cursorPos = examplebis->mapToScene(examplebis->mapFromGlobal(QCursor::pos()));
            canvas->simulate_gentle_move(cursorPos);

            canvas->setInFitImage(examplebis->infitImage());
            canvas->setCurveOffset(examplebis->getCurveOffset());
            canvas->setCurveSize(examplebis->getCurveSize());
        }

    }
}

void Canvas2dVerticalSync::remove(MuratCanvas2dVerticalView* canvas) {
    canvases.removeOne(canvas);
}

MuratCanvas2dVerticalView* Canvas2dVerticalSync::getExample() {
    if (canvases.length()>0) {
        return canvases[0];
    } else {
        return nullptr;
    }
}

void Canvas2dVerticalSync::fitImage(qreal ratio) {
    for (MuratCanvas2dVerticalView* e : canvases) {
        e->setYRatio(ratio);
        e->setXRatio(1);
    }
    emit signalFitImage();
}

void Canvas2dVerticalSync::zoomIn() {
    for (MuratCanvas2dVerticalView* e : canvases) {
        e->verticalZoomIn();
    }
    emit signalZoomIn();
}

void Canvas2dVerticalSync::zoomOut() {
    for( MuratCanvas2dVerticalView* e : canvases) {
        e->verticalZoomOut();
    }
    emit signalZoomOut();
}

void Canvas2dVerticalSync::simulate_gentle_zoom(double factor, qreal y_target_scene_pos, MuratCanvas2dVerticalView* canvas) {
    for (MuratCanvas2dVerticalView* e : canvases) {
        if (e != canvas) {
            e->simulate_gentle_zoom(factor, y_target_scene_pos);
        }
    }
    if (canvas) {
        emit gentle_zoom(factor, y_target_scene_pos);
    }
}

void Canvas2dVerticalSync::simulate_gentle_move(qreal y_target_scene_pos, MuratCanvas2dVerticalView* canvas) {
    for (MuratCanvas2dVerticalView* e : canvases) {
        if (e != canvas) {
            e->simulate_gentle_y_move(y_target_scene_pos);
        }
    }
    if (canvas) {
        emit gentle_move(y_target_scene_pos);
    }
}

void Canvas2dVerticalSync::simulate_scrollbar_value_changed(int value, Qt::Orientation orientation, MuratCanvas2dVerticalView* canvas) {
    if (orientation == Qt::Vertical) {
        for (MuratCanvas2dView* e : canvases) {
            if (e != canvas) {
                e->verticalScrollBar()->setValue(value);
            }
        }
        if (canvas) {
            emit scrollbar_value_changed(value, orientation);
        }
    }
}


void Canvas2dVerticalSync::setCurveOffset(int value) {
    for (MuratCanvas2dView* e : canvases) {
        e->setCurveOffset(value);
    }
}

void Canvas2dVerticalSync::setCurveSize(int value) {
    for (MuratCanvas2dView* e : canvases) {
        e->setCurveSize(value);
    }
}

int Canvas2dVerticalSync::getCurveOffset() {
    if (canvases.length()>0) {
        return canvases[0]->getCurveOffset();
    } else {
        return 0;
    }
}

int Canvas2dVerticalSync::getCurveSize() {
    if (canvases.length()>0) {
        return canvases[0]->getCurveSize();
    } else {
        return 0;
    }
}

