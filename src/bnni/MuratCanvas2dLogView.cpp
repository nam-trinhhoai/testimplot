/*
 * MuratCanvas2dLogView.cpp
 *
 *  Created on: 26 janv. 2018
 *      Author: j0483271
 */

#include "MuratCanvas2dLogView.h"
#include "MuratCanvas2dVerticalView.h"
#include "MuratCanvas2dScene.h"

#include <QPixmap>

MuratCanvas2dLogView::MuratCanvas2dLogView(QWidget* parent) : MuratCanvas2dVerticalView(parent) {
    // TODO Auto-generated constructor stub
}

MuratCanvas2dLogView::~MuratCanvas2dLogView() {
	// TODO Auto-generated destructor stub
}

void MuratCanvas2dLogView::initScene() {
	MuratCanvas2dVerticalView::initScene();
    getCanvas()->setVerticalCursor(true);
	getCanvas()->setVerticalCurve(false);
	getCanvas()->setHorizontalCursor(true);
	getCanvas()->setHorizontalCurve(false);
}

MuratCanvas2dScene* MuratCanvas2dLogView::getCanvas() {
	return canvas;
}

void MuratCanvas2dLogView::setImage(QImage img, Matrix2DInterface* mat) {
	MuratCanvas2dVerticalView::setImage(img, mat);
	getCanvas()->updateCurvesAndCursor(QPointF(0,0));

    connect(getCanvas(), &MuratCanvas2dScene::linesChanged, this, [this](const std::vector<double>& lines) {
        emit linesChanged(lines);
    });
}

std::vector<double> MuratCanvas2dLogView::getLines() {
    return getCanvas()->getLines();
}

void MuratCanvas2dLogView::setLines(const std::vector<double>& lines) {
    getCanvas()->setLines(lines);
}

void MuratCanvas2dLogView::scale(qreal sx, qreal sy) {
    if (sx==1) {
        MuratCanvas2dVerticalView::scale(sx, sy);
    }
}
