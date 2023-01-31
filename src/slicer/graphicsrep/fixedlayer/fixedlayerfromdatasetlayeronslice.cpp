#include "fixedlayerfromdatasetlayeronslice.h"

#include <QGraphicsScene>
#include "fixedlayerfromdatasetreponslice.h"
#include "cudaimagepaletteholder.h"
#include "fixedlayerfromdataset.h"
#include "curve.h"
#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"

FixedLayerFromDatasetLayerOnSlice::FixedLayerFromDatasetLayerOnSlice(FixedLayerFromDatasetRepOnSlice *rep,
		SliceDirection dir, const IGeorefImage *const transfoProvider, int startValue,
		QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth), m_transfoProvider(transfoProvider) {
	m_rep = rep;
	m_dir = dir;
	m_slicePosition = startValue;

	m_curve = new Curve(m_scene, parent);
	QPen pen = m_curve->getPen();
	pen.setColor(Qt::red);
	m_curve->setPen(pen);
	m_curve->setZValue(m_defaultZDepth);

	// apply transform
	QTransform mainTransform;
	if (m_dir==SliceDirection::Inline) {
		mainTransform = QTransform(dynamic_cast<Seismic3DAbstractDataset*>(m_rep->fixedLayer()->dataset())->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
	} else {
		mainTransform = QTransform(dynamic_cast<Seismic3DAbstractDataset*>(m_rep->fixedLayer()->dataset())->ijToInlineXlineTransfoForXline()->imageToWorldTransformation().toTransform());
	}
	m_curve->setTransform(mainTransform);
	setSliceIJPosition(m_slicePosition);
	refresh();
	show();

	connect(m_rep->fixedLayer(), &FixedLayerFromDataset::propertyModified, this, &FixedLayerFromDatasetLayerOnSlice::recomputeCurve);
}

FixedLayerFromDatasetLayerOnSlice::~FixedLayerFromDatasetLayerOnSlice() {
	delete m_curve;
}

void FixedLayerFromDatasetLayerOnSlice::setSliceIJPosition(int imageVal) {
	m_slicePosition = imageVal;
	QList<QPolygon> polygons;
	CUDAImagePaletteHolder* holder = m_rep->fixedLayer()->image(FixedLayerFromDataset::ISOCHRONE);
	if (holder) {
		std::size_t dimJ = m_rep->fixedLayer()->dataset()->height();
		std::size_t holderWidth = holder->width();
		float noValue = -9999.0;
		float pasech = m_rep->fixedLayer()->getStepSample();
		float tdeb = m_rep->fixedLayer()->getOriginSample();

		std::size_t dimI;
		if (SliceDirection::Inline==m_dir) {
			dimI = m_rep->fixedLayer()->dataset()->width();
		} else {
			dimI = m_rep->fixedLayer()->dataset()->depth();
		}

		holder->lockPointer();
		float* tab = static_cast<float*>(holder->backingPointer());
		QPolygon currentPolygon;
		for (std::size_t i=0; i<dimI; i++) {
			float val;
			if (m_dir==SliceDirection::Inline) {
				val = tab[holderWidth*m_slicePosition+i];
			} else {
				val = tab[holderWidth*i+m_slicePosition];
			}
			if (val!=noValue) {
				currentPolygon << QPoint(i, (val-tdeb)/pasech);
			} else if (val==noValue && currentPolygon.size()>0) {
				polygons << currentPolygon;
				currentPolygon = QPolygon();
			}
		}
		holder->unlockPointer();
		if (currentPolygon.size()>0) {
			polygons << currentPolygon;
		}

	}
	m_curve->setPolygons(polygons);
}

void FixedLayerFromDatasetLayerOnSlice::recomputeCurve() {
	setSliceIJPosition(m_slicePosition);
}

void FixedLayerFromDatasetLayerOnSlice::show() {
	m_isShown = true;
	m_curve->addToScene();
}
void FixedLayerFromDatasetLayerOnSlice::hide() {
	m_isShown = false;
	m_curve->removeFromScene();
}

QRectF FixedLayerFromDatasetLayerOnSlice::boundingRect() const {
	return m_curve->boundingRect();
}

void FixedLayerFromDatasetLayerOnSlice::refresh() {

	m_curve->redraw();
	if (m_isShown) {
		show();
	} else {
		hide();
	}
}

void FixedLayerFromDatasetLayerOnSlice::setPenColor(QColor color) {
	QPen pen = m_curve->getPen();
	pen.setColor(color);
	m_curve->setPen(pen);
}
