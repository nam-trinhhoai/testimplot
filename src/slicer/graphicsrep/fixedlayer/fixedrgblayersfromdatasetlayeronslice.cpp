#include "fixedrgblayersfromdatasetlayeronslice.h"
#include <QGraphicsScene>
#include "fixedrgblayersfromdatasetreponslice.h"
#include "fixedrgblayersfromdataset.h"
#include "qglisolineitem.h"
#include "cudaimagepaletteholder.h"
#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"

FixedRGBLayersFromDatasetLayerOnSlice::FixedRGBLayersFromDatasetLayerOnSlice(FixedRGBLayersFromDatasetRepOnSlice *rep, SliceDirection dir,
		const IGeorefImage *const transfoProvider, int startValue,
		QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth), m_transfoProvider(transfoProvider) {
	m_rep = rep;

//	m_lineItem = new QGLIsolineItem(transfoProvider,
//			m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder(),
//			1, dir, parent);
//	m_lineItem->setZValue(defaultZDepth);
//
//	m_lineItem->updateWindowSize(1);
//	m_lineItem->updateSlice(startValue);
//	m_lineItem->setColor(Qt::red);

	m_curveMain.reset(new Curve(m_scene, parent));
	m_curveMain->setZValue(m_defaultZDepth);
	QPen pen;
	pen.setCosmetic(true);
	pen.setColor(Qt::red);
	m_curveMain->setPen(pen);

	QTransform mainTransform;
	if (m_rep->direction()==SliceDirection::Inline) {
		mainTransform = QTransform(m_rep->fixedRGBLayersFromDataset()->dataset()->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
	} else {
		mainTransform = QTransform(m_rep->fixedRGBLayersFromDataset()->dataset()->ijToInlineXlineTransfoForXline()->imageToWorldTransformation().toTransform());
	}
	m_curveMain->setTransform(mainTransform);

	connect(m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder(), SIGNAL(dataChanged()), this,
			SLOT(refresh()));

	show();
}

FixedRGBLayersFromDatasetLayerOnSlice::~FixedRGBLayersFromDatasetLayerOnSlice() {
}

void FixedRGBLayersFromDatasetLayerOnSlice::setSliceIJPosition(int imageVal) {
	refresh();//m_lineItem->updateSlice(imageVal);
}

void FixedRGBLayersFromDatasetLayerOnSlice::show() {
	//m_scene->addItem(m_lineItem);
	m_curveMain->addToScene();
	refresh();
}
void FixedRGBLayersFromDatasetLayerOnSlice::hide() {
	//m_scene->removeItem(m_lineItem);
	m_curveMain->removeFromScene();
}

QRectF FixedRGBLayersFromDatasetLayerOnSlice::boundingRect() const {
	//return m_lineItem->boundingRect();

	// copied from CUDAImageBuffer worldExtent to give the same result
	double width;
	if (m_rep->direction()==SliceDirection::Inline) {
		width = m_rep->fixedRGBLayersFromDataset()->dataset()->width();
	} else {
		width = m_rep->fixedRGBLayersFromDataset()->dataset()->depth();
	}

	double ij[8] = { 0.0, 0.0, (double) width, 0.0,
			0.0, (double) m_rep->fixedRGBLayersFromDataset()->dataset()->height(),
			(double) width, (double) m_rep->fixedRGBLayersFromDataset()->dataset()->height() };

	double xMin = std::numeric_limits<double>::max();
	double yMin = std::numeric_limits<double>::max();

	double xMax = std::numeric_limits<double>::min();
	double yMax = std::numeric_limits<double>::min();
	double x, y;
	for (int i = 0; i < 4; i++) {
		x = ij[2 * i];
		y = ij[2 * i + 1];

		if (m_rep->direction()==SliceDirection::Inline) {
			m_rep->fixedRGBLayersFromDataset()->dataset()->ijToInlineXlineTransfoForInline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
		} else {
			m_rep->fixedRGBLayersFromDataset()->dataset()->ijToInlineXlineTransfoForXline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
		}

		xMin = std::min(xMin, x);
		yMin = std::min(yMin, y);

		xMax = std::max(xMax, x);
		yMax = std::max(yMax, y);
	}
	QRectF rect(xMin, yMin, xMax - xMin, yMax - yMin);
	return rect;
}

void FixedRGBLayersFromDatasetLayerOnSlice::refresh() {
	//m_lineItem->update():
	// extract polygon
	QPolygon poly;
	long slice = m_rep->currentIJSliceRep();
	long width = m_rep->fixedRGBLayersFromDataset()->dataset()->width();
	long depth = m_rep->fixedRGBLayersFromDataset()->dataset()->depth();
	double tdeb = m_rep->fixedRGBLayersFromDataset()->dataset()->sampleTransformation()->b();
	double pasech = m_rep->fixedRGBLayersFromDataset()->dataset()->sampleTransformation()->a();
	CUDAImagePaletteHolder* isoSurface = m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder();
	isoSurface->lockPointer();
	float* tab = (float*) isoSurface->backingPointer();
	if (m_rep->direction()==SliceDirection::Inline) {
		for (long i=0; i<width; i++) {
			float val = (tab[i+slice*width] - tdeb) / pasech;
			poly << QPoint(i, val);
		}
	} else {
		for (long i=0; i<depth; i++) {
			float val = (tab[slice+i*width] - tdeb) / pasech;
			poly << QPoint(i, val);
		}
	}
	isoSurface->unlockPointer();


	m_curveMain->setPolygon(poly);
}
