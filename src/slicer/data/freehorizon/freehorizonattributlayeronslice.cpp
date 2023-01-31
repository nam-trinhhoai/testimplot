

#include <math.h>
#include <cmath>
#include <limits>
#include <QDebug>
#include <QTransform>

#include <sliceutils.h>
#include "freehorizonattributreponslice.h"
#include <QGraphicsScene>
#include "freehorizonattributlayeronslice.h"
// #include "fixedrgblayersfromdatasetandcube.h"
#include "qglisolineitem.h"
#include "cudaimagepaletteholder.h"
#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "GraphEditor_PolyLineShape.h"

#define TIME_NO_VALUE -9999

FreeHorizonAttributLayerOnSlice::FreeHorizonAttributLayerOnSlice(FreeHorizonAttributRepOnSlice *rep, SliceDirection dir,
		int startValue, QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;

	m_curveMain.reset(new Curve(m_scene, parent));
	m_curveMain->setZValue(m_defaultZDepth);
	QPen pen;
	pen.setCosmetic(true);
	pen.setWidth(3);
	pen.setColor(Qt::white);
	m_curveMain->setPen(pen);


	if (m_rep->direction()==SliceDirection::Inline) {
		// m_mainTransform = QTransform(m_rep->fixedRGBLayersFromDataset()->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
	} else {
		// m_mainTransform = QTransform(m_rep->fixedRGBLayersFromDataset()->ijToInlineXlineTransfoForXline()->imageToWorldTransformation().toTransform());
	}

	// m_mainTransform = QTransform(m_rep->fixedRGBLayersFromDataset()->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
	m_curveMain->setTransform(m_mainTransform);

	// connect(m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder(), SIGNAL(dataChanged()), this, SLOT(refresh()));

	show();
}

FreeHorizonAttributLayerOnSlice::~FreeHorizonAttributLayerOnSlice() {
	qDebug() << "erase";
}

void FreeHorizonAttributLayerOnSlice::setSliceIJPosition(int imageVal) {
	refresh();//m_lineItem->updateSlice(imageVal);
}

void FreeHorizonAttributLayerOnSlice::show() {
	m_curveMain->addToScene();
	// GraphEditor_MultiPolyLineShape *shape = static_cast<GraphEditor_MultiPolyLineShape*>(m_rep->getHorizonShape());
	// if ( shape ) m_scene->addItem(shape);
	refresh();
}
void FreeHorizonAttributLayerOnSlice::hide() {
	m_curveMain->removeFromScene();
	// GraphEditor_MultiPolyLineShape *shape = static_cast<GraphEditor_MultiPolyLineShape*>(m_rep->getHorizonShape());
	// if ( shape ) m_scene->removeItem(shape);
}

QRectF FreeHorizonAttributLayerOnSlice::boundingRect() const {
	/*
	// copied from CUDAImageBuffer worldExtent to give the same result
	double width;
	if (m_rep->direction()==SliceDirection::Inline) {
		width = m_rep->fixedRGBLayersFromDataset()->width();
	} else {
		width = m_rep->fixedRGBLayersFromDataset()->depth();
	}

	double ij[8] = { 0.0, 0.0, (double) width, 0.0,
			0.0, (double) m_rep->fixedRGBLayersFromDataset()->heightFor3D(),
			(double) width, (double) m_rep->fixedRGBLayersFromDataset()->heightFor3D() };

	double xMin = std::numeric_limits<double>::max();
	double yMin = std::numeric_limits<double>::max();

	double xMax = std::numeric_limits<double>::min();
	double yMax = std::numeric_limits<double>::min();
	double x, y;
	for (int i = 0; i < 4; i++) {
		x = ij[2 * i];
		y = ij[2 * i + 1];

		if (m_rep->direction()==SliceDirection::Inline) {
			m_rep->fixedRGBLayersFromDataset()->ijToInlineXlineTransfoForInline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
		} else {
			m_rep->fixedRGBLayersFromDataset()->ijToInlineXlineTransfoForXline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
		}

		xMin = std::min(xMin, x);
		yMin = std::min(yMin, y);

		xMax = std::max(xMax, x);
		yMax = std::max(yMax, y);
	}
	QRectF rect(xMin, yMin, xMax - xMin, yMax - yMin);
	*/
	QRectF rect;
	return rect;
}

void FreeHorizonAttributLayerOnSlice::refresh() {
	fprintf(stderr, "refresh\n");
	// extract polygon
	/*
	QPolygon poly;
	long slice = m_rep->currentIJSliceRep();
	long width = m_rep->fixedRGBLayersFromDataset()->width();
	long depth = m_rep->fixedRGBLayersFromDataset()->depth();
	double tdeb = m_rep->fixedRGBLayersFromDataset()->sampleTransformation()->b();
	double pasech = m_rep->fixedRGBLayersFromDataset()->sampleTransformation()->a();
	CPUImagePaletteHolder* isoSurface = m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder();
	QColor col = m_rep->fixedRGBLayersFromDataset()->getHorizonColor();
	QPen pen;
	pen.setCosmetic(true);
	pen.setWidth(3);
	pen.setColor(col);
	m_curveMain->setPen(pen);
	isoSurface->lockPointer();
	const short* tab = (const short*) isoSurface->constBackingPointer();
	if (m_rep->direction()==SliceDirection::Inline) {
		for (long i=0; i<width; i++) {
			short val;
			if (m_rep->fixedRGBLayersFromDataset()->isIsoInT()) {
				val = (tab[i+slice*width] - tdeb) / pasech;
			} else {
				val = tab[i+slice*width];
			}
			QPointF valF;
			if ( tab[i+slice*width] != TIME_NO_VALUE )
			{
				valF = m_mainTransform.map(QPointF(i, val));
			}
			else
			{
				valF = QPointF(i, GraphEditor_MultiPolyLineShape::NOVALUE );
			}
			poly << valF.toPoint();
		}
	} else {
		for (long i=0; i<depth; i++) {
			short val;
			if (m_rep->fixedRGBLayersFromDataset()->isIsoInT()) {
				val = (tab[slice+i*width] - tdeb) / pasech;
			} else {
				val = tab[slice+i*width];
			}
			QPointF valF;
			if ( tab[i+slice*width] != TIME_NO_VALUE )
			{
				valF = m_mainTransform.map(QPointF(i, val));
			}
			else
			{
				valF = QPointF(i, GraphEditor_MultiPolyLineShape::NOVALUE );
			}
			poly << valF.toPoint();
		}
	}
	isoSurface->unlockPointer();

	// std::vector<QPolygonF> v_polygon = qPolygonSplit(poly);
	// GraphEditor_PolyLineShape *shape0 = static_cast<GraphEditor_PolyLineShape*>(m_rep->fixedRGBLayersFromDataset()->getHorizonShape());
	GraphEditor_MultiPolyLineShape *shape0 = static_cast<GraphEditor_MultiPolyLineShape*>(m_rep->getHorizonShape());
	if ( shape0 )
	{
		QBrush brush;
		shape0->setPen(pen);
		shape0->setBrush(brush);
		shape0->setSelected(false);
		// shape0->setPolygon(v_polygon[1]);
		// shape0->setPolygon(v_polygon);
		QPolygonF p(poly);
		shape0->setPolygon(p);
		shape0->setZValue(m_defaultZDepth+10);
	}
	else
	{
		m_curveMain->setPolygon(poly);
	}
	*/
}
