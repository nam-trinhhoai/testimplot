
#include <math.h>
#include <cmath>
#include <limits>

#include "fixedrgblayersfromdatasetandcubelayeronslice.h"
#include <QAction>
#include <QGraphicsScene>
#include <QMenu>
#include "fixedrgblayersfromdatasetandcubereponslice.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "qglisolineitem.h"
#include "cudaimagepaletteholder.h"
#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "GraphEditor_MultiPolyLineShape.h"

#define TIME_NO_VALUE -9999

FixedRGBLayersFromDatasetAndCubeLayerOnSlice::FixedRGBLayersFromDatasetAndCubeLayerOnSlice(FixedRGBLayersFromDatasetAndCubeRepOnSlice *rep, SliceDirection dir,
		int startValue, QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;

	if (m_rep->direction()==SliceDirection::Inline) {
		m_mainTransform = QTransform(m_rep->fixedRGBLayersFromDataset()->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
	} else {
		m_mainTransform = QTransform(m_rep->fixedRGBLayersFromDataset()->ijToInlineXlineTransfoForXline()->imageToWorldTransformation().toTransform());
	}

	actionMenuCreate();
	m_itemMenu = new QMenu("Item Menu");
	m_itemMenu->addAction(m_actionColor);
	m_itemMenu->addAction(m_actionProperties);
	m_itemMenu->addAction(m_actionLocation);

	m_polylineShape = new GraphEditor_MultiPolyLineShape();
	m_polylineShape->setReadOnly(true);
	m_polylineShape->setDisplayPerimetre(false);
	m_polylineShape->setToolTip(m_rep->fixedRGBLayersFromDataset()->getName());
	m_polylineShape->setMenu(m_itemMenu);

	connect(m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder(), SIGNAL(dataChanged()), this,
			SLOT(refresh()));

	show();
}

FixedRGBLayersFromDatasetAndCubeLayerOnSlice::~FixedRGBLayersFromDatasetAndCubeLayerOnSlice() {
}

void FixedRGBLayersFromDatasetAndCubeLayerOnSlice::setSliceIJPosition(int imageVal) {
	refresh();//m_lineItem->updateSlice(imageVal);
}

void FixedRGBLayersFromDatasetAndCubeLayerOnSlice::show() {
	if ( m_polylineShape ) m_scene->addItem(m_polylineShape);
	refresh();
}
void FixedRGBLayersFromDatasetAndCubeLayerOnSlice::hide() {
	if ( m_polylineShape ) m_scene->removeItem(m_polylineShape);
}

QRectF FixedRGBLayersFromDatasetAndCubeLayerOnSlice::boundingRect() const {
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
	return rect;
}

void FixedRGBLayersFromDatasetAndCubeLayerOnSlice::refresh() {
	// extract polygon
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
	isoSurface->lockPointer();
	const float* tab = (const float*) isoSurface->constBackingPointer();
	if (m_rep->direction()==SliceDirection::Inline) {
		for (long i=0; i<width; i++) {
			float val;
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

	if (m_polylineShape) {
		QBrush brush;
		m_polylineShape->setPen(pen);
		m_polylineShape->setBrush(brush);
		m_polylineShape->setSelected(false);
		QPolygonF p(poly);
		m_polylineShape->setPolygon(p);
		m_polylineShape->setZValue(m_defaultZDepth+10);
	}
}

void FixedRGBLayersFromDatasetAndCubeLayerOnSlice::actionMenuCreate()
{
	m_actionColor = new QAction(QIcon(":/slicer/icons/graphic_tools/paint_bucket.png"), tr("color"), this);
	connect(m_actionColor, &QAction::triggered, m_rep, &FixedRGBLayersFromDatasetAndCubeRepOnSlice::trt_changeColor);
	m_actionProperties = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"), tr("properties"), this);
	connect(m_actionProperties, &QAction::triggered, m_rep, &FixedRGBLayersFromDatasetAndCubeRepOnSlice::trt_properties);
	m_actionLocation = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"), tr("folder"), this);
	connect(m_actionLocation, &QAction::triggered, m_rep, &FixedRGBLayersFromDatasetAndCubeRepOnSlice::trt_location);
}
