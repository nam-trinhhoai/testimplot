#include "fixedrgblayersfromdatasetandcubelayeronrandom.h"
#include <QAction>
#include <QGraphicsScene>
#include <QMenu>
#include "fixedrgblayersfromdatasetandcubereponrandom.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "qglisolineitem.h"
#include "cudaimagepaletteholder.h"
#include "seismic3dabstractdataset.h"
#include "randomlineview.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "geometry2dtoolbox.h"
#include "GraphEditor_MultiPolyLineShape.h"
#include "interpolation.h"

#include <cmath>

#define TIME_NO_VALUE -9999

FixedRGBLayersFromDatasetAndCubeLayerOnRandom::FixedRGBLayersFromDatasetAndCubeLayerOnRandom(
		FixedRGBLayersFromDatasetAndCubeRepOnRandom *rep,
		QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;

	double tdeb = m_rep->fixedRGBLayersFromDataset()->sampleTransformation()->b();
	double pasech = m_rep->fixedRGBLayersFromDataset()->sampleTransformation()->a();
	m_mainTransform.translate(0, tdeb);
	m_mainTransform.scale(1, pasech);

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

FixedRGBLayersFromDatasetAndCubeLayerOnRandom::~FixedRGBLayersFromDatasetAndCubeLayerOnRandom() {
}

void FixedRGBLayersFromDatasetAndCubeLayerOnRandom::show() {
	if ( m_polylineShape ) m_scene->addItem(m_polylineShape);
	refresh();
}

void FixedRGBLayersFromDatasetAndCubeLayerOnRandom::hide() {
	if ( m_polylineShape ) m_scene->removeItem(m_polylineShape);
}

QRectF FixedRGBLayersFromDatasetAndCubeLayerOnRandom::boundingRect() const {
	//return m_lineItem->boundingRect();

	// copied from CUDAImageBuffer worldExtent to give the same result
	double width = dynamic_cast<RandomLineView*>(m_rep->view())->discreatePolyLine().size();

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

		m_rep->fixedRGBLayersFromDataset()->sampleTransformation()->direct(ij[2 * i + 1], y);

		xMin = std::min(xMin, x);
		yMin = std::min(yMin, y);

		xMax = std::max(xMax, x);
		yMax = std::max(yMax, y);
	}
	QRectF rect(xMin, yMin, xMax - xMin, yMax - yMin);
	return rect;
}

void FixedRGBLayersFromDatasetAndCubeLayerOnRandom::refresh() {
	QPolygon poly;
	double tdeb = m_rep->fixedRGBLayersFromDataset()->sampleTransformation()->b();
	double pasech = m_rep->fixedRGBLayersFromDataset()->sampleTransformation()->a();
	CPUImagePaletteHolder* isoSurface = m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder();
	long width = isoSurface->width();
	long height = isoSurface->height();
	isoSurface->lockPointer();
	float* tab = (float*) isoSurface->backingPointer();

	QPolygonF polyLine = dynamic_cast<RandomLineView*>(m_rep->view())->worldDiscreatePolyLine();
	for (long k=0; k<polyLine.size(); k++) {
		double val = 0;

		QPointF worldPoint = polyLine[k];
		double i, j;
		m_rep->fixedRGBLayersFromDataset()->ijToXYTransfo()->worldToImage(worldPoint.x(),
				worldPoint.y(), i, j);

		std::vector<BilinearPoint> interpPoints = bilinearInterpolationPoints(i, j, 0.0,
				0.0, width-1, height-1, 1.0, 1.0);

		double w = 0;
		for (BilinearPoint pt : interpPoints) {
			float tabVal = tab[((long)pt.i)+((long)pt.j)*width];
			if (tabVal!=TIME_NO_VALUE) {
				val += pt.w * tabVal;
				w += pt.w;
			}
		}
		if (w!=0.0) {
			val /= w;
			if (m_rep->fixedRGBLayersFromDataset()->isIsoInT()) {
				val = (val - tdeb) / pasech;
			}
		} else {
			val = TIME_NO_VALUE;
		}

		if (val!=TIME_NO_VALUE) {
			QPointF pt = m_mainTransform.map(QPointF(k, val));
			poly << pt.toPoint();
		} else {
			poly << QPoint(k, GraphEditor_MultiPolyLineShape::NOVALUE );
		}
	}
	isoSurface->unlockPointer();

	if (m_polylineShape) {
		QColor col = m_rep->fixedRGBLayersFromDataset()->getHorizonColor();
		QPen pen;
		pen.setCosmetic(true);
		pen.setWidth(3);
		pen.setColor(col);

		QBrush brush;
		m_polylineShape->setPen(pen);
		m_polylineShape->setBrush(brush);
		m_polylineShape->setSelected(false);
		QPolygonF p(poly);
		m_polylineShape->setPolygon(p);
		m_polylineShape->setZValue(m_defaultZDepth+10);
	}
}

void FixedRGBLayersFromDatasetAndCubeLayerOnRandom::actionMenuCreate()
{
	m_actionColor = new QAction(QIcon(":/slicer/icons/graphic_tools/paint_bucket.png"), tr("color"), this);
	connect(m_actionColor, &QAction::triggered, m_rep, &FixedRGBLayersFromDatasetAndCubeRepOnRandom::trt_changeColor);
	m_actionProperties = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"), tr("properties"), this);
	connect(m_actionProperties, &QAction::triggered, m_rep, &FixedRGBLayersFromDatasetAndCubeRepOnRandom::trt_properties);
	m_actionLocation = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"), tr("folder"), this);
	connect(m_actionLocation, &QAction::triggered, m_rep, &FixedRGBLayersFromDatasetAndCubeRepOnRandom::trt_location);
}
