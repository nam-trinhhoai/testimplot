
#include <math.h>
#include <cmath>
#include <limits>

#include "horizonfolderlayeronslice.h"
#include <QGraphicsScene>
#include "horizonfolderreponslice.h"
#include "fixedrgblayersfromdatasetandcube.h"

#include "qglisolineitem.h"
#include "cudaimagepaletteholder.h"
#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "GraphEditor_PolyLineShape.h"
#include "horizonfolderdata.h"
#include "cpuimagepaletteholder.h"

#define TIME_NO_VALUE -9999

HorizonFolderLayerOnSlice::HorizonFolderLayerOnSlice(HorizonFolderRepOnSlice *rep, SliceDirection dir,
		int startValue, QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
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
	pen.setWidth(6);
	pen.setColor(Qt::white);
	m_curveMain->setPen(pen);

/*	if (m_rep->direction()==SliceDirection::Inline) {
		m_mainTransform = QTransform(m_rep->horizonFolderData()->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
	} else {
		m_mainTransform = QTransform(m_rep->horizonFolderData()->ijToInlineXlineTransfoForXline()->imageToWorldTransformation().toTransform());
	}
	m_curveMain->setTransform(m_mainTransform);*/

//	connect(m_rep->isoSurfaceHolder(), SIGNAL(dataChanged()), this,	SLOT(refresh()));

	//show();
}

HorizonFolderLayerOnSlice::~HorizonFolderLayerOnSlice() {
}

void HorizonFolderLayerOnSlice::setSliceIJPosition(int imageVal) {
	refresh();//m_lineItem->updateSlice(imageVal);
}



void HorizonFolderLayerOnSlice::setBuffer(CPUImagePaletteHolder* isoSurfaceHolder)
{
	if(isoSurfaceHolder != nullptr)
	{
		if (m_rep->direction()==SliceDirection::Inline) {
				m_mainTransform = QTransform(m_rep->horizonFolderData()->ijToInlineXlineTransfoForInline()->imageToWorldTransformation().toTransform());
			} else {
				m_mainTransform = QTransform(m_rep->horizonFolderData()->ijToInlineXlineTransfoForXline()->imageToWorldTransformation().toTransform());
			}
			m_curveMain->setTransform(m_mainTransform);

		if(m_showInternal )internalHide();

		connect(isoSurfaceHolder, SIGNAL(dataChanged()),this, SLOT(refresh()));

		if(m_showOK )
		{
			internalShow();
		}
	}
	else
	{
		if(!m_showOK )internalHide();
	}



	if(m_lastiso != nullptr)
	{
		disconnect(m_lastiso, SIGNAL(dataChanged()),this, SLOT(refresh()));
	}

	m_lastiso =isoSurfaceHolder;
}


void HorizonFolderLayerOnSlice::show() {

	m_showOK =true;
	internalShow();
}

void HorizonFolderLayerOnSlice::internalShow() {
	m_showInternal =true;
	//m_scene->addItem(m_lineItem);
	m_curveMain->addToScene();
	// GraphEditor_PolyLineShape *shape = static_cast<GraphEditor_PolyLineShape*>(m_rep->fixedRGBLayersFromDataset()->getHorizonShape());
	GraphEditor_MultiPolyLineShape *shape = static_cast<GraphEditor_MultiPolyLineShape*>(m_rep->getHorizonShape());
	if ( shape ) m_scene->addItem(shape);
	refresh();
}

void HorizonFolderLayerOnSlice::hide() {

	m_showOK =false;
	internalHide();
}

void HorizonFolderLayerOnSlice::internalHide() {
	m_showInternal=false;
	//m_scene->removeItem(m_lineItem);
	m_curveMain->removeFromScene();
	// GraphEditor_PolyLineShape *shape = static_cast<GraphEditor_PolyLineShape*>(m_rep->fixedRGBLayersFromDataset()->getHorizonShape());
	GraphEditor_MultiPolyLineShape *shape = static_cast<GraphEditor_MultiPolyLineShape*>(m_rep->getHorizonShape());
	if ( shape ) m_scene->removeItem(shape);
}

QRectF HorizonFolderLayerOnSlice::boundingRect() const {
	//return m_lineItem->boundingRect();

	// copied from CUDAImageBuffer worldExtent to give the same result

	if( m_rep->horizonFolderData()->currentLayer()== nullptr)
	{
		return QRectF();
		//int size =m_rep->horizonFolderData()->currentLayer()->m_attribut.size();
	}

	double width;
	if (m_rep->direction()==SliceDirection::Inline) {
		width = m_rep->horizonFolderData()->currentLayer()->m_attribut[0].width();
	} else {
		width = m_rep->horizonFolderData()->currentLayer()->m_attribut[0].depth();
	}

	double ij[8] = { 0.0, 0.0, (double) width, 0.0,
			0.0, (double) m_rep->horizonFolderData()->currentLayer()->m_attribut[0].heightFor3D(),
			(double) width, (double) m_rep->horizonFolderData()->currentLayer()->m_attribut[0].heightFor3D() };

	double xMin = std::numeric_limits<double>::max();
	double yMin = std::numeric_limits<double>::max();

	double xMax = std::numeric_limits<double>::min();
	double yMax = std::numeric_limits<double>::min();
	double x, y;
	for (int i = 0; i < 4; i++) {
		x = ij[2 * i];
		y = ij[2 * i + 1];

		if (m_rep->direction()==SliceDirection::Inline) {
			m_rep->horizonFolderData()->currentLayer()->m_attribut[0].ijToInlineXlineTransfoForInline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
		} else {
			m_rep->horizonFolderData()->currentLayer()->m_attribut[0].ijToInlineXlineTransfoForXline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
		}

		xMin = std::min(xMin, x);
		yMin = std::min(yMin, y);

		xMax = std::max(xMax, x);
		yMax = std::max(yMax, y);
	}
	QRectF rect(xMin, yMin, xMax - xMin, yMax - yMin);
	return rect;
}

void HorizonFolderLayerOnSlice::refresh() {
	//m_lineItem->update():
	// extract polygon
	QPolygon poly;

	if(m_rep->horizonFolderData()->currentLayer() == nullptr )return;
	long slice = m_rep->currentIJSliceRep();
	long width = m_rep->horizonFolderData()->currentLayer()->m_attribut[0].width();
	long depth = m_rep->horizonFolderData()->currentLayer()->m_attribut[0].depth();
	double tdeb = m_rep->horizonFolderData()->sampleTransformation()->b();
	double pasech = m_rep->horizonFolderData()->sampleTransformation()->a();
	CPUImagePaletteHolder* isoSurface = m_rep->isoSurfaceHolder();
	if(isoSurface ==nullptr) return;
	QColor col = m_rep->horizonFolderData()->getHorizonColor();
	QPen pen;
	pen.setCosmetic(true);
	pen.setWidth(6);
	pen.setColor(col);
	m_curveMain->setPen(pen);
	isoSurface->lockPointer();
	const short* tab = (const short*) isoSurface->constBackingPointer();
	if (m_rep->direction()==SliceDirection::Inline) {
		for (long i=0; i<width; i++) {
			short val;
			if (m_rep->horizonFolderData()->currentLayer()->m_attribut[0].isIsoInT()) {
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
			if (m_rep->horizonFolderData()->currentLayer()->m_attribut[0].isIsoInT()) {
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


	//setBuffer(isoSurface);
}
