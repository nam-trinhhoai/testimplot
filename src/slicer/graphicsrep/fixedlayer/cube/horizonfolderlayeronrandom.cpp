
#include <math.h>
#include <cmath>
#include <limits>

#include "horizonfolderlayeronrandom.h"
#include <QGraphicsScene>
#include "horizonfolderreponrandom.h"
#include "fixedrgblayersfromdatasetandcube.h"

#include "qglisolineitem.h"
#include "cudaimagepaletteholder.h"
#include "seismic3dabstractdataset.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "GraphEditor_PolyLineShape.h"
#include "horizonfolderdata.h"
#include "cpuimagepaletteholder.h"
#include "randomlineview.h"

#define TIME_NO_VALUE -9999

HorizonFolderLayerOnRandom::HorizonFolderLayerOnRandom(HorizonFolderRepOnRandom *rep,
		int startValue, QGraphicsScene *scene, int defaultZDepth, QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;



	/*m_curveMain.reset(new Curve(m_scene, parent));
	m_curveMain->setZValue(m_defaultZDepth);
	QPen pen;
	pen.setCosmetic(true);
	pen.setWidth(6);
	pen.setColor(Qt::white);
	m_curveMain->setPen(pen);*/

	computeTransform();

}

HorizonFolderLayerOnRandom::~HorizonFolderLayerOnRandom() {
}

void HorizonFolderLayerOnRandom::setSliceIJPosition(int imageVal) {
	refresh();//m_lineItem->updateSlice(imageVal);
}


void HorizonFolderLayerOnRandom::computeTransform()
{
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_rep->view());
	if (randomView)
	{
		if(m_transformation != nullptr)
		{
			m_transformation->deleteLater();
			m_transformation = nullptr;
		}

		m_discreatPolyline = randomView->discreatePolyLine();
		const AffineTransformation* sampleTransform = m_rep->horizonFolderData()->sampleTransformation();
		std::array<double, 6> transform;

		transform[0]=0;
		transform[1]=1;
		transform[2]=0;

		transform[3]=sampleTransform->b();
		transform[4]=0;
		transform[5]=sampleTransform->a();

		m_transformation = new Affine2DTransformation(randomView->discreatePolyLine().size(),  m_rep->horizonFolderData()->height(), transform, this);

		m_mainTransform = m_transformation->imageToWorldTransformation().toTransform();

	}
}


void HorizonFolderLayerOnRandom::setBuffer(CPUImagePaletteHolder* isoSurfaceHolder)
{
	if(isoSurfaceHolder != nullptr)
	{

		computeTransform();


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


void HorizonFolderLayerOnRandom::show() {

	m_showOK =true;
	internalShow();
}

void HorizonFolderLayerOnRandom::internalShow() {
	m_showInternal =true;
	//m_scene->addItem(m_lineItem);
	//m_curveMain->addToScene();
	// GraphEditor_PolyLineShape *shape = static_cast<GraphEditor_PolyLineShape*>(m_rep->fixedRGBLayersFromDataset()->getHorizonShape());
	GraphEditor_MultiPolyLineShape *shape = static_cast<GraphEditor_MultiPolyLineShape*>(m_rep->getHorizonShape());
	if ( shape ) m_scene->addItem(shape);
	refresh();
}

void HorizonFolderLayerOnRandom::hide() {

	m_showOK =false;
	internalHide();
}

void HorizonFolderLayerOnRandom::internalHide() {
	m_showInternal=false;
	//m_scene->removeItem(m_lineItem);
	//m_curveMain->removeFromScene();
	// GraphEditor_PolyLineShape *shape = static_cast<GraphEditor_PolyLineShape*>(m_rep->fixedRGBLayersFromDataset()->getHorizonShape());
	GraphEditor_MultiPolyLineShape *shape = static_cast<GraphEditor_MultiPolyLineShape*>(m_rep->getHorizonShape());
	if ( shape ) m_scene->removeItem(shape);
}

QRectF HorizonFolderLayerOnRandom::boundingRect() const {
	//return m_lineItem->boundingRect();

	// copied from CUDAImageBuffer worldExtent to give the same result

	if( m_rep->horizonFolderData()->currentLayer()== nullptr)
	{
		return QRectF();
		//int size =m_rep->horizonFolderData()->currentLayer()->m_attribut.size();
	}

//	double width=m_rep->horizonFolderData()->currentLayer()->m_transformation->width();
	double width = m_transformation->width();
//	if (m_rep->direction()==SliceDirection::Inline) {
//		width = m_rep->horizonFolderData()->currentLayer()->m_attribut[0].width();
/*	} else {
		width = m_rep->horizonFolderData()->currentLayer()->m_attribut[0].depth();
	}*/

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
		m_transformation->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
	//	m_rep->horizonFolderData()->currentLayer()->
	//	if (m_rep->direction()==SliceDirection::Inline) {
		//	m_rep->horizonFolderData()->currentLayer()->m_attribut[0].ijToInlineXlineTransfoForInline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
	/*	} else {
			m_rep->horizonFolderData()->currentLayer()->m_attribut[0].ijToInlineXlineTransfoForXline()->imageToWorld(ij[2 * i], ij[2 * i + 1], x, y);
		}*/

		xMin = std::min(xMin, x);
		yMin = std::min(yMin, y);

		xMax = std::max(xMax, x);
		yMax = std::max(yMax, y);
	}
	QRectF rect(xMin, yMin, xMax - xMin, yMax - yMin);
	return rect;
}

void HorizonFolderLayerOnRandom::refresh() {
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
	//m_curveMain->setPen(pen);
	isoSurface->lockPointer();
	const short* tab = (const short*) isoSurface->constBackingPointer();



//	if (m_rep->direction()==SliceDirection::Inline) {
		for (long i=0; i<m_discreatPolyline.size(); i++) {
			short val;
			QPoint point = m_discreatPolyline[i];
			if (m_rep->horizonFolderData()->currentLayer()->m_attribut[0].isIsoInT()) {

				val = (tab[point.x()+point.y()*width] - tdeb) / pasech;
				//val = (tab[i+slice*width] - tdeb) / pasech;
			} else {

				val = tab[point.x()+point.y()*width] ;
				//val = tab[i+slice*width];
			}
			QPointF valF;
			if ( tab[point.x()+point.y()*width] != TIME_NO_VALUE )
			{
				valF = m_mainTransform.map(QPointF(i, val));
			}
			else
			{
				valF = QPointF(i, GraphEditor_MultiPolyLineShape::NOVALUE );
			}
			poly << valF.toPoint();
		}
/*	} else {
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
	}*/
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
	/*else
	{
		m_curveMain->setPolygon(poly);
	}*/


	//setBuffer(isoSurface);
}
