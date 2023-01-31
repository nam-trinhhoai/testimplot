#include "horizonfolderlayeronmap.h"
#include <QGraphicsScene>
#include "rgbinterleavedqglcudaimageitem.h"
#include "cpuimagepaletteholder.h"
#include "cudargbinterleavedimage.h"
//#include "fixedrgblayersfromdatasetandcuberep.h"
#include "horizondatarep.h"
#include "fixedrgblayersfromdatasetandcube.h"

HorizonFolderLayerOnMap::HorizonFolderLayerOnMap(HorizonDataRep *rep, QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) :
		GraphicLayer(scene, defaultZDepth) {
	m_rep = rep;
	m_parent = parent;
	m_item = nullptr;

	if(m_rep->image() != nullptr && m_rep->isoSurfaceHolder() != nullptr)
	{
		setBuffer(m_rep->image(), m_rep->isoSurfaceHolder());
	}


	//m_item = new RGBInterleavedQGLCUDAImageItem(rep->isoSurfaceHolder(),rep->image(), 0,parent);
	//m_item->setZValue(defaultZDepth);
	/*connect(m_rep->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(refresh()));
	connect(m_rep->image(),
			SIGNAL(rangeChanged(unsigned int , const QVector2D & )), this,
			SLOT(refresh()));

	connect(m_rep->image(), SIGNAL(dataChanged()), this,
			SLOT(refresh()));*/

//	connect(m_rep, SIGNAL(minimumValueActivated(bool)), this, SLOT(minValueActivated(bool)));
//	connect(m_rep, SIGNAL(minimumValueChanged(float)), this, SLOT(minValueChanged(float)));
}

HorizonFolderLayerOnMap::~HorizonFolderLayerOnMap() {
}



void HorizonFolderLayerOnMap::setBuffer(CUDARGBInterleavedImage* image ,CPUImagePaletteHolder* isoSurfaceHolder)
{
	//QMutexLocker lock(&m_mutex);
	RGBInterleavedQGLCUDAImageItem *lastitem =nullptr;
	if(image == m_lastimage) return;
	if(image != nullptr && isoSurfaceHolder != nullptr)
	{
		lastitem = m_item;

		m_item = new RGBInterleavedQGLCUDAImageItem(isoSurfaceHolder, image, 0,m_parent);
		m_item->setZValue(m_defaultZDepth);

		emit boundingRectChanged(boundingRect());

		refresh();


		if(m_showInternal )internalHide();



		connect(image,SIGNAL(rangeChanged(unsigned int, const QVector2D &)), this,SLOT(rangeChanged(unsigned int, const QVector2D &)));
		//connect(image, SIGNAL(opacityChanged(float)), this,SLOT(opacityChanged(float)));
		connect(image, SIGNAL(dataChanged()), this,SLOT(updateRgb()));

		connect(isoSurfaceHolder, SIGNAL(dataChanged()),this, SLOT(updateIsoSurface()));


		if(m_showOK )
		{
			internalShow();
		}
	}
	else
	{
		if(!m_showOK )internalHide();
	}

	if(m_lastimage != nullptr)
	{

		disconnect(m_lastimage,SIGNAL(rangeChanged(unsigned int, const QVector2D &)), this,SLOT(rangeChanged(unsigned int, const QVector2D &)));
		//disconnect(m_lastimage, SIGNAL(opacityChanged(float)), this,SLOT(opacityChanged(float)));
		disconnect(m_lastimage, SIGNAL(dataChanged()), this,SLOT(updateRgb()));
	}

	if(m_lastiso != nullptr)
	{
		disconnect(m_lastiso, SIGNAL(dataChanged()),this, SLOT(updateIsoSurface()));

	}
	m_lastimage = image;
	m_lastiso =isoSurfaceHolder;

	if(lastitem) delete lastitem;


}

void HorizonFolderLayerOnMap::updateRgb()
{
	refresh();
}

void HorizonFolderLayerOnMap::updateIsoSurface()
{
	refresh();
}

void HorizonFolderLayerOnMap::rangeChanged(unsigned int i, const QVector2D v)
{
	refresh();
}


void HorizonFolderLayerOnMap::internalShow() {
	m_showInternal =true;
	show();

}
void HorizonFolderLayerOnMap::internalHide() {
	m_showInternal =false;
	hide();
}



void HorizonFolderLayerOnMap::show() {
	m_showOK =true;

	if(m_item)m_scene->addItem(m_item);

}
void HorizonFolderLayerOnMap::hide() {
	m_showOK =false;
	if(m_item)m_scene->removeItem(m_item);
}

QRectF HorizonFolderLayerOnMap::boundingRect() const {

	if(m_rep->isoSurfaceHolder() == nullptr) return QRectF();
	return m_rep->isoSurfaceHolder()->worldExtent();
}
void HorizonFolderLayerOnMap::refresh() {
	if(m_item)m_item->update();
}

void HorizonFolderLayerOnMap::minValueActivated(bool activated) {
	if(m_item)m_item->setMinimumValueActive(activated);
	refresh();
}

void HorizonFolderLayerOnMap::minValueChanged(float value) {
	if(m_item)m_item->setMinimumValue(value);
	refresh();
}

