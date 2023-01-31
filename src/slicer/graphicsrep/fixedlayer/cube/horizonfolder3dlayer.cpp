#include "horizonfolder3dlayer.h"
#include "horizondatarep.h"

#include <iostream>
#include <cmath>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QEntity>
#include <Qt3DExtras/QPlaneMesh>
#include <Qt3DRender/QTechnique>
#include <Qt3DRender/QParameter>
#include <Qt3DInput/QMouseHandler>
#include <Qt3DRender/QPickEvent>
#include <Qt3DRender/QPickTriangleEvent>
#include <Qt3DCore/QBuffer>
#include <QMouseDevice>
#include <QWindow>
#include <QRenderPass>
#include <QMaterial>
#include <QEffect>
#include <QObjectPicker>
#include <QMouseEvent>
#include <QCamera>
#include <QMutexLocker>

#include <QPropertyAnimation>

#include "iimagepaletteholder.h"
#include "cudargbimage.h"
#include "volumeboundingmesh.h"
#include "seismic3dabstractdataset.h"
#include "fixedrgblayersfromdatasetandcuberep.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "colortabletexture.h"
#include "cudaimagetexture.h"
#include "qt3dhelpers.h"

#include "viewqt3d.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "surfacemeshcacheutils.h"
#include <QDebug>

HorizonFolder3DLayer::HorizonFolder3DLayer(HorizonDataRep *rep, QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) :
		Graphic3DLayer(parent, root, camera) {


	m_rep = rep;
//	m_transform=nullptr;
//	m_sliceEntity = nullptr;
	m_colorTexture = nullptr;

	m_cudaRgbTexture = nullptr;

//	m_actifRayon= false;
//	m_EntityMesh = nullptr;
	m_rgbMaterial = nullptr;
	m_cudaSurfaceTexture = nullptr;
////	m_material = nullptr;
//m_opacityParameter = nullptr;

	m_surface= nullptr;
//	m_paletteRedRangeParameter = m_paletteGreenRangeParameter =
//			m_paletteBlueRangeParameter = nullptr;

/*	connect(m_rep->horizonFolderData()->image(),
			SIGNAL(rangeChanged(unsigned int, const QVector2D &)), this,
			SLOT(rangeChanged(unsigned int, const QVector2D &)));
	connect(m_rep->horizonFolderData()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(opacityChanged(float)));

	connect(m_rep->horizonFolderData()->image(), SIGNAL(dataChanged()), this,
			SLOT(updateRgb()));

	connect(m_rep->horizonFolderData()->isoSurfaceHolder(), SIGNAL(dataChanged()),
			this, SLOT(updateIsoSurface()));*/

//	connect(m_rep->horizonFolderData(), SIGNAL(minimumValueActivated(bool)), this, SLOT(minValueActivated(bool)));
//	connect(m_rep->horizonFolderData(), SIGNAL(minimumValueChanged(float)), this, SLOT(minValueChanged(float)));
}

HorizonFolder3DLayer::~HorizonFolder3DLayer() {

}

void HorizonFolder3DLayer::opacityChanged(float val) {
	if (m_surface!=nullptr) {
		m_surface->setOpacity(val);
	}
}

void HorizonFolder3DLayer::rangeChanged(unsigned int i, const QVector2D &value) {

	if (m_rgbMaterial!=nullptr && m_rep->horizonFolderData() != nullptr) {
		QVector2D rangeRatio(0, 1);
		if (i==0) {
			rangeRatio = m_rep->image()->redRangeRatio();
		} else if (i==1) {
			rangeRatio = m_rep->image()->greenRangeRatio();
		} else if (i==2) {
			rangeRatio = m_rep->image()->blueRangeRatio();
		}
		m_rgbMaterial->rangeChanged(i, rangeRatio);
	}
}


float HorizonFolder3DLayer::distanceSigned(QVector3D position, bool* ok)
{
	if(m_surface == nullptr)
	{
		*ok = false;
		return 0.0f;
	}
	float zscale = dynamic_cast<ViewQt3D*>(m_rep->view())->zScale();
	QVector3D positionScaler(position.x(),position.y()/zscale,position.z());
	QMatrix4x4 transformInverse = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransformInverse();
	QVector3D posTr = transformInverse * positionScaler;
	double iImage,kImage,jImage;
	m_rep->currentLayerWithAttribut().ijToXYTransfo()->worldToImage(posTr.x(), posTr.z(), iImage, kImage);


	float multDist = 1.0f;
	if (m_rep->currentLayerWithAttribut().isIsoInT())
	{
		jImage = posTr.y();
	}
	else
	{
		data()->sampleTransformation()->indirect(posTr.y(), jImage);
		multDist = data()->sampleTransformation()->a();
	}

	QVector3D ijk(iImage,jImage,kImage);
	float distance = m_surface->distanceSigned(ijk,ok);
	return multDist*distance;

}

void HorizonFolder3DLayer::updateTexture(CudaImageTexture *texture,
		IImagePaletteHolder *img) {
	if (texture == nullptr )//|| m_sliceEntity==nullptr)
		return;


	//auto start = std::chrono::steady_clock::now();
	texture->setData(img->getDataAsByteArray());

	//auto end = std::chrono::steady_clock::now();
	//qDebug() << "update texture : " << std::chrono::duration<double, std::milli>(end - start).count();
}

void HorizonFolder3DLayer::updateTexture(CudaImageTexture *texture,
		CUDARGBInterleavedImage *img) {
	if (texture == nullptr )//|| m_sliceEntity==nullptr)
		return;
	//auto start = std::chrono::steady_clock::now();
	size_t pointerSize = img->internalPointerSize();
	img->lockPointer();
	texture->setData(img->byteArray());
	img->unlockPointer();

}

void HorizonFolder3DLayer::updateRgb() {

	if(m_rep->horizonFolderData() != nullptr)
	{
		updateTexture(m_cudaRgbTexture, m_rep->image());

		rangeChanged(0, m_rep->image()->redRange());
		rangeChanged(1, m_rep->image()->greenRange());
		rangeChanged(2, m_rep->image()->blueRange());
	}
}


void HorizonFolder3DLayer::setBuffer(CUDARGBInterleavedImage* image ,CPUImagePaletteHolder* isoSurfaceHolder)
{
	QMutexLocker lock(&m_mutex);
//	bool createOKImage = (image == nullptr);
//	bool createOKIso = (isoSurfaceHolder == nullptr);
	if(image == m_lastimage) return;
	if(image != nullptr && isoSurfaceHolder != nullptr)
	{


		if(m_showInternal )internalHide();


		connect(image,SIGNAL(rangeChanged(unsigned int, const QVector2D &)), this,SLOT(rangeChanged(unsigned int, const QVector2D &)));
		connect(image, SIGNAL(opacityChanged(float)), this,SLOT(opacityChanged(float)));
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
		disconnect(m_lastimage, SIGNAL(opacityChanged(float)), this,SLOT(opacityChanged(float)));
		disconnect(m_lastimage, SIGNAL(dataChanged()), this,SLOT(updateRgb()));
	}

	if(m_lastiso != nullptr)
	{
		disconnect(m_lastiso, SIGNAL(dataChanged()),this, SLOT(updateIsoSurface()));

	}
	m_lastimage = image;
	m_lastiso =isoSurfaceHolder;
}


void HorizonFolder3DLayer::updateIsoSurface() {
	if( m_rep->horizonFolderData() != nullptr && m_rep->currentLayerWithAttribut().getData() != nullptr)
	{
		updateTexture(m_cudaSurfaceTexture,m_rep->isoSurfaceHolder());

		bool cacheExist =m_rep->currentLayerWithAttribut().isIndexCache(m_rep->currentLayerWithAttribut().currentImageIndex());
		if(cacheExist && m_surface != nullptr)
		{
			SurfaceMeshCache* meshCache = m_rep->currentLayerWithAttribut().getMeshCache(m_rep->currentLayerWithAttribut().currentImageIndex());
			if(meshCache != nullptr) m_surface->reloadFromCache(*meshCache);
			else cacheExist = false;

		}
		 if(!cacheExist && m_surface != nullptr)
		{
			m_surface->update(m_rep->isoSurfaceHolder(), ""/*,m_rep->fixedRGBLayersFromDataset()->getCurrentObjFile()*/,m_rep->currentLayerWithAttribut().getSimplifyMeshSteps(),m_rep->currentLayerWithAttribut().getCompressionMesh() );
		}

	}
}

HorizonFolderData* HorizonFolder3DLayer::data() const {
	return m_rep->horizonFolderData();
}

void HorizonFolder3DLayer::show() {

	m_showOK =true;
	internalShow();
}

void HorizonFolder3DLayer::generateCacheAnimation(CUDARGBInterleavedImage* image ,CPUImagePaletteHolder* isoSurfaceHolder)
{
	internalHide();
	createEntityCache(image,isoSurfaceHolder);

}

void HorizonFolder3DLayer::clearCacheAnimation()
{
	for(int i=0;i<m_listSurface.count();i++)
	{
		m_listSurface[i]->hide();
		m_listSurface[i]->deleteLater();

	}
	m_listSurface.clear();

	if(m_lastimage != nullptr)
	{
		//m_lastimage->deleteLater();
		m_lastimage=  nullptr;
	}
	if(m_lastiso != nullptr)
	{
		//m_lastiso->deleteLater();
		m_lastiso=  nullptr;
	}
}

void HorizonFolder3DLayer::setVisible(int index)
{
	if(index>=0 && index < m_listSurface.count())
	{
		int prec = index+1;
		if(prec == m_listSurface.count() ) prec = 0;
		if(m_listSurface[prec] != nullptr)
		{
			m_listSurface[prec]->setVisible(false);
		}
		if(m_listSurface[index] != nullptr)
		{
			m_listSurface[index]->setVisible(true);
		}
	}
}

void HorizonFolder3DLayer::createEntityCache(CUDARGBInterleavedImage* image ,CPUImagePaletteHolder* isoSurfaceHolder)
{
	int width = m_rep->currentLayerWithAttribut().width();
	int depth = m_rep->currentLayerWithAttribut().depth();

	QMatrix4x4 sceneTransform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
	//	Qt3DRender::QLayer* layerTr = dynamic_cast<ViewQt3D*>(m_rep->view())->getLayerTransparent();
		QMatrix4x4 ijToXYTranform(m_rep->currentLayerWithAttribut().ijToXYTransfo()->imageToWorldTransformation());

		// swap axis of ijToXYTranform (i,j,k) -> (i,k,j) i:Y, j:Z, k:sample
		const float* tbuf = ijToXYTranform.constData();
		QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
										 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
										 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
										 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);


		QMatrix4x4 transform = sceneTransform * ijToXYTranformSwapped;

		// Set different parameters on the materials
		m_colorTexture = new ColorTableTexture();
		//CUDARGBInterleavedImage *imgRgb = m_rep->image();
		m_cudaRgbTexture = new CudaImageTexture(image->colorFormat(), image->sampleType(), image->width(),image->height());

		//CPUImagePaletteHolder *imgSurf = m_rep->isoSurfaceHolder();
		m_cudaSurfaceTexture = new CudaImageTexture(isoSurfaceHolder->colorFormat(), isoSurfaceHolder->sampleType(), isoSurfaceHolder->width(),isoSurfaceHolder->height());


		tbuf = transform.constData();
		if (m_rep->currentLayerWithAttribut().isIsoInT()) {
			//m_cubeOrigin =tbuf[3*4+1];
			//m_cubeScale = tbuf[1*4+1];
			m_cubeOrigin =0;
			m_cubeScale = 1;
		} else {
			const AffineTransformation* sampleTransform = data()->sampleTransformation();
			//m_cubeOrigin = tbuf[1*4+1] * sampleTransform->b() + tbuf[3*4+1];
			//m_cubeScale = tbuf[1*4+1] * sampleTransform->a();
			m_cubeOrigin = sampleTransform->b();
			m_cubeScale = sampleTransform->a();
		}


		const AffineTransformation* sampleTransform = data()->sampleTransformation();
		 float heightThreshold = ( m_rep->currentLayerWithAttribut().heightFor3D()-2) *sampleTransform->a()  +sampleTransform->b() ;
		 float opacite = image->opacity();




		 QVector2D ratioRed = image->redRangeRatio();
		 QVector2D ratioGreen =image->greenRangeRatio();
		 QVector2D ratioBlue = image->blueRangeRatio();

		 QCoreApplication::processEvents();
		updateRgb();
		updateIsoSurface();

		GenericSurface3DLayer* surfaceCache = new GenericSurface3DLayer(sceneTransform,ijToXYTranformSwapped, ""/*,m_rep->fixedRGBLayersFromDataset()->getCurrentObjFile()*/);
		RGBInterleavedMaterialInitializer* rgbMaterial = new RGBInterleavedMaterialInitializer(image->sampleType(),ratioRed,ratioGreen,ratioBlue,m_cudaRgbTexture);

		surfaceCache->Show(m_root,transform,width,depth,m_cudaSurfaceTexture,rgbMaterial,isoSurfaceHolder,heightThreshold,m_cubeScale,m_cubeOrigin,m_camera,opacite,isoSurfaceHolder->sampleType(),m_rep->currentLayerWithAttribut().getSimplifyMeshSteps(),m_rep->currentLayerWithAttribut().getCompressionMesh()/*,layerTr*/);

		surfaceCache->setVisible(false);

		m_listSurface.push_back(surfaceCache);


}


void HorizonFolder3DLayer::internalShow() {
	m_showInternal =true;
	//QMutexLocker lock(&m_mutex);

	if(m_rep->currentLayerWithAttribut().getData() == nullptr)
	{
		return;
	}
	int width = m_rep->currentLayerWithAttribut().width();
	int depth = m_rep->currentLayerWithAttribut().depth();


	QMatrix4x4 sceneTransform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
//	Qt3DRender::QLayer* layerTr = dynamic_cast<ViewQt3D*>(m_rep->view())->getLayerTransparent();
	QMatrix4x4 ijToXYTranform(m_rep->currentLayerWithAttribut().ijToXYTransfo()->imageToWorldTransformation());

	// swap axis of ijToXYTranform (i,j,k) -> (i,k,j) i:Y, j:Z, k:sample
	const float* tbuf = ijToXYTranform.constData();
	QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
									 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
									 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
									 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);


	QMatrix4x4 transform = sceneTransform * ijToXYTranformSwapped;

	// Set different parameters on the materials
	m_colorTexture = new ColorTableTexture();
	CUDARGBInterleavedImage *imgRgb = m_rep->image();
	m_cudaRgbTexture = new CudaImageTexture(imgRgb->colorFormat(), imgRgb->sampleType(), imgRgb->width(),
			imgRgb->height());

	CPUImagePaletteHolder *imgSurf = m_rep->isoSurfaceHolder();
	m_cudaSurfaceTexture = new CudaImageTexture(imgSurf->colorFormat(), imgSurf->sampleType(), imgSurf->width(),
			imgSurf->height());


	tbuf = transform.constData();
	if (m_rep->currentLayerWithAttribut().isIsoInT()) {
		//m_cubeOrigin =tbuf[3*4+1];
		//m_cubeScale = tbuf[1*4+1];
		m_cubeOrigin =0;
		m_cubeScale = 1;
	} else {
		const AffineTransformation* sampleTransform = data()->sampleTransformation();
		//m_cubeOrigin = tbuf[1*4+1] * sampleTransform->b() + tbuf[3*4+1];
		//m_cubeScale = tbuf[1*4+1] * sampleTransform->a();
		m_cubeOrigin = sampleTransform->b();
		m_cubeScale = sampleTransform->a();
	}


	const AffineTransformation* sampleTransform = data()->sampleTransformation();
	 float heightThreshold = ( m_rep->currentLayerWithAttribut().heightFor3D()-2) *sampleTransform->a()  +sampleTransform->b() ;
	 float opacite = m_rep->image()->opacity();




	 QVector2D ratioRed = m_rep->image()->redRangeRatio();
	 QVector2D ratioGreen =m_rep->image()->greenRangeRatio();
	 QVector2D ratioBlue = m_rep->image()->blueRangeRatio();

	 QCoreApplication::processEvents();
	updateRgb();
	updateIsoSurface();

	m_surface = new GenericSurface3DLayer(sceneTransform,ijToXYTranformSwapped, ""/*,m_rep->fixedRGBLayersFromDataset()->getCurrentObjFile()*/);
	m_rgbMaterial = new RGBInterleavedMaterialInitializer(imgRgb->sampleType(),ratioRed,ratioGreen,ratioBlue,m_cudaRgbTexture);

	 m_surface->Show(m_root,transform,width,depth,m_cudaSurfaceTexture,m_rgbMaterial,imgSurf,
			 heightThreshold,m_cubeScale,m_cubeOrigin,m_camera,opacite,imgSurf->sampleType(),m_rep->currentLayerWithAttribut().getSimplifyMeshSteps(),m_rep->currentLayerWithAttribut().getCompressionMesh()/*,layerTr*/);

	 connect(m_surface,SIGNAL(sendPositionTarget(QVector3D, QVector3D)),this,SLOT(receiveInfosCam(QVector3D,QVector3D)) );

	 ViewQt3D *view3d = dynamic_cast<ViewQt3D*>(m_rep->view());
	if(view3d!= nullptr)
	{

		connect(m_surface,SIGNAL(sendPositionCam(int, QVector3D)), view3d,SLOT(setAnimationCamera(int,QVector3D)),Qt::QueuedConnection );
	}


}

void HorizonFolder3DLayer::zScale(float val)
{
	if (m_surface!=nullptr) {
		m_surface->zScale(val);
	}
}

void HorizonFolder3DLayer::hide() {
	m_showOK= false;

	internalHide();

}

void HorizonFolder3DLayer::internalHide() {
	m_showInternal= false;
	//QMutexLocker lock(&m_mutex);
	if(m_surface==nullptr){
		return;
	}


	m_surface->hide();
	m_surface->deleteLater();
	m_surface = nullptr;

	m_cudaRgbTexture->deleteLater();
	m_cudaRgbTexture = nullptr;

	m_cudaSurfaceTexture->deleteLater();
	m_cudaSurfaceTexture = nullptr;

	m_rgbMaterial->hide();
	delete m_rgbMaterial;
	m_rgbMaterial = nullptr;



}

QRect3D HorizonFolder3DLayer::boundingRect() const {

	if(m_rep->currentLayerWithAttribut().getData() == nullptr) return QRect3D();

	int width = m_rep->currentLayerWithAttribut().width();
	int height = m_rep->currentLayerWithAttribut().heightFor3D();
	int depth = m_rep->currentLayerWithAttribut().depth();

	// fill list
	double xmin = std::numeric_limits<double>::max();
	double xmax = std::numeric_limits<double>::lowest();
	double ymin = std::numeric_limits<double>::max();
	double ymax = std::numeric_limits<double>::lowest();
	double zmin = std::numeric_limits<double>::max();
	double zmax = std::numeric_limits<double>::lowest();

	QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();

	for (int i=0; i<=width; i+=width) { // xline
		for(int j=0; j<=height; j+=height) { // sample
			for (int k=0; k<=depth; k+= depth) { // inline
				// apply transform
				double iWorld, jWorld, kWorld;
				m_rep->currentLayerWithAttribut().ijToXYTransfo()->imageToWorld(i, k, iWorld, kWorld);
				data()->sampleTransformation()->direct(j, jWorld);

				QVector3D oriPt(iWorld, jWorld, kWorld);
				QVector3D newPoint = transform*oriPt;

				// get min max
				if (xmin>newPoint.x()) {
					xmin = newPoint.x();
				}
				if (xmax<newPoint.x()) {
					xmax = newPoint.x();
				}
				if (ymin>newPoint.y()) {
					ymin = newPoint.y();
				}
				if (ymax<newPoint.y()) {
					ymax = newPoint.y();
				}
				if (zmin>newPoint.z()) {
					zmin = newPoint.z();
				}
				if (zmax<newPoint.z()) {
					zmax = newPoint.z();
				}
			}
		}
	}

	QRect3D worldBox = QRect3D(xmin, ymin, zmin, xmax-xmin, ymax-ymin, zmax-zmin);
	//	qDebug()<<"worldBox y : "<<ymax-ymin<<"   worldBox z :"<<zmax-zmin ;
	return worldBox;
}

void HorizonFolder3DLayer::refresh() {

}

void HorizonFolder3DLayer::minValueActivated(bool activated) {
	if (m_rgbMaterial) {
		m_rgbMaterial->setMinimumValueActive(activated);
		refresh();
	}
}

void HorizonFolder3DLayer::minValueChanged(float value) {
	if (m_rgbMaterial) {
		m_rgbMaterial->setMinimumValue(value);
		refresh();
	}
}

