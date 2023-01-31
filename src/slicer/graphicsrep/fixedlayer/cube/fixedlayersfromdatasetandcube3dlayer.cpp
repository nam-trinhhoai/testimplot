#include "fixedlayersfromdatasetandcube3dlayer.h"
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

#include <QPropertyAnimation>

#include "cudaimagepaletteholder.h"
#include "cpuimagepaletteholder.h"
#include "volumeboundingmesh.h"
#include "seismic3dabstractdataset.h"
#include "fixedlayersfromdatasetandcuberep.h"
#include "fixedlayersfromdatasetandcube.h"
#include "colortabletexture.h"
#include "cudaimagetexture.h"
#include "qt3dhelpers.h"

#include "viewqt3d.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "genericsurface3Dlayer.h"
#include "surfacemeshcacheutils.h"
#include <QDebug>

FixedLayersFromDatasetAndCube3DLayer::FixedLayersFromDatasetAndCube3DLayer(
		FixedLayersFromDatasetAndCubeRep *rep, QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) :
		Graphic3DLayer(parent, root, camera) {
	m_rep = rep;
	m_colorTexture = nullptr;

	m_cudaAttributeTexture = nullptr;

	m_cudaSurfaceTexture = nullptr;
	m_grayMaterial =nullptr;
	m_surface = nullptr;

	connect(m_rep->fixedLayersFromDataset()->image(),
			SIGNAL(rangeChanged(const QVector2D &)), this,
			SLOT(rangeChanged(const QVector2D &)));
	connect(m_rep->fixedLayersFromDataset()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(opacityChanged(float)));
	connect(m_rep->fixedLayersFromDataset()->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), this,
			SLOT(updateLookupTable(const LookupTable &)));

	connect(m_rep->fixedLayersFromDataset()->image(), SIGNAL(dataChanged()), this,
			SLOT(updateAttribute()));

	connect(m_rep->fixedLayersFromDataset()->isoSurfaceHolder(), SIGNAL(dataChanged()),
			this, SLOT(updateIsoSurface()));
}

FixedLayersFromDatasetAndCube3DLayer::~FixedLayersFromDatasetAndCube3DLayer() {
	if (m_surface!=nullptr) {
		hide();
	}
}

void FixedLayersFromDatasetAndCube3DLayer::opacityChanged(float val) {
	if (m_surface!=nullptr) {
		m_surface->setOpacity(val);
	}
}

void FixedLayersFromDatasetAndCube3DLayer::rangeChanged(const QVector2D &value) {
	if (m_grayMaterial!=nullptr) {
		m_grayMaterial->rangeChanged(m_rep->fixedLayersFromDataset()->image()->rangeRatio());
	}
}

void FixedLayersFromDatasetAndCube3DLayer::updateTexture(CudaImageTexture *texture,
		CPUImagePaletteHolder *img) {
	if (texture == nullptr )//|| m_sliceEntity==nullptr)
		return;

//	size_t pointerSize = img->internalPointerSize();
	//img->lockPointer();

	texture->setData(img->getDataAsByteArray());
	//		byteArrayFromRawData((const char*) img->backingPointer(),
	//				pointerSize));
	//img->unlockPointer();
}

void FixedLayersFromDatasetAndCube3DLayer::updateLookupTable(const LookupTable &table) {
	if (m_colorTexture == nullptr)
		return;
	CPUImagePaletteHolder *img = m_rep->fixedLayersFromDataset()->image();
	m_colorTexture->setLookupTable(img->lookupTable());
}

void FixedLayersFromDatasetAndCube3DLayer::updateAttribute() {
	updateTexture(m_cudaAttributeTexture, m_rep->fixedLayersFromDataset()->image());
	rangeChanged(m_rep->fixedLayersFromDataset()->image()->range());
}

void FixedLayersFromDatasetAndCube3DLayer::updateIsoSurface() {
	updateTexture(m_cudaSurfaceTexture,
			m_rep->fixedLayersFromDataset()->isoSurfaceHolder());

	bool cacheExist =m_rep->fixedLayersFromDataset()->isIndexCache(m_rep->fixedLayersFromDataset()->currentImageIndex());
		if(cacheExist && m_surface != nullptr)
		{
			SurfaceMeshCache* meshCache = m_rep->fixedLayersFromDataset()->getMeshCache(m_rep->fixedLayersFromDataset()->currentImageIndex());
			if(meshCache != nullptr) m_surface->reloadFromCache(*meshCache);
			else cacheExist = false;

		}
		 if(!cacheExist && m_surface != nullptr)
		{
			m_surface->update(m_rep->fixedLayersFromDataset()->isoSurfaceHolder(),m_rep->fixedLayersFromDataset()->getCurrentObjFile(),m_rep->fixedLayersFromDataset()->getSimplifyMeshSteps(),m_rep->fixedLayersFromDataset()->getCompressionMesh() );
		}
}

FixedLayersFromDatasetAndCube* FixedLayersFromDatasetAndCube3DLayer::data() const {
	return m_rep->fixedLayersFromDataset();
}

float FixedLayersFromDatasetAndCube3DLayer::distanceSigned(QVector3D position, bool* ok)
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
	data()->ijToXYTransfo()->worldToImage(posTr.x(), posTr.z(), iImage, kImage);


	float multDist = 1.0f;
	if (m_rep->fixedLayersFromDataset()->isIsoInT())
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

void FixedLayersFromDatasetAndCube3DLayer::show() {
	int width = data()->width();
	int depth = data()->depth();

	QMatrix4x4 sceneTransform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();



//	Qt3DRender::QLayer* layerTr = dynamic_cast<ViewQt3D*>(m_rep->view())->getLayerTransparent();
	QMatrix4x4 ijToXYTranform(data()->ijToXYTransfo()->imageToWorldTransformation());

	// swap axis of ijToXYTranform (i,j,k) -> (i,k,j) i:Y, j:Z, k:sample
	const float* tbuf = ijToXYTranform.constData();
	QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
									 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
									 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
									 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);

	QMatrix4x4 transform = sceneTransform * ijToXYTranformSwapped;

	ImageFormats::QSampleType sampleType = m_rep->fixedLayersFromDataset()->image()->sampleType();
	ImageFormats::QSampleType sampleTypeIso = m_rep->fixedLayersFromDataset()->isoSurfaceHolder()->sampleType();

	// Set different parameters on the materials
	m_colorTexture = new ColorTableTexture();
	CPUImagePaletteHolder *imgAttr = m_rep->fixedLayersFromDataset()->image();
	m_cudaAttributeTexture = new CudaImageTexture(imgAttr->colorFormat(),
			imgAttr->sampleType(), imgAttr->width(), imgAttr->height());

	CPUImagePaletteHolder *imgSurf = m_rep->fixedLayersFromDataset()->isoSurfaceHolder();
	m_cudaSurfaceTexture = new CudaImageTexture(imgSurf->colorFormat(),
			imgSurf->sampleType(), imgSurf->width(), imgSurf->height());

	tbuf = transform.constData();
	if (m_rep->fixedLayersFromDataset()->isIsoInT()) {
		m_cubeOrigin =0;//tbuf[3*4+1];
		m_cubeScale = 1;//tbuf[1*4+1];
	} else {
		const AffineTransformation* sampleTransform = data()->sampleTransformation();
		m_cubeOrigin =sampleTransform->b();// tbuf[1*4+1] * sampleTransform->b() + tbuf[3*4+1];
		m_cubeScale = sampleTransform->a();//tbuf[1*4+1] * sampleTransform->a();
	}
	const AffineTransformation* sampleTransform = data()->sampleTransformation();
	float heightThreshold = ( m_rep->fixedLayersFromDataset()->heightFor3D()-2) *sampleTransform->a()  +sampleTransform->b() ;
	float opacite = m_rep->fixedLayersFromDataset()->image()->opacity();




	QVector2D ratioAttr = m_rep->fixedLayersFromDataset()->image()->rangeRatio();

	updateAttribute();

	updateIsoSurface();

	updateLookupTable(m_rep->fixedLayersFromDataset()->image()->lookupTable());

	m_surface = new GenericSurface3DLayer(sceneTransform,ijToXYTranformSwapped,m_rep->fixedLayersFromDataset()->getCurrentObjFile());
	m_grayMaterial = new GrayMaterialInitializer(sampleType,ratioAttr,m_cudaAttributeTexture,m_colorTexture);
	m_surface->Show(m_root,transform,width,depth,m_cudaSurfaceTexture,m_grayMaterial,
			imgSurf,heightThreshold,m_cubeScale,m_cubeOrigin,m_camera,opacite,sampleTypeIso,m_rep->fixedLayersFromDataset()->getSimplifyMeshSteps(), m_rep->fixedLayersFromDataset()->getCompressionMesh()/*,layerTr*/);

	connect(m_surface,SIGNAL(sendPositionTarget(QVector3D, QVector3D)),this,SLOT(receiveInfosCam(QVector3D,QVector3D)) );

	ViewQt3D *view3d = dynamic_cast<ViewQt3D*>(m_rep->view());
	if(view3d!= nullptr)
	{

		connect(m_surface,SIGNAL(sendPositionCam(int, QVector3D)), view3d,SLOT(setAnimationCamera(int,QVector3D)),Qt::QueuedConnection );
	}

}

void FixedLayersFromDatasetAndCube3DLayer::zScale(float val)
{
	if (m_surface!=nullptr) {
		m_surface->zScale(val);
	}
}

void FixedLayersFromDatasetAndCube3DLayer::hide() {
	m_surface->hide();
	m_surface->deleteLater();
	m_surface = nullptr;

	m_colorTexture->deleteLater();
	m_colorTexture = nullptr;

	m_cudaAttributeTexture->deleteLater();
	m_cudaAttributeTexture = nullptr;

	m_cudaSurfaceTexture->deleteLater();
	m_cudaSurfaceTexture = nullptr;

	m_grayMaterial->hide();
	delete m_grayMaterial;
	m_grayMaterial = nullptr;

}

QRect3D FixedLayersFromDatasetAndCube3DLayer::boundingRect() const {
	int width = data()->width();
	int height = data()->heightFor3D();
	int depth = data()->depth();

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
				data()->ijToXYTransfo()->imageToWorld(i, k, iWorld, kWorld);
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
	return worldBox;
}

void FixedLayersFromDatasetAndCube3DLayer::refresh() {

}

