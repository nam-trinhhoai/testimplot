#include "layerrgt3Dlayer.h"

#include <iostream>
#include <cmath>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QEntity>
#include <Qt3DExtras/QPlaneMesh>
#include <Qt3DRender/QTechnique>
#include <Qt3DRender/QParameter>
#include <Qt3DInput/QMouseHandler>
#include <Qt3DRender/QPickEvent>
#include <QMouseDevice>
#include <QWindow>
#include <QRenderPass>
#include <QMaterial>
#include <QEffect>
#include <QObjectPicker>
#include <QMouseEvent>
#include <QCamera>
#include <QMatrix4x4>
#include <QDebug>
#include <Qt3DRender/QObjectPicker>
#include <Qt3DRender/QPickingSettings>
#include <Qt3DRender/QPickTriangleEvent>

#include "cudaimagepaletteholder.h"
#include "volumeboundingmesh.h"
#include "seismic3dabstractdataset.h"
#include "LayerSlice.h"
#include "colortabletexture.h"
#include "cudaimagetexture.h"
#include "qt3dhelpers.h"
#include "surfacemesh.h"
#include "layerrgtrep.h"
#include "viewqt3d.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "surfacemeshcacheutils.h"

LayerRGT3DLayer::LayerRGT3DLayer(LayerRGTRep *rep, QWindow *parent,
		Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) :
		Graphic3DLayer(parent, root, camera) {
	m_rep = rep;
	m_colorTexture = nullptr;
	m_cudaTexture = nullptr;
	m_cudaSurfaceTexture = nullptr;
	m_surface= nullptr;
	m_grayMaterial = nullptr;

	connect(m_rep->layerSlice()->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), this,
			SLOT(updateLookupTable(const LookupTable &)));
	connect(m_rep->layerSlice()->image(),
			SIGNAL(rangeChanged(const QVector2D &)), this,
			SLOT(rangeChanged()));
	connect(m_rep->layerSlice()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(opacityChanged(float)));

	connect(m_rep->layerSlice()->image(), SIGNAL(dataChanged()), this,
			SLOT(update()));
	connect(m_rep->layerSlice()->isoSurfaceHolder(), SIGNAL(dataChanged()),
			this, SLOT(updateIsoSurface()), Qt::AutoConnection);
}

LayerRGT3DLayer::~LayerRGT3DLayer() {


}

void LayerRGT3DLayer::opacityChanged(float val) {
	if (m_surface!=nullptr) {
		m_surface->setOpacity(val);
	}
}

void LayerRGT3DLayer::rangeChanged() {
	if (m_grayMaterial!=nullptr) {
		m_grayMaterial->rangeChanged(m_rep->layerSlice()->image()->rangeRatio());
	}
}

float LayerRGT3DLayer::distanceSigned(QVector3D position, bool* ok)
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
	Seismic3DDataset* dataset = layerSlice()->seismic();
	dataset->ijToXYTransfo()->worldToImage(posTr.x(), posTr.z(), iImage, kImage);


	dataset->sampleTransformation()->indirect(posTr.y(), jImage);
	float multDist =dataset->sampleTransformation()->a();

	QVector3D ijk(iImage,jImage,kImage);
	float distance = m_surface->distanceSigned(ijk,ok);
	return multDist*distance;
}

void LayerRGT3DLayer::updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img ) {
	if (texture == nullptr)
		return;

	size_t pointerSize = img->internalPointerSize();
	img->lockPointer();
	texture->setData(
			byteArrayFromRawData((const char*) img->backingPointer(),
					pointerSize));
	img->unlockPointer();
}

void LayerRGT3DLayer::update() {
	updateTexture(m_cudaTexture,m_rep->layerSlice()->image());
	rangeChanged();
}

void LayerRGT3DLayer::updateIsoSurface() {
	if (m_cudaSurfaceTexture==nullptr) {
		return;
	}
	updateTexture(m_cudaSurfaceTexture,m_rep->layerSlice()->isoSurfaceHolder());

	if (m_surface) {
		m_surface->update(m_rep->layerSlice()->isoSurfaceHolder(), "",
				m_rep->layerSlice()->getSimplifyMeshSteps(),
				m_rep->layerSlice()->getCompressionMesh());
	}
}

void LayerRGT3DLayer::updateLookupTable(const LookupTable &table) {
	if (m_colorTexture == nullptr)
		return;
	CUDAImagePaletteHolder *img = m_rep->layerSlice()->image();
	m_colorTexture->setLookupTable(img->lookupTable());
}

LayerSlice* LayerRGT3DLayer::layerSlice() const {
	return ((LayerSlice*) m_rep->data());
}

void LayerRGT3DLayer::show() {
	int width = layerSlice()->width();
	int height = layerSlice()->height();
	int depth = layerSlice()->depth();

	QMatrix4x4 sceneTransform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
//	Qt3DRender::QLayer* layerTr = dynamic_cast<ViewQt3D*>(m_rep->view())->getLayerTransparent();
	QMatrix4x4 ijToXYTranform(layerSlice()->seismic()->ijToXYTransfo()->imageToWorldTransformation());
	const AffineTransformation* sampleTransform = layerSlice()->seismic()->sampleTransformation();

	// swap axis of ijToXYTranform (i,j,k) -> (i,k,j) i:Y, j:Z, k:sample
	const float* tbuf = ijToXYTranform.constData();
	QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
									 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
									 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
									 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);

	QMatrix4x4 transform = sceneTransform * ijToXYTranformSwapped;

	// Set different parameters on the materials
	m_colorTexture = new ColorTableTexture();
	CUDAImagePaletteHolder *img = m_rep->layerSlice()->image();
	m_cudaTexture = new CudaImageTexture(img->colorFormat(),
			img->sampleType(), img->width(), img->height());

	CUDAImagePaletteHolder *imgSurf = m_rep->layerSlice()->isoSurfaceHolder();
	m_cudaSurfaceTexture = new CudaImageTexture(imgSurf->colorFormat(),
			imgSurf->sampleType(), imgSurf->width(), imgSurf->height());

	tbuf = transform.constData();

	QVector2D ratio = m_rep->layerSlice()->image()->rangeRatio();
	ImageFormats::QSampleType sampleTypeIma = m_rep->layerSlice()->image()->sampleType();
	ImageFormats::QSampleType sampleTypeIso = m_rep->layerSlice()->isoSurfaceHolder()->sampleType();
	float cubeOrigin =  sampleTransform->b() ;
	float cubeScale = sampleTransform->a();

	m_surface = new GenericSurface3DLayer(sceneTransform,ijToXYTranformSwapped);

	update();
	updateIsoSurface();
	updateLookupTable(m_rep->layerSlice()->image()->lookupTable());

	float heightThreshold = m_rep->layerSlice()->height()-2;
	float opacite = m_rep->layerSlice()->image()->opacity();
	m_grayMaterial= new GrayMaterialInitializer(sampleTypeIma,ratio,m_cudaTexture,m_colorTexture);

	m_surface->Show(m_root,transform,width,depth,m_cudaSurfaceTexture,m_grayMaterial,imgSurf,heightThreshold,
			cubeScale,cubeOrigin,m_camera,opacite, sampleTypeIso,m_rep->layerSlice()->getSimplifyMeshSteps(),m_rep->layerSlice()->getCompressionMesh()/*,layerTr*/);

	connect(m_surface,SIGNAL(sendPositionTarget(QVector3D, QVector3D)),this,SLOT(receiveInfosCam(QVector3D,QVector3D)) );

	ViewQt3D *view3d = dynamic_cast<ViewQt3D*>(m_rep->view());
	if(view3d!= nullptr)
	{

		connect(m_surface,SIGNAL(sendPositionCam(int, QVector3D)), view3d,SLOT(setAnimationCamera(int,QVector3D)),Qt::QueuedConnection );
	}



}
void LayerRGT3DLayer::hide() {
	if (m_surface==nullptr) {
		return;
	}

	m_surface->hide();
	m_surface->deleteLater();
	m_surface = nullptr;

	m_cudaSurfaceTexture->deleteLater();
	m_cudaSurfaceTexture = nullptr;
	m_cudaTexture->deleteLater();
	m_cudaTexture = nullptr;

	m_grayMaterial->hide();
	delete m_grayMaterial;
	m_grayMaterial = nullptr;
}

QRect3D LayerRGT3DLayer::boundingRect() const {

	int width = layerSlice()->width();
	int height = layerSlice()->height();
	int depth = layerSlice()->depth();

	QRect3D oriBox = QRect3D(0, 0, 0, width, height, depth);

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
				Seismic3DDataset* dataset = layerSlice()->seismic();
				dataset->ijToXYTransfo()->imageToWorld(i, k, iWorld, kWorld);
				dataset->sampleTransformation()->direct(j, jWorld);

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

void LayerRGT3DLayer::refresh() {

}

void LayerRGT3DLayer::zScale(float val)
{
	if (m_surface) {
		m_surface->zScale(val);
	}
}


