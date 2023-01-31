#include "rgblayerrgt3dlayer.h"
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
#include <QDebug>
#include <Qt3DRender/QPickTriangleEvent>
#include <Qt3DCore/QBuffer>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DExtras/QPerVertexColorMaterial>

#include <QPropertyAnimation>

#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "volumeboundingmesh.h"
#include "seismic3dabstractdataset.h"
#include "rgblayerrgtrep.h"
#include "rgblayerslice.h"
#include "LayerSlice.h"
#include "colortabletexture.h"
#include "cudaimagetexture.h"
#include "qt3dhelpers.h"
#include "surfacemesh.h"
#include "viewqt3d.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "surfacemeshcacheutils.h"
#include <QApplication>

RGBLayerRGT3DLayer::RGBLayerRGT3DLayer(RGBLayerRGTRep *rep, QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) :
		Graphic3DLayer(parent, root, camera) {
	m_rep = rep;

	m_colorTexture = nullptr;

	m_cudaRedTexture = nullptr;
	m_cudaGreenTexture = nullptr;
	m_cudaBlueTexture = nullptr;

	m_cudaSurfaceTexture = nullptr;
	m_surface= nullptr;
	m_rgbMaterial = nullptr;

	connect(m_rep->rgbLayerSlice()->image(),
			SIGNAL(rangeChanged(unsigned int, const QVector2D &)), this,
			SLOT(rangeChanged(unsigned int, const QVector2D &)));
	connect(m_rep->rgbLayerSlice()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(opacityChanged(float)));

	connect(m_rep->rgbLayerSlice()->image()->get(0), SIGNAL(dataChanged()), this,
			SLOT(updateRed()));
	connect(m_rep->rgbLayerSlice()->image()->get(1), SIGNAL(dataChanged()), this,
			SLOT(updateGreen()));
	connect(m_rep->rgbLayerSlice()->image()->get(2), SIGNAL(dataChanged()), this,
			SLOT(updateBlue()));

	connect(m_rep->rgbLayerSlice()->layerSlice()->isoSurfaceHolder(), SIGNAL(dataChanged()),
			this, SLOT(updateIsoSurface()));

	connect(m_rep->rgbLayerSlice(), SIGNAL(minimumValueActivated(bool)), this, SLOT(minValueActivated(bool)));
	connect(m_rep->rgbLayerSlice(), SIGNAL(minimumValueChanged(float)), this, SLOT(minValueChanged(float)));
}

RGBLayerRGT3DLayer::~RGBLayerRGT3DLayer() {

}

void RGBLayerRGT3DLayer::opacityChanged(float val) {
	if (m_surface!=nullptr) {
		m_surface->setOpacity(val);
	}
}

void RGBLayerRGT3DLayer::rangeChanged(unsigned int i, const QVector2D &value) {
	if (m_rgbMaterial!=nullptr) {
		m_rgbMaterial->rangeChanged(i, m_rep->rgbLayerSlice()->image()->get(i)->rangeRatio());
	}
}

float RGBLayerRGT3DLayer::distanceSigned(QVector3D position, bool* ok)
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
	Seismic3DDataset* dataset = rgbLayerSlice()->layerSlice()->seismic();
	dataset->ijToXYTransfo()->worldToImage(posTr.x(), posTr.z(), iImage, kImage);


	dataset->sampleTransformation()->indirect(posTr.y(), jImage);
	float multDist =dataset->sampleTransformation()->a();
	/*if (m_rep->fixedRGBLayersFromDataset()->isIsoInT())
	{
		jImage = posTr.y();
	}
	else
	{
		data()->sampleTransformation()->indirect(posTr.y(), jImage);
		multDist = data()->sampleTransformation()->a();
	}*/

	QVector3D ijk(iImage,jImage,kImage);
	float distance = m_surface->distanceSigned(ijk,ok);
	return multDist*distance;

}

void RGBLayerRGT3DLayer::updateTexture(CudaImageTexture *texture,
		CUDAImagePaletteHolder *img) {
	if (texture == nullptr)
		return;

	size_t pointerSize = img->internalPointerSize();
	img->lockPointer();
	texture->setData(
			byteArrayFromRawData((const char*) img->backingPointer(),
					pointerSize));
	img->unlockPointer();

	//QApplication::processEvents();

	//qDebug()<<" RGBLayerRGT3DLayer update texture";
}

void RGBLayerRGT3DLayer::updateRed() {

	updateTexture(m_cudaRedTexture, m_rep->rgbLayerSlice()->image()->get(0));
	rangeChanged(0, m_rep->rgbLayerSlice()->image()->get(0)->range());


}

void RGBLayerRGT3DLayer::updateGreen() {

	updateTexture(m_cudaGreenTexture, m_rep->rgbLayerSlice()->image()->get(1));
	rangeChanged(1, m_rep->rgbLayerSlice()->image()->get(1)->range());


}

void RGBLayerRGT3DLayer::updateBlue() {

	updateTexture(m_cudaBlueTexture, m_rep->rgbLayerSlice()->image()->get(2));
	rangeChanged(2, m_rep->rgbLayerSlice()->image()->get(2)->range());


}

void RGBLayerRGT3DLayer::updateIsoSurface() {

	if (m_cudaSurfaceTexture==nullptr) {
		return;
	}

	updateTexture(m_cudaSurfaceTexture,
			m_rep->rgbLayerSlice()->layerSlice()->isoSurfaceHolder());
	if (m_surface) {
		m_surface->update(m_rep->rgbLayerSlice()->layerSlice()->isoSurfaceHolder(), "",
				m_rep->rgbLayerSlice()->layerSlice()->getSimplifyMeshSteps(),
				m_rep->rgbLayerSlice()->layerSlice()->getCompressionMesh());
	}


}

RGBLayerSlice* RGBLayerRGT3DLayer::rgbLayerSlice() const {
	return m_rep->rgbLayerSlice();
}

void RGBLayerRGT3DLayer::show() {


	int width = rgbLayerSlice()->layerSlice()->width();
	int height = rgbLayerSlice()->layerSlice()->height();
	int depth = rgbLayerSlice()->layerSlice()->depth();




	QMatrix4x4 sceneTransform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
	//Qt3DRender::QLayer* layerTr = dynamic_cast<ViewQt3D*>(m_rep->view())->getLayerTransparent();
	QMatrix4x4 ijToXYTranform(rgbLayerSlice()->layerSlice()->seismic()->ijToXYTransfo()->imageToWorldTransformation());
	const AffineTransformation* sampleTransform = rgbLayerSlice()->layerSlice()->seismic()->sampleTransformation();


	// swap axis of ijToXYTranform (i,j,k) -> (i,k,j) i:Y, j:Z, k:sample
	const float* tbuf = ijToXYTranform.constData();
	QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
									 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
									 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
									 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);

	QMatrix4x4 transform = sceneTransform * ijToXYTranformSwapped;

	// Set different parameters on the materials
	m_colorTexture = new ColorTableTexture();
	CUDAImagePaletteHolder *imgRed = m_rep->rgbLayerSlice()->image()->get(0);
	m_cudaRedTexture = new CudaImageTexture(imgRed->colorFormat(),
			imgRed->sampleType(), imgRed->width(), imgRed->height());


		CUDAImagePaletteHolder *imgGreen = m_rep->rgbLayerSlice()->image()->get(1);
	m_cudaGreenTexture = new CudaImageTexture(imgGreen->colorFormat(),
			imgGreen->sampleType(), imgGreen->width(), imgGreen->height());



	CUDAImagePaletteHolder *imgBlue = m_rep->rgbLayerSlice()->image()->get(2);
	m_cudaBlueTexture = new CudaImageTexture(imgBlue->colorFormat(),
			imgBlue->sampleType(), imgBlue->width(), imgBlue->height());



	CUDAImagePaletteHolder *imgSurf = m_rep->rgbLayerSlice()->layerSlice()->isoSurfaceHolder();
	m_cudaSurfaceTexture = new CudaImageTexture(ImageFormats::QColorFormat::GRAY,
			imgSurf->sampleType(), imgSurf->width(), imgSurf->height());

	tbuf = transform.constData();

	QVector2D ratioRed = m_rep->rgbLayerSlice()->image()->get(0)->rangeRatio();
	QVector2D ratioGreen = m_rep->rgbLayerSlice()->image()->get(1)->rangeRatio();
	QVector2D ratioBlue = m_rep->rgbLayerSlice()->image()->get(2)->rangeRatio();


	ImageFormats::QSampleType sampleTypeIma = m_rep->rgbLayerSlice()->image()->get(0)->sampleType();
	ImageFormats::QSampleType sampleTypeIso = m_rep->rgbLayerSlice()->layerSlice()->isoSurfaceHolder()->sampleType();
  //  m_cubeOrigin = tbuf[1*4+1] * sampleTransform->b() + tbuf[3*4+1];
  //  m_cubeScale = tbuf[1*4+1] * sampleTransform->a();
    m_cubeOrigin =  sampleTransform->b() ;
       m_cubeScale = sampleTransform->a();

    m_surface = new GenericSurface3DLayer(sceneTransform,ijToXYTranformSwapped);

    QApplication::processEvents();
	updateRed();
	updateGreen();
	updateBlue();

    updateIsoSurface();

	QApplication::processEvents();

    float heightThreshold = m_rep->rgbLayerSlice()->layerSlice()->height()-2;
    float opacite = m_rep->rgbLayerSlice()->image()->opacity();
    m_rgbMaterial= new RGBMaterialInitializer(sampleTypeIma,ratioRed,ratioGreen,ratioBlue,m_cudaRedTexture,m_cudaGreenTexture,m_cudaBlueTexture);

    m_surface->Show(m_root,transform,width,depth,m_cudaSurfaceTexture,m_rgbMaterial,imgSurf,heightThreshold,
            m_cubeScale,m_cubeOrigin,m_camera,opacite, sampleTypeIso,m_rep->rgbLayerSlice()->layerSlice()->getSimplifyMeshSteps(),m_rep->rgbLayerSlice()->layerSlice()->getCompressionMesh()/*,layerTr*/);

    connect(m_surface,SIGNAL(sendPositionTarget(QVector3D, QVector3D)),this,SLOT(receiveInfosCam(QVector3D,QVector3D)) );

    ViewQt3D *view3d = dynamic_cast<ViewQt3D*>(m_rep->view());
	if(view3d!= nullptr)
	{

		connect(m_surface,SIGNAL(sendPositionCam(int, QVector3D)), view3d,SLOT(setAnimationCamera(int,QVector3D)),Qt::QueuedConnection );
	}

}

void RGBLayerRGT3DLayer::zScale(float val)
{
	//m_transform->setScale3D(QVector3D(1, val, 1));
	m_surface->zScale(val);

}

void RGBLayerRGT3DLayer::hide() {
	if (m_surface==nullptr) {
		return;
	}

	m_surface->hide();
	m_surface->deleteLater();
	m_surface = nullptr;

	m_cudaSurfaceTexture->deleteLater();
	m_cudaSurfaceTexture = nullptr;
	m_cudaRedTexture->deleteLater();
	m_cudaRedTexture = nullptr;
	m_cudaGreenTexture->deleteLater();
	m_cudaGreenTexture = nullptr;
	m_cudaBlueTexture->deleteLater();
	m_cudaBlueTexture = nullptr;

	m_rgbMaterial->hide();
	delete m_rgbMaterial;
	m_rgbMaterial = nullptr;
}

QRect3D RGBLayerRGT3DLayer::boundingRect() const {

	int width = rgbLayerSlice()->layerSlice()->width();
	int height = rgbLayerSlice()->layerSlice()->height();
	int depth = rgbLayerSlice()->layerSlice()->depth();

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
				Seismic3DDataset* dataset = rgbLayerSlice()->layerSlice()->seismic();
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

void RGBLayerRGT3DLayer::refresh() {

}

void RGBLayerRGT3DLayer::minValueActivated(bool activated) {
	if (m_rgbMaterial) {
		m_rgbMaterial->setMinimumValueActive(activated);
		refresh();
	}
}

void RGBLayerRGT3DLayer::minValueChanged(float value) {
	if (m_rgbMaterial) {
		m_rgbMaterial->setMinimumValue(value);
		refresh();
	}
}

