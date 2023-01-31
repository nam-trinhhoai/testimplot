#include "fixedrgblayersfromdatasetandcube3dlayer.h"
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

FixedRGBLayersFromDatasetAndCube3DLayer::FixedRGBLayersFromDatasetAndCube3DLayer(FixedRGBLayersFromDatasetAndCubeRep *rep, QWindow *parent, Qt3DCore::QEntity *root,
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

	connect(m_rep->fixedRGBLayersFromDataset()->image(),
			SIGNAL(rangeChanged(unsigned int, const QVector2D &)), this,
			SLOT(rangeChanged(unsigned int, const QVector2D &)));
	connect(m_rep->fixedRGBLayersFromDataset()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(opacityChanged(float)));

	connect(m_rep->fixedRGBLayersFromDataset()->image(), SIGNAL(dataChanged()), this,
			SLOT(updateRgb()));

	connect(m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder(), SIGNAL(dataChanged()),
			this, SLOT(updateIsoSurface()));

	connect(m_rep->fixedRGBLayersFromDataset(), SIGNAL(minimumValueActivated(bool)), this, SLOT(minValueActivated(bool)));
	connect(m_rep->fixedRGBLayersFromDataset(), SIGNAL(minimumValueChanged(float)), this, SLOT(minValueChanged(float)));
}

FixedRGBLayersFromDatasetAndCube3DLayer::~FixedRGBLayersFromDatasetAndCube3DLayer() {
	hide();
}

void FixedRGBLayersFromDatasetAndCube3DLayer::opacityChanged(float val) {
	if (m_surface!=nullptr) {
		m_surface->setOpacity(val);
	}
}

void FixedRGBLayersFromDatasetAndCube3DLayer::rangeChanged(unsigned int i, const QVector2D &value) {
	if (m_rgbMaterial!=nullptr) {
		QVector2D rangeRatio(0, 1);
		if (i==0) {
			rangeRatio = m_rep->fixedRGBLayersFromDataset()->image()->redRangeRatio();
		} else if (i==1) {
			rangeRatio = m_rep->fixedRGBLayersFromDataset()->image()->greenRangeRatio();
		} else if (i==2) {
			rangeRatio = m_rep->fixedRGBLayersFromDataset()->image()->blueRangeRatio();
		}
		m_rgbMaterial->rangeChanged(i, rangeRatio);
	}
}


float FixedRGBLayersFromDatasetAndCube3DLayer::distanceSigned(QVector3D position, bool* ok)
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
	if (m_rep->fixedRGBLayersFromDataset()->isIsoInT())
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

void FixedRGBLayersFromDatasetAndCube3DLayer::updateTexture(CudaImageTexture *texture,
		IImagePaletteHolder *img) {
	if (texture == nullptr )//|| m_sliceEntity==nullptr)
		return;


	//auto start = std::chrono::steady_clock::now();
	texture->setData(img->getDataAsByteArray());

	//auto end = std::chrono::steady_clock::now();
	//qDebug() << "update texture : " << std::chrono::duration<double, std::milli>(end - start).count();
}

void FixedRGBLayersFromDatasetAndCube3DLayer::updateTexture(CudaImageTexture *texture,
		CUDARGBInterleavedImage *img) {
	if (texture == nullptr )//|| m_sliceEntity==nullptr)
		return;
	//auto start = std::chrono::steady_clock::now();
	size_t pointerSize = img->internalPointerSize();
	img->lockPointer();
	texture->setData(img->byteArray());
	img->unlockPointer();



	//auto end = std::chrono::steady_clock::now();
	//qDebug() << "update texture RGB: " << std::chrono::duration<double, std::milli>(end - start).count();
}

void FixedRGBLayersFromDatasetAndCube3DLayer::updateRgb() {

	updateTexture(m_cudaRgbTexture, m_rep->fixedRGBLayersFromDataset()->image());

	rangeChanged(0, m_rep->fixedRGBLayersFromDataset()->image()->redRange());
	rangeChanged(1, m_rep->fixedRGBLayersFromDataset()->image()->greenRange());
	rangeChanged(2, m_rep->fixedRGBLayersFromDataset()->image()->blueRange());
}

/*
template<typename InputType>
struct UpdateIsoSurfaceKernel2 {
	static void run(int meshWidth, int meshHeight, int textureWidth, int textureHeight,
			CUDAImagePaletteHolder* palette, float heightThreshold, float cubeOrigin,
			float cubeScale, float* rawVertexArray){//, float* rawVertexArrayOld) {


		qDebug()<<"============> RUN 2 ";

		palette->lockPointer();
		InputType* tab = static_cast<InputType*>(palette->backingPointer());

		#pragma omp parallel for
		for (int i = 0; i < meshWidth; i++)
		{
			for (int j = 0; j < meshHeight; j++)
			{
				int bufferIndexMesh = j * meshWidth + i;

				int textureIndexWidth = ((float) i / (float ) meshWidth) * (float) textureWidth;
				textureIndexWidth = fmin(textureIndexWidth, textureWidth - 1);
				int textureIndexHeight = ((float) j / (float ) meshHeight) * (float) textureHeight;
				textureIndexHeight = fmin(textureIndexHeight, textureHeight - 1);



				double heightValue = tab[textureIndexWidth + textureIndexHeight * textureWidth];
				//palette->valueAt(textureIndexWidth, textureIndexHeight, heightValue);
				// workaround for holes: search for valid height in the neighborhood
				int xIndex = textureIndexWidth;
				int yIndex = textureIndexHeight;

				bool isValid=  true;



				if(heightValue > heightThreshold )
				{
					double lastvalue = heightValue;
					int xDir, yDir;
					textureIndexWidth > (textureWidth / 2) ? xDir = -1 : xDir = 1;
					textureIndexHeight > (textureHeight / 2) ? yDir = -1 : yDir = 1;
					xIndex += xDir;
					yIndex += yDir;
					if (xIndex > textureWidth && yIndex > textureHeight) {
						//qDebug()  << "ERROR";
						isValid = false;
//                        errorCountSearchInit++;
					}
					while (xIndex < textureWidth && xIndex>=0 && yIndex < textureHeight && yIndex>=0) {

						//palette->valueAt(xIndex, yIndex, heightValue);
						heightValue = tab[xIndex + yIndex * textureWidth];

						//qDebug() << xIndex << yIndex << heightValue;
						if(heightValue < heightThreshold) {
						//	qDebug()<<" heightThreshold: " <<heightThreshold<< " , last heightValue: " <<lastvalue<< " , new heightValue: " <<heightValue;
							//qDebug() << "success";
							break;
						}
						xIndex += xDir;
						yIndex += yDir;
					}
					if (xIndex >= textureWidth || yIndex >= textureHeight || xIndex<0 || yIndex<0) {
						//qDebug()  << "ERROR";
						isValid = false;
//                        errorCountNoValidHeight++;
					}
				}

				float nbelem= 1.0f;
				if(textureIndexWidth < textureWidth-1)
				{
					if(tab[textureIndexWidth+1 + textureIndexHeight * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth+1 + textureIndexHeight * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth > 0)
				{
					if(tab[textureIndexWidth-1 + textureIndexHeight * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth-1 + textureIndexHeight * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexHeight > 0)
				{
					if(tab[textureIndexWidth + (textureIndexHeight-1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth + (textureIndexHeight-1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexHeight < textureHeight-1)
				{
					if(tab[textureIndexWidth + (textureIndexHeight+1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth + (textureIndexHeight+1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth < textureWidth-1 && textureIndexHeight > 0)
				{
					if(tab[textureIndexWidth+1 + (textureIndexHeight-1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth+1 + (textureIndexHeight-1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth > 0 && textureIndexHeight < textureHeight-1)
				{
					if(tab[textureIndexWidth-1 + (textureIndexHeight+1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth-1 + (textureIndexHeight+1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth > 0 && textureIndexHeight > 0)
				{
					if(tab[textureIndexWidth-1 + (textureIndexHeight-1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth-1 + (textureIndexHeight-1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth < textureWidth-1 && textureIndexHeight < textureHeight-1)
				{
					if(tab[textureIndexWidth+1 + (textureIndexHeight+1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth+1 + (textureIndexHeight+1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				heightValue/= nbelem;
				float newHeightValue =  cubeOrigin + cubeScale *heightValue;
			//	if (m_rep->fixedRGBLayersFromDataset()->isIsoInT())
			//		newHeightValue = heightValue;




			//	rawVertexArray[(bufferIndexMesh * 3)] = rawVertexArrayOld[bufferIndexMesh * 3];
				rawVertexArray[(bufferIndexMesh * 3) + 1] = newHeightValue;
			//   rawVertexArray[(bufferIndexMesh * 3) + 2] = rawVertexArrayOld[bufferIndexMesh * 3 + 2];

			}
		}
		palette->unlockPointer();
	}
};*/

void FixedRGBLayersFromDatasetAndCube3DLayer::updateIsoSurface() {
	//std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	updateTexture(m_cudaSurfaceTexture,m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder());

	bool cacheExist =m_rep->fixedRGBLayersFromDataset()->isIndexCache(m_rep->fixedRGBLayersFromDataset()->currentImageIndex());
	if(cacheExist && m_surface != nullptr)
	{
		SurfaceMeshCache* meshCache = m_rep->fixedRGBLayersFromDataset()->getMeshCache(m_rep->fixedRGBLayersFromDataset()->currentImageIndex());
		if(meshCache != nullptr) m_surface->reloadFromCache(*meshCache);
		else cacheExist = false;

	}
	 if(!cacheExist && m_surface != nullptr)
	{
		m_surface->update(m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder(), ""/*,m_rep->fixedRGBLayersFromDataset()->getCurrentObjFile()*/,m_rep->fixedRGBLayersFromDataset()->getSimplifyMeshSteps(),m_rep->fixedRGBLayersFromDataset()->getCompressionMesh() );
	}
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//qDebug() << "layer update iso : " << std::chrono::duration<double, std::milli>(end-start).count();

}

FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCube3DLayer::data() const {
	return m_rep->fixedRGBLayersFromDataset();
}

void FixedRGBLayersFromDatasetAndCube3DLayer::show() {

	qDebug()<<"start FixedRGBLayersFromDatasetAndCube3DLayer";
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	int width = data()->width();
	int depth = data()->depth();

/*	m_sliceEntity = new Qt3DCore::QEntity(m_root);
	m_EntityMesh = new SurfaceMesh();
	m_EntityMesh->setDimensions(QVector2D(width, depth));*/
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

	// Set different parameters on the materials
	m_colorTexture = new ColorTableTexture();
	CUDARGBInterleavedImage *imgRgb = m_rep->fixedRGBLayersFromDataset()->image();
	m_cudaRgbTexture = new CudaImageTexture(imgRgb->colorFormat(), imgRgb->sampleType(), imgRgb->width(),
			imgRgb->height());

	CPUImagePaletteHolder *imgSurf = m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder();
	m_cudaSurfaceTexture = new CudaImageTexture(imgSurf->colorFormat(), imgSurf->sampleType(), imgSurf->width(),
			imgSurf->height());


	tbuf = transform.constData();
	if (m_rep->fixedRGBLayersFromDataset()->isIsoInT()) {
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
	 float heightThreshold = ( m_rep->fixedRGBLayersFromDataset()->heightFor3D()-2) *sampleTransform->a()  +sampleTransform->b() ;
	 float opacite = m_rep->fixedRGBLayersFromDataset()->image()->opacity();




	 QVector2D ratioRed = m_rep->fixedRGBLayersFromDataset()->image()->redRangeRatio();
	 QVector2D ratioGreen =m_rep->fixedRGBLayersFromDataset()->image()->greenRangeRatio();
	 QVector2D ratioBlue = m_rep->fixedRGBLayersFromDataset()->image()->blueRangeRatio();

	updateRgb();
	updateIsoSurface();

	m_surface = new GenericSurface3DLayer(sceneTransform,ijToXYTranformSwapped, ""/*,m_rep->fixedRGBLayersFromDataset()->getCurrentObjFile()*/);
	m_rgbMaterial = new RGBInterleavedMaterialInitializer(imgRgb->sampleType(),ratioRed,ratioGreen,ratioBlue,m_cudaRgbTexture);

	 m_surface->Show(m_root,transform,width,depth,m_cudaSurfaceTexture,m_rgbMaterial,imgSurf,
			 heightThreshold,m_cubeScale,m_cubeOrigin,m_camera,opacite,imgSurf->sampleType(),m_rep->fixedRGBLayersFromDataset()->getSimplifyMeshSteps(),m_rep->fixedRGBLayersFromDataset()->getCompressionMesh()/*,layerTr*/);

	 connect(m_surface,SIGNAL(sendPositionTarget(QVector3D, QVector3D)),this,SLOT(receiveInfosCam(QVector3D,QVector3D)) );

	 ViewQt3D *view3d = dynamic_cast<ViewQt3D*>(m_rep->view());
	if(view3d!= nullptr)
	{

		connect(m_surface,SIGNAL(sendPositionCam(int, QVector3D)), view3d,SLOT(setAnimationCamera(int,QVector3D)),Qt::QueuedConnection );
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	qDebug() << "FixedRGBLayersFromDatasetAndCube3DLayer::show: " << std::chrono::duration<double, std::milli>(end-start).count();
}

void FixedRGBLayersFromDatasetAndCube3DLayer::zScale(float val)
{
	if (m_surface!=nullptr) {
		m_surface->zScale(val);
	}
}

void FixedRGBLayersFromDatasetAndCube3DLayer::hide() {

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

QRect3D FixedRGBLayersFromDatasetAndCube3DLayer::boundingRect() const {

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
	//	qDebug()<<"worldBox y : "<<ymax-ymin<<"   worldBox z :"<<zmax-zmin ;
	return worldBox;
}

void FixedRGBLayersFromDatasetAndCube3DLayer::refresh() {

}

void FixedRGBLayersFromDatasetAndCube3DLayer::minValueActivated(bool activated) {
	if (m_rgbMaterial) {
		m_rgbMaterial->setMinimumValueActive(activated);
		refresh();
	}
}

void FixedRGBLayersFromDatasetAndCube3DLayer::minValueChanged(float value) {
	if (m_rgbMaterial) {
		m_rgbMaterial->setMinimumValue(value);
		refresh();
	}
}

