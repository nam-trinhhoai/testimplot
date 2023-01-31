#include "fixedrgblayersfromdataset3dlayer.h"
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

#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "volumeboundingmesh.h"
#include "seismic3dabstractdataset.h"
#include "fixedrgblayersfromdatasetrep.h"
#include "fixedrgblayersfromdataset.h"
#include "colortabletexture.h"
#include "cudaimagetexture.h"
#include "qt3dhelpers.h"
#include "surfacemesh.h"
#include "viewqt3d.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "surfacemeshcacheutils.h"
#include <QDebug>

FixedRGBLayersFromDataset3DLayer::FixedRGBLayersFromDataset3DLayer(FixedRGBLayersFromDatasetRep *rep, QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) :
		Graphic3DLayer(parent, root, camera) {
	m_rep = rep;
	m_transform=nullptr;
	m_sliceEntity = nullptr;
	m_colorTexture = nullptr;

	m_cudaRedTexture = nullptr;
	m_cudaGreenTexture = nullptr;
	m_cudaBlueTexture = nullptr;

	m_cudaSurfaceTexture = nullptr;
	m_material = nullptr;
	m_opacityParameter = nullptr;
	m_paletteRedRangeParameter = m_paletteGreenRangeParameter =
			m_paletteBlueRangeParameter = nullptr;

	connect(m_rep->fixedRGBLayersFromDataset()->image(),
			SIGNAL(rangeChanged(unsigned int, const QVector2D &)), this,
			SLOT(rangeChanged(unsigned int, const QVector2D &)));
	connect(m_rep->fixedRGBLayersFromDataset()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(opacityChanged(float)));

	connect(m_rep->fixedRGBLayersFromDataset()->image()->get(0), SIGNAL(dataChanged()), this,
			SLOT(updateRed()));
	connect(m_rep->fixedRGBLayersFromDataset()->image()->get(1), SIGNAL(dataChanged()), this,
			SLOT(updateGreen()));
	connect(m_rep->fixedRGBLayersFromDataset()->image()->get(2), SIGNAL(dataChanged()), this,
			SLOT(updateBlue()));

	connect(m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder(), SIGNAL(dataChanged()),
			this, SLOT(updateIsoSurface()));
}

FixedRGBLayersFromDataset3DLayer::~FixedRGBLayersFromDataset3DLayer() {

}

void FixedRGBLayersFromDataset3DLayer::opacityChanged(float val) {
	m_opacityParameter->setValue(val);
}

void FixedRGBLayersFromDataset3DLayer::rangeChanged(unsigned int i, const QVector2D &value) {
	if (i == 0)
		m_paletteRedRangeParameter->setValue(
				m_rep->fixedRGBLayersFromDataset()->image()->get(0)->rangeRatio());
	else if (i == 1)
		m_paletteGreenRangeParameter->setValue(
				m_rep->fixedRGBLayersFromDataset()->image()->get(1)->rangeRatio());
	else if (i == 2)
		m_paletteBlueRangeParameter->setValue(
				m_rep->fixedRGBLayersFromDataset()->image()->get(2)->rangeRatio());
}

void FixedRGBLayersFromDataset3DLayer::updateTexture(CudaImageTexture *texture,
		CUDAImagePaletteHolder *img) {
	if (texture == nullptr)
		return;

	size_t pointerSize = img->internalPointerSize();
	img->lockPointer();
	texture->setData(
			byteArrayFromRawData((const char*) img->backingPointer(),
					pointerSize));
	img->unlockPointer();
}

void FixedRGBLayersFromDataset3DLayer::updateRed() {
	updateTexture(m_cudaRedTexture, m_rep->fixedRGBLayersFromDataset()->image()->get(0));
	rangeChanged(0, m_rep->fixedRGBLayersFromDataset()->image()->get(0)->range());
}

void FixedRGBLayersFromDataset3DLayer::updateGreen() {
	updateTexture(m_cudaGreenTexture, m_rep->fixedRGBLayersFromDataset()->image()->get(1));
	rangeChanged(1, m_rep->fixedRGBLayersFromDataset()->image()->get(1)->range());
}

void FixedRGBLayersFromDataset3DLayer::updateBlue() {
	updateTexture(m_cudaBlueTexture, m_rep->fixedRGBLayersFromDataset()->image()->get(2));
	rangeChanged(2, m_rep->fixedRGBLayersFromDataset()->image()->get(2)->range());
}

void FixedRGBLayersFromDataset3DLayer::updateIsoSurface() {
	updateTexture(m_cudaSurfaceTexture,
			m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder());
}

FixedRGBLayersFromDataset* FixedRGBLayersFromDataset3DLayer::data() const {
	return m_rep->fixedRGBLayersFromDataset();
}

void FixedRGBLayersFromDataset3DLayer::show() {
	int width = data()->dataset()->width();
	int height = data()->dataset()->height();
	int depth = data()->dataset()->depth();

	m_sliceEntity = new Qt3DCore::QEntity(m_root);
	SurfaceMesh *mesh = new SurfaceMesh();
	mesh->setDimensions(QVector2D(width, depth));
	QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
	QMatrix4x4 ijToXYTranform(data()->dataset()->ijToXYTransfo()->imageToWorldTransformation());

	// swap axis of ijToXYTranform (i,j,k) -> (i,k,j) i:Y, j:Z, k:sample
	const float* tbuf = ijToXYTranform.constData();
	QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
									 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
									 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
									 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);

	transform = transform * ijToXYTranformSwapped;
	mesh->setTransform(transform);

	//Create a material
	m_material = new Qt3DRender::QMaterial();

	// Set the effect on the materials
	if (m_rep->fixedRGBLayersFromDataset()->image()->get(0)->sampleType()==ImageFormats::QSampleType::FLOAT32 &&
			m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder()->sampleType()==ImageFormats::QSampleType::FLOAT32) {
		m_material->setEffect(
				Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/RGBColor_simple.frag",
						"qrc:/shaders/qt3d/simpleHorizonColor.vert"));
	} else if (m_rep->fixedRGBLayersFromDataset()->image()->get(0)->sampleType()==ImageFormats::QSampleType::FLOAT32 &&
			m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder()->sampleType()!=ImageFormats::QSampleType::FLOAT32) {
		m_material->setEffect(
				Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/RGBColor_simple.frag",
						"qrc:/shaders/qt3d/isimpleHorizonColor.vert"));
	} else if (m_rep->fixedRGBLayersFromDataset()->image()->get(0)->sampleType()!=ImageFormats::QSampleType::FLOAT32 &&
			m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder()->sampleType()==ImageFormats::QSampleType::FLOAT32) {
		m_material->setEffect(
				Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/iRGBColor_simple.frag",
						"qrc:/shaders/qt3d/simpleHorizonColor.vert"));
	} else {
		m_material->setEffect(
				Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/iRGBColor_simple.frag",
						"qrc:/shaders/qt3d/isimpleHorizonColor.vert"));
	}

	// Set different parameters on the materials
	m_colorTexture = new ColorTableTexture();
	CUDAImagePaletteHolder *img = m_rep->fixedRGBLayersFromDataset()->image()->get(0);
	m_cudaRedTexture = new CudaImageTexture(img->colorFormat(),
			img->sampleType(), img->width(), img->height());

	img = m_rep->fixedRGBLayersFromDataset()->image()->get(1);
	m_cudaGreenTexture = new CudaImageTexture(img->colorFormat(),
			img->sampleType(), img->width(), img->height());

	img = m_rep->fixedRGBLayersFromDataset()->image()->get(2);
	m_cudaBlueTexture = new CudaImageTexture(img->colorFormat(),
			img->sampleType(), img->width(), img->height());

	img = m_rep->fixedRGBLayersFromDataset()->isoSurfaceHolder();
	m_cudaSurfaceTexture = new CudaImageTexture(img->colorFormat(),
			img->sampleType(), img->width(), img->height());

	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("redMap"),
					m_cudaRedTexture));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("greenMap"),
					m_cudaGreenTexture));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("blueMap"),
					m_cudaBlueTexture));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("surfaceMap"),
					m_cudaSurfaceTexture));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("colormap"),
					m_colorTexture));

	m_paletteRedRangeParameter = new Qt3DRender::QParameter(
			QStringLiteral("redRange"),
			m_rep->fixedRGBLayersFromDataset()->image()->get(0)->rangeRatio());
	m_material->addParameter(m_paletteRedRangeParameter);

	m_paletteGreenRangeParameter = new Qt3DRender::QParameter(
			QStringLiteral("greenRange"),
			m_rep->fixedRGBLayersFromDataset()->image()->get(1)->rangeRatio());
	m_material->addParameter(m_paletteGreenRangeParameter);

	m_paletteBlueRangeParameter = new Qt3DRender::QParameter(
			QStringLiteral("blueRange"),
			m_rep->fixedRGBLayersFromDataset()->image()->get(2)->rangeRatio());
	m_material->addParameter(m_paletteBlueRangeParameter);

	m_opacityParameter = new Qt3DRender::QParameter(QStringLiteral("opacity"),
			m_rep->fixedRGBLayersFromDataset()->image()->opacity());
	m_material->addParameter(m_opacityParameter);
	tbuf = transform.constData();
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("cubeOrigin"),
					tbuf[3*4+1]));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("cubeScale"),
					tbuf[1*4+1]));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("heightThreshold"),
					m_rep->fixedRGBLayersFromDataset()->dataset()->height()-2));

	updateRed();
	updateGreen();
	updateBlue();
	updateIsoSurface();

	m_transform = new Qt3DCore::QTransform();
	m_transform->setScale3D(QVector3D(1, 1, 1));


	m_sliceEntity->addComponent(mesh);
	m_sliceEntity->addComponent(m_material);
	m_sliceEntity->addComponent(m_transform);
}

void FixedRGBLayersFromDataset3DLayer::zScale(float val)
{
	m_transform->setScale3D(QVector3D(1, val, 1));
}

void FixedRGBLayersFromDataset3DLayer::hide() {
	if(m_sliceEntity != nullptr){
		m_sliceEntity->setParent((Qt3DCore::QEntity*) nullptr);
		m_sliceEntity->deleteLater();
		m_sliceEntity = nullptr;
	}
}

QRect3D FixedRGBLayersFromDataset3DLayer::boundingRect() const {

	int width = data()->dataset()->width();
	int height = data()->dataset()->height();
	int depth = data()->dataset()->depth();

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
				Seismic3DAbstractDataset* dataset = data()->dataset();
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

void FixedRGBLayersFromDataset3DLayer::refresh() {

}

