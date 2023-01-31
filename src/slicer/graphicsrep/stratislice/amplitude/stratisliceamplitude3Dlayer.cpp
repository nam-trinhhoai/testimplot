#include "stratisliceamplitude3Dlayer.h"
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
#include "volumeboundingmesh.h"
#include "seismic3dabstractdataset.h"
#include "stratisliceamplituderep.h"
#include "amplitudestratisliceattribute.h"
#include "stratislice.h"
#include "colortabletexture.h"
#include "cudaimagetexture.h"
#include "qt3dhelpers.h"
#include "surfacemesh.h"
#include "viewqt3d.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "surfacemeshcacheutils.h"
#include <QDebug>

StratiSliceAmplitude3DLayer::StratiSliceAmplitude3DLayer(StratiSliceAmplitudeRep *rep, QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) :
		Graphic3DLayer(parent, root, camera) {
	m_rep = rep;
	m_transform=nullptr;
	m_sliceEntity = nullptr;
	m_colorTexture = nullptr;
	m_cudaTexture = nullptr;
	m_cudaSurfaceTexture = nullptr;
	m_material = nullptr;
	m_opacityParameter = nullptr;
	m_paletteRangeParameter = nullptr;

	connect(m_rep->stratiSliceAttribute()->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), this,
			SLOT(updateLookupTable(const LookupTable &)));
	connect(m_rep->stratiSliceAttribute()->image(),
			SIGNAL(rangeChanged(const QVector2D &)), this,
			SLOT(rangeChanged()));
	connect(m_rep->stratiSliceAttribute()->image(), SIGNAL(opacityChanged(float)), this,
			SLOT(opacityChanged(float)));

	connect(m_rep->stratiSliceAttribute()->image(), SIGNAL(dataChanged()), this,
			SLOT(update()));
	connect(m_rep->stratiSliceAttribute()->isoSurfaceHolder(), SIGNAL(dataChanged()),
			this, SLOT(updateIsoSurface()));
}

StratiSliceAmplitude3DLayer::~StratiSliceAmplitude3DLayer() {


}

void StratiSliceAmplitude3DLayer::opacityChanged(float val) {
	m_opacityParameter->setValue(val);
}

void StratiSliceAmplitude3DLayer::rangeChanged() {
	m_paletteRangeParameter->setValue(
			m_rep->stratiSliceAttribute()->image()->rangeRatio());
}


void StratiSliceAmplitude3DLayer::updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img ) {
	if (texture == nullptr)
		return;

	size_t pointerSize = img->internalPointerSize();
	img->lockPointer();
	texture->setData(
			byteArrayFromRawData((const char*) img->backingPointer(),
					pointerSize));
	img->unlockPointer();
}

void StratiSliceAmplitude3DLayer::update() {
	updateTexture(m_cudaTexture,m_rep->stratiSliceAttribute()->image());
	rangeChanged();
}

void StratiSliceAmplitude3DLayer::updateIsoSurface() {
	updateTexture(m_cudaSurfaceTexture,m_rep->stratiSliceAttribute()->isoSurfaceHolder());
}

void StratiSliceAmplitude3DLayer::updateLookupTable(const LookupTable &table) {
	if (m_colorTexture == nullptr)
		return;
	CUDAImagePaletteHolder *img = m_rep->stratiSliceAttribute()->image();
	m_colorTexture->setLookupTable(img->lookupTable());
}

StratiSlice* StratiSliceAmplitude3DLayer::stratiSlice() const {
	return m_rep->stratiSliceAttribute()->stratiSlice();
}
void StratiSliceAmplitude3DLayer::show() {
	int width = stratiSlice()->width();
	int height = stratiSlice()->height();
	int depth = stratiSlice()->depth();

	m_sliceEntity = new Qt3DCore::QEntity(m_root);
	SurfaceMesh *mesh = new SurfaceMesh();
	mesh->setDimensions(QVector2D(width, depth));
	QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
	QMatrix4x4 ijToXYTranform(stratiSlice()->seismic()->ijToXYTransfo()->imageToWorldTransformation());
	const AffineTransformation* sampleTransform = stratiSlice()->seismic()->sampleTransformation();

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
	m_material->setEffect(
			Qt3DHelpers::generateImageEffect(
					"qrc:/shaders/qt3d/simpleColor.frag",
					"qrc:/shaders/qt3d/isimpleHorizonColor.vert"));

	// Set different parameters on the materials
	m_colorTexture = new ColorTableTexture();
	CUDAImagePaletteHolder *img = m_rep->stratiSliceAttribute()->image();
	m_cudaTexture = new CudaImageTexture(ImageFormats::QColorFormat::GRAY,
			img->sampleType(), img->width(), img->height());

	img = m_rep->stratiSliceAttribute()->isoSurfaceHolder();
	m_cudaSurfaceTexture = new CudaImageTexture(ImageFormats::QColorFormat::GRAY,
			img->sampleType(), img->width(), img->height());

	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("elementMap"),
					m_cudaTexture));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("surfaceMap"),
					m_cudaSurfaceTexture));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("colormap"),
					m_colorTexture));
	m_paletteRangeParameter = new Qt3DRender::QParameter(
			QStringLiteral("paletteRange"),
			m_rep->stratiSliceAttribute()->image()->rangeRatio());
	m_material->addParameter(m_paletteRangeParameter);
	m_opacityParameter = new Qt3DRender::QParameter(QStringLiteral("opacity"),
			m_rep->stratiSliceAttribute()->image()->opacity());
	m_material->addParameter(m_opacityParameter);
	tbuf = transform.constData();
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("cubeOrigin"),
					tbuf[1*4+1] * sampleTransform->b() + tbuf[3*4+1]));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("cubeScale"),
					tbuf[1*4+1] * sampleTransform->a()));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("heightThreshold"),
					m_rep->stratiSliceAttribute()->stratiSlice()->height()-2));

	update();
	updateIsoSurface();
	updateLookupTable(m_rep->stratiSliceAttribute()->image()->lookupTable());

	m_transform = new Qt3DCore::QTransform();
	m_transform->setScale3D(QVector3D(1, 1, 1));

	m_sliceEntity->addComponent(mesh);
	m_sliceEntity->addComponent(m_material);
	m_sliceEntity->addComponent(m_transform);
}

void StratiSliceAmplitude3DLayer::zScale(float val)
{
	m_transform->setScale3D(QVector3D(1, val, 1));
}

void StratiSliceAmplitude3DLayer::hide() {
	if(m_sliceEntity != nullptr){
		m_sliceEntity->setParent((Qt3DCore::QEntity*) nullptr);
		m_sliceEntity->deleteLater();
		m_sliceEntity = nullptr;
	}
}

QRect3D StratiSliceAmplitude3DLayer::boundingRect() const {

	int width = stratiSlice()->width();
	int height = stratiSlice()->height();
	int depth = stratiSlice()->depth();

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
				Seismic3DAbstractDataset* dataset = stratiSlice()->seismic();
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

void StratiSliceAmplitude3DLayer::refresh() {

}

