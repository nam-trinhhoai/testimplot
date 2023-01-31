#include "seismicdataset3Dlayer.h"
#include <cmath>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QEntity>
#include <QDiffuseSpecularMaterial>
#include "volumeboundingmesh.h"
#include "seismic3dabstractdataset.h"
#include "datasetrep.h"
#include "viewqt3d.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"

SeismicDataset3DLayer::SeismicDataset3DLayer(DatasetRep *rep,QWindow * parent,
		Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) :
		Graphic3DLayer(parent,root,camera) {
	m_rep = rep;
	m_volumeEntity=nullptr;
	m_transform=nullptr;
}

SeismicDataset3DLayer::~SeismicDataset3DLayer() {
	hide(); // to cleanup
}

Seismic3DAbstractDataset* SeismicDataset3DLayer::dataset() const
{
	return ((Seismic3DAbstractDataset*) m_rep->data());
}
void SeismicDataset3DLayer::show() {

	int width =dataset()->width();
	int height = dataset()->height();
	int depth = dataset()->depth();



	m_volumeEntity = new Qt3DCore::QEntity(m_root);
	VolumeBoundingMesh *mesh = new VolumeBoundingMesh;
	mesh->setDimensions(QVector3D(width, height, depth));

	Qt3DExtras::QDiffuseSpecularMaterial *material =
			new Qt3DExtras::QDiffuseSpecularMaterial(m_volumeEntity);
	material->setDiffuse(QVariant::fromValue(QColor(Qt::white)));
	material->setAmbient(Qt::white);

	QMatrix4x4 ijToXYTranform(dataset()->ijToXYTransfo()->imageToWorldTransformation());

	// swap axis of ijToXYTranform (i,j,k) -> (i,k,j) i:Y, j:Z, k:sample
	const float* tbuf = ijToXYTranform.constData();
	QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
									 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
									 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
									 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);

	const AffineTransformation* sampleTransform = dataset()->sampleTransformation();
	double a = sampleTransform->a();
	double b = sampleTransform->b();
	QMatrix4x4 sampleTransfo(1, 0, 0, 0,
							 0, a, 0, b,
							 0, 0, 1, 0,
							 0, 0, 0, 1);
	QMatrix4x4 sceneTransform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
	m_transformMatrixOri = sceneTransform * sampleTransfo * ijToXYTranformSwapped;


	m_transform = new Qt3DCore::QTransform();
	m_transform->setMatrix(m_transformMatrixOri);

	//Qt3DRender::QLayer* layerTr = dynamic_cast<ViewQt3D*>(m_rep->view())->getLayerOpaque();

	m_volumeEntity->addComponent(mesh);
	m_volumeEntity->addComponent(material);
	m_volumeEntity->addComponent(m_transform);
	//m_volumeEntity->addComponent(layerTr);

}


void SeismicDataset3DLayer::zScale(float val)
{
	QMatrix4x4 scale;
	scale.scale(1, val, 1);
	QMatrix4x4 transform = scale * m_transformMatrixOri;
	m_transform->setMatrix(transform);
}

void SeismicDataset3DLayer::hide() {
	if(m_volumeEntity != nullptr){
		m_volumeEntity->setParent((Qt3DCore::QEntity*) nullptr);
		m_volumeEntity->deleteLater();
		m_volumeEntity = nullptr;
	}
}

QRect3D SeismicDataset3DLayer::boundingRect() const {
	int width = dataset()->width();
	int height = dataset()->height();
	int depth = dataset()->depth();

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
				dataset()->ijToXYTransfo()->imageToWorld(i, k, iWorld, kWorld);
				dataset()->sampleTransformation()->direct(j, jWorld);

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

void SeismicDataset3DLayer::refresh() {

}

