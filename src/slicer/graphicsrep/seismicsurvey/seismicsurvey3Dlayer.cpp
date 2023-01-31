#include "seismicsurvey3Dlayer.h"
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QEntity>
#include <QDiffuseSpecularMaterial>
#include <QMatrix4x4>
#include "volumeboundingmesh.h"
#include "seismicsurvey.h"
#include "seismicsurveyrep.h"
#include "viewqt3d.h"

SeismicSurvey3DLayer::SeismicSurvey3DLayer(SeismicSurveyRep *rep,QWindow * parent,
		Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) :
		Graphic3DLayer(parent,root,camera) {
	m_rep = rep;
	m_transform=nullptr;
	m_volumeEntity=nullptr;
}

SeismicSurvey3DLayer::~SeismicSurvey3DLayer() {
}
void SeismicSurvey3DLayer::show() {

	int width = ((SeismicSurvey*) m_rep->data())->ijToXYTransfo()->width();
	int height = 1;
	int depth = ((SeismicSurvey*) m_rep->data())->ijToXYTransfo()->height();

	m_volumeEntity = new Qt3DCore::QEntity(m_root);
	VolumeBoundingMesh *mesh = new VolumeBoundingMesh;
	mesh->setDimensions(QVector3D(width, height, depth));

	Qt3DExtras::QDiffuseSpecularMaterial *material =
			new Qt3DExtras::QDiffuseSpecularMaterial(m_volumeEntity);
	material->setDiffuse(QVariant::fromValue(QColor(Qt::white)));
	material->setAmbient(Qt::white);

	QMatrix4x4 ijToXYTranform(((SeismicSurvey*) m_rep->data())->ijToXYTransfo()->imageToWorldTransformation());

	// swap axis of ijToXYTranform (i,j,k) -> (i,k,j) i:Y, j:Z, k:sample
	const float* tbuf = ijToXYTranform.constData();
	QMatrix4x4 ijToXYTranformSwapped(tbuf[ 0], tbuf[ 8], tbuf[ 4], tbuf[ 12],
									 tbuf[ 2], tbuf[10], tbuf[ 6], tbuf[14],
									 tbuf[ 1], tbuf[ 9], tbuf[ 5], tbuf[ 13],
									 tbuf[ 3], tbuf[11], tbuf[ 7], tbuf[15]);


	QMatrix4x4 sceneTransform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
	m_transformMatrixOri = sceneTransform * ijToXYTranformSwapped;

	m_transform = new Qt3DCore::QTransform();
	//m_transform->setScale3D(QVector3D(1, 1, 1));
	m_transform->setMatrix(m_transformMatrixOri);

//	Qt3DRender::QLayer* layer = dynamic_cast<ViewQt3D*>(m_rep->view())->getLayerOpaque();

	m_volumeEntity->addComponent(mesh);
	m_volumeEntity->addComponent(material);
	m_volumeEntity->addComponent(m_transform);


//	m_volumeEntity->addComponent(layer);
}

void SeismicSurvey3DLayer::zScale(float val)
{
	QMatrix4x4 scale;
	scale.scale(1, val, 1);
	QMatrix4x4 transform = scale * m_transformMatrixOri;
	m_transform->setMatrix(transform);
}

void SeismicSurvey3DLayer::hide() {
	if(m_volumeEntity != nullptr){
		m_volumeEntity->setParent((Qt3DCore::QEntity*) nullptr);
		m_volumeEntity->deleteLater();
		m_volumeEntity = nullptr;
	}
}

QRect3D SeismicSurvey3DLayer::boundingRect() const {

	int width = ((SeismicSurvey*) m_rep->data())->ijToXYTransfo()->width();
	int height = ((SeismicSurvey*) m_rep->data())->ijToXYTransfo()->height();
	int depth = 10;

	QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();

	// fill list
	double xmin = std::numeric_limits<double>::max();
	double xmax = std::numeric_limits<double>::lowest();
	double ymin = std::numeric_limits<double>::max();
	double ymax = std::numeric_limits<double>::lowest();
	double zmin = std::numeric_limits<double>::max();
	double zmax = std::numeric_limits<double>::lowest();

	for (int i=0; i<=width; i+=width) { // xline
		for(int j=0; j<=height; j+=height) { // sample
			for (int k=0; k<=depth; k+= depth) { // inline
				// apply transform
				double iWorld, jWorld, kWorld;
				((SeismicSurvey*) m_rep->data())->ijToXYTransfo()->imageToWorld(i, k, iWorld, kWorld);
				jWorld = j;

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

void SeismicSurvey3DLayer::refresh() {

}

