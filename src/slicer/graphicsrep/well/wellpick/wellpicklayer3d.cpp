#include "wellpicklayer3d.h"

#include "wellpickrep.h"
#include "wellpick.h"
#include "marker.h"
#include "viewqt3d.h"
#include "qt3dhelpers.h"
#include "mtlengthunit.h"

#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QCamera>

#include <Qt3DExtras/QCylinderMesh>

WellPickLayer3D::WellPickLayer3D(WellPickRep *rep, QWindow * parent,
		Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera)  :
		Graphic3DLayer(parent, root, camera) {
	m_rep = rep;


	  connect(this,SIGNAL(showPickInfosSignal(const IToolTipProvider*,QString,int,int,QVector3D)),dynamic_cast<ViewQt3D*>(m_rep->view()),SLOT(showTooltipPick(const IToolTipProvider*,QString,int,int,QVector3D)));
/*	m_entityL1 = nullptr;
	m_transformL1 = nullptr;
	m_entityL2 = nullptr;
	m_transformL2 = nullptr;
	*/
	m_disqueEntity = nullptr;
	m_transformDisque = nullptr;
}

WellPickLayer3D::~WellPickLayer3D() {

}

void WellPickLayer3D::show() {
	bool isValid;
	QVector3D _point = m_rep->getCurrentPoint(&isValid);
	bool ok;
	QVector3D _dir = m_rep->getDirection(m_rep->sampleUnit() ,&ok);

	QVector3D point(_point.x(), _point.z(), _point.y());

	if (isValid) {
		m_rep->searchWellBoreRep();
		isValid = m_rep->isLinkedRepValid() && m_rep->linkedRepShown();
	}

	if (isValid) {
		QColor color = wellPick()->currentMarker()->color();
		QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
		float zscale = dynamic_cast<ViewQt3D*>(m_rep->view())->zScale();
		QMatrix4x4 transformScale;
		transformScale.scale(1.0f,zscale,1.0f);
		point = transformScale* transform * point;

	//	QVector3D startL1(point.x()-m_r, point.y(), point.z()-m_r);
	//	QVector3D endL1(point.x()+m_r, point.y(), point.z()+m_r);

		m_disqueEntity = new Qt3DCore::QEntity(m_root);
		m_transformDisque = new Qt3DCore::QTransform();
		m_transformDisque->setTranslation(point);
		QQuaternion quat =QQuaternion::rotationTo(QVector3D(0,1,0), _dir);
		m_transformDisque->setRotation(quat);
		m_transformDisque->setScale3D(QVector3D(m_r,m_thickness,m_r));
		Qt3DExtras::QCylinderMesh* disqueMesh = new Qt3DExtras::QCylinderMesh(m_root);
		//disqueMesh->setLength(m_thickness);
		//disqueMesh->setRadius(m_r);
		disqueMesh->setLength(1.0f);
		disqueMesh->setRadius(1.0f);
		disqueMesh->setRings(2);
		disqueMesh->setSlices(8);

		 material = new Qt3DExtras::QPhongMaterial(m_root);
		material->setAmbient(color);
		material->setDiffuse(color);
		material->setSpecular(QColor(0, 0, 0, 0));

		m_disqueEntity->addComponent(material);
		m_disqueEntity->addComponent(m_transformDisque);
		m_disqueEntity->addComponent(disqueMesh);



		Qt3DRender::QObjectPicker *sPicker = new Qt3DRender::QObjectPicker(m_disqueEntity);
		Qt3DRender::QPickingSettings * sPickingSettings = new Qt3DRender::QPickingSettings(sPicker);
	   sPickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	   sPickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
	   sPickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
	   sPickingSettings->setEnabled(true);
	   sPicker->setEnabled(true);
	   //sPicker->setHoverEnabled(true);
	   m_disqueEntity->addComponent(sPicker);

	   connect(sPicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {

	   		  if(e->button() == Qt3DRender::QPickEvent::Buttons::RightButton)
	   		  {
	   			// m_selected = ! m_selected;

	   			 selectPick(e->position().x(),e->position().y(),e->worldIntersection());


	   			//  if(m_selected) this->selectWell(e->entity()->objectName(),e->position().x(),e->position().y(),e->worldIntersection());
	   			// else deselectWell();

	   		  }
	   });

	/*	m_entityL1 = Qt3DHelpers::drawLine(startL1, endL1, color, m_root);
		m_transformL1 = new Qt3DCore::QTransform();
		m_entityL1->addComponent(m_transformL1);

		QVector3D startL2(point.x()-m_r, point.y(), point.z()+m_r);
		QVector3D endL2(point.x()+m_r, point.y(), point.z()-m_r);

		m_entityL2 = Qt3DHelpers::drawLine(startL2, endL2, color, m_root);
		m_transformL2 = new Qt3DCore::QTransform();
		m_entityL2->addComponent(m_transformL2);*/
	}
	m_isShown = true;
}

QString WellPickLayer3D::generateToolTipInfo() const {
	QString namepick = wellPick()->markerName();
	QString kindpick = wellPick()->kind();
	double value = wellPick()->value();

	double displayValue;
	if ((wellPick()->kindUnit()==WellUnit::MD || wellPick()->kindUnit()==WellUnit::TVD) && dynamic_cast<ViewQt3D*>(m_rep->view())) {
		// depth value in meter, conversion needed to view depth unit
		const MtLengthUnit* viewDepthLengthUnit = dynamic_cast<ViewQt3D*>(m_rep->view())->depthLengthUnit();
		displayValue = MtLengthUnit::convert(MtLengthUnit::METRE, *viewDepthLengthUnit, value);
	} else {
		// value does not represent a depth but a time, no conversion to do
		// or fall back because view is not as expected
		displayValue = value;
	}

	QString nameall = namepick+"|"+kindpick+"|"+QString::number(displayValue);
	return nameall;
}

void WellPickLayer3D::selectPick(int posX, int posY,QVector3D posGlobal)
{
	QString nameall = generateToolTipInfo();
	emit showPickInfosSignal(this,nameall,posX,posY,posGlobal);

}

void WellPickLayer3D::deselectPick()
{
	QColor color = wellPick()->currentMarker()->color();
	material->setAmbient(color);
	material->setDiffuse(color);
}


void WellPickLayer3D::setDiameter(int value)
{

	if(!qFuzzyCompare(m_r,value))
	{
		m_r = (double)(value);
		if(m_transformDisque!= nullptr)m_transformDisque->setScale3D(QVector3D(m_r,m_thickness,m_r));
	}

}

void WellPickLayer3D::setThickness(int value)
{
	if(!qFuzzyCompare(m_thickness,value))
	{
		m_thickness = (float)(value);
		if(m_transformDisque!= nullptr)m_transformDisque->setScale3D(QVector3D(m_r,m_thickness,m_r));
	}
}


void WellPickLayer3D::hide() {
	if (m_disqueEntity!=nullptr) {
		m_disqueEntity->setParent((Qt3DCore::QEntity*) nullptr);
		m_disqueEntity->deleteLater();
		m_transformDisque = nullptr;
			m_disqueEntity = nullptr;
		}
/*	if (m_entityL1!=nullptr) {
		m_entityL1->setParent((Qt3DCore::QEntity*) nullptr);
		m_entityL1->deleteLater();
		m_transformL1 = nullptr;
		m_entityL1 = nullptr;
	}
	if (m_entityL2!=nullptr) {
		m_entityL2->setParent((Qt3DCore::QEntity*) nullptr);
		m_entityL2->deleteLater();
		m_transformL2 = nullptr;
		m_entityL2 = nullptr;
	}*/
	m_isShown = false;
}

QRect3D WellPickLayer3D::boundingRect() const {
	QRect3D rect;

	bool isValid;
	QVector3D _point = m_rep->getCurrentPoint(&isValid);
	QVector3D point(_point.x(), _point.z(), _point.y());

	if (isValid) {
		QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
		point = transform * point;

		rect = QRect3D(point.x()-m_r, point.y()-m_r, point.z()-m_r, m_r*2, m_r*2, m_r*2);
	}
	return rect;
}

void WellPickLayer3D::refresh() {
	if (m_isShown) {
		hide();
		show();
	}
}


void WellPickLayer3D::zScale(float val) {

	bool isValid;
	QVector3D _point = m_rep->getCurrentPoint(&isValid);
	QVector3D point(_point.x(), _point.z(), _point.y());

	if (isValid && m_transformDisque) {
		QMatrix4x4 transform = dynamic_cast<ViewQt3D*>(m_rep->view())->sceneTransform();
		QMatrix4x4 transformScale;
		transformScale.scale(1.0f,val,1.0f);
		point = transformScale* transform * point;
		m_transformDisque->setTranslation(point);
	}
	/*if(m_transformDisque)
	{
		m_transformDisque->setScale3D(QVector3D(1, val, 1));
	}*/
/*	if (m_transformL1) {
		m_transformL1->setScale3D(QVector3D(1, val, 1));
	}
	if (m_transformL2) {
		m_transformL2->setScale3D(QVector3D(1, val, 1));
	}*/
}

WellPick * WellPickLayer3D::wellPick() const {
	return dynamic_cast<WellPick*>(m_rep->data());
}
