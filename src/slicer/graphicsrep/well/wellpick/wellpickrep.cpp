#include "wellpickrep.h"
#include "wellpick.h"
#include "wellbore.h"
#include "marker.h"
#include "wellpicklayer3d.h"
#include "abstractsectionview.h"
#include "affine2dtransformation.h"
#include "wellborerepon3d.h"
#include "viewqt3d.h"

WellPickRep::WellPickRep(WellPick* wellPick, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = wellPick;
	m_layer3D = nullptr;
	connect(m_data->wellBore(), &WellBore::boreUpdated, this, &WellPickRep::reExtractPosition);
}

WellPickRep::~WellPickRep() {
	if (m_layer3D!=nullptr) {
		delete m_layer3D;
	}
}

IData* WellPickRep::data() const {
	return m_data;
}

QString WellPickRep::name() const {
	if (m_data->wellBore()) {
		return m_data->wellBore()->name();
	} else {
		return m_data->name();
	}
}


SampleUnit WellPickRep::sampleUnit() const
{
	return m_sectionType;
}

bool WellPickRep::isCurrentPointSet() const {
	return m_isPointSet;
}

QVector3D WellPickRep::getCurrentPoint(bool* ok) const {
	*ok = isCurrentPointSet();
	return m_point;
}

QVector3D WellPickRep::getDirection(SampleUnit unit , bool* ok) const
{

	//QVector3D dir = m_data->wellBore()->getDirectionFromMd(m_data->value(), unit, ok);

	return m_direction;
}

//AbstractGraphicRep
QWidget* WellPickRep::propertyPanel() {
	return nullptr;
}

GraphicLayer * WellPickRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

Graphic3DLayer * WellPickRep::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) {
	if (m_layer3D==nullptr) {

		ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(m_parent);


		m_layer3D = new WellPickLayer3D(this, parent, root, camera);

		if(view3D != nullptr)
		{
			connect(view3D, SIGNAL(signalDiameterPick(int)), m_layer3D, SLOT(setDiameter(int)) ) ;
			connect(view3D, SIGNAL(signalThicknessPick(int)), m_layer3D, SLOT(setThickness(int)) ) ;

			view3D->sendDiameterPick();
			view3D->sendThicknessPick();

		}
	}


	return m_layer3D;
}

void WellPickRep::updateLayer() {
	if (m_layer3D) {
		m_layer3D->refresh();
	}
}

bool WellPickRep::setSampleUnit(SampleUnit unit) {
	bool isValid = false;
	if (unit==SampleUnit::TIME || unit==SampleUnit::DEPTH) {
		WellUnit wellUnit = m_data->kindUnit();
		double value = m_data->value();
		double depth, x, y;
		WellBore* wellBore = m_data->wellBore();
		depth = wellBore->getDepthFromWellUnit(value, wellUnit, unit, &isValid);
		if (isValid) {
			x = wellBore->getXFromWellUnit(value, wellUnit, &isValid);
		}
		if (isValid) {
			y = wellBore->getYFromWellUnit(value, wellUnit, &isValid);
		}
		double mdVal;
		if (isValid) {
			mdVal = wellBore->getMdFromWellUnit(value, wellUnit, &isValid);
		}

		if (isValid) {
			m_sectionType = unit;
			m_point = QVector3D(x, y, depth);
			bool ok;
			m_direction = m_data->wellBore()->getDirectionFromMd(mdVal,unit,&ok);

			m_isPointSet = true;
		} else {
			setSampleUnit(SampleUnit::NONE);
		}
	} else {
		m_sectionType = SampleUnit::NONE;
		m_isPointSet = false;
		isValid = unit==SampleUnit::NONE;
	}
	updateLayer();
	return isValid;
}

QList<SampleUnit> WellPickRep::getAvailableSampleUnits() const {
	QList<SampleUnit> list;
	if (m_data->wellBore()->isTfpDefined()) {
		list.push_back(SampleUnit::TIME);
	}
	list.push_back(SampleUnit::DEPTH);
	return list;
}

QString WellPickRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	if (list.contains(sampleUnit)) {
		return "Failure to load supported unit";
	} else{
		return "Unknown unit";
	}
}

bool WellPickRep::linkedRepShown() const {
	return m_linkedRepShown;
}

bool WellPickRep::isLinkedRepValid() const {
	return m_linkedRep!=nullptr;
}

void WellPickRep::wellBoreRepDeleted() {
	m_linkedRepShown = false;
	m_linkedRep = nullptr;
	updateLayer();
}

void WellPickRep::wellBoreLayerChanged(bool toggle, WellBoreRepOn3D* originObj) {
	if (originObj==m_linkedRep || m_linkedRep!=nullptr) {
		m_linkedRepShown = toggle;
		updateLayer();
	} else if (m_linkedRep==nullptr) {
		m_linkedRep = originObj;
		m_linkedRepShown = toggle;
		connect(m_linkedRep, &WellBoreRepOn3D::destroyed, this, &WellPickRep::wellBoreRepDeleted);
		updateLayer();
	}
}

void WellPickRep::searchWellBoreRep() {
	// get well bore rep
	const QList<AbstractGraphicRep*>& reps = m_parent->getVisibleReps();
	std::size_t index = 0;
	while (index<reps.size() && m_linkedRep==nullptr) {
		WellBoreRepOn3D* wellRep = dynamic_cast<WellBoreRepOn3D*>(reps[index]);
		if (wellRep!=nullptr && wellRep->data()==m_data->wellBore()) {
			m_linkedRep = wellRep;
			connect(m_linkedRep, &WellBoreRepOn3D::destroyed, this, &WellPickRep::wellBoreRepDeleted);
		}
		index++;
	}

	if (m_linkedRep!=nullptr) {
		// is layer of well bore rep shown ?
		m_linkedRepShown = m_linkedRep->isLayerShown();
	}
}

AbstractGraphicRep::TypeRep WellPickRep::getTypeGraphicRep() {
	return Courbe;
}

void WellPickRep::reExtractPosition() {
	setSampleUnit(m_sectionType);
	if (m_layer3D) {
		m_layer3D->refresh();
	}
}
