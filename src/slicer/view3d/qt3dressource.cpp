#include "qt3dressource.h"

Qt3DRessource::Qt3DRessource(Qt3DCore::QNode *parent) :
		Qt3DCore::QEntity(parent) {
	m_zScale = 100;
}

Qt3DRessource::~Qt3DRessource() {
}

double Qt3DRessource::zScale() const {
	return m_zScale;
}
void Qt3DRessource::setzScale(double val) {
	if (m_zScale!=val) {
		m_zScale = val;
		emit zScaleChanged(val);
	}
}

SampleUnit Qt3DRessource::sectionType() const {
	return convertFromQml(m_sectionType);
}

void Qt3DRessource::setSectionType(SampleUnit val) {
	QMLEnumWrappers::SampleUnit newSectionType = convertToQml(val);
	if (newSectionType!=m_sectionType) {
		m_sectionType = newSectionType;
		emit sectionTypeQMLChanged(m_sectionType);
		emit sectionTypeChanged(val);
	}
}

const MtLengthUnit* Qt3DRessource::depthLengthUnit() const {
	return m_depthLengthUnit;
}

void Qt3DRessource::setDepthLengthUnit(const MtLengthUnit* val) {
	if (m_depthLengthUnit!=nullptr && val==nullptr) {
		m_depthLengthUnit = nullptr;
		m_depthLengthUnitWrapper = MtLengthUnitWrapperQML();
		emit depthLengthUnitChanged(m_depthLengthUnit);
		emit depthLengthUnitQMLChanged(&m_depthLengthUnitWrapper);
	} else if ((m_depthLengthUnit==nullptr && val!=nullptr) || *val!=*m_depthLengthUnit) {
		m_depthLengthUnit = val;
		m_depthLengthUnitWrapper = MtLengthUnitWrapperQML(m_depthLengthUnit);
		emit depthLengthUnitChanged(m_depthLengthUnit);
		emit depthLengthUnitQMLChanged(&m_depthLengthUnitWrapper);
	}
}

QMLEnumWrappers::SampleUnit Qt3DRessource::sectionTypeQML() const {
	return m_sectionType;
}

void Qt3DRessource::setSectionTypeQML(QMLEnumWrappers::SampleUnit val) {
	SampleUnit newSectionType = convertFromQml(val);
	if (val!=m_sectionType) {
		m_sectionType = val;
		emit sectionTypeQMLChanged(m_sectionType);
		emit sectionTypeChanged(newSectionType);
	}
}

MtLengthUnitWrapperQML* Qt3DRessource::depthLengthUnitQML() {
	return &m_depthLengthUnitWrapper;
}

void Qt3DRessource::setDepthLengthUnitQML(MtLengthUnitWrapperQML* val) {
	if (*val!=m_depthLengthUnitWrapper) {
		m_depthLengthUnit = &MtLengthUnit::fromModelUnit(val->getSymbol());
		m_depthLengthUnitWrapper = *val;
		emit depthLengthUnitChanged(m_depthLengthUnit);
		emit depthLengthUnitQMLChanged(&m_depthLengthUnitWrapper);
	}
}
