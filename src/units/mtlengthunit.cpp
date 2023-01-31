/*
 * MtLengthUnit.cpp
 *
 *  Created on: 5 mars 2019
 *      Author: l0222891
 *
 *  From TarumApp, 22 mars 2022
 */

#include "mtlengthunit.h"

#include <cmath>
#include <QDebug>

#include <QQmlEngine>
#include <QJSEngine>


const MtLengthUnit MtLengthUnit::METRE("Meter", "m");
const MtLengthUnit MtLengthUnit::FEET("Feet", "ft");

MtLengthUnitWrapperQML MtLengthUnitWrapperQML::METRE(&MtLengthUnit::METRE);
MtLengthUnitWrapperQML MtLengthUnitWrapperQML::FEET(&MtLengthUnit::FEET);

const double MtLengthUnit::FEET_TO_METER_RATIO = 0.3048;

MtLengthUnit::MtLengthUnit(QString name, QString symbol) {
	this->m_name = name;
	this->m_symbol = symbol;
}

MtLengthUnit::~MtLengthUnit() {
	// TODO Auto-generated destructor stub
}

bool MtLengthUnit::operator==(const MtLengthUnit& other) const {
  return (m_name == other.m_name && m_symbol == other.m_symbol);
}

bool MtLengthUnit::operator!=(const MtLengthUnit& other) const {
  return (m_name != other.m_name || m_symbol != other.m_symbol);
}

const MtLengthUnit& MtLengthUnit::fromModelUnit(const QString& symbol) {
	if (symbol == "m") return MtLengthUnit::METRE;
	else if (symbol == "ft") return MtLengthUnit::FEET;
	else return MtLengthUnit::METRE;
}

double MtLengthUnit::convert(const MtLengthUnit& inUnit, const MtLengthUnit& outUnit, double value) {
	double outValue = 0;
	if (inUnit==outUnit) {
		outValue = value;
	} else if (inUnit==MtLengthUnit::METRE && outUnit==MtLengthUnit::FEET) {
		outValue = value / MtLengthUnit::FEET_TO_METER_RATIO;
	} else if (inUnit==MtLengthUnit::FEET && outUnit==MtLengthUnit::METRE) {
		outValue = value * MtLengthUnit::FEET_TO_METER_RATIO;
	}
	return outValue;
}

MtLengthUnitWrapperQML::MtLengthUnitWrapperQML(QString name, QString symbol, QObject* parent) :
			QObject(parent) {
	m_object = new MtLengthUnit(name, symbol);
	m_ownPointer = true;
	m_valid = true;
}

MtLengthUnitWrapperQML::MtLengthUnitWrapperQML(const MtLengthUnit* lengthUnit, QObject* parent) : QObject(parent) {
	m_object = lengthUnit;
	m_ownPointer = false;
	m_valid = m_object!=nullptr;
}

MtLengthUnitWrapperQML::MtLengthUnitWrapperQML(const MtLengthUnitWrapperQML& lengthUnit, QObject* parent) : QObject(parent) {
	m_valid = lengthUnit.m_valid;
	if (lengthUnit.m_ownPointer && m_valid) {
		m_object = new MtLengthUnit(lengthUnit.getName(), lengthUnit.getSymbol());
		m_ownPointer = true;
	} else {
		m_object = lengthUnit.m_object;
		m_ownPointer = false;
	}
}

MtLengthUnitWrapperQML::MtLengthUnitWrapperQML(QObject* parent) : QObject(parent) {
	m_valid = false;
	m_ownPointer = false;
	m_object = nullptr;
}

MtLengthUnitWrapperQML::~MtLengthUnitWrapperQML() {
	if (m_ownPointer) {
		delete m_object;
	}
}

const MtLengthUnitWrapperQML& MtLengthUnitWrapperQML::operator=(const MtLengthUnitWrapperQML& other) {
	// cleanup
	if (m_ownPointer) {
		delete m_object;
		m_object = nullptr;
	}
	m_valid = other.m_valid;
	if (other.m_ownPointer && m_valid) {
		m_object = new MtLengthUnit(other.getName(), other.getSymbol());
		m_ownPointer = true;
	} else {
		m_object = other.m_object;
		m_ownPointer = false;
	}
	return *this;
}

bool MtLengthUnitWrapperQML::operator==(const MtLengthUnitWrapperQML& other) const {
	return *m_object==*(other.m_object);
}

bool MtLengthUnitWrapperQML::operator!=(const MtLengthUnitWrapperQML& other) const {
	return *m_object!=*(other.m_object);
}

bool MtLengthUnitWrapperQML::isValid() const {
	return m_valid;
}

QString MtLengthUnitWrapperQML::getSymbol() const {
	return m_object->getSymbol();
}

QString MtLengthUnitWrapperQML::getName() const {
	return m_object->getName();
}

double MtLengthUnitWrapperQML::convertFrom(const MtLengthUnitWrapperQML& inUnit, double value) {
	return MtLengthUnit::convert(*(inUnit.m_object), *m_object, value);
}

MtLengthUnitWrapperQML* MtLengthUnitWrapperQML::getMetre(QQmlEngine *engine, QJSEngine *scriptEngine) {
	Q_UNUSED(engine)
	Q_UNUSED(scriptEngine)
	return new MtLengthUnitWrapperQML(MtLengthUnitWrapperQML::METRE);
}

MtLengthUnitWrapperQML* MtLengthUnitWrapperQML::getFeet(QQmlEngine *engine, QJSEngine *scriptEngine) {
	Q_UNUSED(engine)
	Q_UNUSED(scriptEngine)
	return new MtLengthUnitWrapperQML(MtLengthUnitWrapperQML::FEET);
}
