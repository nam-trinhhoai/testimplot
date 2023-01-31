#include "qmlenumwrappers.h"

QMLEnumWrappers::SampleUnit convertToQml(const SampleUnit& unit) {
	QMLEnumWrappers::SampleUnit qmlUnit = QMLEnumWrappers::SampleUnit::NONE;
	switch (unit) {
	case SampleUnit::TIME:
		qmlUnit = QMLEnumWrappers::SampleUnit::TIME;
		break;
	case SampleUnit::DEPTH:
		qmlUnit = QMLEnumWrappers::SampleUnit::DEPTH;
		break;
	default:
		qmlUnit = QMLEnumWrappers::SampleUnit::NONE;
	}
	return qmlUnit;
}

SampleUnit convertFromQml(const QMLEnumWrappers::SampleUnit& qmlUnit) {
	SampleUnit unit = SampleUnit::NONE;
	switch (qmlUnit) {
	case QMLEnumWrappers::SampleUnit::TIME:
		unit = SampleUnit::TIME;
		break;
	case QMLEnumWrappers::SampleUnit::DEPTH:
		unit = SampleUnit::DEPTH;
		break;
	default:
		unit = SampleUnit::NONE;
	}
	return unit;
}
