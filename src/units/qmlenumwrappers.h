/*
 * qmlenumwrappers.h
 *
 *  Created on: 25 mars 2022
 *      Author: l0483271
 *
 */

#include "viewutils.h"

#include <QObject>

#ifndef NEXTVISION_SRC_UNITS_QMLENUMWRAPPERS_H_
#define NEXTVISION_SRC_UNITS_QMLENUMWRAPPERS_H_

namespace QMLEnumWrappers {
	Q_NAMESPACE

	enum SampleUnit {
		NONE,
		TIME,
		DEPTH
	};

	Q_ENUM_NS(SampleUnit)
}; /* namespace QMLEnumWrappers */

Q_DECLARE_METATYPE(QMLEnumWrappers::SampleUnit)

QMLEnumWrappers::SampleUnit convertToQml(const SampleUnit& unit);
SampleUnit convertFromQml(const QMLEnumWrappers::SampleUnit& unit);

#endif /* NEXTVISION_SRC_UNITS_QMLENUMWRAPPERS_H_ */
