/*
 * SimpleDoubleValidator.h
 *
 *  Created on: 19 mai 2017
 *      Author: a
 */

#ifndef MURATAPP_SRC_DIALOG_SIMPLEDOUBLEVALIDATOR_H_
#define MURATAPP_SRC_DIALOG_SIMPLEDOUBLEVALIDATOR_H_

#include <QValidator>

class SimpleDoubleValidator: public QDoubleValidator {
public:
	SimpleDoubleValidator(QObject * parent = 0) :
			QDoubleValidator(parent) {
		setLocale(QLocale(QLocale::English));
		setNotation(QDoubleValidator::StandardNotation);
	}
	SimpleDoubleValidator(int decimals, QObject * parent) :
			QDoubleValidator(std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), decimals, parent) {
		setLocale(QLocale(QLocale::English));
		setNotation(QDoubleValidator::StandardNotation);
	}

	QValidator::State validate(QString & s, int & pos) const {
		if (s.isEmpty() || s == "-") {
			// allow empty field or minus sign
			return QValidator::Intermediate;
		}
		// check length of decimal places
		//QChar
		QString point = locale().decimalPoint();
		if (s.indexOf(point) != -1) {
			int lengthDecimals = s.length() - s.indexOf(point) - 1;
			if (lengthDecimals > decimals()) {
				return QValidator::Invalid;
			}
		}
		// check range of value
		bool isNumber;
		double value = locale().toDouble(s, &isNumber);
		if (isNumber) {
			return QValidator::Acceptable;
		}
		return QValidator::Invalid;
	}
};

#endif /* MURATAPP_SRC_DIALOG_SIMPLEDOUBLEVALIDATOR_H_ */
