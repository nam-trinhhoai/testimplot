/*
 * SimpleIntValidator.h
 *
 *  Created on: 19 mai 2017
 *      Author: a
 */

#ifndef MURATAPP_SRC_DIALOG_DATAMANAGEMENT_SIMPLEINTVALIDATOR_H_
#define MURATAPP_SRC_DIALOG_DATAMANAGEMENT_SIMPLEINTVALIDATOR_H_

#include <QValidator>


class SimpleIntValidator: public QIntValidator {
public:
	SimpleIntValidator(QObject * parent = 0) :
			QIntValidator(parent) {
		setLocale(QLocale::C);

	}

	SimpleIntValidator(int bottom, int top, QObject *parent = 0) :
			QIntValidator(bottom, top, parent) {
		setLocale(QLocale::C);
	}

};

#endif /* MURATAPP_SRC_DIALOG_DATAMANAGEMENT_SIMPLEINTVALIDATOR_H_ */
