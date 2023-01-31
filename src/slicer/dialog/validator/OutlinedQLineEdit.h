/*
 * OutlinedQLineEdit.h
 *
 *  Created on: 19 mai 2017
 *      Author: a
 */

#ifndef MURATAPP_SRC_DIALOG_OutlinedQLineEdit_H_
#define MURATAPP_SRC_DIALOG_OutlinedQLineEdit_H_

#include <QLineEdit>
#include <QMessageBox>
#include <QLocale>

class OutlinedQLineEdit: public QLineEdit {
//Q_OBJECT
public:
	OutlinedQLineEdit(QWidget* parent = 0) :
			QLineEdit(parent), isEdited(false) {
		connect(this, &QLineEdit::textChanged, this, &OutlinedQLineEdit::edit);
		setLocale(QLocale::C);
	}
	OutlinedQLineEdit(const QString &s, QWidget* parent = 0) :
			QLineEdit(s, parent), isEdited(false) {
		connect(this, &QLineEdit::textChanged, this, &OutlinedQLineEdit::edit);
		setLocale(QLocale::C);
	}

	~OutlinedQLineEdit(){

	}

	inline bool edited() const {
		return isEdited;
	}

	bool getValue(std::string &defaultValue) {
		if (edited()) {
			if (!hasAcceptableInput()) {
				QMessageBox::information(this, "Error",
						"Invalid value enterred");
				return false;
			}
		}

		defaultValue = text().toStdString();
		return true;
	}

	bool getValue(double &defaultValue) {
		if (edited()) {
			if (!hasAcceptableInput()) {
				QMessageBox::information(this, "Error",
						"Invalid value enterred");
				return false;
			}
		}
		defaultValue = locale().toDouble(text());
		return true;
	}

	bool getValue(float &defaultValue) {
		if (edited()) {
			if (!hasAcceptableInput()) {
				QMessageBox::information(this, "Error",
						"Invalid value enterred");
				return false;
			}
		}
		defaultValue = locale().toFloat(text());
		return true;
	}

	bool getValue(int &defaultValue) {
		if (edited()) {
			if (!hasAcceptableInput()) {
				QMessageBox::information(this, "Error",
						"Invalid value enterred");
				return false;
			}
		}
		defaultValue = locale().toInt(text());
		return true;
	}

	bool getValue(unsigned int &defaultValue) {
		if (edited()) {
			if (!hasAcceptableInput()) {
				QMessageBox::information(this, "Error",
						"Invalid value enterred");
				return false;
			}
		}
		defaultValue = locale().toUInt(text());
		return true;
	}


private slots:
	void edit() {
		isEdited = true;
		setStyleSheet("border: 1px solid orange");
	}

private:
	bool isEdited;
};

#endif /* MURATAPP_SRC_DIALOG_OutlinedQLineEdit_H_ */
