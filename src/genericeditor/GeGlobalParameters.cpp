/*
 * GeGlobalParameters.cpp
 *
 *  Created on: 7 août 2019
 *      Author: Georges
 */

#include "GeGlobalParameters.h"

GeGlobalParameters::GeGlobalParameters() {
	// TODO Auto-generated constructor stub

}

GeGlobalParameters::GeGlobalParameters(const GeGlobalParameters& other) {
	grabberThickness = other.grabberThickness;
	displayText = other.displayText;
	font = other.font;
}

GeGlobalParameters::~GeGlobalParameters() {
	// TODO Auto-generated destructor stub
}

int GeGlobalParameters::getGrabberThickness() const {
	return grabberThickness;
}

void GeGlobalParameters::setGrabberThickness(int grabberThicknes) {
	this->grabberThickness = grabberThickness;
}

bool GeGlobalParameters::isDisplayText() const {
	return displayText;
}

void GeGlobalParameters::setDisplayText(bool displayText) {
	this->displayText = displayText;
}

const QFont& GeGlobalParameters::getFont() const {
	return font;
}

void GeGlobalParameters::setFont(const QFont& font) {
	this->font = font;
}
