/*
 * GeGlobalParameters.h
 *
 *  Created on: 7 août 2019
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_VIEWERS_VIDEOEDITOR_SVGLOBALPARAMETERS_H_
#define TARUMAPP_SRC_VIEWERS_VIDEOEDITOR_SVGLOBALPARAMETERS_H_

#include <QFont>

class GeGlobalParameters {
public:
	GeGlobalParameters();
	GeGlobalParameters(const GeGlobalParameters& other);
	virtual ~GeGlobalParameters();

	int getGrabberThickness() const;

	void setGrabberThickness(int grabberThickness = 1);

	bool isDisplayText() const;

	void setDisplayText(bool displayText = true);

	const QFont& getFont() const;

	void setFont(const QFont& font);

private:
	int grabberThickness = 2;
	bool displayText = false;
	QFont font;
};

#endif /* TARUMAPP_SRC_VIEWERS_VIDEOEDITOR_SVGLOBALPARAMETERS_H_ */
