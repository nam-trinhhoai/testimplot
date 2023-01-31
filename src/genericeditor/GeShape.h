/*
 * VEShape.h
 *
 *  Created on: 6 août 2019
 *      Author: l0222891
 */

#ifndef VIDEOEDITOR_MTSHAPE_H_
#define VIDEOEDITOR_MTSHAPE_H_

#include <QColor>

//#include "data/MtRepresentation.h"
#include "GeObjectId.h"
#include "GeGlobalParameters.h"

class GeGlobalParameters;

class GeShape /*: public data::MtRepresentation*/{
public:
	GeShape(/*data::MtData* vaData,*/ GeGlobalParameters& globalParameters,
			const GeObjectId& m_objectId, QColor& color,
    		bool m_pickedHere);
	virtual ~GeShape();

	const GeObjectId& getObjectId() const {
		return m_objectId;
	}

	bool isPickedHere() const {
		return m_pickedHere;
	}

	void setPickedHere(bool m_pickedHere) {
		this->m_pickedHere = m_pickedHere;
	}
	const QColor& getColor() const {
		return m_color;
	}

	void setColor(const QColor& color) {
		m_color = color;
	}

	int getAlphaForFill() const {
		return m_alphaForFill;
	}

	void setAlphaForFill(int alphaForFill = 127) {
		m_alphaForFill = alphaForFill;
	}

	Qt::PenStyle getLinePenStyle() const {
		return m_linePenStyle;
	}

	void setLinePenStyle(Qt::PenStyle linePenStyle = Qt::SolidLine) {
		m_linePenStyle = linePenStyle;
	}

protected:
    GeObjectId m_objectId;
    QColor m_color;
    int m_alphaForFill = 127;
    Qt::PenStyle m_linePenStyle = Qt::SolidLine;
    bool m_pickedHere;
    GeGlobalParameters m_globalParameters;
};

#endif /* VIDEOEDITOR_MTSHAPE_H_ */
