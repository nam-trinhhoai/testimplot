/*
 * VEShape.cpp
 *
 *  Created on: 6 août 2019
 *      Author: l0222891
 */

#include "GeShape.h"

#include "genericeditor/GeGlobalParameters.h"


GeShape::GeShape(/*data::MtData* vaData,*/ GeGlobalParameters& globalParameters,
		const GeObjectId& objectId, QColor& color,
		bool pickedHere) :
		/* MtRepresentation( vaData),*/ m_objectId ( objectId), m_color ( color),
				m_pickedHere(pickedHere), m_globalParameters (globalParameters) {

}

GeShape::~GeShape() {
	// TODO Auto-generated destructor stub
}
