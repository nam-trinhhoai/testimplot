/*
 * GeObjectId.h
 * Generic object id to identify a sub part (rectangle, polygon, ellipse, ...) in a data
 *
 *  Created on: 24 juin 2019
 *      Author: l0222891
 */

#ifndef TARUMAPP_SRC_VIEWERS_VIDEOEDITOR_VAOBJECTID_H_
#define TARUMAPP_SRC_VIEWERS_VIDEOEDITOR_VAOBJECTID_H_

class GeObjectId {
public:
	GeObjectId(int objectId, int slice) : objectId(objectId), slice(slice) {}
	GeObjectId( const GeObjectId & par) {
		this->slice = par.getSlice();
		this->objectId = par.getObjectId();
	}
	virtual ~GeObjectId(){}

	GeObjectId& operator=(const GeObjectId& par) {
		this->slice = par.getSlice();
		this->objectId = par.getObjectId();
		return *this;
	}

	int getSlice() const {
		return slice;
	}

	int getObjectId() const {
		return objectId;
	}

	void setObjectId(int objectId) {
		this->objectId = objectId;
	}

	void setSlice(int slice) {
		this->slice = slice;
	}

private:
	int objectId;
	int slice;
};

#endif /* TARUMAPP_SRC_VIEWERS_VIDEOEDITOR_VAOBJECTID_H_ */
