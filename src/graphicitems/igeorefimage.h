#ifndef IGeorefImage_H
#define IGeorefImage_H

#include <QMatrix4x4>
#include "igeorefgrid.h"

class IGeorefImage : public IGeorefGrid {
public:

	virtual ~IGeorefImage();

	virtual bool valueAt(int i, int j, double &value)const=0;

	virtual void valuesAlongJ(int j, bool* valid,double* values)const=0;
	virtual void valuesAlongI(int i, bool* valid,double* values)const=0;

	static bool value(const IGeorefImage * const image,double worldX, double worldY,int &i, int &j,double &value);
protected:
	IGeorefImage();
};

#endif /* RGTSEISMICSLICER_SRC_BASEMAP_TRANSFORMATIONPROVIDER_H_ */
