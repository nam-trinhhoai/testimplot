#ifndef SURFACECOLLISSION_H
#define SURFACECOLLISSION_H

#include <QVector3D>

class SurfaceCollision
{
public:
	// altitude between  position (X,Z) and surface( X,Z)
	virtual float distanceSigned(QVector3D position, bool* ok)=0;


};

#endif
