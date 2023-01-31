#ifndef _SURFACEMESHCACHE_
#define _SURFACEMESHCACHE_

#include <QByteArray>
#include "cudaimagetexture.h"

struct SurfaceMeshCache
{
	QByteArray vertexArray;
	QByteArray normalArray;
	QByteArray textureArray;
	QByteArray indexArray;
	int nbVertex;
	int nbTri;
};

// this function create a deep copy of the raw data
// else look at QByteArray::fromRawData
QByteArray byteArrayFromRawData(const char* data, long size);

#endif
