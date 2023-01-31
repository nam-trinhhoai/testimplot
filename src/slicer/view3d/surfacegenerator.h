/*
 * surfacegenerator.h
 *
 *  Created on: 21 févr. 2020
 *      Author: a
 */

#ifndef RGTSEISMICSLICER_SRC_SURFACEGENERATOR_H_
#define RGTSEISMICSLICER_SRC_SURFACEGENERATOR_H_

#include <QMatrix4x4>

class SurfaceGenerator {
public:
	SurfaceGenerator(int width,int height, const QMatrix4x4& transform);
	virtual ~SurfaceGenerator();

	int getVerticesCount();

	void getVertices(float * vertices,float * texCoord, float * normals);
	void getIndices( unsigned int* indices);

    int getWidth();
    int getHeight();




private:
	int m_width;
	int m_height;

	int m_sensIndex;

	QMatrix4x4 m_transform;
};

#endif /* RGTSEISMICSLICER_SRC_SURFACEGENERATOR_H_ */
