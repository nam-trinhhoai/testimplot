/*
 * surfacegenerator.cpp
 *
 *  Created on: 21 févr. 2020
 *      Author: a
 */

#include "surfacegenerator.h"
#include <iostream>
#include <cmath>
#include <omp.h>
#include <QDebug>

SurfaceGenerator::SurfaceGenerator(int width, int height, const QMatrix4x4& transform) :
	m_transform(transform)
{


	m_width = width+1;
	m_height = height+1;

	const float*transfo = transform.constData();
	float xscale = transfo[0];
	float zscale = transfo[10];

	if((xscale > 0 && zscale > 0) || (xscale < 0 && zscale < 0))
	{
		m_sensIndex = 1;
	}else{
		m_sensIndex = -1;
	}

}

SurfaceGenerator::~SurfaceGenerator() {
	// TODO Auto-generated destructor stub
}

int SurfaceGenerator::getVerticesCount() {
	return m_width * m_height;
}
void SurfaceGenerator::getVertices(float *vertices,float * texCoord, float * normals) {

	#pragma omp parallel for
	for (int row = 0; row < m_height; row+=1) {
		for (int col = 0; col < m_width; col+=1) {
			QVector3D inVect(col, 0.0f, row);
			QVector3D outVect = m_transform * inVect;
			long iv = (col + row * m_width) * 3;
			long it = (col + row * m_width) * 2;
			long in = (col + row * m_width) * 3;
			vertices[iv++] = outVect.x();
			vertices[iv++] = outVect.y();
			vertices[iv++] = outVect.z();

			texCoord[it++]=col*1.0/(m_width-1);
			texCoord[it++]=row*1.0/(m_height-1);

			normals[in++] = 0.0f;
			normals[in++] = 1.0f;
			normals[in++] = 0.0f;
		}
	}
}

void SurfaceGenerator::getIndices(unsigned int *indices) {
	if(m_sensIndex > 0)
	{
		#pragma omp parallel for
		for (int row = 0; row < m_height-1; row+=1) {
			for (int col = 0; col < m_width-1; col+=1) {
				long i = (col + row * (m_width-1)) * 6;
				indices[i++] = col + row * m_width;
				indices[i++] = col + 1 + row * m_width;
				indices[i++] = col + (row + 1) * m_width;

				indices[i++] = col + (row + 1) * m_width;
				indices[i++] = col + 1 + row * m_width;
				indices[i++] = col + 1 + (row + 1) * m_width;
			}
		}
	}
	else
	{
		#pragma omp parallel for
		for (int row = 0; row < m_height-1; row+=1) {
			for (int col = 0; col < m_width-1; col+=1) {
				long i = (col + row * (m_width-1)) * 6;
				indices[i++] = col + row * m_width;
				indices[i++] = col + (row + 1) * m_width;
				indices[i++] = col + 1 + row * m_width;


				indices[i++] = col + (row + 1) * m_width;
				indices[i++] = col + 1 + (row + 1) * m_width;
				indices[i++] = col + 1 + row * m_width;

			}
		}
	}
}



int SurfaceGenerator::getWidth()
{
    return m_width;
}
int SurfaceGenerator::getHeight()
{
    return m_height;
}

//https://stackoverflow.com/questions/34437153/how-to-correctly-draw-a-flat-grid-of-triangles-for-terrain-direct-x-11-c
//http://www.mbsoftworks.sk/tutorials/opengl4/016-heightmap-pt1-random-terrain/
//http://www.rastertek.com/tertut02.html
