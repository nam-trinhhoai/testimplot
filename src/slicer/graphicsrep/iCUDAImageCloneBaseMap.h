/*
 * iCUDAImageClone.h
 *
 *  Created on: Jan 27, 2022
 *      Author: l1046262
 */

#ifndef SRC_SLICER_GRAPHICSREP_ICUDAIMAGECLONEBASEMAP_H_
#define SRC_SLICER_GRAPHICSREP_ICUDAIMAGECLONEBASEMAP_H_

#include "iCUDAImageClone.h"

class BaseMapSurface;

class iCUDAImageCloneBaseMap : public iCUDAImageClone
{
public:
	virtual BaseMapSurface* cloneCUDAImageWithMaskOnBaseMap(QGraphicsItem *parent) = 0;
};


#endif /* SRC_SLICER_GRAPHICSREP_ICUDAIMAGECLONE_H_ */
