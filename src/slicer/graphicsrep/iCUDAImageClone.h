/*
 * iCUDAImageClone.h
 *
 *  Created on: Jan 27, 2022
 *      Author: l1046262
 */

#ifndef SRC_SLICER_GRAPHICSREP_ICUDAIMAGECLONE_H_
#define SRC_SLICER_GRAPHICSREP_ICUDAIMAGECLONE_H_

class QGraphicsObject;
class QGraphicsItem;

class iCUDAImageClone
{
public:
	virtual QGraphicsObject* cloneCUDAImageWithMask(QGraphicsItem *parent) =0;
};


#endif /* SRC_SLICER_GRAPHICSREP_ICUDAIMAGECLONE_H_ */
