/*
 * iRepGraphicItem.h
 *
 *  Created on: Nov 26, 2021
 *      Author: l1046262
 */

#ifndef SRC_SLICER_GRAPHICSREP_WELL_IREPGRAPHICITEM_H_
#define SRC_SLICER_GRAPHICSREP_WELL_IREPGRAPHICITEM_H_

#include <QGraphicsItem>

class iRepGraphicItem
{
public:
	virtual QGraphicsItem* graphicsItem() const =0;
	virtual void autoDeleteRep() =0;
};


#endif /* SRC_SLICER_GRAPHICSREP_WELL_IREPGRAPHICITEM_H_ */
