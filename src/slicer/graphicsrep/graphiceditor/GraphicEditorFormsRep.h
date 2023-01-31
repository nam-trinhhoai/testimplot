/*
 * GraphicEditorFormsRep.h
 *
 *  Created on: Nov 15, 2021
 *      Author: l1046262
 */

#ifndef SRC_GRAPHICEDITOR_GraphicEditorFormsRep_H_
#define SRC_GRAPHICEDITOR_GraphicEditorFormsRep_H_

#include "GraphicTool_GraphicLayer.h"
#include "idata.h"
#include "abstractgraphicrep.h"

class GraphicEditorFormsRep : public AbstractGraphicRep
{
	Q_OBJECT
	public:
	GraphicEditorFormsRep(GraphicTool_GraphicLayer *data,AbstractInnerView * parent);

		~GraphicEditorFormsRep();

		QWidget* propertyPanel() override;

		IData* data() const override;
		GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

		TypeRep getTypeGraphicRep() override;

	private:
		GraphicTool_GraphicLayer *m_Data;
		GraphicLayer *m_layer;
};


#endif /* SRC_GRAPHICEDITOR_GraphicEditorFormsRep_H_ */
