/*
 * GraphicEditorFormsRepFactory.h
 *
 *  Created on: Nov 18, 2021
 *      Author: l1046262
 */

#ifndef SRC_GRAPHICEDITOR_GraphicEditorFormsRepFACTORY_H_
#define SRC_GRAPHICEDITOR_GraphicEditorFormsRepFACTORY_H_

#include <QObject>
#include "igraphicrepfactory.h"
#include <QGraphicsScene>
#include <QGraphicsItem>
#include "abstract2Dinnerview.h"

class GraphicTool_GraphicLayer;

class GraphicEditorFormsRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	GraphicEditorFormsRepFactory(GraphicTool_GraphicLayer *data, Abstract2DInnerView* view);
	virtual ~GraphicEditorFormsRepFactory();
	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent) override;

private:
	GraphicTool_GraphicLayer *m_data;
	Abstract2DInnerView* m_View;

};



#endif /* SRC_GRAPHICEDITOR_GraphicEditorFormsRepFACTORY_H_ */
