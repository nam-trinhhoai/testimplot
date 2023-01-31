/*
 * GraphicEditorFormsRepFactory.cpp
 *
 *  Created on: Nov 18, 2021
 *      Author: l1046262
 */

#include "GraphicEditorFormsRepFactory.h"
#include "GraphicEditorFormsRep.h"
#include <QDebug>
#include "GraphicTool_GraphicLayer.h"

GraphicEditorFormsRepFactory::GraphicEditorFormsRepFactory(GraphicTool_GraphicLayer *data, Abstract2DInnerView* view) : IGraphicRepFactory()
{
	m_View = view;
	m_data = data;
}

GraphicEditorFormsRepFactory::~GraphicEditorFormsRepFactory()
{

}

AbstractGraphicRep * GraphicEditorFormsRepFactory::rep(ViewType type,AbstractInnerView * parent)
{
	if (parent == m_View)
	{
		return new GraphicEditorFormsRep (m_data,parent);
	}

	return nullptr;
}

