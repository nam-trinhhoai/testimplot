/*
 * GraphicEditorFormsRep.cpp
 *
 *  Created on: Nov 18, 2021
 *      Author: l1046262
 */

#include "GraphicEditorFormsLayer.h"
#include "GraphicEditorFormsRep.h"

GraphicEditorFormsRep::GraphicEditorFormsRep(GraphicTool_GraphicLayer *data,AbstractInnerView * parent):AbstractGraphicRep(parent)
{
	m_Data =data ;
	if(m_Data!=nullptr){
	  m_name = m_Data->name();
	}
	m_layer = nullptr;
}

GraphicEditorFormsRep::~GraphicEditorFormsRep() {}

QWidget* GraphicEditorFormsRep::propertyPanel()
{
	return nullptr;
}

IData* GraphicEditorFormsRep::data() const
{
	return m_Data;
}
GraphicLayer * GraphicEditorFormsRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)
{
	if (m_layer == nullptr) {
		m_layer = new GraphicEditorFormsLayer(this,scene,defaultZDepth,parent);
	}

	return m_layer;
}

AbstractGraphicRep::TypeRep GraphicEditorFormsRep::getTypeGraphicRep() {
	return AbstractGraphicRep::Courbe;
}

