/*
 * GraphicEditorFormsLayer.cpp
 *
 *  Created on: Nov 18, 2021
 *      Author: l1046262
 */

#include "GraphicEditorFormsLayer.h"
#include <QDebug>
#include <QGraphicsScene>
#include "GraphicSceneEditor.h"
#include "GraphicEditorFormsRep.h"
#include "GraphEditor_Item.h"

GraphicEditorFormsLayer::GraphicEditorFormsLayer(GraphicEditorFormsRep *rep,QGraphicsScene *scene,
		int defaultZDepth,QGraphicsItem * parent) : GraphicLayer(scene, defaultZDepth)
{
	m_rep = rep;
}

GraphicEditorFormsLayer::~GraphicEditorFormsLayer()
{

}

void GraphicEditorFormsLayer::show()
{
	GraphicTool_GraphicLayer *layer = dynamic_cast<GraphicTool_GraphicLayer *> (m_rep->data());
	if (layer->type() == eLoadedLayer)
	{
		foreach(QGraphicsItem *p , m_scene->items())
		{
			if (dynamic_cast<GraphEditor_Item *>(p))
			{
				if (dynamic_cast<GraphEditor_Item *>(p)->getID() == layer->name())
					p->show();
			}
		}
	}
	else
	{
		dynamic_cast<GraphicSceneEditor* >(m_scene)->showItemsLayer((m_rep->data()->name()).toInt());
	}

}
void GraphicEditorFormsLayer::hide()
{
	GraphicTool_GraphicLayer *layer = dynamic_cast<GraphicTool_GraphicLayer *> (m_rep->data());
	if (layer->type() == eLoadedLayer)
	{
		foreach(QGraphicsItem *p , m_scene->items())
		{
			if (dynamic_cast<GraphEditor_Item *>(p))
			{
				if (dynamic_cast<GraphEditor_Item *>(p)->getID() == layer->name())
					p->hide();
			}

		}
	}
	else
	{
		dynamic_cast<GraphicSceneEditor* >(m_scene)->hideItemsLayer((m_rep->data()->name()).toInt());
	}
}

QRectF GraphicEditorFormsLayer::boundingRect() const
{
	return m_scene->sceneRect();
}

void GraphicEditorFormsLayer::refresh()
{

}

