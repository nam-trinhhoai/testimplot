#include "graphiclayer.h"

#include <QGraphicsScene>

GraphicLayer::GraphicLayer(QGraphicsScene *scene, int defaultZDepth):QObject(scene)
{
	m_scene=scene;
	m_defaultZDepth=defaultZDepth;
}

GraphicLayer::~GraphicLayer()
{

}
