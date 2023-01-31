#include "randomlayer.h"
#include "randomrep.h"
#include "qglfullcudaimageitem.h"
#include "cudaimagepaletteholder.h"
#include "qglcolorbar.h"
#include <QGraphicsScene>

RandomLayer::RandomLayer(RandomRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) :GraphicLayer(scene, defaultZDepth) {
	m_rep=rep,
	m_item = new QGLFullCUDAImageItem(rep->image(),parent);
	m_item->setZValue(defaultZDepth);
	m_colorScaleItem=nullptr;
	connect(rep->image(), SIGNAL(opacityChanged(float)), this,SLOT(refresh()));
	connect(rep->image(), SIGNAL(rangeChanged(const QVector2D &)), this,SLOT(refresh()));
	connect(rep->image(), SIGNAL(lookupTableChanged(const LookupTable &)), this,SLOT(refresh()));
}

void RandomLayer::updateImage()
{
	m_item->updateImage(m_rep->image());


	//warning: disconection not done because image suppose deleted
	connect(m_rep->image(), SIGNAL(opacityChanged(float)), this,SLOT(refresh()));
	connect(m_rep->image(), SIGNAL(rangeChanged(const QVector2D &)), this,SLOT(refresh()));
	connect(m_rep->image(), SIGNAL(lookupTableChanged(const LookupTable &)), this,SLOT(refresh()));


	refresh();

}

void RandomLayer::showColorScale(bool val)
{
	if(val)
	{
		m_colorScaleItem=new QGLColorBar ( m_item->boundingRect());
		updateColorScale();
		m_colorScaleItem->setZValue(m_defaultZDepth);
		m_scene->addItem(m_colorScaleItem);
	}else
	{
		if(m_colorScaleItem!=nullptr)
			m_scene->removeItem(m_colorScaleItem);
		m_colorScaleItem=nullptr;
	}
}

void RandomLayer::updateColorScale()
{
	m_colorScaleItem->setLookupTable(m_rep->image()->lookupTable());
	m_colorScaleItem->setOpacity(m_rep->image()->opacity());
	m_colorScaleItem->setRange(m_rep->image()->range());
}

RandomLayer::~RandomLayer() {
}
void RandomLayer::show()
{
	m_scene->addItem(m_item);
	if(m_colorScaleItem)
		m_scene->addItem(m_colorScaleItem);
}
void RandomLayer::hide()
{
	m_scene->removeItem(m_item);
	if(m_colorScaleItem)
		m_scene->removeItem(m_colorScaleItem);

	emit hidden();
}

QRectF RandomLayer::boundingRect() const
{
	return m_rep->image()->worldExtent();
}

void RandomLayer::refresh() {
	m_item->update();
	if(m_colorScaleItem)
	{
		updateColorScale();
		m_colorScaleItem->update();
	}
}

