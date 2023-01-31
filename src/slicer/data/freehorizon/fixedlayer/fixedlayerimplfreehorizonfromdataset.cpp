
#include "fixedlayerfromdataset.h"

#include <fixedlayerfromdatasetgraphicrepfactory.h>
#include <fixedlayerimplfreehorizonfromdataset.h>



FixedLayerImplFreeHorizonFromDataset::FixedLayerImplFreeHorizonFromDataset(QString name, WorkingSetManager *workingSet,
		Seismic3DAbstractDataset* dataset, QObject *parent) :
FixedLayerFromDataset(name, workingSet, dataset, parent)
{
	m_dataset = dataset;
	m_name = name;
	m_color = Qt::blue;
	m_repFactory.reset(new FixedLayerFromDatasetGraphicRepFactory(this));
}

FixedLayerImplFreeHorizonFromDataset::~FixedLayerImplFreeHorizonFromDataset()
{

}


IGraphicRepFactory* FixedLayerImplFreeHorizonFromDataset::graphicRepFactory()
{
	return m_repFactory.get();
}

QString FixedLayerImplFreeHorizonFromDataset::name()
{
	return m_name;
}

void FixedLayerImplFreeHorizonFromDataset::deleteGraphicItemDataContent(QGraphicsItem* item)
{

}
