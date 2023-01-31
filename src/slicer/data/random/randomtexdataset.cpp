#include "randomtexdataset.h"
#include "randomview3d.h"
#include "randomtexgraphicrepfactory.h"

RandomTexDataset::RandomTexDataset(WorkingSetManager * workingSet,const QString &name,CudaImageTexture* texture,QVector2D range, RandomView3D *parent) :
		IData(workingSet, (QObject*)(parent)), m_name(name) {

	m_uuid = QUuid::createUuid();
	m_repFactory = new RandomTexGraphicRepFactory(this);
	m_texture = texture;
	m_range = range;

	m_randomParent = parent;

}

RandomTexDataset::~RandomTexDataset() {

}

//IData
IGraphicRepFactory *RandomTexDataset::graphicRepFactory() {
	return m_repFactory;
}

QUuid RandomTexDataset::dataID() const {
	return m_uuid;
}

CudaImageTexture* RandomTexDataset::texture()
{
	return m_texture;
}

QVector2D RandomTexDataset::range()
{
	return m_range;
}

RandomView3D* RandomTexDataset::parentRandom()
{
	return m_randomParent;
}
