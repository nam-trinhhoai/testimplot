#include "nurbsdataset.h"

#include "nurbsgraphicrepfactory.h"

NurbsDataset::NurbsDataset(WorkingSetManager * workingSet,Manager* nurbs,const QString &name, QObject *parent) :
		IData(workingSet, parent), m_name(name) {

	m_uuid = QUuid::createUuid();
	m_repFactory = new NurbsGraphicRepFactory(this);
	m_nurbs = nurbs;

}

NurbsDataset::~NurbsDataset() {

}

//IData
IGraphicRepFactory *NurbsDataset::graphicRepFactory() {
	return m_repFactory;
}

QUuid NurbsDataset::dataID() const {
	return m_uuid;
}

Manager* NurbsDataset::getNurbs3d()
{
	return m_nurbs;
}
