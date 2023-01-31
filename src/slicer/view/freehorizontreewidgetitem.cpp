#include "freehorizontreewidgetitem.h"
#include "datatreewidgetitem.h"
#include "freehorizon.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include <fixedlayerimplfreehorizonfromdataset.h>
#include <fixedattributimplfreehorizonfromdirectories.h>
#include "rgblayerfreehorizongraphicrepfactory.h"
#include <rgblayerimplfreehorizonslice.h>
#include "fixedlayerimplfreehorizonfromdatasetandcube.h"

#include <QDebug>

FreeHorizonTreeWidgetItem::FreeHorizonTreeWidgetItem(FreeHorizon* freeHorizon,
		const std::vector<ViewType>& viewTypes, QTreeWidgetItem *parent, QObject* parentObj) :
		DataTreeWidgetItem(freeHorizon, viewTypes, parent, parentObj) {
	m_freeHorizon = freeHorizon;
//	{
//		QSignalBlocker block(parent->treeWidget()->model());
//		setData(0, Qt::DisplayRole, QVariant::fromValue(marker->name()));
//		setData(0, Qt::UserRole, QVariant::fromValue(marker));
//		setData(0, Qt::CheckStateRole, QVariant::fromValue(marker->displayPreference()));
//	}
//	connect(m_freeHorizon, SIGNAL(wellBoreAdded(WellBore *)), this,
//			SLOT(insertChildBore(WellBore *)));

	bool restrictToSections = true;
	int idxViewType = 0;
	while (restrictToSections && idxViewType<viewTypes.size()) {
		restrictToSections = viewTypes[idxViewType]==ViewType::InlineView ||
				viewTypes[idxViewType]==ViewType::XLineView ||
				viewTypes[idxViewType]==ViewType::RandomView;
		idxViewType++;
	}

	// std::vector<FixedRGBLayersFromDatasetAndCube*> attributs = m_freeHorizon->m_attribut;
	std::vector<FreeHorizon::Attribut> attributs = m_freeHorizon->m_attribut;
	QList<QTreeWidgetItem*> itemsToAdd;
	for (FreeHorizon::Attribut attribut : attributs) {
		IData *myData = attribut.getData();

		if ( myData && ( !restrictToSections || myData->name().compare("isochrone")==0) ) {
			DataTreeWidgetItem *el = new DataTreeWidgetItem(myData, getViewTypes(), this);
			itemsToAdd.push_back(el);
		}
	}
	addChildren(itemsToAdd);
//	setExpanded(false);

//	connect(m_marker, &IData::displayPreferenceChanged, this, &MarkerTreeWidgetItem::dataDisplayPreferenceChanged);
}

//void FreeHorizonTreeWidgetItem::dataDisplayPreferenceChanged(bool newPreference) {
//	if (this->data(0, Qt::CheckStateRole).toBool()!=newPreference) {
//		Qt::CheckState state = newPreference ? Qt::Checked : Qt::Unchecked;
//		this->setCheckState(0, state);
//	}
//}

void FreeHorizonTreeWidgetItem::insertChildBore(FixedRGBLayersFromDatasetAndCube *layer) {
	DataTreeWidgetItem *childNode = new DataTreeWidgetItem(layer, getViewTypes(), this);
	addChild(childNode);
}

FreeHorizonTreeWidgetItem::~FreeHorizonTreeWidgetItem() {
}

const FreeHorizon* FreeHorizonTreeWidgetItem::getFreeHorizon() const {
	return m_freeHorizon;
}

FreeHorizon* FreeHorizonTreeWidgetItem::getFreeHorizon() {
	return m_freeHorizon;
}
