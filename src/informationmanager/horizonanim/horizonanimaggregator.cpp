#include "horizonanimaggregator.h"

#include "GeotimeProjectManagerWidget.h"
#include "horizonaniminformation.h"
#include "horizonanimiterator.h"

#include "workingsetmanager.h"
#include "horizondatarep.h"

#include <iterator>

void HorizonAnimAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

HorizonAnimAggregator::HorizonAnimAggregator(WorkingSetManager* manager, QObject* parent) :
		IInformationAggregator(parent), m_manager(manager) {

	std::vector<QString> dataFullname = manager->getManagerWidget()->get_horizonanim_fullnames0();
	std::vector<QString> dataTinyname = manager->getManagerWidget()->get_horizonanim_names0();

	long N = std::min(dataFullname.size(), dataTinyname.size());
	for (long i=0; i<N; i++) {
		QString name = dataTinyname[i];
		name.replace(".hor", "");
		HorizonAnimInformation* info = new HorizonAnimInformation(name, dataFullname[i], manager);
		insert(i, info);
	}
}

HorizonAnimAggregator::~HorizonAnimAggregator() {

}

bool HorizonAnimAggregator::isCreatable() const {
	return true;
}

bool HorizonAnimAggregator::createStorage() {
	HorizonAnimInformation* info   = HorizonDataRep::newAnimationHorizon(m_manager);
	if(info != nullptr)
	{
		info->m_horizonFolderData = new HorizonFolderData(m_manager,info->name(),info->listHorizons());
		m_manager->addHorizonAnimData(info->m_horizonFolderData);
		info->m_horizonFolderData->setDisplayPreferences({InlineView,XLineView,RandomView},true);
		insert(0,info);
	}
	return true;
}


bool HorizonAnimAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	HorizonAnimInformation* info = dynamic_cast<HorizonAnimInformation*>(information);
	if (info==nullptr) {
		return false;
	}

	bool res = info->deleteStorage(errorMsg);
	if (res) {
		long i = 0;
		long N = size();
		bool foundIdx = false;
		auto it = m_informations.begin();
		while (!foundIdx && i<N) {
			foundIdx = it->obj==info;
			if (!foundIdx) {
				i++;
				it++;
			}
		}
		if (i<N) {
			remove(i);
		}
		info->deleteLater();
	}
	return res;
}

HorizonAnimInformation* HorizonAnimAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const HorizonAnimInformation* HorizonAnimAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> HorizonAnimAggregator::begin() {
	HorizonAnimIterator* it = new HorizonAnimIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> HorizonAnimAggregator::end() {
	HorizonAnimIterator* it = new HorizonAnimIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long HorizonAnimAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> HorizonAnimAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE};
}

void HorizonAnimAggregator::insert(long i, HorizonAnimInformation* information) {
	ListItem item;
	item.obj = information;
	item.originParent = information->parent();
	information->setParent(this);

	if (i>=size()) {
		i = size();
		m_informations.push_back(item);
	} else {
		if (i<0) {
			i = 0;
		}
		auto it = m_informations.begin();
		if (i>0) {
			std::advance(it, i);
		}
		m_informations.insert(it, item);
	}

	emit horizonAnimInformationAdded(i, information);
	emit informationAdded(information);

}

void HorizonAnimAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	HorizonAnimInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	emit horizonAnimInformationRemoved(i, information);
	emit informationRemoved(information);
}
