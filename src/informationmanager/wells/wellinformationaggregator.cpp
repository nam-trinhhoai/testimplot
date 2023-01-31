#include "wellinformationaggregator.h"

#include "GeotimeProjectManagerWidget.h"
#include "wellinformation.h"
#include "wellinformationiterator.h"
#include "workingsetmanager.h"

#include <iterator>

void WellInformationAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

WellInformationAggregator::WellInformationAggregator(WorkingSetManager* manager, QObject* parent) :
		IInformationAggregator(parent), m_manager(manager) {
	manager->getManagerWidget()->well_database_update();
	std::vector<PMANAGER_WELL_DISPLAY> wellDisplayList = m_manager->getManagerWidget()->get_display_well_list();

	for (long i=0; i<wellDisplayList.size(); i++) {
		const std::vector<PMANAGER_BORE_DISPLAY>& bores = wellDisplayList[i].bore;
		for (long j=0; j<bores.size(); j++) {
			WellInformation* info = new WellInformation(bores[j].bore_fullname, manager);
			insert(i, info);
		}
	}
}

WellInformationAggregator::~WellInformationAggregator() {

}

bool WellInformationAggregator::isCreatable() const {
	return false;
}

bool WellInformationAggregator::createStorage() {
	return false;
}

bool WellInformationAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	WellInformation* info = dynamic_cast<WellInformation*>(information);
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

WellInformation* WellInformationAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const WellInformation* WellInformationAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> WellInformationAggregator::begin() {
	WellInformationIterator* it = new WellInformationIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> WellInformationAggregator::end() {
	WellInformationIterator* it = new WellInformationIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long WellInformationAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> WellInformationAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE, information::Property::WELL_BORE,
		information::Property::WELL_HEAD, information::Property::WELL_KIND,
		information::Property::WELL_LOG_NAME, information::Property::WELL_PICK_NAME,
		information::Property::WELL_TFP_NAME};
}

void WellInformationAggregator::insert(long i, WellInformation* information) {
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

	emit wellInformationAdded(i, information);
	emit informationAdded(information);
}

void WellInformationAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	WellInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	emit wellInformationRemoved(i, information);
	emit informationRemoved(information);
}
