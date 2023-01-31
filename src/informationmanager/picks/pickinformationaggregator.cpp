#include "pickinformationaggregator.h"

#include "GeotimeProjectManagerWidget.h"
#include "nextvisiondbmanager.h"
#include "pickinformation.h"
#include "pickinformationiterator.h"

#include <QDir>
#include <QFileInfo>

#include <iterator>

void PickInformationAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

PickInformationAggregator::PickInformationAggregator(WorkingSetManager* manager, QObject* parent) :
		IInformationAggregator(parent), m_manager(manager) {
	manager->getManagerWidget()->pick_database_update();
	std::vector<QString> tiny = manager->getManagerWidget()->get_all_picks_names();
	std::vector<QString> full = manager->getManagerWidget()->get_all_picks_fullnames();
	std::vector<QBrush> color = manager->getManagerWidget()->get_all_picks_colors();

	long N = std::min(std::min(tiny.size(), full.size()), color.size());
	for (long i=0; i<N; i++) {
		PickInformation* info = new PickInformation(tiny[i], full[i], color[i].color(), m_manager);
		insert(i, info);
	}
}

PickInformationAggregator::~PickInformationAggregator() {

}

bool PickInformationAggregator::isCreatable() const {
	return false;
}

bool PickInformationAggregator::createStorage() {
	return false;
}

bool PickInformationAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	PickInformation* info = dynamic_cast<PickInformation*>(information);
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

PickInformation* PickInformationAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const PickInformation* PickInformationAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> PickInformationAggregator::begin() {
	PickInformationIterator* it = new PickInformationIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> PickInformationAggregator::end() {
	PickInformationIterator* it = new PickInformationIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long PickInformationAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> PickInformationAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE};
}

void PickInformationAggregator::insert(long i, PickInformation* information) {
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

	emit picksInformationAdded(i, information);
	emit informationAdded(information);
}

void PickInformationAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	PickInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	emit picksInformationRemoved(i, information);
	emit informationRemoved(information);
}
