#include "isohorizoninformationaggregator.h"

#include "GeotimeProjectManagerWidget.h"
#include "isohorizoninformation.h"
#include "isohorizoninformationiterator.h"
// #include "nextvisionhorizonwidget.h"
#include "workingsetmanager.h"

#include <iterator>

void IsoHorizonInformationAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

IsoHorizonInformationAggregator::IsoHorizonInformationAggregator(WorkingSetManager* manager, QObject* parent) :
		IInformationAggregator(parent), m_manager(manager) {

	std::vector<QString> dataFullname = manager->getManagerWidget()->get_isohorizon_fullnames();
	std::vector<QString> dataTinyname = manager->getManagerWidget()->get_isohorizon_names();

	long N = std::min(dataFullname.size(), dataTinyname.size());
	for (long i=0; i<N; i++) {
		QString name = dataTinyname[i];
		name.replace(".txt", "");
		IsoHorizonInformation* info = new IsoHorizonInformation(name, dataFullname[i], manager);
		insert(i, info);
	}
}

IsoHorizonInformationAggregator::~IsoHorizonInformationAggregator() {

}

bool IsoHorizonInformationAggregator::isCreatable() const {
	return true;
}

bool IsoHorizonInformationAggregator::createStorage() {
	// NextvisionHorizonWidget::showWidget();
	return true;
}

bool IsoHorizonInformationAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	IsoHorizonInformation* info = dynamic_cast<IsoHorizonInformation*>(information);
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

IsoHorizonInformation* IsoHorizonInformationAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const IsoHorizonInformation* IsoHorizonInformationAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> IsoHorizonInformationAggregator::begin() {
	IsoHorizonInformationIterator* it = new IsoHorizonInformationIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> IsoHorizonInformationAggregator::end() {
	IsoHorizonInformationIterator* it = new IsoHorizonInformationIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long IsoHorizonInformationAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> IsoHorizonInformationAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE};
}

void IsoHorizonInformationAggregator::insert(long i, IsoHorizonInformation* information) {
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

	emit isoHorizonInformationAdded(i, information);
	emit informationAdded(information);
}

void IsoHorizonInformationAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	IsoHorizonInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	emit isoHorizonInformationRemoved(i, information);
	emit informationRemoved(information);
}
