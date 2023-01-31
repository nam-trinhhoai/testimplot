#include "nurbinformationaggregator.h"

#include "GeotimeProjectManagerWidget.h"
#include "nurbinformation.h"
#include "nurbinformationiterator.h"
#include "nurbswidget.h"
#include "workingsetmanager.h"

#include <iterator>

void NurbInformationAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

NurbInformationAggregator::NurbInformationAggregator(WorkingSetManager* manager, QObject* parent) :
		IInformationAggregator(parent), m_manager(manager) {
	manager->getManagerWidget()->trt_nurbs_database_update();

	std::vector<QString> dataFullname = manager->getManagerWidget()->get_nurbs_fullnames0();
	std::vector<QString> dataTinyname = manager->getManagerWidget()->get_nurbs_names0();

	long N = std::min(dataFullname.size(), dataTinyname.size());
	for (long i=0; i<N; i++) {
		QString name = dataTinyname[i];
		name.replace(".txt", "");
		//qDebug()<<"dataFullname -->"<<dataFullname[i];
		NurbInformation* info = new NurbInformation(name, dataFullname[i], manager);
		insert(i, info);
	}
}

NurbInformationAggregator::~NurbInformationAggregator() {

}

bool NurbInformationAggregator::isCreatable() const {
	return true;
}

bool NurbInformationAggregator::createStorage() {
	QString name = NurbsWidget::newNurbsSimple();
	if(name!= "")
	{
		QString dataFullname = m_manager->getManagerWidget()->get_nurbs_path0()+name+".txt";
		NurbInformation* info = new NurbInformation(name, dataFullname, m_manager);
		insert(0, info);
	}
	return true;
}

bool NurbInformationAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	NurbInformation* info = dynamic_cast<NurbInformation*>(information);
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

NurbInformation* NurbInformationAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const NurbInformation* NurbInformationAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> NurbInformationAggregator::begin() {
	NurbInformationIterator* it = new NurbInformationIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> NurbInformationAggregator::end() {
	NurbInformationIterator* it = new NurbInformationIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long NurbInformationAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> NurbInformationAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE};
}

void NurbInformationAggregator::insert(long i, NurbInformation* information) {
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

	emit nurbsInformationAdded(i, information);
	emit informationAdded(information);
}

void NurbInformationAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	NurbInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	emit nurbsInformationRemoved(i, information);
	emit informationRemoved(information);
}
