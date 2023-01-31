#include "trainingsetinformationaggregator.h"

#include "sismagedbmanager.h"
#include "trainingsetinformation.h"
#include "trainingsetinformationiterator.h"
#include "trainingsetparameterwidget.h"

#include <QDir>
#include <QFileInfo>

#include <iterator>

void TrainingSetInformationAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

TrainingSetInformationAggregator::TrainingSetInformationAggregator(const QString& projectPath, QObject* parent) :
		IInformationAggregator(parent), m_projectPath(projectPath) {
	std::string neuronPath = SismageDBManager::projectPath2NeuronPath(m_projectPath.toStdString());

	QDir neuronDir(QString::fromStdString(neuronPath));
	QFileInfoList trainingSets = neuronDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot);

	for (long i=0; i<trainingSets.size(); i++) {
		TrainingSetInformation* info = new TrainingSetInformation(trainingSets[i].completeBaseName(), trainingSets[i].absoluteFilePath());
		insert(i, info);
	}
}

TrainingSetInformationAggregator::~TrainingSetInformationAggregator() {

}

bool TrainingSetInformationAggregator::isCreatable() const {
	return true;
}

bool TrainingSetInformationAggregator::createStorage() {
	TrainingSetParameterWidget* widget = new TrainingSetParameterWidget;
	widget->show();
	return true;
}

bool TrainingSetInformationAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	TrainingSetInformation* info = dynamic_cast<TrainingSetInformation*>(information);
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

TrainingSetInformation* TrainingSetInformationAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const TrainingSetInformation* TrainingSetInformationAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> TrainingSetInformationAggregator::begin() {
	TrainingSetInformationIterator* it = new TrainingSetInformationIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> TrainingSetInformationAggregator::end() {
	TrainingSetInformationIterator* it = new TrainingSetInformationIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long TrainingSetInformationAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> TrainingSetInformationAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE};
}

void TrainingSetInformationAggregator::insert(long i, TrainingSetInformation* information) {
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

	emit trainingSetsInformationAdded(i, information);
	emit informationAdded(information);
}

void TrainingSetInformationAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	TrainingSetInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	emit trainingSetsInformationRemoved(i, information);
	emit informationRemoved(information);
}
