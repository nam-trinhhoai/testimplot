#include "videoinformationaggregator.h"

#include "nextvisiondbmanager.h"
#include "videoinformation.h"
#include "videoinformationiterator.h"
#include <geotimepath.h>

#include <QDir>
#include <QFileInfo>

#include <iterator>

void VideoInformationAggregator::doDeleteLater(QObject* object) {
	object->deleteLater();
}

VideoInformationAggregator::VideoInformationAggregator(const QString& surveyPath, QObject* parent) :
		IInformationAggregator(parent), m_surveyPath(surveyPath) {
	std::string nvDatasetDir = NextVisionDBManager::surveyPath2NvDatasetsDir(m_surveyPath.toStdString());

	QDir datasetsDir(QString::fromStdString(nvDatasetDir));
	QFileInfoList datasetDirs = datasetsDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot);

	long counter = 0;
	for (long i=0; i<datasetDirs.size(); i++) {
		const QFileInfo& datasetDirFileInfo = datasetDirs[i];

		std::string videosDirStr = NextVisionDBManager::nvDatasetDir2RGTToRGBDir(datasetDirFileInfo.absoluteFilePath().toStdString());
		QDir videosDir(QString::fromStdString(videosDirStr));
		QFileInfoList videos = videosDir.entryInfoList(QStringList() << "*.avi", QDir::Files);
		for (long j=0; j<videos.size(); j++) {
			VideoInformation* info = new VideoInformation(videos[j].completeBaseName(), videos[j].absoluteFilePath());
			insert(counter, info);
			counter++;
		}
	}
	// nextvision dir
	QString nextVisionVideoPath = surveyPath + "/" + QString::fromStdString(GeotimePath::NEXTVISION_VIDEO_PATH) + "/";
	QDir videosDir(nextVisionVideoPath);
	QFileInfoList videos = videosDir.entryInfoList(QStringList() << "*.avi", QDir::Files);
	for (long j=0; j<videos.size(); j++) {
		VideoInformation* info = new VideoInformation(videos[j].completeBaseName(), videos[j].absoluteFilePath());
		insert(counter, info);
		counter++;
	}
}

VideoInformationAggregator::~VideoInformationAggregator() {

}

bool VideoInformationAggregator::isCreatable() const {
	return false;
}

bool VideoInformationAggregator::createStorage() {
	return false;
}

bool VideoInformationAggregator::deleteInformation(IInformation* information, QString* errorMsg) {
	VideoInformation* info = dynamic_cast<VideoInformation*>(information);
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

VideoInformation* VideoInformationAggregator::at(long idx) {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

const VideoInformation* VideoInformationAggregator::cat(long idx) const {
	if (idx<0 || idx>=size()) {
		return nullptr;
	}

	auto it = m_informations.begin();
	if (idx>0) {
		std::advance(it, idx);
	}
	return it->obj;
}

std::shared_ptr<IInformationIterator> VideoInformationAggregator::begin() {
	VideoInformationIterator* it = new VideoInformationIterator(this, 0);
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

std::shared_ptr<IInformationIterator> VideoInformationAggregator::end() {
	VideoInformationIterator* it = new VideoInformationIterator(this, size());
	std::shared_ptr<IInformationIterator> ptr(it, doDeleteLater);
	return ptr;
}

long VideoInformationAggregator::size() const {
	return m_informations.size();
}

std::list<information::Property> VideoInformationAggregator::availableProperties() const {
	return {information::Property::CREATION_DATE, information::Property::MODIFICATION_DATE, information::Property::NAME,
		information::Property::OWNER, information::Property::STORAGE_TYPE};
}

void VideoInformationAggregator::insert(long i, VideoInformation* information) {
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

	emit videosInformationAdded(i, information);
	emit informationAdded(information);
}

void VideoInformationAggregator::remove(long i) {
	if (i<0 || i>=size()) {
		return;
	}

	auto it = m_informations.begin();
	if (i>0) {
		std::advance(it, i);
	}
	VideoInformation* information = it->obj;
	if (it->originParent.isNull()) {
		information->setParent(nullptr);
	} else {
		information->setParent(it->originParent.data());
	}
	m_informations.erase(it);

	emit videosInformationRemoved(i, information);
	emit informationRemoved(information);
}
