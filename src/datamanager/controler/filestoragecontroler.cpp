#include "filestoragecontroler.h"
#include "leafcontainer.h"
#include "deleteableleaf.h"
#include "datamanager/util_filesystem.h"
#include "globalconfig.h"

#include <algorithm>
#include <QDebug>

FileStorageControler::FileStorageControler(QObject* parent) : QObject(parent) {

}

FileStorageControler::~FileStorageControler() {

}

std::size_t FileStorageControler::addContainerDuo(ContainerDuo duo) {
	std::size_t id = nextIndex();
	m_containers[id] = duo;
	QMetaObject::Connection connMain = connect(duo.main, &LeafContainer::destroyed, [this, id]() {
		this->removeContainerDuo(id);
	});
	QMetaObject::Connection connTrash = connect(duo.trash, &LeafContainer::destroyed, [this, id]() {
		this->removeContainerDuo(id);
	});
	QList<QMetaObject::Connection> conns({connMain, connTrash});
	m_containerConnections[id] = conns;
	return id;
}

bool FileStorageControler::removeContainerDuo(std::size_t id) {
	std::map<std::size_t, QList<QMetaObject::Connection>>::iterator itConnection = m_containerConnections.find(id);
	if (itConnection!=m_containerConnections.end()) {
		for (QMetaObject::Connection conn : itConnection->second) {
			QObject::disconnect(conn);
		}
		m_containerConnections.erase(id);
	}
	return m_containers.erase(id)>0;
}

std::size_t FileStorageControler::nextIndex() const {
	return m_nextIndex++;
}

bool FileStorageControler::removeLeafFromMainContainer(std::size_t leafKey, std::size_t containerKey) {
	bool out = false;

	std::map<std::size_t, ContainerDuo>::iterator itDuo = m_containers.find(containerKey);

	out = itDuo!=m_containers.end();
	if (out) {
		out = itDuo->second.main->containId(leafKey);
	}

	if (out) {
		DeletableLeaf& leaf = itDuo->second.main->at(leafKey);
		std::pair<bool, DeletableLeaf> pair = ijkMoveToTrash(leaf);
		out = pair.first;
		if (out) {
			itDuo->second.main->removeLeaf(leafKey);
			itDuo->second.trash->addLeaf(pair.second);
		}
	}

	return out;
}

bool FileStorageControler::restoreLeafFromTrashContainer(std::size_t leafKey, std::size_t containerKey) {
	bool out = false;

	std::map<std::size_t, ContainerDuo>::iterator itDuo = m_containers.find(containerKey);

	out = itDuo!=m_containers.end();
	if (out) {
		out = itDuo->second.trash->containId(leafKey);
	}

	if (out) {
		DeletableLeaf& leaf = itDuo->second.trash->at(leafKey);
		std::pair<bool, DeletableLeaf> pair = ijkRestoreFromTrash(leaf);
		out = pair.first;
		if (out) {
			itDuo->second.trash->removeLeaf(leafKey);
			itDuo->second.main->addLeaf(pair.second);
		}
	}

	return out;
}

bool FileStorageControler::restoreLeafFromTrashContainer(std::size_t leafKey, LeafContainer* trashContainer) {
	bool out = false;

	std::map<std::size_t, ContainerDuo>::const_iterator it = std::find_if(m_containers.begin(), m_containers.end(), [trashContainer](const std::pair<std::size_t, ContainerDuo>& val) {
		return val.second.trash==trashContainer;
	});
	out = it != m_containers.end();
	if (out) {
		std::size_t duoId = it->first;
		out = restoreLeafFromTrashContainer(leafKey, duoId);
	}

	return out;
}

bool FileStorageControler::deleteLeafFromTrashContainer(std::size_t leafKey, std::size_t containerKey) {
	bool out = false;

	if (m_projectPathSet) {
		std::map<std::size_t, ContainerDuo>::iterator itDuo = m_containers.find(containerKey);

		out = itDuo!=m_containers.end();
		if (out) {
			out = itDuo->second.trash->containId(leafKey);
		}

		if (out) {
			DeletableLeaf& leaf = itDuo->second.trash->at(leafKey);
			std::pair<bool, DeletableLeaf> outPair = ijkDeleteFromTrash(leaf, m_logPath);
			out = outPair.first;
			if (out) {
				itDuo->second.trash->removeLeaf(leafKey);
			} else {
				LeafContainer* trashContainer = itDuo->second.trash;
				trashContainer->at(leafKey) = outPair.second;
			}
		}
	} else {
		qDebug() << "Reject deletion because project not set";
	}

	return out;
}

bool FileStorageControler::deleteLeafFromTrashContainer(std::size_t leafKey, LeafContainer* trashContainer) {
	bool out = false;

	std::map<std::size_t, ContainerDuo>::const_iterator it = std::find_if(m_containers.begin(), m_containers.end(), [trashContainer](const std::pair<std::size_t, ContainerDuo>& val) {
		return val.second.trash==trashContainer;
	});
	out = it != m_containers.end();
	if (out) {
		std::size_t duoId = it->first;
		out = deleteLeafFromTrashContainer(leafKey, duoId);
	}

	return out;
}

QString FileStorageControler::logPath() const {
	return m_logPath;
}

void FileStorageControler::setProjectPath(QString projectPath) {
	GlobalConfig config = GlobalConfig::getConfig();

	m_projectPath = projectPath;
	m_logPath = config.getDeleteLogPathFromProjectPath(m_projectPath);
	m_projectPathSet = true;
}

void FileStorageControler::clearProjectPath() {
	m_projectPath = "";
	m_logPath = "";
	m_projectPathSet = false;
}
