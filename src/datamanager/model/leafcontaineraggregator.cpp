#include "leafcontaineraggregator.h"
#include "leafcontainer.h"

LeafContainerAggregator::LeafContainerAggregator(QObject* parent) : QObject(parent) {

}

LeafContainerAggregator::~LeafContainerAggregator() {

}

std::size_t LeafContainerAggregator::addContainer(LeafContainer* container) { // return container id
	std::size_t id = getNextId();
	m_map[id] = container;
	QMetaObject::Connection connAdd = connect(container, &LeafContainer::dataAdded, [this, id](const QList<std::size_t>&  newLeafsId) {
		QList<AggregatorKey> keys;
		for (std::size_t leafId : newLeafsId) {
			AggregatorKey key;
			key.containerId = id;
			key.leafId = leafId;
			keys << key;
		}
		emit dataAdded(keys);
	});
	QMetaObject::Connection connRemoved = connect(container, &LeafContainer::dataRemoved, [this, id](std::size_t leafId, DeletableLeaf leaf) {
		AggregatorKey key;
		key.containerId = id;
		key.leafId = leafId;
		emit dataRemoved(key, leaf);
	});
	QMetaObject::Connection connCleared = connect(container, &LeafContainer::dataCleared, [this, id]() {
		emit containerDataCleared(id);
	});
	QMetaObject::Connection connDelete = connect(container, &LeafContainer::destroyed, [this, id]() {
		this->removeContainer(id);
	});
	QList<QMetaObject::Connection> conns({connAdd, connRemoved, connCleared, connDelete});
	m_containerConnections[id] = conns;
	return id;
}

bool LeafContainerAggregator::removeContainer(std::size_t containerId) {
	std::map<std::size_t, QList<QMetaObject::Connection>>::iterator itConnection = m_containerConnections.find(containerId);
	if (itConnection!=m_containerConnections.end()) {
		for (QMetaObject::Connection conn : itConnection->second) {
			QObject::disconnect(conn);
		}
		m_containerConnections.erase(containerId);
	}
	return m_map.erase(containerId)>0;
}

LeafContainer* LeafContainerAggregator::container(std::size_t containerId) {
	LeafContainer* out = nullptr;
	std::map<std::size_t, LeafContainer*>::const_iterator it = m_map.find(containerId);
	if (it!=m_map.end()) {
		out = m_map.at(containerId);
	}
	return out;
}

const LeafContainer* LeafContainerAggregator::container(std::size_t containerId) const {
	const LeafContainer* out = nullptr;
	std::map<std::size_t, LeafContainer*>::const_iterator it = m_map.find(containerId);
	if (it!=m_map.end()) {
		out = m_map.at(containerId);
	}
	return out;
}

const DeletableLeaf& LeafContainerAggregator::at(AggregatorKey id) const {
	return m_map.at(id.containerId)->at(id.leafId);
}

DeletableLeaf& LeafContainerAggregator::at(AggregatorKey id) {
	return m_map.at(id.containerId)->at(id.leafId);
}

bool LeafContainerAggregator::containId(AggregatorKey id) const {
	bool out;

	std::map<std::size_t, LeafContainer*>::const_iterator it = m_map.find(id.containerId);
	out = it!=m_map.end() && it->second->containId(id.leafId);

	return out;
}

bool LeafContainerAggregator::removeLeaf(AggregatorKey id) {
	bool out = containId(id);
	if (out) {
		DeletableLeaf leaf = m_map.at(id.containerId)->at(id.leafId);
		out = m_map.at(id.containerId)->removeLeaf(id.leafId);
		emit dataRemoved(id, leaf);
	}
	return out;
}

void LeafContainerAggregator::clearContainersList() {
	m_map.clear();
	for (const std::pair<size_t, QList<QMetaObject::Connection>>& pair : m_containerConnections) {
		for (QMetaObject::Connection conn : pair.second) {
			QObject::disconnect(conn);
		}
	}
	m_containerConnections.clear();
	emit dataCleared();
}

void LeafContainerAggregator::clearContainersContent() {
	for (const std::pair<std::size_t, LeafContainer*>& pair : m_map) {
		pair.second->clear();
	}
	emit dataCleared();
}

std::size_t LeafContainerAggregator::count() const {
	std::size_t sum = 0;
	for (const std::pair<std::size_t, LeafContainer*>& pair : m_map) {
		sum += pair.second->count();
	}
	return sum;
}

QList<LeafContainerAggregator::AggregatorKey> LeafContainerAggregator::ids() const {
	QList<LeafContainerAggregator::AggregatorKey> out;
	for (const std::pair<std::size_t, LeafContainer*>& pair : m_map) {
		QList<std::size_t> leafIds  = pair.second->ids();
		for (std::size_t leafId : leafIds) {
			AggregatorKey key;
			key.containerId = pair.first;
			key.leafId = leafId;
			out << key;
		}
	}
}

std::size_t LeafContainerAggregator::getNextId() const {// id for containers
	return m_nextId++;
}

