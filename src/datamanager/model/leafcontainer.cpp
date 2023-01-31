#include "leafcontainer.h"

LeafContainer::LeafContainer(QObject* parent) : QObject(parent) {

}

LeafContainer::~LeafContainer() {

}

std::size_t LeafContainer::addLeaf(DeletableLeaf leaf) {
	std::size_t id = getNextId();
	m_map[id] = leaf;
	emit dataAdded(QList<std::size_t>() << id);
	return id;
}

QList<std::size_t> LeafContainer::addLeafs(const QList<DeletableLeaf>& leafs) {
	QList<std::size_t> out;
	for (const DeletableLeaf& leaf : leafs) {
		std::size_t id = getNextId();
		m_map[id] = leaf;
		out << id;
	}
	emit dataAdded(out);
	return out;
}

const DeletableLeaf& LeafContainer::at(std::size_t id) const {
	return m_map.at(id);
}

DeletableLeaf& LeafContainer::at(std::size_t id) {
	return m_map.at(id);
}

bool LeafContainer::containId(std::size_t id) const {
	return m_map.find(id)!=m_map.end();
}

bool LeafContainer::removeLeaf(std::size_t id) {
	bool contain = containId(id);
	if (contain) {
		DeletableLeaf leaf = m_map[id];
		m_map.erase(id);
		emit dataRemoved(id, leaf);
	}
	return contain;
}

void LeafContainer::clear() {
	m_map.clear();
	emit dataCleared();
}

std::size_t LeafContainer::getNextId() const {
	return m_nextId++;
}

std::size_t LeafContainer::count() const {
	return m_map.size();
}

QList<std::size_t> LeafContainer::ids() const {
	QList<std::size_t> ids;
	for (const std::pair<std::size_t, DeletableLeaf>& leaf : m_map) {
		ids.push_back(leaf.first);
	}
	return ids;
}
