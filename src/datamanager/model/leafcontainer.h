#ifndef DATAMANAGER_LEAFCONTAINER_H_
#define DATAMANAGER_LEAFCONTAINER_H_

#include "deleteableleaf.h"
#include <QObject>
#include <QList>

#include <map>

class LeafContainer : public QObject {
	Q_OBJECT
public:
	LeafContainer(QObject* parent=nullptr);
	~LeafContainer();

	std::size_t addLeaf(DeletableLeaf leaf);
	QList<std::size_t> addLeafs(const QList<DeletableLeaf>& leafs);
	const DeletableLeaf& at(std::size_t id) const;
	DeletableLeaf& at(std::size_t id);
	bool containId(std::size_t id) const;
	bool removeLeaf(std::size_t id);
	void clear();

	std::size_t count() const;
	QList<std::size_t> ids() const;

signals:
	void dataAdded(const QList<std::size_t>&  newLeafsId);
	void dataRemoved(std::size_t id, DeletableLeaf leaf);
	void dataCleared();

private:
	std::size_t getNextId() const;

	std::map<std::size_t, DeletableLeaf> m_map;
	mutable std::size_t m_nextId = 1;
};

#endif
