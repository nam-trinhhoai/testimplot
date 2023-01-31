#ifndef DATAMANAGER_LEAFCONTAINERAGGREGATOR_H_
#define DATAMANAGER_LEAFCONTAINERAGGREGATOR_H_

#include "deleteableleaf.h"
#include <QObject>
#include <QList>

#include <map>

class LeafContainer;

class LeafContainerAggregator : public QObject {
	Q_OBJECT
public:
	typedef struct AggregatorKey {
		std::size_t containerId;
		std::size_t leafId;
	} AggregatorKey;

	LeafContainerAggregator(QObject* parent=nullptr);
	~LeafContainerAggregator();

	std::size_t addContainer(LeafContainer* container); // return container id
	bool removeContainer(std::size_t containerId); // return container id
	LeafContainer* container(std::size_t containerId);
	const LeafContainer* container(std::size_t containerId) const;
	const DeletableLeaf& at(AggregatorKey id) const;
	DeletableLeaf& at(AggregatorKey id);
	bool containId(AggregatorKey id) const;
	bool removeLeaf(AggregatorKey id);
	void clearContainersList();
	void clearContainersContent();

	std::size_t count() const;
	QList<AggregatorKey> ids() const;

signals:
	void dataAdded(const QList<AggregatorKey>&  newLeafsId);
	void dataRemoved(AggregatorKey id, DeletableLeaf leaf);
	void dataCleared();
	void containerDataCleared(std::size_t containerId);

private:
	std::size_t getNextId() const;// id for containers

	std::map<std::size_t, LeafContainer*> m_map;
	std::map<std::size_t, QList<QMetaObject::Connection>> m_containerConnections;
	mutable std::size_t m_nextId = 1;
};

Q_DECLARE_METATYPE(LeafContainerAggregator::AggregatorKey);

#endif
