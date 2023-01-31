#ifndef SRC_PROCESSRELAY_PROCESSRELAY_H
#define SRC_PROCESSRELAY_PROCESSRELAY_H

#include <QObject>
#include <QMutex>

#include <map>

class IComputationOperator;

/**
 * ProcessRelay does not take ownership of the processes,
 * The processes need to be removed before being destroyed
 *
 * The same process cannot be added more than once
 */
class ProcessRelay : public QObject {
	Q_OBJECT
public:
	ProcessRelay(QObject* parent = nullptr);
	~ProcessRelay();

	std::size_t addProcess(IComputationOperator* obj);
	bool removeProcess(IComputationOperator* obj);
	bool removeProcess(std::size_t id);

	const std::map<std::size_t, IComputationOperator*>& data() const;

	static std::size_t INVALID_ID;

signals:
	void processAdded(long id, IComputationOperator* obj);
	void processRemoved(long id, IComputationOperator* obj);

private:
	std::size_t getNextId();

	std::map<std::size_t, IComputationOperator*> m_data;
	mutable QMutex m_mutex;

	std::size_t m_nextId = 1;
};

#endif // INCLUDE_PROCESSRELAY_PROCESSRELAY_H
