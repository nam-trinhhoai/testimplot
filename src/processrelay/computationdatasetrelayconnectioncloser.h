#ifndef COMPUTATIONDATASETRELAYCONNECTIONCLOSER_H
#define COMPUTATIONDATASETRELAYCONNECTIONCLOSER_H

#include <QObject>

#include <cstddef>

class ComputationOperatorDataset;
class IComputationOperator;
class ProcessRelay;
class WorkingSetManager;

class ComputationDatasetRelayConnectionCloser : public QObject {
	Q_OBJECT
public:
	ComputationDatasetRelayConnectionCloser(WorkingSetManager* manager,
			ComputationOperatorDataset* data, ProcessRelay* relay, QObject* parent=0);

	~ComputationDatasetRelayConnectionCloser();

public slots:
	void removeDataFromRelay(long id, IComputationOperator* op);
	void dataDestroyed();
	void relayDestroyed();
	void workingSetDestroyed();

private:
	void clearConnections();

	WorkingSetManager* m_manager;
	ComputationOperatorDataset* m_data;
	IComputationOperator* m_op;
	ProcessRelay* m_relay;
};

#endif
