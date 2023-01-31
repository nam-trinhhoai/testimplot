#include "computationdatasetrelayconnectioncloser.h"

#include "computationoperatordataset.h"
#include "ivolumecomputationoperator.h"
#include "processrelay.h"
#include "workingsetmanager.h"

ComputationDatasetRelayConnectionCloser::ComputationDatasetRelayConnectionCloser(WorkingSetManager* manager,
		ComputationOperatorDataset* data, ProcessRelay* relay, QObject* parent) : QObject(parent) {
	m_manager = manager;
	m_data = data;
	m_relay = relay;
	m_op = m_data->computationOperator();

	connect(m_relay, &ProcessRelay::processRemoved, this, &ComputationDatasetRelayConnectionCloser::removeDataFromRelay);
	connect(m_data, &QObject::destroyed, this, &ComputationDatasetRelayConnectionCloser::dataDestroyed);
	connect(m_relay, &ProcessRelay::destroyed, this, &ComputationDatasetRelayConnectionCloser::relayDestroyed);
	connect(m_manager, &ProcessRelay::destroyed, this, &ComputationDatasetRelayConnectionCloser::workingSetDestroyed);
}

ComputationDatasetRelayConnectionCloser::~ComputationDatasetRelayConnectionCloser() {
	//clearConnections();
}

void ComputationDatasetRelayConnectionCloser::removeDataFromRelay(long id, IComputationOperator* op) {
	if (op==m_op) {
		clearConnections();
		m_manager->removeComputationOperatorDataset(m_data);
		m_data->deleteLater(); // may not be necessary because WorkingSetManager does it already
		deleteLater();
	}
}

void ComputationDatasetRelayConnectionCloser::dataDestroyed() {
	// the one deleting the data should remove it from the working set first
	// cleanup
	clearConnections();
	deleteLater();
}

void ComputationDatasetRelayConnectionCloser::relayDestroyed() {
	clearConnections();
	m_manager->removeComputationOperatorDataset(m_data);
	m_data->deleteLater(); // may not be necessary because WorkingSetManager does it already
	deleteLater();
}

void ComputationDatasetRelayConnectionCloser::workingSetDestroyed() {
	clearConnections();
	m_data->deleteLater(); // may not be necessary because WorkingSetManager does it already
	deleteLater();
}

void ComputationDatasetRelayConnectionCloser::clearConnections() {
	disconnect(m_relay, &ProcessRelay::processRemoved, this, &ComputationDatasetRelayConnectionCloser::removeDataFromRelay);
	disconnect(m_data, &QObject::destroyed, this, &ComputationDatasetRelayConnectionCloser::dataDestroyed);
	disconnect(m_relay, &ProcessRelay::destroyed, this, &ComputationDatasetRelayConnectionCloser::relayDestroyed);
	disconnect(m_manager, &ProcessRelay::destroyed, this, &ComputationDatasetRelayConnectionCloser::workingSetDestroyed);
}
