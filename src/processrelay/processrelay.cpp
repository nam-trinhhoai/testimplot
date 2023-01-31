#include "processrelay.h"

#include "icomputationoperator.h"

#include <QMutexLocker>

std::size_t ProcessRelay::INVALID_ID = 0;

ProcessRelay::ProcessRelay(QObject* parent) : QObject(parent) , m_mutex() {

}

ProcessRelay::~ProcessRelay() {

}

std::size_t ProcessRelay::addProcess(IComputationOperator* obj) {
	std::size_t id = INVALID_ID;
	{
		QMutexLocker locker(&m_mutex);
		auto it = std::find_if(m_data.begin(), m_data.end(), [obj](const std::pair<std::size_t, IComputationOperator*>& pair) {
			return obj==pair.second;
		});

		if (it==m_data.end()) {
			id = getNextId();
			m_data[id] = obj;
		}
	}

	if (id!=INVALID_ID) {
		emit processAdded(id, obj);
	}

	return id;
}

bool ProcessRelay::removeProcess(IComputationOperator* obj) {
	std::size_t id = INVALID_ID;
	bool ok = false;
	{
		QMutexLocker locker(&m_mutex);
		auto it = std::find_if(m_data.begin(), m_data.end(), [obj](const std::pair<std::size_t, IComputationOperator*>& pair) {
			return obj==pair.second;
		});

		ok = it!=m_data.end();
		if (ok) {
			id = it->first;
			m_data.erase(it);
		}
	}

	if (ok) {
		emit processRemoved(id, obj);
	}

	return ok;
}

bool ProcessRelay::removeProcess(std::size_t id) {
	IComputationOperator* obj = nullptr;
	bool ok = false;
	{
		QMutexLocker locker(&m_mutex);
		auto it = m_data.find(id);

		ok = it!=m_data.end();
		if (ok) {
			obj = it->second;
			m_data.erase(it);
		}
	}

	if (ok) {
		emit processRemoved(id, obj);
	}

	return ok;
}

const std::map<std::size_t, IComputationOperator*>& ProcessRelay::data() const {
	QMutexLocker locker(&m_mutex);
	return m_data;
}

std::size_t ProcessRelay::getNextId() {
	return m_nextId++;
}
