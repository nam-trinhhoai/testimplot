
#include "loginformationiterator.h"
#include "loginformation.h"
#include "loginformationaggregator.h"

LogInformationIterator::LogInformationIterator(LogInformationAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &LogInformationAggregator::logInformationAdded, this, &LogInformationIterator::informationAdded);
		connect(aggregator, &LogInformationAggregator::logInformationRemoved, this, &LogInformationIterator::informationRemoved);
	}
}

LogInformationIterator::~LogInformationIterator() {

}

bool LogInformationIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* LogInformationIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* LogInformationIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool LogInformationIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool LogInformationIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> LogInformationIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		LogInformationIterator* realPtr = dynamic_cast<LogInformationIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void LogInformationIterator::informationAdded(long i, LogInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void LogInformationIterator::informationRemoved(long i, LogInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
