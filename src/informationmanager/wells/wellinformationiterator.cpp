#include "wellinformationiterator.h"
#include "wellinformation.h"
#include "wellinformationaggregator.h"

WellInformationIterator::WellInformationIterator(WellInformationAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &WellInformationAggregator::wellInformationAdded, this, &WellInformationIterator::informationAdded);
		connect(aggregator, &WellInformationAggregator::wellInformationRemoved, this, &WellInformationIterator::informationRemoved);
	}
}

WellInformationIterator::~WellInformationIterator() {

}

bool WellInformationIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* WellInformationIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* WellInformationIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool WellInformationIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool WellInformationIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> WellInformationIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		WellInformationIterator* realPtr = dynamic_cast<WellInformationIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void WellInformationIterator::informationAdded(long i, WellInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void WellInformationIterator::informationRemoved(long i, WellInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
