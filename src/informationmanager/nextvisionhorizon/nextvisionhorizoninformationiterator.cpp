#include "nextvisionhorizoninformationiterator.h"
#include "nextvisionhorizoninformation.h"
#include "nextvisionhorizoninformationaggregator.h"

NextvisionHorizonInformationIterator::NextvisionHorizonInformationIterator(NextvisionHorizonInformationAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &NextvisionHorizonInformationAggregator::nextvisionHorizonInformationAdded, this, &NextvisionHorizonInformationIterator::informationAdded);
		connect(aggregator, &NextvisionHorizonInformationAggregator::nextvisionHorizonInformationRemoved, this, &NextvisionHorizonInformationIterator::informationRemoved);
	}
}

NextvisionHorizonInformationIterator::~NextvisionHorizonInformationIterator() {

}

bool NextvisionHorizonInformationIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* NextvisionHorizonInformationIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* NextvisionHorizonInformationIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool NextvisionHorizonInformationIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool NextvisionHorizonInformationIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> NextvisionHorizonInformationIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		NextvisionHorizonInformationIterator* realPtr = dynamic_cast<NextvisionHorizonInformationIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void NextvisionHorizonInformationIterator::informationAdded(long i, NextvisionHorizonInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void NextvisionHorizonInformationIterator::informationRemoved(long i, NextvisionHorizonInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
