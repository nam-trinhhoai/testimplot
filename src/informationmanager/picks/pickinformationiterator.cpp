#include "pickinformationiterator.h"
#include "pickinformation.h"
#include "pickinformationaggregator.h"

PickInformationIterator::PickInformationIterator(PickInformationAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &PickInformationAggregator::picksInformationAdded, this, &PickInformationIterator::informationAdded);
		connect(aggregator, &PickInformationAggregator::picksInformationRemoved, this, &PickInformationIterator::informationRemoved);
	}
}

PickInformationIterator::~PickInformationIterator() {

}

bool PickInformationIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* PickInformationIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* PickInformationIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool PickInformationIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool PickInformationIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> PickInformationIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		PickInformationIterator* realPtr = dynamic_cast<PickInformationIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void PickInformationIterator::informationAdded(long i, PickInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void PickInformationIterator::informationRemoved(long i, PickInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
