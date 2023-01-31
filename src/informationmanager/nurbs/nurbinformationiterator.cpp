#include "nurbinformationiterator.h"
#include "nurbinformation.h"
#include "nurbinformationaggregator.h"

NurbInformationIterator::NurbInformationIterator(NurbInformationAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &NurbInformationAggregator::nurbsInformationAdded, this, &NurbInformationIterator::informationAdded);
		connect(aggregator, &NurbInformationAggregator::nurbsInformationRemoved, this, &NurbInformationIterator::informationRemoved);
	}
}

NurbInformationIterator::~NurbInformationIterator() {

}

bool NurbInformationIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* NurbInformationIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* NurbInformationIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool NurbInformationIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool NurbInformationIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> NurbInformationIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		NurbInformationIterator* realPtr = dynamic_cast<NurbInformationIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void NurbInformationIterator::informationAdded(long i, NurbInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void NurbInformationIterator::informationRemoved(long i, NurbInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
