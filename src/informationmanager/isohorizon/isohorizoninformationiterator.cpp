#include "isohorizoninformationiterator.h"
#include "isohorizoninformation.h"
#include "isohorizoninformationaggregator.h"

IsoHorizonInformationIterator::IsoHorizonInformationIterator(IsoHorizonInformationAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &IsoHorizonInformationAggregator::isoHorizonInformationAdded, this, &IsoHorizonInformationIterator::informationAdded);
		connect(aggregator, &IsoHorizonInformationAggregator::isoHorizonInformationRemoved, this, &IsoHorizonInformationIterator::informationRemoved);
	}
}

IsoHorizonInformationIterator::~IsoHorizonInformationIterator() {

}

bool IsoHorizonInformationIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* IsoHorizonInformationIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* IsoHorizonInformationIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool IsoHorizonInformationIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool IsoHorizonInformationIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> IsoHorizonInformationIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		IsoHorizonInformationIterator* realPtr = dynamic_cast<IsoHorizonInformationIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void IsoHorizonInformationIterator::informationAdded(long i, IsoHorizonInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void IsoHorizonInformationIterator::informationRemoved(long i, IsoHorizonInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
