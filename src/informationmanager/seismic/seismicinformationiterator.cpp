
#include "seismicinformationiterator.h"
#include "seismicinformation.h"
#include "seismicinformationaggregator.h"

SeismicInformationIterator::SeismicInformationIterator(SeismicInformationAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &SeismicInformationAggregator::seismicInformationAdded, this, &SeismicInformationIterator::informationAdded);
		connect(aggregator, &SeismicInformationAggregator::seismicInformationRemoved, this, &SeismicInformationIterator::informationRemoved);
	}
}

SeismicInformationIterator::~SeismicInformationIterator() {

}

bool SeismicInformationIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* SeismicInformationIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* SeismicInformationIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool SeismicInformationIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool SeismicInformationIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> SeismicInformationIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		SeismicInformationIterator* realPtr = dynamic_cast<SeismicInformationIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void SeismicInformationIterator::informationAdded(long i, SeismicInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void SeismicInformationIterator::informationRemoved(long i, SeismicInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
