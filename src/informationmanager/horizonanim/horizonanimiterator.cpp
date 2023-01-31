#include "horizonanimiterator.h"
#include "horizonaniminformation.h"
#include "horizonanimaggregator.h"

HorizonAnimIterator::HorizonAnimIterator(HorizonAnimAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &HorizonAnimAggregator::horizonAnimInformationAdded, this, &HorizonAnimIterator::informationAdded);
		connect(aggregator, &HorizonAnimAggregator::horizonAnimInformationRemoved, this, &HorizonAnimIterator::informationRemoved);
	}
}

HorizonAnimIterator::~HorizonAnimIterator() {

}

bool HorizonAnimIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* HorizonAnimIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* HorizonAnimIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool HorizonAnimIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool HorizonAnimIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> HorizonAnimIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		HorizonAnimIterator* realPtr = dynamic_cast<HorizonAnimIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void HorizonAnimIterator::informationAdded(long i, HorizonAnimInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void HorizonAnimIterator::informationRemoved(long i, HorizonAnimInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
