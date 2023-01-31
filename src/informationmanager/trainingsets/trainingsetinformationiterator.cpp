#include "trainingsetinformationiterator.h"
#include "trainingsetinformation.h"
#include "trainingsetinformationaggregator.h"

TrainingSetInformationIterator::TrainingSetInformationIterator(TrainingSetInformationAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &TrainingSetInformationAggregator::trainingSetsInformationAdded, this, &TrainingSetInformationIterator::informationAdded);
		connect(aggregator, &TrainingSetInformationAggregator::trainingSetsInformationRemoved, this, &TrainingSetInformationIterator::informationRemoved);
	}
}

TrainingSetInformationIterator::~TrainingSetInformationIterator() {

}

bool TrainingSetInformationIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* TrainingSetInformationIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* TrainingSetInformationIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool TrainingSetInformationIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool TrainingSetInformationIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> TrainingSetInformationIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		TrainingSetInformationIterator* realPtr = dynamic_cast<TrainingSetInformationIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void TrainingSetInformationIterator::informationAdded(long i, TrainingSetInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void TrainingSetInformationIterator::informationRemoved(long i, TrainingSetInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
