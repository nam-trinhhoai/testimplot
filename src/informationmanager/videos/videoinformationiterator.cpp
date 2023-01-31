#include "videoinformationiterator.h"
#include "videoinformation.h"
#include "videoinformationaggregator.h"

VideoInformationIterator::VideoInformationIterator(VideoInformationAggregator* aggregator, long idx, QObject* parent) :
		QObject(parent), IInformationIterator(), m_aggregator(aggregator), m_idx(idx) {
	if (aggregator) {
		connect(aggregator, &VideoInformationAggregator::videosInformationAdded, this, &VideoInformationIterator::informationAdded);
		connect(aggregator, &VideoInformationAggregator::videosInformationRemoved, this, &VideoInformationIterator::informationRemoved);
	}
}

VideoInformationIterator::~VideoInformationIterator() {

}

bool VideoInformationIterator::isValid() {
	return m_idx>=0 && !m_aggregator.isNull() && m_idx<m_aggregator->size();
}

const IInformation* VideoInformationIterator::cobject() const {
	const IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->cat(m_idx);
	}
	return info;
}

IInformation* VideoInformationIterator::object() {
	IInformation* info = nullptr;
	if (!m_aggregator.isNull()) {
		info = m_aggregator->at(m_idx);
	}
	return info;
}

bool VideoInformationIterator::hasNext() const {
	return m_idx+1>=0 && !m_aggregator.isNull() && m_idx+1<m_aggregator->size();
}

bool VideoInformationIterator::next() {
	m_idx++;
	return isValid();
}

std::shared_ptr<IInformationIterator> VideoInformationIterator::copy() const {
	// use aggregator to allow the aggregator to have control if needed in the future
	// This could be removed if needed
	std::shared_ptr<IInformationIterator> it;
	if (!m_aggregator.isNull()) {
		it = m_aggregator->begin();
		VideoInformationIterator* realPtr = dynamic_cast<VideoInformationIterator*>(it.get());
		if (realPtr) {
			realPtr->m_idx = m_idx;
		}
	}
	return it;
}

void VideoInformationIterator::informationAdded(long i, VideoInformation* information) {
	if (i<=m_idx) {
		m_idx++;
	}
}

void VideoInformationIterator::informationRemoved(long i, VideoInformation* information) {
	if (i<m_idx) {
		m_idx--;
	}
}
