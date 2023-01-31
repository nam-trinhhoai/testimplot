#ifndef SRC_INFORMATIONMANAGER_VIDEOS_VIDEOINFORMATIONITERATOR_H
#define SRC_INFORMATIONMANAGER_VIDEOS_VIDEOINFORMATIONITERATOR_H

#include "iinformationiterator.h"

#include <QObject>
#include <QPointer>

class VideoInformation;
class VideoInformationAggregator;

class VideoInformationIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	VideoInformationIterator(VideoInformationAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~VideoInformationIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, VideoInformation* information);
	void informationRemoved(long i, VideoInformation* information);

private:
	QPointer<VideoInformationAggregator> m_aggregator;
	long m_idx;
};

#endif // SRC_INFORMATIONMANAGER_VIDEOS_VIDEOINFORMATIONITERATOR_H
