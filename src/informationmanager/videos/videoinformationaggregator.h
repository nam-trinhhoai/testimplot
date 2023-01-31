#ifndef SRC_INFORMATIONMANAGER_VIDEOS_VIDEOINFORMATIONAGGREGATOR_H
#define SRC_INFORMATIONMANAGER_VIDEOS_VIDEOINFORMATIONAGGREGATOR_H

#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>

#include <list>

class VideoInformation;

// VideoInformationAggregator has ownership of the informations
class VideoInformationAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	VideoInformationAggregator(const QString& surveyPath, QObject* parent=0);
	virtual ~VideoInformationAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	VideoInformation* at(long idx);
	const VideoInformation* cat(long idx) const;

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);

signals:
	void videosInformationAdded(long i, VideoInformation* information);
	void videosInformationRemoved(long i, VideoInformation* information);

private:
	void insert(long i, VideoInformation* information);
	void remove(long i);

	struct ListItem {
		VideoInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QString m_surveyPath;
};

#endif // SRC_INFORMATIONMANAGER_VIDEOS_VIDEOINFORMATIONAGGREGATOR_H
