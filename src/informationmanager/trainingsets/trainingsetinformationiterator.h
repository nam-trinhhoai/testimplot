#ifndef SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATIONITERATOR_H
#define SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATIONITERATOR_H

#include "iinformationiterator.h"

#include <QObject>
#include <QPointer>

class TrainingSetInformation;
class TrainingSetInformationAggregator;

class TrainingSetInformationIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	TrainingSetInformationIterator(TrainingSetInformationAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~TrainingSetInformationIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, TrainingSetInformation* information);
	void informationRemoved(long i, TrainingSetInformation* information);

private:
	QPointer<TrainingSetInformationAggregator> m_aggregator;
	long m_idx;
};

#endif // SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATIONITERATOR_H
