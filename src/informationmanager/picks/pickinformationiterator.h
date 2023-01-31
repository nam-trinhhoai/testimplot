#ifndef SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATIONITERATOR_H
#define SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATIONITERATOR_H

#include "iinformationiterator.h"

#include <QObject>
#include <QPointer>

class PickInformation;
class PickInformationAggregator;

class PickInformationIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	PickInformationIterator(PickInformationAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~PickInformationIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, PickInformation* information);
	void informationRemoved(long i, PickInformation* information);

private:
	QPointer<PickInformationAggregator> m_aggregator;
	long m_idx;
};

#endif // SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATIONITERATOR_H
