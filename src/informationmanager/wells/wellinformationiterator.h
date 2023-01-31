#ifndef SRC_INFORMATIONMANAGER_WELLS_WELLINFORMATIONITERATOR_H
#define SRC_INFORMATIONMANAGER_WELLS_WELLINFORMATIONITERATOR_H

#include "iinformationiterator.h"

#include <QObject>
#include <QPointer>

class WellInformation;
class WellInformationAggregator;

class WellInformationIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	WellInformationIterator(WellInformationAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~WellInformationIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, WellInformation* information);
	void informationRemoved(long i, WellInformation* information);

private:
	QPointer<WellInformationAggregator> m_aggregator;
	long m_idx;
};

#endif // SRC_INFORMATIONMANAGER_WELLS_WELLINFORMATIONITERATOR_H
