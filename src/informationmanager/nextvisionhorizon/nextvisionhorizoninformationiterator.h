#ifndef SRC_INFORMATIONMANAGER_NEXTVISIONHORIZON_NECTVISIONHORIZONINFORMATIONITERATOR_H
#define SRC_INFORMATIONMANAGER_NEXTVISIONHORIZON_NECTVISIONHORIZONINFORMATIONITERATOR_H

#include "iinformationiterator.h"

#include <QObject>
#include <QPointer>

class NextvisionHorizonInformation;
class NextvisionHorizonInformationAggregator;

class NextvisionHorizonInformationIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	NextvisionHorizonInformationIterator(NextvisionHorizonInformationAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~NextvisionHorizonInformationIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, NextvisionHorizonInformation* information);
	void informationRemoved(long i, NextvisionHorizonInformation* information);

private:
	QPointer<NextvisionHorizonInformationAggregator> m_aggregator;
	long m_idx;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONITERATOR_H
