#ifndef SRC_INFORMATIONMANAGER_ISOHORIZON_ISOHORIZONINFORMATIONITERATOR_H
#define SRC_INFORMATIONMANAGER_ISOHORIZON_ISOHORIZONINFORMATIONITERATOR_H

#include "iinformationiterator.h"

#include <QObject>
#include <QPointer>

class IsoHorizonInformation;
class IsoHorizonInformationAggregator;

class IsoHorizonInformationIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	IsoHorizonInformationIterator(IsoHorizonInformationAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~IsoHorizonInformationIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, IsoHorizonInformation* information);
	void informationRemoved(long i, IsoHorizonInformation* information);

private:
	QPointer<IsoHorizonInformationAggregator> m_aggregator;
	long m_idx;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONITERATOR_H
