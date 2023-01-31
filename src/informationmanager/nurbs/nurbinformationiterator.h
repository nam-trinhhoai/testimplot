#ifndef SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONITERATOR_H
#define SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONITERATOR_H

#include "iinformationiterator.h"

#include <QObject>
#include <QPointer>

class NurbInformation;
class NurbInformationAggregator;

class NurbInformationIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	NurbInformationIterator(NurbInformationAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~NurbInformationIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, NurbInformation* information);
	void informationRemoved(long i, NurbInformation* information);

private:
	QPointer<NurbInformationAggregator> m_aggregator;
	long m_idx;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONITERATOR_H
