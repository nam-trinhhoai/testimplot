
#ifndef SRC_INFORMATIONMANAGER_SEISMIC_SEISMICINFORMATIONITERATOR_H
#define SRC_INFORMATIONMANAGER_SEISMIC_SEISMICINFORMATIONITERATOR_H

#include "iinformationiterator.h"

#include <QObject>
#include <QPointer>

class SeismicInformation;
class SeismicInformationAggregator;

class SeismicInformationIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	SeismicInformationIterator(SeismicInformationAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~SeismicInformationIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, SeismicInformation* information);
	void informationRemoved(long i, SeismicInformation* information);

private:
	QPointer<SeismicInformationAggregator> m_aggregator;
	long m_idx;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONITERATOR_H
