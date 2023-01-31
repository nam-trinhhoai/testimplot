#ifndef HORIZONANINITERATOR_H
#define HORIZONANINITERATOR_H

#include "iinformationiterator.h"
#include "horizonanimaggregator.h"

#include <QObject>
#include <QPointer>

class HorizonAnimInformation;
class HorizonAnimAggregator;

class HorizonAnimIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	HorizonAnimIterator(HorizonAnimAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~HorizonAnimIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, HorizonAnimInformation* information);
	void informationRemoved(long i, HorizonAnimInformation* information);

private:
	QPointer<HorizonAnimAggregator> m_aggregator;
	long m_idx;
};

#endif // HORIZONANINITERATOR_H
