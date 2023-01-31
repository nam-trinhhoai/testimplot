
#ifndef __LOGINFORMATIONITERATOR__
#define __LOGINFORMATIONITERATOR__

#include "iinformationiterator.h"

#include <QObject>
#include <QPointer>

class LogInformation;
class LogInformationAggregator;

class LogInformationIterator : public QObject, public IInformationIterator {
	Q_OBJECT
public:
	LogInformationIterator(LogInformationAggregator* aggregator, long idx, QObject* parent=0);
	virtual ~LogInformationIterator();

	virtual bool isValid() override;
	virtual const IInformation* cobject() const override;
	virtual IInformation* object() override;

	virtual bool hasNext() const override;
	virtual bool next() override;

	virtual std::shared_ptr<IInformationIterator> copy() const override;

public slots:
	void informationAdded(long i, LogInformation* information);
	void informationRemoved(long i, LogInformation* information);

private:
	QPointer<LogInformationAggregator> m_aggregator;
	long m_idx;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONITERATOR_H
