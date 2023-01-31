#ifndef SRC_INFORMATIONMANAGER_IINFORMATIONAGGREGATOR_H
#define SRC_INFORMATIONMANAGER_IINFORMATIONAGGREGATOR_H

#include "iinformationiterator.h"
#include "informationutils.h"

#include <QObject>

#include <list>
#include <memory>

class IInformationAggregator : public QObject {
	Q_OBJECT
public:
	IInformationAggregator(QObject* parent=0);
	virtual ~IInformationAggregator();

	virtual bool isCreatable() const = 0;
	virtual bool createStorage() = 0;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) = 0;

	virtual std::shared_ptr<IInformationIterator> begin() = 0;
	virtual std::shared_ptr<IInformationIterator> end() = 0;
	virtual long size() const = 0;

	virtual std::list<information::Property> availableProperties() const = 0;

signals:
	void informationAdded(IInformation* information);
	void informationRemoved(IInformation* information);
};

#endif // SRC_INFORMATIONMANAGER_IINFORMATIONAGGREGATOR_H
