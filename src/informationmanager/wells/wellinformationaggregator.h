#ifndef SRC_INFORMATIONMANAGER_WELLS_WELLINFORMATIONAGGREGATOR_H
#define SRC_INFORMATIONMANAGER_WELLS_WELLINFORMATIONAGGREGATOR_H

#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>

#include <list>

class WellInformation;
class WorkingSetManager;

// NurbInformationAggregator has ownership of the informations
class WellInformationAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	WellInformationAggregator(WorkingSetManager* manager, QObject* parent=0);
	virtual ~WellInformationAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	WellInformation* at(long idx);
	const WellInformation* cat(long idx) const;

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);

signals:
	void wellInformationAdded(long i, WellInformation* information);
	void wellInformationRemoved(long i, WellInformation* information);

private:
	void insert(long i, WellInformation* information);
	void remove(long i);

	struct ListItem {
		WellInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QPointer<WorkingSetManager> m_manager;
};

#endif // SRC_INFORMATIONMANAGER_WELLS_WELLINFORMATIONAGGREGATOR_H
