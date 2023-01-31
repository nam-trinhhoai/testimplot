#ifndef SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATIONAGGREGATOR_H
#define SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATIONAGGREGATOR_H

#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>

#include <list>

class PickInformation;
class WorkingSetManager;

// PickInformationAggregator has ownership of the informations
class PickInformationAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	PickInformationAggregator(WorkingSetManager* manager, QObject* parent=0);
	virtual ~PickInformationAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	PickInformation* at(long idx);
	const PickInformation* cat(long idx) const;

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);

signals:
	void picksInformationAdded(long i, PickInformation* information);
	void picksInformationRemoved(long i, PickInformation* information);

private:
	void insert(long i, PickInformation* information);
	void remove(long i);

	struct ListItem {
		PickInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QPointer<WorkingSetManager> m_manager;
};

#endif // SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATIONAGGREGATOR_H
