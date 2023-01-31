#ifndef SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONAGGREGATOR_H
#define SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONAGGREGATOR_H

#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>

#include <list>

class NurbInformation;
class WorkingSetManager;

// NurbInformationAggregator has ownership of the informations
class NurbInformationAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	NurbInformationAggregator(WorkingSetManager* manager, QObject* parent=0);
	virtual ~NurbInformationAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	NurbInformation* at(long idx);
	const NurbInformation* cat(long idx) const;

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);

signals:
	void nurbsInformationAdded(long i, NurbInformation* information);
	void nurbsInformationRemoved(long i, NurbInformation* information);

private:
	void insert(long i, NurbInformation* information);
	void remove(long i);

	struct ListItem {
		NurbInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QPointer<WorkingSetManager> m_manager;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONAGGREGATOR_H
