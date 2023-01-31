#ifndef SRC_INFORMATIONMANAGER_ISOHORIZON_ISOHORIZONINFORMATIONAGGREGATOR_H
#define SRC_INFORMATIONMANAGER_ISOHORIZON_ISOHORIZONINFORMATIONAGGREGATOR_H

#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>

#include <list>

class IsoHorizonInformation;
class WorkingSetManager;

// NurbInformationAggregator has ownership of the informations
class IsoHorizonInformationAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	IsoHorizonInformationAggregator(WorkingSetManager* manager, QObject* parent=0);
	virtual ~IsoHorizonInformationAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	IsoHorizonInformation* at(long idx);
	const IsoHorizonInformation* cat(long idx) const;

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);

 signals:
	void isoHorizonInformationAdded(long i, IsoHorizonInformation* information);
	void isoHorizonInformationRemoved(long i, IsoHorizonInformation* information);

private:
	void insert(long i, IsoHorizonInformation* information);
	void remove(long i);

	struct ListItem {
		IsoHorizonInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QPointer<WorkingSetManager> m_manager;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONAGGREGATOR_H
