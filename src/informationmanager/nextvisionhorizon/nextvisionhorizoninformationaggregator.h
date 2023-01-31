#ifndef SRC_INFORMATIONMANAGER_NEXTVISIONHORIZON_NEXTVISIONHORIZONINFORMATIONAGGREGATOR_H
#define SRC_INFORMATIONMANAGER_NEXTVISIONHORIZON_NEXTVISIONHORIZONINFORMATIONAGGREGATOR_H

#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>
#include <QString>

#include <list>

class NextvisionHorizonInformation;
class WorkingSetManager;

// NurbInformationAggregator has ownership of the informations
class NextvisionHorizonInformationAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	NextvisionHorizonInformationAggregator(WorkingSetManager* manager, bool enableToggleAction = true, QObject* parent=0);
	virtual ~NextvisionHorizonInformationAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	NextvisionHorizonInformation* at(long idx);
	const NextvisionHorizonInformation* cat(long idx) const;

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);
	QString surveyName();
	QString projectName();

 signals:
	void nextvisionHorizonInformationAdded(long i, NextvisionHorizonInformation* information);
	void nextvisionHorizonInformationRemoved(long i, NextvisionHorizonInformation* information);

private:
	void insert(long i, NextvisionHorizonInformation* information);
	void remove(long i);

	struct ListItem {
		NextvisionHorizonInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QPointer<WorkingSetManager> m_manager;
	bool m_enableToggleAction = true;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONAGGREGATOR_H
