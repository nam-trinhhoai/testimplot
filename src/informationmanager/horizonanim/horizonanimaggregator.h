#ifndef HORIZONANIMAGGREGATOR_H
#define HORIZONANIMAGGREGATOR_H

#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>

#include <list>

class HorizonAnimInformation;
class WorkingSetManager;

// NurbInformationAggregator has ownership of the informations
class HorizonAnimAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	HorizonAnimAggregator(WorkingSetManager* manager, QObject* parent=0);
	virtual ~HorizonAnimAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	HorizonAnimInformation* at(long idx);
	const HorizonAnimInformation* cat(long idx) const;

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);

signals:
	void horizonAnimInformationAdded(long i, HorizonAnimInformation* information);
	void horizonAnimInformationRemoved(long i, HorizonAnimInformation* information);

private:
	void insert(long i, HorizonAnimInformation* information);
	void remove(long i);

	struct ListItem {
		HorizonAnimInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QPointer<WorkingSetManager> m_manager;
};

#endif // HORIZONANIMAGGREGATOR_H
