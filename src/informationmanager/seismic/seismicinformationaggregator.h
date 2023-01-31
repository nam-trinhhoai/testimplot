

#ifndef __SEISMICINFORMATIONAGGREGATOR__
#define __SEISMICINFORMATIONAGGREGATOR__

#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>
#include <QString>

#include <list>

class SeismicInformation;
class WorkingSetManager;

// NurbInformationAggregator has ownership of the informations
class SeismicInformationAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	SeismicInformationAggregator(WorkingSetManager* manager, bool enableToggleAction = true, QObject* parent=0);
	SeismicInformationAggregator(WorkingSetManager* manager, int dimx, int dimy, int dimz, bool enableToggleAction = true, QObject* parent=0);
	virtual ~SeismicInformationAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	SeismicInformation* at(long idx);
	const SeismicInformation* cat(long idx) const;
	QString surveyName();
	QString projectName();

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);

signals:
	void seismicInformationAdded(long i, SeismicInformation* information);
	void seismicInformationRemoved(long i, SeismicInformation* information);

private:
	void insert(long i, SeismicInformation* information);
	void remove(long i);

	struct ListItem {
		SeismicInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QPointer<WorkingSetManager> m_manager;
	QString getUserName();
	QString m_userName = "";
	bool m_enableToggleAction = true;
};





#endif
