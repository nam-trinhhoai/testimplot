
#ifndef __LOGINFORMATIONAGGREGATOR__
#define __LOGINFORMATIONAGGREGATOR__


#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>
#include <QString>

#include <list>

class LogInformation;
class WorkingSetManager;
class WellBore;

// NurbInformationAggregator has ownership of the informations
class LogInformationAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	LogInformationAggregator(WellBore* manager, QObject* parent=0);
	virtual ~LogInformationAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	LogInformation* at(long idx);
	const LogInformation* cat(long idx) const;
	QString surveyName();
	QString projectName();

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);

signals:
	void logInformationAdded(long i, LogInformation* information);
	void logInformationRemoved(long i, LogInformation* information);

private:
	void insert(long i, LogInformation* information);
	void remove(long i);

	struct ListItem {
		LogInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QPointer<WorkingSetManager> m_manager;
	QString getUserName();
	QString m_userName = "";
	WellBore *m_wellBore = nullptr;
};








#endif
