#ifndef SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATIONAGGREGATOR_H
#define SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATIONAGGREGATOR_H

#include "iinformationaggregator.h"

#include <QObject>
#include <QPointer>

#include <list>

class TrainingSetInformation;

// TrainingSetInformationAggregator has ownership of the informations
class TrainingSetInformationAggregator : public IInformationAggregator {
	Q_OBJECT
public:
	TrainingSetInformationAggregator(const QString& projectPath, QObject* parent=0);
	virtual ~TrainingSetInformationAggregator();

	virtual bool isCreatable() const override;
	virtual bool createStorage() override;
	virtual bool deleteInformation(IInformation* information, QString* errorMsg=nullptr) override;

	virtual std::shared_ptr<IInformationIterator> begin() override;
	virtual std::shared_ptr<IInformationIterator> end() override;
	virtual long size() const override;
	TrainingSetInformation* at(long idx);
	const TrainingSetInformation* cat(long idx) const;

	virtual std::list<information::Property> availableProperties() const override;

	static void doDeleteLater(QObject* object);

signals:
	void trainingSetsInformationAdded(long i, TrainingSetInformation* information);
	void trainingSetsInformationRemoved(long i, TrainingSetInformation* information);

private:
	void insert(long i, TrainingSetInformation* information);
	void remove(long i);

	struct ListItem {
		TrainingSetInformation* obj = nullptr;
		QPointer<QObject> originParent;
	};

	std::list<ListItem> m_informations;
	QString m_projectPath;
};

#endif // SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATIONAGGREGATOR_H
