#ifndef SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATION_H
#define SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATION_H

#include "iinformation.h"
#include "iinformationfolder.h"

#include <QObject>

class TrainingSetInformation : public IInformation, public IInformationFolder {
	Q_OBJECT
public:
	TrainingSetInformation(const QString& name, const QString& trainingSetPath, QObject* parent=0);
	virtual ~TrainingSetInformation();

	// for actions
	virtual bool isDeletable() const override;
	virtual bool deleteStorage(QString* errorMsg=nullptr) override;
	virtual bool isSelectable() const override;
	virtual bool isSelected() const override;
	virtual void toggleSelection(bool toggle) override;

	// comments
	virtual bool commentsEditable() const override;
	virtual QString comments() const override;
	virtual void setComments(const QString& txt) override;

	virtual bool hasIcon() const override;
	virtual QIcon icon(int preferedSizeX, int preferedSizeY) const override;

	// for sort and filtering
	virtual QString mainOwner() const override;
	virtual QStringList owners() const override;
	virtual QDateTime mainCreationDate() const override;
	virtual QList<QDateTime> creationDates() const override;
	virtual QDateTime mainModificationDate() const override;
	virtual QList<QDateTime> modificationDates() const override;
	virtual QString name() const override;
	virtual information::StorageType storage() const override;
	void searchFileCache() const;

	virtual bool hasProperty(information::Property property) const override;
	virtual QVariant property(information::Property property) const override;
	virtual bool isCompatible(information::Property property, const QVariant& filter) const override;

	// for gui representation, maybe should be done by another class
	virtual IInformationPanelWidget* buildInformationWidget(QWidget* parent=nullptr) override;
	virtual QWidget* buildMetadataWidget(QWidget* parent=nullptr) override;

	// interface of IInformationFolder
	virtual QString folder() const override;
	virtual QString mainPath() const override;


private:
	mutable QList<QDateTime> m_cacheCreationDates;
	mutable QList<QDateTime> m_cacheModificationDates;
	mutable QStringList m_cacheOwners;
	mutable bool m_cacheSearchDone = false;

	QString m_trainingSetPath;
	QString m_name;
};

#endif // SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATION_H
