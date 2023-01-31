#ifndef SRC_INFORMATIONMANAGER_WELLINFORMATION_H
#define SRC_INFORMATIONMANAGER_WELLINFORMATION_H

#include "GeotimeProjectManagerWidget.h"
#include "iinformation.h"
#include "iinformationfolder.h"

#include <QObject>
#include <QPointer>

class WellBore;
class WorkingSetManager;

class WellInformation : public IInformation, public IInformationFolder {
	Q_OBJECT
public:
	struct WellBoreDescParams {
		QString datum;
		QString domain;
		QString elev;
		QString ihs;
		QString status;
		QString uwi;
		QString velocity;
	};

	WellInformation(const QString& wellBoreDir, WorkingSetManager* manager, QObject* parent=nullptr);
	virtual ~WellInformation();

	// for actions
	virtual bool isDeletable() const override;
	virtual bool deleteStorage(QString* errorMsg=nullptr) override;
	virtual bool isSelectable() const override;
	virtual bool isSelected() const override;
	virtual void toggleSelection(bool toggle) override;

	void searchLoadedData();

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
	void searchFileCache() const;
	virtual QString name() const override;
	QString wellBoreName() const;
	QString wellHeadName() const;
	QStringList wellKinds() const;
	QStringList wellLogs() const;
	QStringList wellPicks() const;
	QStringList wellTfps() const;
	QStringList wellTfpPaths() const;
	virtual information::StorageType storage() const override;

	virtual bool hasProperty(information::Property property) const override;
	virtual QVariant property(information::Property property) const override;
	virtual bool isCompatible(information::Property property, const QVariant& filter) const override;

	// for gui representation, maybe should be done by another class
	virtual IInformationPanelWidget* buildInformationWidget(QWidget* parent=nullptr) override;
	virtual QWidget* buildMetadataWidget(QWidget* parent=nullptr) override;

	static QString getWellTinyName(const QString& wellPath); // could be simplified with WellsManager

	// interface of IInformationFolder
	virtual QString folder() const override;
	virtual QString mainPath() const override;

	QString currentTfpName() const;
	QString currentTfpPath() const;
	QString defaultTfp() const;
	static WellBoreDescParams readDescFile(const QString& descFile);
	QString wellBoreDescFile() const;
	WellBoreDescParams wellBoreDescParams() const;

	static bool selectWell(const QString& wellBoreDir, WorkingSetManager* manager);
	static bool unselectWell(const QString& wellBoreDir, WorkingSetManager* manager);

	// this only work if data is loaded, else there is no way to select the tfp
	bool setCurrentTfp(const QString& tfpPath, const QString& tfpName);

signals:
	void currentTfpChanged(QString tfpPath);

private:
	PMANAGER_BORE_DISPLAY getBoreInList(bool* ok=nullptr) const;
	void searchLogsTfpsPicks() const;

	QPointer<WellBore> m_loadedData;
	QPointer<WorkingSetManager> m_workingSetManager;

	QString m_wellBoreDir; // should not end with a /

	mutable WellBoreDescParams m_wellBoreDescParams;
	mutable bool m_wellBoreDescParamsSet = false;

	// cache
	mutable bool logRetrieved = false;
	mutable QList<QDateTime> m_cacheCreationDates;
	mutable QList<QDateTime> m_cacheModificationDates;
	mutable QStringList m_cacheOwners;
	mutable QString m_cacheWellHead;
	mutable QString m_cacheWellBore;
	mutable std::map<QString, QString> m_cacheWellKinds;
	mutable QStringList m_cacheWellLogNames;
	mutable QStringList m_cacheWellLogPaths;
	mutable QStringList m_cacheWellTfpNames;
	mutable QStringList m_cacheWellTfpPaths;
	mutable QStringList m_cacheWellPickNames;
};

#endif
