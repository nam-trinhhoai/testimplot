#ifndef SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATION_H
#define SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATION_H

#include "iinformation.h"
#include "iinformationfolder.h"
#include "marker.h"
#include "workingsetmanager.h"

#include <QPointer>

class PickInformation : public IInformation, public IInformationFolder {
	Q_OBJECT
public:
	PickInformation(const QString& name, const QString& path, const QColor& color, WorkingSetManager* manager, QObject* parent=0);
	virtual ~PickInformation();

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
	QColor color() const;
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

	static bool selectPick(const QString& pickPath, WorkingSetManager* manager);
	static bool unselectPick(const QString& pickPath, WorkingSetManager* manager);

private:
	void searchLoadedData();

	mutable QList<QDateTime> m_cacheCreationDates;
	mutable QList<QDateTime> m_cacheModificationDates;
	mutable QStringList m_cacheOwners;
	mutable bool m_cacheSearchDone = false;
	mutable long m_cacheDataIdx = -1;

	QPointer<Marker> m_loadedData;
	QPointer<WorkingSetManager> m_manager;
	QColor m_color;
	QString m_name;
	QString m_path;
};

#endif // SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATION_H
