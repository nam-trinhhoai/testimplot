#ifndef SRC_INFORMATIONMANAGER_IINFORMATION_H
#define SRC_INFORMATIONMANAGER_IINFORMATION_H

#include "informationutils.h"

#include <QDateTime>
#include <QIcon>
#include <QObject>
#include <QStringList>
#include <QVariant>

class IInformationPanelWidget;

class QWidget;


class IInformation : public QObject {
	Q_OBJECT
public:
	IInformation(QObject* parent=0);
	virtual ~IInformation();

	// for actions
	virtual bool isDeletable() const = 0;
	virtual bool deleteStorage(QString* errorMsg=nullptr) = 0;
	virtual bool isSelectable() const = 0;
	virtual bool isSelected() const = 0;
	virtual void toggleSelection(bool toggle) = 0;

	// comments
	virtual bool commentsEditable() const = 0;
	virtual QString comments() const = 0;
	virtual void setComments(const QString& txt) = 0;

	virtual bool hasIcon() const = 0;
	virtual QIcon icon(int preferedSizeX, int preferedSizeY) const = 0;

	// for sort and filtering
	virtual QString mainOwner() const = 0;
	virtual QStringList owners() const = 0;
	virtual QDateTime mainCreationDate() const = 0;
	virtual QList<QDateTime> creationDates() const = 0;
	virtual QDateTime mainModificationDate() const = 0;
	virtual QList<QDateTime> modificationDates() const = 0;
	virtual QString name() const = 0;
	virtual information::StorageType storage() const = 0;

	virtual bool hasProperty(information::Property property) const = 0;
	virtual QVariant property(information::Property property) const = 0;
	virtual bool isCompatible(information::Property property, const QVariant& filter) const = 0;

	// for gui representation, maybe should be done by another class
	virtual IInformationPanelWidget* buildInformationWidget(QWidget* parent=nullptr) = 0;
	virtual QWidget* buildMetadataWidget(QWidget* parent=nullptr) = 0;

signals:
	void iconChanged();
};

Q_DECLARE_METATYPE(IInformation*)

#endif // SRC_INFORMATIONMANAGER_IINFORMATION_H
