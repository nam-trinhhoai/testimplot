#ifndef SRC_INFORMATIONMANAGER_INFORMATIONPROPERTYWIDGET_H
#define SRC_INFORMATIONMANAGER_INFORMATIONPROPERTYWIDGET_H

#include "informationutils.h"

#include <QWidget>


class IInformationPropertyWidget : public QWidget {
	Q_OBJECT
public:
	IInformationPropertyWidget(QWidget* parent=nullptr);
	virtual ~IInformationPropertyWidget();

	virtual information::Property property() const = 0;
	virtual QVariant value() const = 0;

public slots:
	virtual bool setValue(const QVariant& val) = 0;

signals:
	void valueChanged(QVariant val);
};

#endif // SRC_INFORMATIONMANAGER_INFORMATIONPROPERTYWIDGET_H
