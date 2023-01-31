#ifndef SRC_INFORMATIONMANAGER_ISOHORIZON_ISOHORIZONINFORMATIONPANELWIDGET_H
#define SRC_INFORMATIONMANAGER_ISOHORIZON_ISOHORIZONINFORMATIONPANELWIDGET_H

#include "iinformationpanelwidget.h"

#include <QColor>
#include <QPointer>

class IsoHorizonInformation;

class QPushButton;

class IsoHorizonInformationPanelWidget : public IInformationPanelWidget {
	Q_OBJECT
public:
	IsoHorizonInformationPanelWidget(IsoHorizonInformation* information, QWidget* parent=nullptr);
	virtual ~IsoHorizonInformationPanelWidget();

	virtual bool saveChanges() override;

	QColor color() const;

public slots:
	void editColor();
	void setColor(QColor color);

private:
	void setButtonColor(const QColor& color);

	QPointer<IsoHorizonInformation> m_information;

	QColor m_color;
	QPushButton* m_colorButton;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONPANELWIDGET_H
