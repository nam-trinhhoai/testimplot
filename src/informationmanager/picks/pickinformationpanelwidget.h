#ifndef SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATIONPANELWIDGET_H
#define SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATIONPANELWIDGET_H

#include "iinformationpanelwidget.h"

#include <QColor>
#include <QPointer>

class PickInformation;

class QPushButton;

class PickInformationPanelWidget : public IInformationPanelWidget {
	Q_OBJECT
public:
	PickInformationPanelWidget(PickInformation* information, QWidget* parent=nullptr);
	virtual ~PickInformationPanelWidget();

	virtual bool saveChanges() override;

private:
	QPointer<PickInformation> m_information;
};

#endif // SRC_INFORMATIONMANAGER_PICKS_PICKINFORMATIONPANELWIDGET_H
