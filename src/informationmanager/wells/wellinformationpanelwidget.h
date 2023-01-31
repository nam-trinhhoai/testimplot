#ifndef SRC_INFORMATIONMANAGER_WELLS_WELLINFORMATIONPANELWIDGET_H
#define SRC_INFORMATIONMANAGER_WELLS_WELLINFORMATIONPANELWIDGET_H

#include "iinformationpanelwidget.h"

#include <QColor>
#include <QPointer>

class WellInformation;

class QComboBox;
class QPushButton;

class WellInformationPanelWidget : public IInformationPanelWidget {
	Q_OBJECT
public:
	WellInformationPanelWidget(WellInformation* information, QWidget* parent=nullptr);
	virtual ~WellInformationPanelWidget();

	virtual bool saveChanges() override;

public slots:
	void tfpPathChanged(QString path);

private:
	QPointer<WellInformation> m_information;

	QComboBox* m_currentTfpComboBox;
};

#endif // SRC_INFORMATIONMANAGER_WELLS_WELLINFORMATIONPANELWIDGET_H
