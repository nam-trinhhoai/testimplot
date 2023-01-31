#ifndef SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATIONPANELWIDGET_H
#define SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATIONPANELWIDGET_H

#include "iinformationpanelwidget.h"

#include <QColor>
#include <QPointer>

class TrainingSetInformation;

class QPushButton;

class TrainingSetInformationPanelWidget : public IInformationPanelWidget {
	Q_OBJECT
public:
	TrainingSetInformationPanelWidget(TrainingSetInformation* information, QWidget* parent=nullptr);
	virtual ~TrainingSetInformationPanelWidget();

	virtual bool saveChanges() override;

private:
	QPointer<TrainingSetInformation> m_information;
};

#endif // SRC_INFORMATIONMANAGER_TRAININGSETS_TRAININGSETINFORMATIONPANELWIDGET_H
