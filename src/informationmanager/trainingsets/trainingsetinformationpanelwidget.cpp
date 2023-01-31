#include "trainingsetinformationpanelwidget.h"
#include "trainingsetinformation.h"

#include <QFormLayout>
#include <QLabel>


TrainingSetInformationPanelWidget::TrainingSetInformationPanelWidget(TrainingSetInformation* information, QWidget* parent) :
		IInformationPanelWidget(parent), m_information(information) {
	QString name = "No name";
	if (!m_information.isNull()) {
		name = m_information->name();
	}

	QFormLayout* mainLayout = new QFormLayout;
	setLayout(mainLayout);
	mainLayout->addRow("Name: ", new QLabel(name));
}

TrainingSetInformationPanelWidget::~TrainingSetInformationPanelWidget() {

}

bool TrainingSetInformationPanelWidget::saveChanges() {
	return false;
}
