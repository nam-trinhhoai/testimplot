#include "pickinformationpanelwidget.h"
#include "pickinformation.h"

#include <QColorDialog>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>

PickInformationPanelWidget::PickInformationPanelWidget(PickInformation* information, QWidget* parent) :
		IInformationPanelWidget(parent), m_information(information) {
	QColor color;
	QString name = "No name";
	if (!m_information.isNull()) {
		color = m_information->color();
		name = m_information->name();
	}

	QFormLayout* mainLayout = new QFormLayout;
	setLayout(mainLayout);
	mainLayout->addRow("Name: ", new QLabel(name));

	QPushButton* colorButton = new QPushButton;
	colorButton->setStyleSheet(QString("QPushButton{ background: %1; }").arg(color.name()));
	mainLayout->addRow("Color: ", colorButton);
}

PickInformationPanelWidget::~PickInformationPanelWidget() {

}

bool PickInformationPanelWidget::saveChanges() {
	return false;
}
