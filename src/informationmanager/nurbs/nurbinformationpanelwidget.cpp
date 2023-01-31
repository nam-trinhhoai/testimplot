#include "nurbinformationpanelwidget.h"
#include "nurbinformation.h"

#include <QColorDialog>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>

NurbInformationPanelWidget::NurbInformationPanelWidget(NurbInformation* information, QWidget* parent) :
		IInformationPanelWidget(parent), m_information(information) {
	QString name = "No name";
	QString nbCurves = "";
	QString precision = "";
	if (!m_information.isNull()) {
		m_color = m_information->color();
		name = m_information->name();
		nbCurves = QString::number(m_information->nbCurves());
		precision = QString::number(m_information->precision());
	}

	QFormLayout* mainLayout = new QFormLayout;
	setLayout(mainLayout);
	mainLayout->addRow("Name: ", new QLabel(name));

	m_colorButton = new QPushButton;
	mainLayout->addRow("Color: ", m_colorButton);
	setButtonColor(m_color);

	mainLayout->addRow("Number of generatrices : ", new QLabel(nbCurves));
	mainLayout->addRow("Precision: ", new QLabel(precision));

	connect(m_colorButton, &QPushButton::clicked, this, &NurbInformationPanelWidget::editColor);
}

NurbInformationPanelWidget::~NurbInformationPanelWidget() {

}

bool NurbInformationPanelWidget::saveChanges() {
	bool valid = !m_information.isNull();
	if (valid) {
		m_information->setColor(m_color);
	}
	return valid;
}

QColor NurbInformationPanelWidget::color() const {
	return m_color;
}

void NurbInformationPanelWidget::setColor(QColor color) {
	m_color = color;
	setButtonColor(m_color);
}

void NurbInformationPanelWidget::editColor() {
	QString name;
	if (!m_information.isNull()) {
		name = m_information->name();
	}

	QColor newColor = QColorDialog::getColor(m_color, this, "Select " + name + " color");
	if (newColor.isValid()) {
		setColor(newColor);
	}
}

void NurbInformationPanelWidget::setButtonColor(const QColor& color) {
	m_colorButton->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_color.name()));
}
