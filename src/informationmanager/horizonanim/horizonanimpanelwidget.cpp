#include "horizonanimpanelwidget.h"
#include "horizonaniminformation.h"
#include "horizondatarep.h"

#include <QColorDialog>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QListWidget>
#include <QFileInfo>

HorizonAnimPanelWidget::HorizonAnimPanelWidget(HorizonAnimInformation* information, QWidget* parent) :
		IInformationPanelWidget(parent), m_information(information) {
	QString name = "No name";
	//QString nbCurves = "";
	//QString precision = "";
	QString nameAttribut="";
	QStringList horizons;

	if (!m_information.isNull()) {
		//m_color = m_information->color();
		name = m_information->name();
		nameAttribut =m_information->nameAttribut();
		horizons = m_information->listHorizons();
		//nbCurves = QString::number(m_information->nbCurves());
		//precision = QString::number(m_information->precision());
	}

	QFormLayout* mainLayout = new QFormLayout;
	setLayout(mainLayout);
	mainLayout->addRow("Name: ", new QLabel(name));

	QListWidget* listWidget = new QListWidget();

	mainLayout->addWidget(listWidget);
	for(int i=0;i<horizons.size();i++)
	{
		QFileInfo fileinfo(horizons[i]);
		QString name = fileinfo.baseName();
		listWidget->addItem(name);
	}

	m_comboAttribut = new QComboBox();

	if(information != nullptr && information->m_horizonFolderData != nullptr)
	{
		m_comboAttribut->addItems(information->m_horizonFolderData->getAttributesAvailable());
		m_comboAttribut->setCurrentText(nameAttribut);
	}

	mainLayout->addRow("Attribut: ",m_comboAttribut);// new QLabel(nameAttribut));

	connect(m_comboAttribut, &QComboBox::currentIndexChanged, this, &HorizonAnimPanelWidget::attributChanged);

	/*m_colorButton = new QPushButton;
	mainLayout->addRow("Color: ", m_colorButton);
	setButtonColor(m_color);

	mainLayout->addRow("Number of generatrices : ", new QLabel(nbCurves));
	mainLayout->addRow("Precision: ", new QLabel(precision));

	connect(m_colorButton, &QPushButton::clicked, this, &HorizonAnimPanelWidget::editColor);*/
}

HorizonAnimPanelWidget::~HorizonAnimPanelWidget() {

}

bool HorizonAnimPanelWidget::saveChanges() {
	bool valid = !m_information.isNull();
	if (valid) {
		m_information->save();
	}
	return valid;
}

void HorizonAnimPanelWidget::attributChanged(int i) {
	QString name;
	if (!m_information.isNull()) {
		QString nameAttribut = m_comboAttribut->itemText(i);
		m_information->setNameAttribut(nameAttribut);

	}
}

/*QColor HorizonAnimPanelWidget::color() const {
	return m_color;
}

void HorizonAnimPanelWidget::setColor(QColor color) {
	m_color = color;
	setButtonColor(m_color);
}

void HorizonAnimPanelWidget::editColor() {
	QString name;
	if (!m_information.isNull()) {
		name = m_information->name();
	}

QColor newColor = QColorDialog::getColor(m_color, this, "Select " + name + " color");
	if (newColor.isValid()) {
		setColor(newColor);
	}
}

void HorizonAnimPanelWidget::setButtonColor(const QColor& color) {
	m_colorButton->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_color.name()));
}*/
