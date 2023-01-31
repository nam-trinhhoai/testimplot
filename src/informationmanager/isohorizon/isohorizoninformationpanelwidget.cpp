#include "isohorizoninformationpanelwidget.h"
#include "isohorizoninformation.h"
#include "isohorizonattributinfo.h"

#include <freeHorizonQManager.h>
#include <QColorDialog>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>

IsoHorizonInformationPanelWidget::IsoHorizonInformationPanelWidget(IsoHorizonInformation* information, QWidget* parent) :
		IInformationPanelWidget(parent), m_information(information) {
	QString name = "No name";
	QString NAttributs = 0;
	QString Dims = "";
	QString NDirectories = "";
	std::vector<QString> attributName = m_information->attributName();
	std::vector<QString> attributPath = m_information->attributPath();
	QString attributDirPath = m_information->attributDirPath();

	if (!m_information.isNull()) {
		m_color = m_information->color();
		name = m_information->name();
		NDirectories = m_information->getNbreDirectories();
		Dims = m_information->Dims();
		NAttributs = QString::number(attributName.size());

		connect(m_information.get(), &IsoHorizonInformation::colorChanged, this, &IsoHorizonInformationPanelWidget::setColor);
	}
	QFormLayout* mainLayout = new QFormLayout;
	setLayout(mainLayout);
	mainLayout->addRow("Name: ", new QLabel(name));
	m_colorButton = new QPushButton;
	mainLayout->addRow("Color: ", m_colorButton);
	setButtonColor(m_color);
	mainLayout->addRow("Dimensions: ", new QLabel(Dims));
	mainLayout->addRow("Number of attributs: ", new QLabel(NAttributs));
	mainLayout->addRow("Number of iso: ", new QLabel(NDirectories));

	for (int i=0; i<attributName.size(); i++)
	{
		/*
		// mainLayout->addRow(" ", new QLabel(" "));
		QFrame *line = new QFrame;
		line->setObjectName(QString::fromUtf8("line"));
		line->setGeometry(QRect(320, 150, 118, 3));
		line->setFrameShape(QFrame::HLine);
		mainLayout->addRow(line);
		mainLayout->addRow("Name: ", new QLabel(attributName[i]));
		QString attributType = m_information->getAttributType(i);
		// mainLayout->addRow("Type", new QLabel(attributType));
		QString sizeOnDisk = m_information->getSizeOnDisk(i);
		mainLayout->addRow("Size on disk: ", new QLabel(sizeOnDisk));
		*/

		IsoHorizonAttributInfo *attributInfo = new IsoHorizonAttributInfo(attributPath[i], attributName[i], attributDirPath);
		mainLayout->addRow(attributInfo);
	}

	connect(m_colorButton, &QPushButton::clicked, this, &IsoHorizonInformationPanelWidget::editColor);
}

IsoHorizonInformationPanelWidget::~IsoHorizonInformationPanelWidget() {
	if (!m_information.isNull()) {
		disconnect(m_information.get(), &IsoHorizonInformation::colorChanged, this, &IsoHorizonInformationPanelWidget::setColor);
	}
}

bool IsoHorizonInformationPanelWidget::saveChanges() {
	bool valid = !m_information.isNull();
	if (valid) {
		m_information->setColor(m_color);
	}
	return valid;
}

QColor IsoHorizonInformationPanelWidget::color() const {
	return m_color;
}

void IsoHorizonInformationPanelWidget::setColor(QColor color) {
	m_color = color;
	setButtonColor(m_color);
}

void IsoHorizonInformationPanelWidget::editColor() {
	QString name;
	if (!m_information.isNull()) {
		name = m_information->name();
	}

	QColor newColor = QColorDialog::getColor(m_color, this, "Select " + name + " color");
	if (newColor.isValid()) {
		setColor(newColor);
	}
}

void IsoHorizonInformationPanelWidget::setButtonColor(const QColor& color) {
	m_colorButton->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_color.name()));
}
