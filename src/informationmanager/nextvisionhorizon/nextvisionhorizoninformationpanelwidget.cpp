#include "nextvisionhorizoninformationpanelwidget.h"
#include "nextvisionhorizoninformation.h"
#include <nextvisionhorizonattributinfo.h>

#include <workingsetmanager.h>
#include <freeHorizonQManager.h>
#include <QColorDialog>
#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>

NextvisionHorizonInformationPanelWidget::NextvisionHorizonInformationPanelWidget(NextvisionHorizonInformation* information,
		WorkingSetManager *workingSetManager,
		QWidget* parent) :
		IInformationPanelWidget(parent), m_information(information) {
	QString name = "No name";
	QString NAttributs = 0;
	QString Dims = "";
	m_workingSetManager = workingSetManager;
	std::vector<QString> attributName = m_information->attributName();
	std::vector<QString> attributPath = m_information->attributPath();

	if (!m_information.isNull()) {
		m_color = m_information->color();
		name = m_information->name();
		Dims = m_information->Dims();
		NAttributs = QString::number(attributName.size());

		connect(m_information.get(), &NextvisionHorizonInformation::colorChanged, this,
				&NextvisionHorizonInformationPanelWidget::setColor);
	}
	QFormLayout* mainLayout = new QFormLayout;
	setLayout(mainLayout);
	QLineEdit *labelName = new QLineEdit(name);
	labelName->setReadOnly(true);
	labelName->setStyleSheet("QLineEdit { border: none }");
	mainLayout->addRow("Name: ", labelName);
	m_colorButton = new QPushButton;
	mainLayout->addRow("Color: ", m_colorButton);
	setButtonColor(m_color);
	mainLayout->addRow("Dimensions: ", new QLabel(Dims));
	mainLayout->addRow("Number of attributs: ", new QLabel(NAttributs));

	int cpt = 0;
	for (int i=0; i<attributName.size(); i++)
	{
		NextvisionHorizonAttributInfo *attributInfo = new NextvisionHorizonAttributInfo(attributPath[i], attributName[i],
				m_workingSetManager, m_information->name(), m_information->path());
		mainLayout->addRow(attributInfo);
	}

	/*
	int cpt = 0;
	for (int i=0; i<attributName.size(); i++)
	{
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
		if ( attributType == "spectrum" )
		{
			QString nFreq = m_information->getNbreSpectrumFrequencies(i);
			mainLayout->addRow("Nbre frequencies: ", new QLabel(nFreq));
		}
		if ( attributType == "gcc" )
		{
			QString ngcc = m_information->getNbreGccScales(i);
			mainLayout->addRow("Nbre scales: ", new QLabel(ngcc));
		}
	}
	QFrame *line = new QFrame;
	line->setObjectName(QString::fromUtf8("line"));
	*/
	connect(m_colorButton, &QPushButton::clicked, this, &NextvisionHorizonInformationPanelWidget::editColor);
}

NextvisionHorizonInformationPanelWidget::~NextvisionHorizonInformationPanelWidget() {
	if (!m_information.isNull()) {
		disconnect(m_information.get(), &NextvisionHorizonInformation::colorChanged, this,
				&NextvisionHorizonInformationPanelWidget::setColor);
	}
}

bool NextvisionHorizonInformationPanelWidget::saveChanges() {
	bool valid = !m_information.isNull();
	if (valid) {
		m_information->setColor(m_color);
	}
	return valid;
}

QColor NextvisionHorizonInformationPanelWidget::color() const {
	return m_color;
}

void NextvisionHorizonInformationPanelWidget::setColor(QColor color) {
	m_color = color;
	setButtonColor(m_color);
}

void NextvisionHorizonInformationPanelWidget::editColor() {
	QString name;
	if (!m_information.isNull()) {
		name = m_information->name();
	}

	QColor newColor = QColorDialog::getColor(m_color, this, "Select " + name + " color");
	if (newColor.isValid()) {
		setColor(newColor);
	}
}

void NextvisionHorizonInformationPanelWidget::setButtonColor(const QColor& color) {
	m_colorButton->setStyleSheet(QString("QPushButton{ background: %1; }").arg(m_color.name()));
}
