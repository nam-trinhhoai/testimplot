#include "seismicinformationmetadatawidget.h"

#include "globalconfig.h"
#include "iinformation.h"
#include "iinformationfolder.h"

#include <QDir>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QProcess>


SeismicInformationMetadataWidget::SeismicInformationMetadataWidget(IInformation* information, QWidget* parent) :
		QWidget(parent), m_information(information) {
	QString mainOwner = "No Owner";
	QString mainCreationDateStr = "";
	QString mainModificationDateStr = "";
	m_hasFolder = false;
	QString mainPath;
	if (m_information) {
		mainOwner = m_information->mainOwner();
		QDateTime time = m_information->mainCreationDate();
		mainCreationDateStr = time.toString("ddd MMMM d yyyy hh:mm:ss");
		QDateTime timeModif = m_information->mainModificationDate();
		mainModificationDateStr = timeModif.toString("ddd MMMM d yyyy hh:mm:ss");

		IInformationFolder* ifolder = dynamic_cast<IInformationFolder*>(m_information);
		if (ifolder) {
			m_openDir = ifolder->folder();
			mainPath = ifolder->mainPath();
			m_hasFolder = true;
		}
	}

	QFormLayout* mainLayout = new QFormLayout;
	setLayout(mainLayout);

	mainLayout->addRow("Owner: ", new QLabel(mainOwner));
	mainLayout->addRow("Creation date: ", new QLabel(mainCreationDateStr));
	mainLayout->addRow("Last modification date: ", new QLabel(mainModificationDateStr));

	if (m_hasFolder) {
		QLineEdit* pathLabel = new QLineEdit(mainPath);
		pathLabel->setToolTip(mainPath);
		pathLabel->setStyleSheet("QLineEdit { border: none }");
		pathLabel->setReadOnly(true);
		m_openButton = new QPushButton("Folder");
		mainLayout->addRow(m_openButton, pathLabel);

		connect(m_openButton, &QPushButton::clicked, this, &SeismicInformationMetadataWidget::openDir);
	} else {
		m_openButton = nullptr;
	}
}

SeismicInformationMetadataWidget::~SeismicInformationMetadataWidget() {

}

void SeismicInformationMetadataWidget::openDir() {
	if (!m_hasFolder) {
		return;
	}

	QDir dir(m_openDir);
	if (!dir.exists()) {
		return;
	}

	GlobalConfig& config = GlobalConfig::getConfig();

	QProcess* process = new QProcess(this);
	process->start(config.fileExplorerProgram(), QStringList() << m_openDir);
}
