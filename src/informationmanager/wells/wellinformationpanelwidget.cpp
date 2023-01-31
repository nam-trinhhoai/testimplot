#include "wellinformationpanelwidget.h"

#include "wellinformation.h"

#include <QComboBox>
#include <QFormLayout>
#include <QLabel>
#include <QListWidget>
#include <QVBoxLayout>

WellInformationPanelWidget::WellInformationPanelWidget(WellInformation* information, QWidget* parent) :
		IInformationPanelWidget(parent), m_information(information) {
	QString wellBore;
	QString wellHead;
	QString defaultTfp;
	QString currentTfpName;
	QString currentTfpPath;
	QString datum;
	QString domain;
	QString elev;
	QString ihs;
	QStringList logs;
	QStringList picks;
	QString status;
	QStringList tfps;
	QStringList tfpPaths;
	QString uwi;
	QString velocity;
	if (m_information!=nullptr) {
		wellBore = m_information->wellBoreName();
		wellHead = m_information->wellHeadName();
		logs = m_information->wellLogs();
		picks = m_information->wellPicks();
		tfps = m_information->wellTfps();
		tfpPaths = m_information->wellTfpPaths();
		currentTfpName = m_information->currentTfpName();
		currentTfpPath = m_information->currentTfpPath();
		defaultTfp = m_information->defaultTfp();

		WellInformation::WellBoreDescParams params = m_information->wellBoreDescParams();
		datum = params.datum;
		domain = params.domain;
		elev = params.elev;
		ihs = params.ihs;
		status = params.status;
		uwi = params.uwi;
		velocity = params.velocity;

		connect(information, &WellInformation::currentTfpChanged, this,
				&WellInformationPanelWidget::tfpPathChanged);
	}

	if (defaultTfp.isNull() || defaultTfp.isEmpty()) {
		defaultTfp = "No default";
	}

	QVBoxLayout* infoLayout = new QVBoxLayout;
	setLayout(infoLayout);
	QFormLayout* formLayout = new QFormLayout;
	infoLayout->addLayout(formLayout);

	formLayout->addRow("Well Head", new QLabel(wellHead));
	formLayout->addRow("Well Bore", new QLabel(wellBore));
	formLayout->addRow("Default tfp", new QLabel(defaultTfp));

	m_currentTfpComboBox = new QComboBox;
	m_currentTfpComboBox->addItem("");
	int currentTfpIdx = 0;
	for (long i=0; i<std::min(tfps.size(), tfpPaths.size()); i++) {
		m_currentTfpComboBox->addItem(tfps[i], tfpPaths[i]);
		if (tfpPaths[i].compare(currentTfpPath)==0) {
			currentTfpIdx = i + 1;
		}
	}
	m_currentTfpComboBox->setCurrentIndex(currentTfpIdx);
	formLayout->addRow("Current tfp", m_currentTfpComboBox);

	if (!datum.isNull() && !datum.isEmpty()) {
		formLayout->addRow("Datum", new QLabel(datum));
	}
	if (!domain.isNull() && !domain.isEmpty()) {
		formLayout->addRow("Domain", new QLabel(domain));
	}
	if (!elev.isNull() && !elev.isEmpty()) {
		formLayout->addRow("Elevation", new QLabel(elev));
	}
	if (!ihs.isNull() && !ihs.isEmpty()) {
		formLayout->addRow("IHS", new QLabel(ihs));
	}
	if (!status.isNull() && !status.isEmpty()) {
		formLayout->addRow("Status", new QLabel(status));
	}
	if (!uwi.isNull() && !uwi.isEmpty()) {
		formLayout->addRow("UWI", new QLabel(uwi));
	}
	if (!velocity.isNull() && !velocity.isEmpty()) {
		formLayout->addRow("Velocity", new QLabel(velocity));
	}

	infoLayout->addWidget(new QLabel("Logs"));
	QListWidget* logListWidget = new QListWidget;
	logListWidget->setStyleSheet("QListWidget{min-height: 3em}");
	logListWidget->setSortingEnabled(true);
	for (int i=0; i<logs.size(); i++) {
		logListWidget->addItem(logs[i]);
	}
	infoLayout->addWidget(logListWidget);

	infoLayout->addWidget(new QLabel("TD laws"));
	QListWidget* tfpListWidget = new QListWidget;
	tfpListWidget->setStyleSheet("QListWidget{min-height: 3em}");
	tfpListWidget->setSortingEnabled(true);
	for (int i=0; i<tfps.size(); i++) {
		tfpListWidget->addItem(tfps[i]);
	}
	infoLayout->addWidget(tfpListWidget);

	infoLayout->addWidget(new QLabel("Picks"));
	QListWidget* pickListWidget = new QListWidget;
	pickListWidget->setStyleSheet("QListWidget{min-height: 3em}");
	pickListWidget->setSortingEnabled(true);
	for (int i=0; i<picks.size(); i++) {
		pickListWidget->addItem(picks[i]);
	}
	infoLayout->addWidget(pickListWidget);
}

WellInformationPanelWidget::~WellInformationPanelWidget() {

}

void WellInformationPanelWidget::tfpPathChanged(QString path) {
	int currentTfpIdx = 0;
	for (long i=0; i<m_currentTfpComboBox->count(); i++) {
		if (m_currentTfpComboBox->itemData(i).toString().compare(path)==0) {
			currentTfpIdx = i;
		}
	}
	m_currentTfpComboBox->setCurrentIndex(currentTfpIdx);
}

bool WellInformationPanelWidget::saveChanges() {
	if (m_information.isNull()) {
		return false;
	}
	QVariant var = m_currentTfpComboBox->currentData();
	QString tfpPath = var.toString();
	QString tfpName = m_currentTfpComboBox->currentText();

	return m_information->setCurrentTfp(tfpPath, tfpName);
}
