#include "bnniconfigcreationdialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>


BnniConfigCreationDialog::BnniConfigCreationDialog(const QStringList& existingConfigs,
		const QStringList& checkpoints, QWidget* parent, Qt::WindowFlags f) : QDialog(parent, f),
		m_existingConfigs(existingConfigs), m_checkpoints(checkpoints) {
	setWindowTitle("Create new configuration");

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QHBoxLayout* configNameLayout = new QHBoxLayout;
	mainLayout->addLayout(configNameLayout);
	configNameLayout->addWidget(new QLabel("New configuration name : "));
	m_nameLineEdit = new QLineEdit;
	configNameLayout->addWidget(m_nameLineEdit);

	QHBoxLayout* checkpointLayout = new QHBoxLayout;
	mainLayout->addLayout(checkpointLayout);
	m_useCheckpointsCheckBox = new QCheckBox("From checkpoint : ");
	m_useCheckpointsCheckBox->setCheckState(m_useCheckpoints ? Qt::Checked : Qt::Unchecked);
	checkpointLayout->addWidget(m_useCheckpointsCheckBox);

	m_checkpointsComboBox = new QComboBox;
	m_checkpointsComboBox->addItem("", -1);
	for (int i=0; i<m_checkpoints.size(); i++) {
		const QString& checkPoint = m_checkpoints[i];
		m_checkpointsComboBox->addItem(checkPoint, i);
	}
	checkpointLayout->addWidget(m_checkpointsComboBox);
	if (m_useCheckpoints) {
		m_checkpointsComboBox->show();
	} else {
		m_checkpointsComboBox->hide();
	}

	QHBoxLayout* buttonsLayout = new QHBoxLayout;
	mainLayout->addLayout(buttonsLayout);
	buttonsLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));
	m_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal);
	buttonsLayout->addWidget(m_buttonBox);

	connect(m_buttonBox, &QDialogButtonBox::accepted, this, &BnniConfigCreationDialog::accept);
	connect(m_buttonBox, &QDialogButtonBox::rejected, this, &BnniConfigCreationDialog::reject);
	connect(m_useCheckpointsCheckBox, &QCheckBox::stateChanged, this, &BnniConfigCreationDialog::useCheckpointsChanged);
	connect(m_checkpointsComboBox, &QComboBox::currentIndexChanged, this, &BnniConfigCreationDialog::checkpointChanged);
	connect(m_nameLineEdit, &QLineEdit::textChanged, this, &BnniConfigCreationDialog::nameChanged);
}

BnniConfigCreationDialog::~BnniConfigCreationDialog() {

}

bool BnniConfigCreationDialog::useCheckpoints() const {
	return m_useCheckpoints;
}

void BnniConfigCreationDialog::toggleUseCheckpoints(bool val) {
	if (m_useCheckpoints!=val) {
		m_useCheckpoints = val;

		if (m_useCheckpoints) {
			m_checkpointsComboBox->show();
		} else {
			m_checkpointsComboBox->hide();
		}

		QSignalBlocker b(m_useCheckpointsCheckBox);
		m_useCheckpointsCheckBox->setCheckState(m_useCheckpoints ? Qt::Checked : Qt::Unchecked);
	}
}

QString BnniConfigCreationDialog::checkpoint() const {
	QString txt;
	if (checkpointValid()) {
		txt = m_checkpoints[m_currentCheckpoint];
	}
	return txt;
}

bool BnniConfigCreationDialog::checkpointValid() const {
	return m_currentCheckpoint>=0 && m_currentCheckpoint<m_checkpoints.size();
}

QString BnniConfigCreationDialog::newName() const {
	return m_newName;
}

void BnniConfigCreationDialog::checkpointChanged(int val) {
	bool ok;
	int index = m_checkpointsComboBox->itemData(val).toInt(&ok);
	if (!ok) {
		index = -1;
	}
	m_currentCheckpoint = index;
}

void BnniConfigCreationDialog::nameChanged() {
	m_newName = m_nameLineEdit->text();

	if (m_existingConfigs.contains(m_newName)) {
		m_nameLineEdit->setToolTip("Config already exists");
		m_nameLineEdit->setStyleSheet("QLineEdit {background-color: red;}");
		m_buttonBox->setEnabled(false);
	} else {
		m_nameLineEdit->setToolTip("");
		m_nameLineEdit->setStyleSheet("");
		m_buttonBox->setEnabled(true);
	}
}

void BnniConfigCreationDialog::useCheckpointsChanged(int val) {
	toggleUseCheckpoints(val==Qt::Checked);
}

