#include "predictionwidget.h"
#include "resourcesselectorwidget.h"
#include "surveyselectiondialog.h"
#include "sshfingerprint.h"
#include "bnnijsondecoder.h"
#include "bnniubjsondecoder.h"
#include "globalconfig.h"
#include "Xt.h"
#include "listselectiondialog.h"
#include "sshhostkey.h"
#include "kerberosauthentification.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QListWidget>
#include <QListWidgetItem>
#include <QFileInfo>
#include <QDir>
#include <QTemporaryFile>
#include <QDebug>
#include <QMessageBox>
#include <QSettings>
#include <QSizeGrip>

const QLatin1String LAST_PROJECT_PATH_IN_SETTINGS("BnniMainWindow/lastProjectPath");
const QLatin1String LAST_TRAININGSET_PATH_IN_SETTINGS("BnniMainWindow/lastTrainingSetPath");
const QLatin1String LAST_CONFIG_PATH_IN_SETTINGS("BnniMainWindow/lastExperimentPath");

PredictionWidget::PredictionWidget(QWidget* parent, Qt::WindowFlags f) :
		QWidget(parent, f), m_volumeBounds(0, 0, 0, 0), m_generationBounds(0, 0, 0, 0) {
	setAttribute(Qt::WA_DeleteOnClose);
	setWindowTitle("BNNI Prediction");

	QHBoxLayout* mainLayout = new QHBoxLayout;
	mainLayout->setContentsMargins(0, 0, 0, 0);
	setLayout(mainLayout);

	QWidget* dataHolder = new QWidget;
	QVBoxLayout* dataLayout = new QVBoxLayout;
	dataHolder->setLayout(dataLayout);
	mainLayout->addWidget(dataHolder);

	QHBoxLayout* projectLayout = new QHBoxLayout;
	dataLayout->addLayout(projectLayout);
	QPushButton* openProjectButton = new QPushButton("Open Project");
	projectLayout->addWidget(openProjectButton);
	m_projectLabel = new QLabel;
	projectLayout->addWidget(m_projectLabel);

	dataLayout->addWidget(new QLabel("Training Set : "));

	m_trainingSetListWidget = new QListWidget;
	m_trainingSetListWidget->setSelectionMode(QAbstractItemView::SingleSelection);
	dataLayout->addWidget(m_trainingSetListWidget);

	dataLayout->addWidget(new QLabel("Experiment : "));

	m_configListWidget = new QListWidget;
	m_configListWidget->setSelectionMode(QAbstractItemView::SingleSelection);
	dataLayout->addWidget(m_configListWidget);

	dataLayout->addWidget(new QLabel("Check Point : "));
	m_checkPointListWidget = new QListWidget;
	m_checkPointListWidget->setSelectionMode(QAbstractItemView::SingleSelection);
	dataLayout->addWidget(m_checkPointListWidget);

	QVBoxLayout* mainSizeGripLayout = new QVBoxLayout;
	mainSizeGripLayout->setContentsMargins(0, 0, 0, 0);
	mainSizeGripLayout->setSpacing(0);
	mainLayout->addLayout(mainSizeGripLayout);

	QWidget* predictionHolder = new QWidget;
	QVBoxLayout* predictionLayout = new QVBoxLayout;
	predictionHolder->setLayout(predictionLayout);
	mainSizeGripLayout->addWidget(predictionHolder);

	QSizeGrip* sizeGrip = new QSizeGrip(this);
	sizeGrip->setContentsMargins(0, 0, 0, 0);
	mainSizeGripLayout->addWidget(sizeGrip, 0, Qt::AlignRight);

	m_resourcesSelector = new ResourcesSelectorWidget;
	predictionLayout->addWidget(m_resourcesSelector);

	QHBoxLayout* yLimitLayout = new QHBoxLayout;
	yLimitLayout->addWidget(new QLabel("XLine first index"));
	m_yMinSpinBox = new QSpinBox;
	m_yMinSpinBox->setMinimum(m_volumeBounds.left());
	m_yMinSpinBox->setMaximum(m_volumeBounds.right());
	m_yMinSpinBox->setValue(m_volumeBounds.left());
	yLimitLayout->addWidget(m_yMinSpinBox);
	yLimitLayout->addWidget(new QLabel("XLine last index"));
	m_yMaxSpinBox = new QSpinBox;
	m_yMaxSpinBox->setMinimum(m_volumeBounds.left());
	m_yMaxSpinBox->setMaximum(m_volumeBounds.right());
	m_yMaxSpinBox->setValue(m_volumeBounds.right());
	yLimitLayout->addWidget(m_yMaxSpinBox);
	predictionLayout->addLayout(yLimitLayout);

	QHBoxLayout* zLimitLayout = new QHBoxLayout;
	zLimitLayout->addWidget(new QLabel("Inline first index"));
	m_zMinSpinBox = new QSpinBox;
	m_zMinSpinBox->setMinimum(m_volumeBounds.top());
	m_zMinSpinBox->setMaximum(m_volumeBounds.bottom());
	m_zMinSpinBox->setValue(m_volumeBounds.top());
	zLimitLayout->addWidget(m_zMinSpinBox);
	zLimitLayout->addWidget(new QLabel("Inline last index"));
	m_zMaxSpinBox = new QSpinBox;
	m_zMaxSpinBox->setMinimum(m_volumeBounds.top());
	m_zMaxSpinBox->setMaximum(m_volumeBounds.bottom());
	m_zMaxSpinBox->setValue(m_volumeBounds.bottom());
	zLimitLayout->addWidget(m_zMaxSpinBox);
	predictionLayout->addLayout(zLimitLayout);

	QHBoxLayout* suffixLayout = new QHBoxLayout;
	suffixLayout->addWidget(new QLabel("Suffix :"));
	m_suffixLineEdit = new QLineEdit(m_generalizeSuffix);
	suffixLayout->addWidget(m_suffixLineEdit);
	predictionLayout->addLayout(suffixLayout);

	QPushButton* predictButton = new QPushButton("Generate Volume");
	predictionLayout->addWidget(predictButton);

	// init process
    m_process = new QProcess(this);
    m_process->setProcessChannelMode(QProcess::ForwardedChannels);

	// data
	connect(openProjectButton, &QPushButton::clicked, this, &PredictionWidget::openProject);
	connect(m_trainingSetListWidget, &QListWidget::itemSelectionChanged, this, &PredictionWidget::trainingSetChanged);
	connect(m_configListWidget, &QListWidget::itemSelectionChanged, this, &PredictionWidget::configChanged);
	connect(m_checkPointListWidget, &QListWidget::itemSelectionChanged, this, &PredictionWidget::checkPointChanged);

	// prediction
	connect(m_yMinSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &PredictionWidget::setYMin);
	connect(m_yMaxSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &PredictionWidget::setYMax);
	connect(m_zMinSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &PredictionWidget::setZMin);
	connect(m_zMaxSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &PredictionWidget::setZMax);
	connect(m_suffixLineEdit, &QLineEdit::editingFinished, this, &PredictionWidget::generalizeSuffixChanged);

	connect(predictButton, &QPushButton::clicked, this, &PredictionWidget::run);

	connect(m_process, &QProcess::stateChanged, this, &PredictionWidget::resetTemporaryFile);

	loadSettings();
}

PredictionWidget::~PredictionWidget() {

}

QRect PredictionWidget::volumeBounds() const {
	return m_volumeBounds;
}

QRect PredictionWidget::generationBounds() const {
	return m_generationBounds;
}

void PredictionWidget::run() {
	qDebug() << "run";

	if (m_checkPointFilePath.isNull() || m_checkPointFilePath.isEmpty()) {
		QMessageBox::information(this, "Failed to run process", "No checkpoint to generalize, please choose a checkpoint before running.");
		return;
	}

	if (m_process->state()!=QProcess::NotRunning) {
		qWarning() << "Cannot launch a new generalization process an old one is running";
		return;
	}

	const std::vector<ComputerResource*>& resources = m_resourcesSelector->getSelectedResources();
	if (resources.size()==0) {
		qWarning() << "No computer to use for generalization";
		return;
	}

	// cleanup just in case, may create issues with threading
	if (!m_processTemporaryFile.isEmpty()) {
		resetTemporaryFile(m_process->state());
	}

	KerberosAuthentification authentifiator;
	if (!authentifiator.isAuthentificated() && !authentifiator.authentificate(this)) {
		return;
	}

	QStringList readytoUseHosts, toApproveHosts, toApproveTexts;
	std::vector<SshFingerprintAndKey> toApproveFingerprint;
	for (int i=0; i<resources.size(); i++) {
		const ComputerResource* resource = resources[i];
		std::vector<SshFingerprintAndKey> sshfingerPrints = resource->getSshFingerprint();
		for (int j=0; j<sshfingerPrints.size(); j++) {
			qDebug() << resource->hostName() << (*sshfingerPrints[j].fingerprint.get());
		}

		bool hostValid = sshfingerPrints.size()>0;
		bool fingerprintMatch = false;
		if (hostValid) {
			// search in known host
			SshFingerprintAndKey knowHostFingerprint = SshFingerprint::getFingerprintFromKnownHosts(resource->hostName());
			if (knowHostFingerprint.fingerprint!=nullptr) {
				int fingerprintIdx = 0;
				while (!fingerprintMatch && fingerprintIdx<sshfingerPrints.size()) {
					fingerprintMatch = *knowHostFingerprint.fingerprint==*sshfingerPrints[fingerprintIdx].fingerprint;
					fingerprintIdx++;
				}
			}
		}

		bool fingerprintMatchCustom = false;
		if (hostValid && !fingerprintMatch) {
			// search in custom known host file
			GlobalConfig& config = GlobalConfig::getConfig();

			SshFingerprintAndKey knowHostFingerprint = SshFingerprint::getFingerprintFromCustomKnownHosts(resource->hostName(),
					config.customKnownHostsFiles());
			if (knowHostFingerprint.fingerprint!=nullptr) {
				int fingerprintIdx = 0;
				while (!fingerprintMatchCustom && fingerprintIdx<sshfingerPrints.size()) {
					fingerprintMatchCustom = *knowHostFingerprint.fingerprint==*sshfingerPrints[fingerprintIdx].fingerprint;
					fingerprintIdx++;
				}
			}
		}

		if (hostValid && !fingerprintMatch && !fingerprintMatchCustom) {
			// need user approval
			toApproveHosts.append(resource->hostName());
			toApproveTexts.append(resource->hostName() + " " + QString(sshfingerPrints[0].fingerprint->hash()));
			toApproveFingerprint.push_back(sshfingerPrints[0]);
		} else if (hostValid && (fingerprintMatch ||  fingerprintMatchCustom)) {
			// ready to use hosts
			readytoUseHosts.append(resource->hostName());
		}
	}

	if (toApproveTexts.size()>0) {
		ListSelectionDialog selectionDialog(toApproveTexts, "Unknown hosts, do you approve them ?");
		int res = selectionDialog.exec();
		if (res==QDialog::Accepted) {
			// items match toApproveTexts size and order.
			const std::vector<ListSelectionDialog::SelectionItem>& items = selectionDialog.getList();
			for (int i=0; i<items.size(); i++) {
				if (items[i].isSelected) {
					toApproveFingerprint[i].hostKey->addToKnownHosts();
					readytoUseHosts.append(toApproveHosts[i]);
				}
			}
		}
	}

	QTemporaryFile hostFile;
	hostFile.setAutoRemove(false);
	bool hostFileValid = hostFile.open();
	if (hostFileValid) {
		QTextStream textStream(&hostFile);
		for (int hostIdx=0; hostIdx<readytoUseHosts.size(); hostIdx++) {
			const QString& host = readytoUseHosts[hostIdx];
			textStream << host << Qt::endl;
		}
	}
	hostFile.close();
	m_processTemporaryFile = hostFile.fileName();


	GlobalConfig& config = GlobalConfig::getConfig();

	QDir programLocationDir(config.bnniProgramLocation());

	QString program = programLocationDir.absoluteFilePath("spliter_z.sh");
	QStringList arguments;
	arguments << QString::number(m_generationBounds.top()) << QString::number(m_generationBounds.bottom()) <<
				QString::number(m_generationBounds.left()) << QString::number(m_generationBounds.right());
	arguments << programLocationDir.absoluteFilePath("launchWithEnv_predict.sh");
	arguments << "--config" << m_configFilePath << "--restore" << m_checkPointFilePath << "--work_dir" << "" << "--xmin" <<
				QString::number(m_xmin) << "--xmax" << QString::number(m_xmax) <<
				"--generalizationsuffix" << m_generalizeSuffix;
	qDebug() << "Program : " << program;
	qDebug() << "Options : " << arguments;
	m_process->setWorkingDirectory(programLocationDir.absolutePath());

	// set environment
	QProcessEnvironment env = m_process->processEnvironment();
	env.insert("HOSTFILE", m_processTemporaryFile);
	env.insert("NV_GLOBAL_KNOWN_HOSTS_FILE", config.customKnownHostsFiles().join(" "));
	m_process->setProcessEnvironment(env);

	m_process->start(program, arguments);
}

void PredictionWidget::setYMin(int val) {
	m_generationBounds.setLeft(val);
	m_yMaxSpinBox->setMinimum(val);
}

void PredictionWidget::setYMax(int val) {
	m_generationBounds.setRight(val);
	m_yMinSpinBox->setMaximum(val);
}

void PredictionWidget::setZMin(int val) {
	m_generationBounds.setTop(val);
	m_zMaxSpinBox->setMinimum(val);
}

void PredictionWidget::setZMax(int val) {
	m_generationBounds.setBottom(val);
	m_zMinSpinBox->setMaximum(val);
}

void PredictionWidget::generalizeSuffixChanged() {
	m_generalizeSuffix = m_suffixLineEdit->text();
}

void PredictionWidget::updateVolumeBounds() {
	// y
	m_yMinSpinBox->setMinimum(m_volumeBounds.left());
	m_yMinSpinBox->setMaximum(m_volumeBounds.right());
	m_yMaxSpinBox->setMinimum(m_volumeBounds.left());
	m_yMaxSpinBox->setMaximum(m_volumeBounds.right());
	m_yMinSpinBox->setValue(m_volumeBounds.left());
	m_yMaxSpinBox->setValue(m_volumeBounds.right());

	// z
	m_zMinSpinBox->setMinimum(m_volumeBounds.top());
	m_zMinSpinBox->setMaximum(m_volumeBounds.bottom());
	m_zMaxSpinBox->setMinimum(m_volumeBounds.top());
	m_zMaxSpinBox->setMaximum(m_volumeBounds.bottom());
	m_zMinSpinBox->setValue(m_volumeBounds.top());
	m_zMaxSpinBox->setValue(m_volumeBounds.bottom());
}

void PredictionWidget::openProject() {
    SurveySelectionDialog dialog(this);
    int code = dialog.exec();
    QString _project = dialog.getProject();
    QString _projectDir = dialog.getDirProject();

    if (!_project.isNull() && !_project.isEmpty()) {
        setProject(_projectDir + "/" + _project);
    }
}

void PredictionWidget::setProject(const QString& projectDirPath) {
	clearTrainingSetListWidget();

	m_projectDirPath = projectDirPath;
    QString projectName = QDir(m_projectDirPath).dirName();
    m_projectLabel->setText(projectName);

    QDir trainingsetDir(projectDirPath+"/DATA/NEURONS/neurons2/LogInversion2Problem3");
    QFileInfoList dirs = trainingsetDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);

    for (const QFileInfo& fileInfo : dirs) {
    	QListWidgetItem* item = new QListWidgetItem(fileInfo.baseName());
    	item->setData(Qt::UserRole, fileInfo.absoluteFilePath());
    	m_trainingSetListWidget->addItem(item);
    }

    QSettings settings;
    settings.setValue(LAST_PROJECT_PATH_IN_SETTINGS, projectDirPath);
}

void PredictionWidget::trainingSetChanged() {
	clearConfigListWidget();

	QList<QListWidgetItem*> selection = m_trainingSetListWidget->selectedItems();
	if (selection.count()==0) {
		m_trainingSetDirPath = "";
		m_volumeBounds = QRect(0, 0, 0, 0);
	} else {
		m_trainingSetDirPath = selection[0]->data(Qt::UserRole).toString();

		QDir configDir(m_trainingSetDirPath);
		QFileInfoList dirs = configDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
	    for (const QFileInfo& fileInfo : dirs) {
	    	QListWidgetItem* item = new QListWidgetItem(fileInfo.baseName());
	    	item->setData(Qt::UserRole, fileInfo.absoluteFilePath());
	    	m_configListWidget->addItem(item);
	    }

	    std::vector<QString> seismics;
	    if (QDir(m_trainingSetDirPath).exists("trainingset.ubjson")) {
	    	seismics = BnniUbjsonDecoder::ubjsonExtractSeismics(m_trainingSetDirPath+"/trainingset.ubjson", m_projectDirPath);
	    } else {
	    	seismics = BnniJsonDecoder::jsonExtractSeismics(m_trainingSetDirPath+"/trainingset.json", m_projectDirPath);
	    }
	    QRect newRect(0, 0, 0, 0);
	    bool valid = seismics.size()>0;
	    if (valid) {
	    	inri::Xt xt(seismics[0].toStdString().c_str());
	    	valid = xt.is_valid();
	    	if (valid) {
	    		newRect = QRect(0, 0, xt.nRecords(), xt.nSlices());
	    		m_xmin = 0;
	    		m_xmax = xt.nSamples();
	    	}
		}
	    m_volumeBounds = newRect;

	    QSettings settings;
	    settings.setValue(LAST_TRAININGSET_PATH_IN_SETTINGS, m_trainingSetDirPath+"/trainingset.json");
	}
	updateVolumeBounds();
}

void PredictionWidget::configChanged() {
	clearCheckPointListWidget();

	QList<QListWidgetItem*> selection = m_configListWidget->selectedItems();

	if (selection.count()==0) {
		m_configFilePath = "";
	} else {
		QDir configDir(selection[0]->data(Qt::UserRole).toString());
		m_configFilePath = configDir.absoluteFilePath("config.txt");

		QFileInfoList files = configDir.entryInfoList(QStringList() << "*.index", QDir::Files | QDir::NoDotAndDotDot, QDir::Time | QDir::Reversed);
	    for (const QFileInfo& fileInfo : files) {
	    	QListWidgetItem* item = new QListWidgetItem(fileInfo.baseName());
	    	item->setData(Qt::UserRole, configDir.absoluteFilePath(fileInfo.completeBaseName()));
	    	m_checkPointListWidget->addItem(item);
	    }

	    QSettings settings;
	    settings.setValue(LAST_CONFIG_PATH_IN_SETTINGS, m_configFilePath);
	}
}

void PredictionWidget::checkPointChanged() {
	QList<QListWidgetItem*> selection = m_checkPointListWidget->selectedItems();

	if (selection.count()==0) {
		m_checkPointFilePath = "";
	} else {
		m_checkPointFilePath = selection[0]->data(Qt::UserRole).toString();
	}
}

void PredictionWidget::clearTrainingSetListWidget() {
	clearConfigListWidget();
	m_trainingSetListWidget->clear();
	m_trainingSetDirPath = "";
	QSettings settings;
	settings.remove(LAST_TRAININGSET_PATH_IN_SETTINGS);

	m_volumeBounds = QRect(0, 0, 0, 0);
	updateVolumeBounds();
}

void PredictionWidget::clearConfigListWidget() {
	clearCheckPointListWidget();
	m_configListWidget->clear();
	m_configFilePath = "";
	QSettings settings;
	settings.remove(LAST_CONFIG_PATH_IN_SETTINGS);
}

void PredictionWidget::clearCheckPointListWidget() {
	m_checkPointListWidget->clear();
	m_checkPointFilePath = "";
}

void PredictionWidget::resetTemporaryFile(QProcess::ProcessState state) {
	if (state==QProcess::NotRunning && !m_processTemporaryFile.isNull() &&
			!m_processTemporaryFile.isEmpty()) {
		if (QFile::exists(m_processTemporaryFile)) {
			QFile::remove(m_processTemporaryFile);
		}
		m_processTemporaryFile = "";
	}
}

void PredictionWidget::loadSettings() {
	// load them all first because they are unset during the loading
	QSettings settings;
	QString projectPath = settings.value(LAST_PROJECT_PATH_IN_SETTINGS, "").toString();
	QString trainingSetPath = settings.value(LAST_TRAININGSET_PATH_IN_SETTINGS, "").toString();
	QString configPath = settings.value(LAST_CONFIG_PATH_IN_SETTINGS, "").toString();

	if (!projectPath.isNull() && !projectPath.isEmpty()) {
		setProject(projectPath);
	}
	if (!trainingSetPath.isNull() && !trainingSetPath.isEmpty()) {
		// search the training set with the right name
		QString trainingSetName = QFileInfo(trainingSetPath).dir().dirName();

		int i=0;
		bool notFound = true;
		while (notFound && i<m_trainingSetListWidget->count()) {
			notFound = m_trainingSetListWidget->item(i)->text().compare(trainingSetName)!=0;

			if (notFound) {
				i++;
			}
		}
		if (!notFound) {
			m_trainingSetListWidget->item(i)->setSelected(true);
		}
	}
	if (!configPath.isNull() && !configPath.isEmpty()) {
		// search the training set with the right name
		QString configName = QFileInfo(configPath).dir().dirName();

		int i=0;
		bool notFound = true;
		while (notFound && i<m_configListWidget->count()) {
			notFound = m_configListWidget->item(i)->text().compare(configName)!=0;

			if (notFound) {
				i++;
			}
		}
		if (!notFound) {
			m_configListWidget->item(i)->setSelected(true);
		}
	}
}
