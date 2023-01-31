#include "trainingsetparameterwidget.h"

#include "BNNIJsonGenerator.h"
#include "fileSelectorDialog.h"
#include "ProjectManagerWidget.h"
#include "seismic3dabstractdataset.h"
#include "imageformats.h"
#include "sampletypebinder.h"
#include "customchartview.h"
#include "globalconfig.h"
#include "mtlengthunit.h"
#include "resamplespline.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSizeGrip>
#include <QSpinBox>
#include <QTabWidget>
#include <QScrollArea>
#include <QCheckBox>
#include <QTreeWidget>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QMenu>
#include <QCursor>

#include <QLineSeries>
#include <QChart>
#include <QChartView>
#include <QMessageBox>

#include <limits>
#include <fftw3.h>

TrainingSetParameterWidget::TrainingSetParameterWidget(QWidget *parent, Qt::WindowFlags f) :
		QWidget(parent, f), m_trainingSet(this) {
	setAttribute(Qt::WA_DeleteOnClose);
	setWindowTitle("Training Set Generator");

	m_depthUnit = &MtLengthUnit::METRE;

	QHBoxLayout* mainLayout = new QHBoxLayout;
	mainLayout->setContentsMargins(0, 0, 0, 0);
	setLayout(mainLayout);

	// data ui
	QVBoxLayout* dataLayout = new QVBoxLayout;
	mainLayout->addLayout(dataLayout);

	m_projectManagerWidget = new ProjectManagerWidget();
	std::vector<ProjectManagerWidget::ManagerTabName> tabNames;
	tabNames.push_back(ProjectManagerWidget::ManagerTabName::SEISMIC);
	tabNames.push_back(ProjectManagerWidget::ManagerTabName::WELL);
	tabNames.push_back(ProjectManagerWidget::ManagerTabName::HORIZON);
	m_projectManagerWidget->onlyShow(tabNames);
	dataLayout->addWidget(m_projectManagerWidget);

	// bnni params
	QVBoxLayout* mainSizeGripLayout = new QVBoxLayout;
	mainSizeGripLayout->setContentsMargins(0, 0, 0, 0);
	mainSizeGripLayout->setSpacing(0);
	mainLayout->addLayout(mainSizeGripLayout);

	QWidget* bnniHolder = new QWidget;
	QVBoxLayout* bnniLayout = new QVBoxLayout;
	bnniHolder->setLayout(bnniLayout);
	mainSizeGripLayout->addWidget(bnniHolder);

	// depth unit
	QHBoxLayout* unitLayout = new QHBoxLayout;
	bnniLayout->addLayout(unitLayout);

	QLabel* unitLabel = new QLabel("Depth Unit : ");
	unitLayout->addWidget(unitLabel);
	m_depthUnitButton = new QPushButton("Meter");
	m_depthUnitButton->setIcon(QIcon(":/slicer/icons/regle_m128_blanc.png"));
	connect(m_depthUnitButton, &QPushButton::clicked, this, &TrainingSetParameterWidget::toggleDepthUnit);
	unitLayout->addWidget(m_depthUnitButton);

	m_logsLayout = new QGridLayout;
	QWidget* logsHolder = new QWidget;
	logsHolder->setLayout(m_logsLayout);
	QScrollArea* logsScrollArea = new QScrollArea;
	logsScrollArea->setWidget(logsHolder);
	logsScrollArea->setWidgetResizable(true);
	bnniLayout->addWidget(logsScrollArea);
	//bnniLayout->addLayout(m_logsLayout);

	QPushButton* addMoreWellsButton = new QPushButton("Add more wells");
	bnniLayout->addWidget(addMoreWellsButton);

	QFormLayout* logsParamsForm = new QFormLayout;
	bnniLayout->addLayout(logsParamsForm);

	m_bandpassCheckBox = new QCheckBox;
	m_bandpassCheckBox->setCheckState(m_trainingSet.useBandPassHighFrequency() ? Qt::Checked : Qt::Unchecked);
	logsParamsForm->addRow("Use band pass filter", m_bandpassCheckBox);

	m_bandPassFreqLabel = new QLabel("Band pass frequency");
	m_bandPassFreqSpinBox = new QDoubleSpinBox;
	m_bandPassFreqSpinBox->setMinimum(std::numeric_limits<double>::min());
	m_bandPassFreqSpinBox->setMaximum(std::numeric_limits<double>::max());
	m_bandPassFreqSpinBox->setDecimals(5);
	m_bandPassFreqSpinBox->setValue(m_trainingSet.bandPassHighFrequency());

	logsParamsForm->addRow(m_bandPassFreqLabel, m_bandPassFreqSpinBox);
	if (!m_trainingSet.useBandPassHighFrequency()) {
		m_bandPassFreqLabel->hide();
		m_bandPassFreqSpinBox->hide();
	}

	QDoubleSpinBox* mdSampleRateSpinBox = new QDoubleSpinBox;
	mdSampleRateSpinBox->setMinimum(std::numeric_limits<double>::min());
	mdSampleRateSpinBox->setMaximum(std::numeric_limits<double>::max());
	mdSampleRateSpinBox->setDecimals(5);
	mdSampleRateSpinBox->setValue(m_trainingSet.mdSamplingRate());
	logsParamsForm->addRow("Md sampling rate", mdSampleRateSpinBox);

	m_augmentationCheckBox = new QCheckBox;
	m_augmentationCheckBox->setCheckState(m_trainingSet.useAugmentation() ? Qt::Checked : Qt::Unchecked);
	logsParamsForm->addRow("Use well augmentation", m_augmentationCheckBox);

	m_augmentationDistanceLabel = new QLabel("Distance of new wells from origin well");
	m_augmentationDistanceSpinBox = new QSpinBox;
	m_augmentationDistanceSpinBox->setMinimum(1);
	m_augmentationDistanceSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_augmentationDistanceSpinBox->setValue(m_trainingSet.augmentationDistance());
	logsParamsForm->addRow(m_augmentationDistanceLabel, m_augmentationDistanceSpinBox);

	m_augmentationNoiseStdLabel = new QLabel("Gaussian noise standard deviation");
	m_augmentationNoiseStdSpinBox = new QDoubleSpinBox;
	m_augmentationNoiseStdSpinBox->setMinimum(0);
	m_augmentationNoiseStdSpinBox->setMaximum(std::numeric_limits<double>::max());
	m_augmentationNoiseStdSpinBox->setDecimals(5);
	m_augmentationNoiseStdSpinBox->setValue(m_trainingSet.gaussianNoiseStd());

	logsParamsForm->addRow(m_augmentationNoiseStdLabel, m_augmentationNoiseStdSpinBox);

	m_useCnxAugmentationCheckBox = new QCheckBox;
	m_useCnxAugmentationCheckBox->setCheckState(m_trainingSet.useCnxAugmentation() ? Qt::Checked : Qt::Unchecked);
	m_useCnxAugmentationLabel = new QLabel("Use CNX augmentation");
	logsParamsForm->addRow(m_useCnxAugmentationLabel, m_useCnxAugmentationCheckBox);
	if (!m_trainingSet.useAugmentation()) {
		m_augmentationDistanceLabel->hide();
		m_augmentationDistanceSpinBox->hide();
		m_augmentationNoiseStdLabel->hide();
		m_augmentationNoiseStdSpinBox->hide();
		m_useCnxAugmentationLabel->hide();
		m_useCnxAugmentationCheckBox->hide();
	}

	QWidget* seismicHolder = new QWidget;
	m_seismicsLayout = new QVBoxLayout;
	seismicHolder->setLayout(m_seismicsLayout);
	QScrollArea* seismicScrollArea = new QScrollArea;
	seismicScrollArea->setWidget(seismicHolder);
	seismicScrollArea->setWidgetResizable(true);
	bnniLayout->addWidget(seismicScrollArea);

	// horizons
	m_horizonLayout = new QHBoxLayout();
	m_topHorizonLabel = new QLabel();
	m_horizonLayout->addWidget(m_topHorizonLabel, 1);

	QPushButton* selectTopHorizonButton = new QPushButton("<");
	selectTopHorizonButton->setMinimumWidth(10);
	connect(selectTopHorizonButton, &QPushButton::clicked, this, &TrainingSetParameterWidget::selectTopHorizon);
	m_horizonLayout->addWidget(selectTopHorizonButton, 0);

	m_topDeltaSpinBox = new QDoubleSpinBox();
	m_topDeltaSpinBox->setMinimum(std::numeric_limits<float>::lowest());
	m_topDeltaSpinBox->setMaximum(std::numeric_limits<float>::max());
	m_topDeltaSpinBox->setValue(0.0);
	connect(m_topDeltaSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TrainingSetParameterWidget::changeTopDelta);
	m_horizonLayout->addWidget(m_topDeltaSpinBox, 0);

	m_bottomHorizonLabel = new QLabel();
	m_horizonLayout->addWidget(m_bottomHorizonLabel, 1);

	QPushButton* selectBottomHorizonButton = new QPushButton("<");
	selectBottomHorizonButton->setMinimumWidth(10);
	connect(selectBottomHorizonButton, &QPushButton::clicked, this, &TrainingSetParameterWidget::selectBottomHorizon);
	m_horizonLayout->addWidget(selectBottomHorizonButton, 0);

	m_bottomDeltaSpinBox = new QDoubleSpinBox();
	m_bottomDeltaSpinBox->setMinimum(std::numeric_limits<float>::lowest());
	m_bottomDeltaSpinBox->setMaximum(std::numeric_limits<float>::max());
	m_bottomDeltaSpinBox->setValue(0.0);
	connect(m_bottomDeltaSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TrainingSetParameterWidget::changeBottomDelta);
	m_horizonLayout->addWidget(m_bottomDeltaSpinBox, 0);

	QWidget* horizonHolderWidget = new QWidget;
	horizonHolderWidget->setContentsMargins(0, 0, 0, 0);
	horizonHolderWidget->setLayout(m_horizonLayout);
	m_horizonHolder = new QScrollArea;
	m_horizonHolder->setWidget(horizonHolderWidget);
	m_horizonHolder->setWidgetResizable(true);
	m_horizonHolder->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	QSizePolicy sizePolicy = m_horizonHolder->sizePolicy();
	sizePolicy.setVerticalPolicy(QSizePolicy::Minimum);
	m_horizonHolder->setSizePolicy(sizePolicy);
	m_horizonHolder->setContentsMargins(0, 0, 0, 0);
	bnniLayout->addWidget(m_horizonHolder, 0);
	m_horizonHolder->hide();

	QHBoxLayout* horizonTestLayout = new QHBoxLayout();
	QLabel* useHorizonLabel = new QLabel("Use horizons : ");
	horizonTestLayout->addWidget(useHorizonLabel);

	QCheckBox* horizonTestCheckBox = new QCheckBox();
	horizonTestCheckBox->setCheckState(Qt::Unchecked);
	connect(horizonTestCheckBox, &QCheckBox::stateChanged, this, &TrainingSetParameterWidget::toggleHorizonInterval);

	horizonTestLayout->addWidget(horizonTestCheckBox);
	bnniLayout->addLayout(horizonTestLayout);


	QFormLayout* seismicsParamsForm = new  QFormLayout;
	bnniLayout->addLayout(seismicsParamsForm);

	QDoubleSpinBox* sampleRateSpinBox = new QDoubleSpinBox;
	sampleRateSpinBox->setMinimum(std::numeric_limits<float>::min());
	sampleRateSpinBox->setMaximum(std::numeric_limits<float>::max());
	sampleRateSpinBox->setDecimals(5);
	sampleRateSpinBox->setValue(m_trainingSet.targetSampleRate());
	seismicsParamsForm->addRow("Target sampling rate", sampleRateSpinBox);

	QHBoxLayout* halfWindowLayout = new QHBoxLayout;
	QPushButton* autoCorrButton = new QPushButton("Auto Corr");
	QSpinBox* halfWindowSpinBox = new QSpinBox;
	halfWindowSpinBox->setMinimum(0);
	halfWindowSpinBox->setMaximum(std::numeric_limits<int>::max());
	halfWindowSpinBox->setValue(m_trainingSet.halfWindow());
	halfWindowLayout->addWidget(halfWindowSpinBox, 1);
	halfWindowLayout->addWidget(autoCorrButton, 0);

	seismicsParamsForm->addRow("Half Window", halfWindowLayout);

	m_trainingSetLineEdit = new QLineEdit(m_trainingSet.trainingSetName());
	seismicsParamsForm->addRow("Training Set", m_trainingSetLineEdit);

	QPushButton* computeButton = new QPushButton("Generate training set");
	bnniLayout->addWidget(computeButton);

	QSizeGrip* sizeGrip = new QSizeGrip(this);
	sizeGrip->setContentsMargins(0, 0, 0, 0);
	mainSizeGripLayout->addWidget(sizeGrip, 0, Qt::AlignRight);

	connect(m_bandpassCheckBox, &QCheckBox::stateChanged, this, &TrainingSetParameterWidget::setBandPassState);
	connect(m_bandPassFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TrainingSetParameterWidget::setBandPassFreq);
	connect(mdSampleRateSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), &m_trainingSet, &BnniTrainingSet::setMdSamplingRate);
	connect(m_augmentationCheckBox, &QCheckBox::stateChanged, this, &TrainingSetParameterWidget::setAugmentationState);
	connect(m_augmentationDistanceSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &TrainingSetParameterWidget::changeAugmentationDistance);
	connect(m_augmentationNoiseStdSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TrainingSetParameterWidget::changeAugmentationNoiseStd);
	connect(m_useCnxAugmentationCheckBox, &QCheckBox::stateChanged, this, &TrainingSetParameterWidget::changeUseCnxAugmentation);
	connect(sampleRateSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TrainingSetParameterWidget::changeSampleRate);
	connect(halfWindowSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), &m_trainingSet, &BnniTrainingSet::setHalfWindow);
	connect(m_trainingSetLineEdit, &QLineEdit::editingFinished, this, &TrainingSetParameterWidget::setTrainingSetName);
	connect(computeButton, &QPushButton::clicked, this, &TrainingSetParameterWidget::compute);

	connect(m_projectManagerWidget->m_projectManager, &ProjectManager::projectChanged, this, &TrainingSetParameterWidget::projectChanged);

	connect(autoCorrButton, &QPushButton::clicked, this, &TrainingSetParameterWidget::openAutoCorrWindow);

	connect(&m_trainingSet, &BnniTrainingSet::outputJsonFileChanged, this,
			&TrainingSetParameterWidget::setTrainingSetNameFromData);

	connect(&m_trainingSet, &BnniTrainingSet::seismicUnitChanged, this,
				&TrainingSetParameterWidget::trainingSetSeismicUnitChanged);

	connect(addMoreWellsButton, &QPushButton::clicked, this, &TrainingSetParameterWidget::addMoreWells);

	initLogsLayout();
	initSeismicsLayout();

	// use auto loaded session
	QString autoLoadProjectPath = m_projectManagerWidget->getProjectPath();
	if (!autoLoadProjectPath.isNull() && !autoLoadProjectPath.isEmpty()) {
		m_trainingSet.setProjectPath(autoLoadProjectPath);
	}
}

TrainingSetParameterWidget::~TrainingSetParameterWidget() {
	for (int i=0; i<m_displayedAutoCorrWidgets.size(); i++) {
		disconnect(m_autoCorrConns[m_displayedAutoCorrWidgets[i]]);
		m_displayedAutoCorrWidgets[i]->deleteLater();
	}
}

QString TrainingSetParameterWidget::getNameFromSampleUnit(SampleUnit unit) {
	QString out;
	switch (unit) {
	case SampleUnit::DEPTH:
		out = "depth";
		break;
	case SampleUnit::TIME:
		out = "time";
		break;
	default:
		out = "no_unit";
		break;
	}
	return out;
}

void TrainingSetParameterWidget::changeSampleRate(double val) {
	m_trainingSet.setTargetSampleRate(val);

	for (int i=0; i<m_displayedAutoCorrWidgets.size(); i++) {
		m_displayedAutoCorrWidgets[i]->setSampleRate(val);
	}
}

void TrainingSetParameterWidget::compute() {
	BnniJsonGenerator* generator = new BnniJsonGenerator;

	// set params
	generator->setHalfWindow(m_trainingSet.halfWindow());
	generator->setPasSampleSurrechantillon(m_trainingSet.targetSampleRate());
	generator->setMdSamplingRate(m_trainingSet.mdSamplingRate());
	if (m_trainingSet.useBandPassHighFrequency()) {
		generator->activateBandPass(m_trainingSet.bandPassHighFrequency());
	} else {
		generator->deactivateBandPass();
	}
	generator->setUseAugmentation(m_trainingSet.useAugmentation());
	generator->setAugmentationDistance(m_trainingSet.augmentationDistance());
	generator->setGaussianNoiseStd(m_trainingSet.gaussianNoiseStd());
	generator->toggleCnxAugmentation(m_trainingSet.useCnxAugmentation());
	generator->setOutputJsonFile(m_trainingSet.outputJsonFile());

	generator->setDepthUnit(m_depthUnit);

	// set seismics
	std::map<long, BnniTrainingSet::SeismicParameter>::const_iterator seismicIt =
			m_trainingSet.seismics().begin();
	while (seismicIt != m_trainingSet.seismics().end()) {
		const BnniTrainingSet::SeismicParameter& param = seismicIt->second;
		std::pair<float, float> dynamic(param.min, param.max);
		generator->addInputVolume(param.path, dynamic);
		seismicIt++;
	}

	// set logs
	std::map<long, BnniTrainingSet::BnniWellHeader>::const_iterator kindIt =
			m_trainingSet.wellHeaders().begin();

	std::vector<BnniTrainingSet::BnniWellHeader> logsHeaders;
	while (kindIt!=m_trainingSet.wellHeaders().end()) {
		BnniTrainingSet::BnniWellHeader header = kindIt->second;
		logsHeaders.push_back(header);
		kindIt++;
	}
	generator->defineLogsNames(logsHeaders);

	std::map<long, BnniTrainingSet::WellParameter>::const_iterator wellIt =
				m_trainingSet.wellBores().begin();
	while (wellIt!=m_trainingSet.wellBores().end()) {
		const BnniTrainingSet::WellParameter& param = wellIt->second;

		std::map<long, BnniTrainingSet::BnniWellHeader>::const_iterator kindItBis =
					m_trainingSet.wellHeaders().begin();
		std::vector<QString> logPaths, logNames;
		logPaths.reserve(m_trainingSet.wellHeaders().size());
		logNames.reserve(m_trainingSet.wellHeaders().size());
		while (kindItBis!=m_trainingSet.wellHeaders().end()) {
			logPaths.push_back(param.logsPathAndName.at(kindItBis->first).first);
			logNames.push_back(param.logsPathAndName.at(kindItBis->first).second);
			kindItBis++;
		}
		generator->addWellBore(param.boreDescFile, param.deviationFile, param.tfpFile,
				param.tfpName, logPaths, logNames, param.headDescFile);
		wellIt++;
	}

	const std::map<long, BnniTrainingSet::HorizonIntervalParameter>& horizonsIntervals = m_trainingSet.intervals();
	std::map<long, BnniTrainingSet::HorizonIntervalParameter>::const_iterator it = horizonsIntervals.begin();
	while (it!=horizonsIntervals.end()) {
		float topDelta = it->second.topHorizon.delta;
		float bottomDelta = it->second.bottomHorizon.delta;

		if (m_trainingSet.seismicUnit()==SampleUnit::DEPTH) {
			// convert deltas from display unit (m_depthUnit) to computation unit (metre)
			topDelta = MtLengthUnit::convert(*m_depthUnit, MtLengthUnit::METRE, topDelta);
			bottomDelta = MtLengthUnit::convert(*m_depthUnit, MtLengthUnit::METRE, bottomDelta);
		}

		generator->addHorizonsInterval(it->second.topHorizon.path, topDelta,
				it->second.bottomHorizon.path, bottomDelta);
		it++;
	}

	std::pair<bool, QString> generationResult = generator->run();
	delete generator;

	if (generationResult.first) {
		const QLatin1String LAST_PROJECT_PATH_IN_SETTINGS("BnniMainWindow/lastProjectPath");
		const QLatin1String LAST_TRAININGSET_PATH_IN_SETTINGS("BnniMainWindow/lastTrainingSetPath");

		QSettings settings;
		settings.setValue(LAST_PROJECT_PATH_IN_SETTINGS, m_trainingSet.projectPath());
		settings.setValue(LAST_TRAININGSET_PATH_IN_SETTINGS, m_trainingSet.outputJsonFile());

		QMessageBox::information(this, tr("Training Set Generation"), tr("Generation Finished"));
	} else {
		QMessageBox dialog(QMessageBox::Information, tr("Training Set Generation"),
				tr("Generation Finished with errors"), QMessageBox::Ok);
		dialog.setDetailedText(generationResult.second);
		dialog.setInformativeText("Training set file has been generated, in the process, errors were detected (see details)");
		dialog.exec();
	}
}

void TrainingSetParameterWidget::setBandPassState(int state) {
	if (state==Qt::Checked) {
		m_trainingSet.activateBandPassHighFrequency(m_bandPassFreqSpinBox->value());
		m_bandPassFreqSpinBox->show();
		m_bandPassFreqLabel->show();
	} else {
		m_trainingSet.deactivateBandPassHighFrequency();
		m_bandPassFreqSpinBox->hide();
		m_bandPassFreqLabel->hide();
	}
}

void TrainingSetParameterWidget::setBandPassFreq(double freq) {
	if (m_bandpassCheckBox->checkState()==Qt::Checked) {
		m_trainingSet.activateBandPassHighFrequency(freq);
	}
}

void TrainingSetParameterWidget::setAugmentationState(int state) {
	bool useAugmentation = state==Qt::Checked;
	m_trainingSet.setUseAugmentation(useAugmentation);
	if (useAugmentation) {
		m_augmentationDistanceLabel->show();
		m_augmentationDistanceSpinBox->show();
		m_augmentationNoiseStdLabel->show();
		m_augmentationNoiseStdSpinBox->show();
		m_useCnxAugmentationLabel->show();
		m_useCnxAugmentationCheckBox->show();
	} else {
		m_augmentationDistanceLabel->hide();
		m_augmentationDistanceSpinBox->hide();
		m_augmentationNoiseStdLabel->hide();
		m_augmentationNoiseStdSpinBox->hide();
		m_useCnxAugmentationLabel->hide();
		m_useCnxAugmentationCheckBox->hide();
	}
}

void TrainingSetParameterWidget::changeAugmentationDistance(int dist) {
	m_trainingSet.setAugmentationDistance(dist);
}

void TrainingSetParameterWidget::changeAugmentationNoiseStd(double val) {
	m_trainingSet.setGaussianNoiseStd(val);
}

void TrainingSetParameterWidget::changeUseCnxAugmentation(int state) {
	bool useCnxAugmentation = state==Qt::Checked;
	m_trainingSet.toggleCnxAugmentation(useCnxAugmentation);
}

void TrainingSetParameterWidget::setTrainingSetName() {
	m_trainingSet.setTrainingSetName(m_trainingSetLineEdit->text());
}

void TrainingSetParameterWidget::setTrainingSetNameFromData(QString file) {
	QSignalBlocker b1(m_trainingSetLineEdit);
	m_trainingSetLineEdit->setText(m_trainingSet.trainingSetName());
}

void TrainingSetParameterWidget::projectChanged() {
	m_trainingSet.setProjectPath(m_projectManagerWidget->getProjectPath());
}

void TrainingSetParameterWidget::initLogsLayout() {
	m_addWell = new QPushButton(QIcon(":/slicer/icons/add.png"), "");
	m_addWell->setToolTip("Add new well");
	m_addKind = new QPushButton(QIcon(":/slicer/icons/add.png"), "");
	m_addKind->setToolTip("Add new kind");
	m_logsLayout->addWidget(m_addKind, 1, 0);
	m_logsLayout->addWidget(m_addWell, 0, 1);

	connect(m_addKind, &QPushButton::clicked, this, &TrainingSetParameterWidget::addNewKind);
	connect(m_addWell, &QPushButton::clicked, this, &TrainingSetParameterWidget::addNewWell);

	addTfpColumn();
	addNewKind();
	addNewWell();
}

long TrainingSetParameterWidget::addNewWell() {
	long wellId = m_trainingSet.createNewWell();
	long lineIndex = m_logWellId;
	WellHeaderCell* headerCell = createWellCell(wellId);
	m_logsLayout->removeWidget(m_addWell);
	m_logsLayout->addWidget(headerCell, lineIndex, 0);
	m_logsLayout->addWidget(m_addWell, lineIndex+1, 0);
	m_logWellId++;

	m_wellHeaderCells[wellId] = headerCell;

	// work because ids are increasing, else changes of order may appear
	const std::map<long, BnniTrainingSet::BnniWellHeader>& map = m_trainingSet.wellHeaders();
	std::map<long, BnniTrainingSet::BnniWellHeader>::const_iterator it = map.begin();
	int columnIndex = 2;
	while (it!=map.end()) {
		WellKindCell* cell = createWellKindCell(wellId, it->first);
		m_logsLayout->addWidget(cell, lineIndex, columnIndex);

		m_wellKindCells[wellId][it->first] = cell;

		connect(cell, &WellKindCell::askChangeLog, [this, cell]() {
			renameWellKindCell(cell);
		});
		columnIndex++;
		it++;
	}

	WellTfpCell* tfpCell = createWellTfpCell(wellId);
	m_logsLayout->addWidget(tfpCell, lineIndex, 1);
	m_wellTfpCells[wellId] = tfpCell;
	connect(tfpCell, &WellTfpCell::askChangeTfp, [this, tfpCell]() {
		renameWellTfpCell(tfpCell);
	});

	connect(headerCell, &WellHeaderCell::askDelete, [this, headerCell, lineIndex]() {
		deleteWellHeaderCell(headerCell, lineIndex);
	});
	connect(headerCell, &WellHeaderCell::askChangeWell, [this, headerCell]() {
		changeWellWellHeaderCell(headerCell);
	});

	return wellId;
}

void TrainingSetParameterWidget::addNewKind() {
	long kindId = m_trainingSet.createNewKind();
	long columnIndex = m_logKindId;
	KindHeaderCell* headerCell = createKindCell(kindId);
	m_logsLayout->removeWidget(m_addKind);
	m_logsLayout->addWidget(headerCell, 0, columnIndex);
	m_logsLayout->addWidget(m_addKind, 0, columnIndex+1);
	m_logKindId++;

	m_kindHeaderCells[kindId] = headerCell;

	// work because ids are increasing, else changes of order may appear
	const std::map<long, BnniTrainingSet::WellParameter>& map = m_trainingSet.wellBores();
	std::map<long, BnniTrainingSet::WellParameter>::const_iterator it = map.begin();
	int lineIndex = 1;
	while (it!=map.end()) {
		WellKindCell* cell = createWellKindCell(it->first, kindId);
		m_logsLayout->addWidget(cell, lineIndex, columnIndex);

		m_wellKindCells[it->first][kindId] = cell;

		connect(cell, &WellKindCell::askChangeLog, [this, cell]() {
			renameWellKindCell(cell);
		});
		lineIndex++;
		it++;
	}

	connect(headerCell, &KindHeaderCell::askDelete, [this, headerCell, columnIndex]() {
		deleteKindHeaderCell(headerCell, columnIndex);
	});

}

void TrainingSetParameterWidget::addTfpColumn() {
	TfpHeaderCell* headerCell = createTfpCell();
	m_logsLayout->addWidget(headerCell, 0, 1);
}

WellHeaderCell* TrainingSetParameterWidget::createWellCell(long wellId) {
	return new WellHeaderCell(&m_trainingSet, wellId);
}

KindHeaderCell* TrainingSetParameterWidget::createKindCell(long kindId) {
	return new KindHeaderCell(&m_trainingSet, kindId);
}

TfpHeaderCell* TrainingSetParameterWidget::createTfpCell() {
	return new TfpHeaderCell(&m_trainingSet);
}

WellKindCell* TrainingSetParameterWidget::createWellKindCell(long wellId, long kindId) {
	return new WellKindCell(&m_trainingSet, wellId, kindId);
}

WellTfpCell* TrainingSetParameterWidget::createWellTfpCell(long wellId) {
	return new WellTfpCell(&m_trainingSet, wellId);
}

SeismicCell* TrainingSetParameterWidget::createSeismicCell(long seismicId) {
	return new SeismicCell(&m_trainingSet, seismicId);
}

void TrainingSetParameterWidget::renameWellKindCell(WellKindCell* cell) {
	bool badParam = m_trainingSet.wellBores().find(cell->wellId())==m_trainingSet.wellBores().end();
	if (!badParam) {
		const BnniTrainingSet::WellParameter& wellParameter = m_trainingSet.wellBores().at(cell->wellId());
		badParam = wellParameter.logsPathAndName.find(cell->kindId())==wellParameter.logsPathAndName.end();
	}
	if (badParam) {
		qDebug() << "renameWellKindCell : Cell does not match data";
		return;
	}

	m_projectManagerWidget->wellDatabaseUpdate();
	std::vector<WELLHEADDATA> wellHeads = m_projectManagerWidget->getMainWellData();
	int wellHeadIdx=-1, wellBoreIdx=-1;
	const BnniTrainingSet::WellParameter& wellParameter = m_trainingSet.wellBores().at(cell->wellId());
	wellHeadIdx = wellParameter.cacheHeadIdx;
	wellBoreIdx = wellParameter.cacheBoreIdx;
	bool valid = wellHeadIdx>=0 && wellHeadIdx<wellHeads.size() && wellBoreIdx>=0 &&
			wellBoreIdx<wellHeads[wellHeadIdx].bore.size();

	if (valid) {
		const WELLBOREDATA& wellBore = wellHeads[wellHeadIdx].bore[wellBoreIdx];

		valid = wellMatch(wellParameter, wellBore);

		const std::vector<QString>& logNames = wellBore.logs.getTiny();
		const std::vector<QString>& logPaths = wellBore.logs.getFull();
//		QStringList logNamesForDialog(logNames.begin(), logNames.end());// = wellBoreList.log_tinyname;
		FileSelectorDialog dialog(&logNames, "Select log");
		int code = dialog.exec();
		bool accepted = code==QDialog::Accepted;
		if (accepted) {
			int newLogIndex = dialog.getSelectedIndex();

			if (newLogIndex>=0 && newLogIndex<logNames.size()) {
				BnniTrainingSet::WellParameter newWellParameter = wellParameter;
				std::pair<QString, QString> newPathAndName(logPaths[newLogIndex],
						logNames[newLogIndex]);
				newWellParameter.logsPathAndName[cell->kindId()] = newPathAndName;
				m_trainingSet.changeWellBore(cell->wellId(), newWellParameter);
				cell->updateName();
			}
		}
	}
	if (!valid) {
		qDebug() << "renameWellKindCell : Invalid cell, well basket and data relation";
	}
}

void TrainingSetParameterWidget::renameWellTfpCell(WellTfpCell* cell) {
	bool badParam = m_trainingSet.wellBores().find(cell->wellId())==m_trainingSet.wellBores().end();
	if (badParam) {
		qDebug() << "renameWellKindCell : Cell does not match data";
		return;
	}

	m_projectManagerWidget->wellDatabaseUpdate();
	std::vector<WELLHEADDATA> wellHeads = m_projectManagerWidget->getMainWellData();
	int wellHeadIdx=-1, wellBoreIdx=-1;
	const BnniTrainingSet::WellParameter& wellParameter = m_trainingSet.wellBores().at(cell->wellId());
	wellHeadIdx = wellParameter.cacheHeadIdx;
	wellBoreIdx = wellParameter.cacheBoreIdx;
	bool valid = wellHeadIdx>=0 && wellHeadIdx<wellHeads.size() && wellBoreIdx>=0 &&
			wellBoreIdx<wellHeads[wellHeadIdx].bore.size();

	if (valid) {
		const WELLBOREDATA& wellBore = wellHeads[wellHeadIdx].bore[wellBoreIdx];

		valid = wellMatch(wellParameter, wellBore);

		const std::vector<QString>& tf2pNames = wellBore.tf2p.getTiny();
		const std::vector<QString>& tf2pPaths = wellBore.tf2p.getFull();
//		QStringList tf2pNamesForDialog(tf2pNames.begin(), tf2pNames.end());// = wellBoreList.log_tinyname;
		FileSelectorDialog dialog(&tf2pNames, "Select log");
		int code = dialog.exec();
		bool accepted = code==QDialog::Accepted;
		if (accepted) {
			int newTf2pIndex = dialog.getSelectedIndex();

			if (newTf2pIndex>=0 && newTf2pIndex<tf2pNames.size()) {
				BnniTrainingSet::WellParameter newWellParameter = wellParameter;
				newWellParameter.tfpName = tf2pNames[newTf2pIndex];
				newWellParameter.tfpFile = tf2pPaths[newTf2pIndex];
				m_trainingSet.changeWellBore(cell->wellId(), newWellParameter);
				cell->updateName();
			}
		}
	}
	if (!valid) {
		qDebug() << "renameWellKindCell : Invalid cell, well basket and data relation";
	}
}

void TrainingSetParameterWidget::deleteKindHeaderCell(KindHeaderCell* headerCell, int columnIndex) {
	long kindId = headerCell->kindId();
	m_kindHeaderCells.erase(kindId);
	std::map<long, std::map<long, WellKindCell*>>::iterator itWell = m_wellKindCells.begin();
	while (itWell!=m_wellKindCells.end()) {
		std::map<long, WellKindCell*>& kindToCellMap = itWell->second;
		kindToCellMap.erase(kindId);
		itWell++;
	}

	// delete ui then data
	for (int row=0; row<m_logsLayout->rowCount(); row++) {
		QLayoutItem* layoutItem = m_logsLayout->itemAtPosition(row, columnIndex);
		if (layoutItem!=nullptr && layoutItem->widget()!=nullptr) {
			layoutItem->widget()->deleteLater();
		}
	}

	m_trainingSet.deleteKind(kindId);
}

void TrainingSetParameterWidget::deleteWellHeaderCell(WellHeaderCell* headerCell, int lineIndex) {
	long wellId = headerCell->wellId();
	m_wellHeaderCells.erase(wellId);
	m_wellKindCells.erase(wellId);

	// delete ui then data
	for (int col=0; col<m_logsLayout->columnCount(); col++) {
		QLayoutItem* layoutItem = m_logsLayout->itemAtPosition(lineIndex, col);
		if (layoutItem!=nullptr && layoutItem->widget()!=nullptr) {
			layoutItem->widget()->deleteLater();
		}
	}

	m_trainingSet.deleteWell(wellId);
}

void TrainingSetParameterWidget::changeWellWellHeaderCell(WellHeaderCell* headerCell) {
	QString wellName = "";

	m_projectManagerWidget->wellDatabaseUpdate();
	std::vector<WELLHEADDATA> wellHeads = m_projectManagerWidget->getMainWellData();

	LogSelectorTreeDialog dialog(wellHeads);

	int outCode = dialog.exec();

	bool valid = outCode==QDialog::Accepted;
	int wellHeadIdx, wellBoreIdx;
	if (valid) {
		wellHeadIdx = dialog.wellHeadIdx();
		wellBoreIdx = dialog.wellBoreIdx();
		valid = wellHeadIdx>=0 && wellHeadIdx<wellHeads.size() && wellBoreIdx>=0 &&
				wellBoreIdx<wellHeads[wellHeadIdx].bore.size();
	}

	if (valid) {
		const WELLBOREDATA& wellBore = wellHeads[wellHeadIdx].bore[wellBoreIdx];
		QString newDeviationFile = wellBore.deviationFullName;
		const std::map<long, BnniTrainingSet::WellParameter>& wellBores =  m_trainingSet.wellBores();
		QString oldDeviationFile = wellBores.at(headerCell->wellId()).deviationFile;

		valid = valid && newDeviationFile.compare(oldDeviationFile)!=0;

		if (valid) {
			changeWellInData(headerCell->wellId(), wellBore, wellHeadIdx, wellBoreIdx);
			headerCell->updateName();

			std::map<long, WellKindCell*>::iterator it = m_wellKindCells[headerCell->wellId()].begin();
			while (it!=m_wellKindCells[headerCell->wellId()].end()) {
				it->second->updateName();
				it++;
			}
		}
	}
}

void TrainingSetParameterWidget::changeWellInData(long wellId, const WELLBOREDATA& wellBore,
		int cacheHeadIdx, int cacheBoreIdx) {
	std::vector<WELLHEADDATA> wellHeads = m_projectManagerWidget->getMainWellData();
	QDir boreDir(wellBore.fullName);
	QStringList boreDescFileList = boreDir.entryList(QStringList() << "*.desc", QDir::Files);
	QDir headDir(wellHeads[cacheHeadIdx].fullName);
	QStringList headDescFileList = headDir.entryList(QStringList() << "*.desc", QDir::Files);

	BnniTrainingSet::WellParameter well;
	well.deviationFile = wellBore.deviationFullName;
	if (boreDescFileList.size()>0) {
		well.boreDescFile = boreDir.absoluteFilePath(boreDescFileList[0]);
	} else {
		well.boreDescFile = "";
	}
	if (headDescFileList.size()>0) {
		well.headDescFile = headDir.absoluteFilePath(headDescFileList[0]);
	} else {
		well.headDescFile = "";
	}
	well.wellName = wellBore.tinyName;
	initTfpsMap(well, wellBore);
	well.cacheHeadIdx = cacheHeadIdx;
	well.cacheBoreIdx = cacheBoreIdx;
	initLogsMap(well, wellBore);
	m_trainingSet.changeWellBore(wellId, well);
}

void TrainingSetParameterWidget::initLogsMap(BnniTrainingSet::WellParameter& well, const WELLBOREDATA& wellBore) {
	const std::map<long, BnniTrainingSet::BnniWellHeader>& wellHeaders = m_trainingSet.wellHeaders();
	for (std::map<long, BnniTrainingSet::BnniWellHeader>::const_iterator it = wellHeaders.begin();
			it!=wellHeaders.end(); it++) {
		QString logPath = "";
		QString logName = "";
		bool logNotFound = true;

		if (!it->second.filterStr.isNull() && !it->second.filterStr.isEmpty()) {
			const std::vector<QString>& logNames = wellBore.logs.getTiny();
			const std::vector<QString>& logPaths = wellBore.logs.getFull();

			int logIndex=0;
			while (logNotFound && logIndex<logPaths.size()) {
				if (it->second.filterType==BnniTrainingSet::WellKind) {
					QString kind = getKind(logPaths[logIndex]);
					logNotFound = kind.compare(it->second.filterStr)!=0;
				} else {
					QString name = logNames[logIndex];
					logNotFound = name.compare(it->second.filterStr)!=0;
				}
				if (logNotFound) {
					logIndex++;
				}
			}
			if (!logNotFound) {
				logName = logNames[logIndex];
				logPath = logPaths[logIndex];
			}
		}

		// always add logPath even if empty
		well.logsPathAndName[it->first] = std::pair<QString, QString>(logPath, logName);
	}
}

void TrainingSetParameterWidget::initTfpsMap(BnniTrainingSet::WellParameter& well, const WELLBOREDATA& wellBore) {
	QString tfpPath = "";
	QString tfpName = "";
	bool tfpNotFound = true;

	// search the one matching the tfp name
	if (!m_trainingSet.tfpFilter().isNull() && !m_trainingSet.tfpFilter().isEmpty()) {
		const std::vector<QString>& tfpNames = wellBore.tf2p.getTiny();
		const std::vector<QString>& tfpPaths = wellBore.tf2p.getFull();

		int tfpIndex=0;
		while (tfpNotFound && tfpIndex<tfpNames.size()) {
			tfpNotFound = m_trainingSet.tfpFilter().compare(tfpNames[tfpIndex])!=0;
			if (tfpNotFound) {
				tfpIndex++;
			}
		}
		if (!tfpNotFound) {
			tfpName = tfpNames[tfpIndex];
			tfpPath = tfpPaths[tfpIndex];
		}
	}

	// fall back with default tfp
	if (tfpPath.isNull() || tfpPath.isEmpty()) {
		QString defaultAbsolutePath = WellBore::getTfpFileFromDescFile(well.boreDescFile);
		QString name = ProjectManagerNames::getKeyTabFromFilename(defaultAbsolutePath, "Name");
		if (!name.isEmpty()) {
			tfpPath = defaultAbsolutePath;
			tfpName = name;
		}
	}

	// always add tfpPath even if empty
	well.tfpName = tfpName;
	well.tfpFile = tfpPath;
}

QString TrainingSetParameterWidget::getKind(QString logPath) {
	QString out;
	std::map<QString, QString>::const_iterator it = m_logKindDetectionCache.find(logPath);
	if (it!=m_logKindDetectionCache.end()) {
		out = it->second;
	} else {
		out = WellBore::getKindFromLogFile(logPath);
		m_logKindDetectionCache[logPath] = out;
	}

	return out;
}


bool TrainingSetParameterWidget::wellMatch(const BnniTrainingSet::WellParameter& wellParameter,
		const WELLBOREDATA& wellBore) {
	bool out = wellParameter.deviationFile.compare(wellBore.deviationFullName)==0 &&
			wellParameter.wellName.compare(wellBore.tinyName)==0;

	return out;
}

void TrainingSetParameterWidget::initSeismicsLayout() {
	m_addSeismic = new QPushButton(QIcon(":/slicer/icons/add.png"), "");
	m_addSeismic->setToolTip("Add new seismic");
	m_seismicsLayout->addWidget(m_addSeismic);

	connect(m_addSeismic, &QPushButton::clicked, this, &TrainingSetParameterWidget::addNewSeismic);

	addNewSeismic();
}

void TrainingSetParameterWidget::addNewSeismic() {
	long seismicId = m_trainingSet.createNewSeismic();
	SeismicCell* cell = createSeismicCell(seismicId);
	m_seismicsLayout->removeWidget(m_addSeismic);
	m_seismicsLayout->addWidget(cell);
	m_seismicsLayout->addWidget(m_addSeismic);

	connect(cell, &SeismicCell::askChangeSeismic, [this, cell]() {
		changeSeismic(cell);
	});

	connect(cell, &SeismicCell::askDelete, [this, cell]() {
		deleteSeismicCell(cell);
	});
}

void TrainingSetParameterWidget::changeSeismic(SeismicCell* cell) {
	m_projectManagerWidget->seimsicDatabaseUpdate();
	std::vector<QString> names = m_projectManagerWidget->getSeismicAllNames();
	std::vector<QString> paths = m_projectManagerWidget->getSeismicAllPath();
//	QStringList seismicList(names.begin(), names.end());

	FileSelectorDialog dialog(&names, "Select seismic");
	dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::seismic);
	int code = dialog.exec();
	int newSeismicIndex = dialog.getSelectedIndex();
	if (code==QDialog::Accepted && newSeismicIndex>=0 && newSeismicIndex<names.size()) {
		// get dynamic
		bool minSet = false;
		bool maxSet = false;
		float min=0, max = 1;

		FILE *pFile = fopen(paths[newSeismicIndex].toStdString().c_str(), "r");
		if ( pFile == NULL ) return;
		char str[10000];
		fseek(pFile, 0x4c, SEEK_SET);
		int n = 0, cont = 1;
		while ( cont )
		{
			int nbre = fscanf(pFile, "VMIN=\t%f\n", &min);
			if ( nbre > 0 ) {
				cont = 0;
				minSet = true;
			} else
				fgets(str, 10000, pFile);
			n++;
			if ( n > 40 )
			{
				cont = 0;
				strcpy(str, "Other");
			}
		}
		fseek(pFile, 0x4c, SEEK_SET);
		n = 0, cont = 1;
		while ( cont )
		{
			int nbre = fscanf(pFile, "VMAX=\t%f\n", &max);
			if ( nbre > 0 ) {
				cont = 0;
				maxSet = true;
			} else
				fgets(str, 10000, pFile);
			n++;
			if ( n > 40 )
			{
				cont = 0;
				strcpy(str, "Other");
			}
		}
		fclose(pFile);

		// modify object
		BnniTrainingSet::SeismicParameter param = m_trainingSet.seismics().at(cell->seismicId());
		param.name = names[newSeismicIndex];
		param.path = paths[newSeismicIndex];
		switch (SeismicManager::filextGetAxis(param.path)) {
		case 0:
			param.unit = SampleUnit::TIME;
			break;
		case 1:
			param.unit = SampleUnit::DEPTH;
			break;
		default:
			param.unit = SampleUnit::NONE;
			break;
		}
		if (minSet && maxSet) {
			param.min = min;
			param.max = max;
		}
		bool accepted = m_trainingSet.changeSeismic(cell->seismicId(), param);
		if (accepted) {
			cell->updateName();
			if (minSet && maxSet) {
				cell->changeMin(min);
				cell->changeMax(max);
			}
		} else if (param.unit!=m_trainingSet.seismicUnit() && param.unit!=SampleUnit::NONE && m_trainingSet.seismicUnit()!=NONE) {
			QString trainingSetUnit = getNameFromSampleUnit(m_trainingSet.seismicUnit());
			QString seismicUnit = getNameFromSampleUnit(param.unit);

			QMessageBox::warning(this, "Invalid seismic change", "Mismatch of sample unit : trainig set unit is " +
					trainingSetUnit + " and seismic unit is " + seismicUnit);
		}
	}
}

void TrainingSetParameterWidget::deleteSeismicCell(SeismicCell* cell) {
	long seismicId = cell->seismicId();
	cell->deleteLater();

	m_trainingSet.deleteSeismic(seismicId);
}

void TrainingSetParameterWidget::openAutoCorrWindow() {
	const std::map<long , BnniTrainingSet::SeismicParameter>& seismics = m_trainingSet.seismics();
	std::vector<QString> seismicNames;
	std::vector<QString> seismicPaths;

	for (auto it=seismics.begin(); it!=seismics.end(); it++) {
		seismicNames.push_back(it->second.name);
		seismicPaths.push_back(it->second.path);
	}
	AutoCorrControlWidget* widget = new AutoCorrControlWidget(seismicNames, seismicPaths);
	widget->setSampleRate(m_trainingSet.targetSampleRate());
	widget->show();

	m_displayedAutoCorrWidgets.push_back(widget);
	QMetaObject::Connection conn = connect(widget, &QWidget::destroyed, [widget, this]() {
		std::vector<AutoCorrControlWidget*>::iterator it = std::find(m_displayedAutoCorrWidgets.begin(),
				m_displayedAutoCorrWidgets.end(), widget);
		if (it!=m_displayedAutoCorrWidgets.end()) {
			m_displayedAutoCorrWidgets.erase(it);
			m_autoCorrConns.erase(widget);
		}
	});
	m_autoCorrConns[widget] = conn;
}

std::vector<long> TrainingSetParameterWidget::detectEmptyWells() {
	std::vector<long> emptyWellIds;
	std::map<long, BnniTrainingSet::WellParameter>::const_iterator it = m_trainingSet.wellBores().begin();
	while (it!=m_trainingSet.wellBores().end()) {
		if (it->second.boreDescFile.isNull() || it->second.boreDescFile.isEmpty()) {
			emptyWellIds.push_back(it->first);
		}
		it++;
	}
	return emptyWellIds;
}

void TrainingSetParameterWidget::addMoreWells() {
	// Can be optimized by first extracting a list of WellParameter
	// and if we assume that there is no duplicate in WELLMASTER
	// That will avoid checking newly added wells
	m_projectManagerWidget->wellDatabaseUpdate();
	fillIncompleteWells();

	// Detect empty lines
	std::vector<long> emptyWellIds = detectEmptyWells();

	std::vector<WELLHEADDATA> wellHeads = m_projectManagerWidget->getMainWellData();

//	const WELLMASTER& wellMaster = m_projectManagerWidget->get_well_list();
//	std::vector<std::vector<QString>> logsTinyNames = m_projectManagerWidget->getWellBasketLogPicksTf2pNames("log", "tiny");
//	std::vector<std::vector<QString>> logsFullNames = m_projectManagerWidget->getWellBasketLogPicksTf2pNames("log", "full");

	for (int idxHead=0; idxHead<wellHeads.size(); idxHead++) {
		const WELLHEADDATA& wellHead = wellHeads[idxHead];
		for (int idxBore=0; idxBore<wellHead.bore.size(); idxBore++) {
			const WELLBOREDATA& wellBore = wellHead.bore[idxBore];

			// check if already added
			bool isNotAdded = true;
			std::map<long, BnniTrainingSet::WellParameter>::const_iterator itAdded = m_trainingSet.wellBores().begin();
			while (isNotAdded && itAdded!=m_trainingSet.wellBores().end()) {
				isNotAdded = !wellMatch(itAdded->second, wellBore);
				itAdded++;
			}

			// get logs
			const std::vector<QString>& logsFullNames = wellBore.logs.getFull();
			const std::vector<QString>& logsTinyNames = wellBore.logs.getTiny();

			bool isLogValid = true;
			std::map<long, int> idToIndex;
			if (isNotAdded) {
				std::map<long, BnniTrainingSet::BnniWellHeader>::const_iterator logIt = m_trainingSet.wellHeaders().begin();
				while (isLogValid && logIt!=m_trainingSet.wellHeaders().end()) {
					QString itFilterName = logIt->second.filterStr.toLower();
					isLogValid = false;
					int iLog = 0;
					while (!isLogValid && iLog<logsFullNames.size()) {//iLog<wellBoreList.log_tinyname.size()) {
						if (logIt->second.filterType==BnniTrainingSet::WellFilter::WellName) {
							isLogValid = itFilterName.compare(logsTinyNames[iLog].toLower())==0;
						} else {
							QString kind = getKind(logsFullNames[iLog]);
							isLogValid = itFilterName.compare(kind.toLower())==0;
						}
						if (isLogValid) {
							idToIndex[logIt->first] = iLog;
						}
						iLog++;
					}
					logIt++;
				}
			}

			QString tfpPath = "";
			QString tfpName = "";
			bool foundTfp = false;
			if (isNotAdded && isLogValid && !m_trainingSet.tfpFilter().isNull() && !m_trainingSet.tfpFilter().isEmpty()) {
				// search the one matching the tfp name
				bool tfpNotFound = true;
				std::vector<QString> tfpNames = wellBore.tf2p.getTiny();
				std::vector<QString> tfpPaths = wellBore.tf2p.getFull();

				int tfpIndex=0;
				while (tfpNotFound && tfpIndex<tfpNames.size()) {
					tfpNotFound = m_trainingSet.tfpFilter().compare(tfpNames[tfpIndex])!=0;
					if (tfpNotFound) {
						tfpIndex++;
					}
				}
				if (!tfpNotFound) {
					tfpName = tfpNames[tfpIndex];
					tfpPath = tfpPaths[tfpIndex];
					foundTfp = true;
				} else {
					// if tfp filter is valid but the tfp was not found, do not add the well
					isLogValid = false;
				}
			}

			// check log
			if (isNotAdded && isLogValid) {
				// add
				long id;
				if (emptyWellIds.size()>0) {
					id = emptyWellIds[0];
					emptyWellIds.erase(emptyWellIds.begin());
				} else {
					id = addNewWell();
				}
				changeWellInData(id, wellBore, idxHead, idxBore);
				BnniTrainingSet::WellParameter wellParam = m_trainingSet.wellBores().at(id);
				for (std::pair<long, int> idAndIndex : idToIndex) {
					std::pair<QString, QString> newPathAndName(logsFullNames[idAndIndex.second],
									logsTinyNames[idAndIndex.second]);
					wellParam.logsPathAndName[idAndIndex.first] = newPathAndName;
				}
				if (foundTfp) {
					wellParam.tfpName = tfpName;
					wellParam.tfpFile = tfpPath;
				}
				m_trainingSet.changeWellBore(id, wellParam);
				m_wellHeaderCells[id]->updateName();
				std::map<long, WellKindCell*>::iterator it = m_wellKindCells[id].begin();
				while (it!=m_wellKindCells[id].end()) {
					it->second->updateName();
					it++;
				}
				m_wellTfpCells[id]->updateName();
			}
		}
	}
}

void TrainingSetParameterWidget::fillIncompleteWells() {
	std::vector<WELLHEADDATA> wellHeads = m_projectManagerWidget->getMainWellData();

	std::map<long, BnniTrainingSet::WellParameter>::const_iterator wellIt = m_trainingSet.wellBores().begin();
	int numberOfKinds = m_trainingSet.wellHeaders().size();

	std::map<long, BnniTrainingSet::WellParameter>::const_iterator wellItEnd = m_trainingSet.wellBores().end();
	while (wellIt!=wellItEnd) {
		// expecting the m_trainingSet structure to be well organised
		bool isComplete = wellIt->second.logsPathAndName.size()==numberOfKinds;
		{
			std::map<long, BnniTrainingSet::BnniWellHeader>::const_iterator kindIt = m_trainingSet.wellHeaders().begin();
			std::map<long, BnniTrainingSet::BnniWellHeader>::const_iterator kindItEnd = m_trainingSet.wellHeaders().end();

			while (isComplete && kindIt!=kindItEnd) {
				isComplete = !BnniTrainingSet::isLogEmpty(kindIt->first, wellIt->second.logsPathAndName);
				kindIt++;
			}
		}

		if (!isComplete) {
			int headIndex = wellIt->second.cacheHeadIdx;
			int boreIndex = wellIt->second.cacheBoreIdx;
			bool cacheValid = headIndex>=0 && boreIndex>=0 && headIndex<wellHeads.size() &&
					boreIndex<wellHeads[headIndex].bore.size();
			if (cacheValid) {
				const WELLBOREDATA& wellBore = wellHeads[headIndex].bore[boreIndex];
				cacheValid = wellMatch(wellIt->second, wellBore);
			}

			bool notFound = true;
			if (!cacheValid) {
				headIndex = 0;
				boreIndex = 0;
				while (notFound && headIndex<wellHeads.size()) {
					boreIndex = 0;

					while (notFound && boreIndex<wellHeads[headIndex].bore.size()) {
						const WELLBOREDATA& wellBore = wellHeads[headIndex].bore[boreIndex];
						notFound = !wellMatch(wellIt->second, wellBore);

						if (notFound) {
							boreIndex++;
						}
					}
					if (notFound) {
						headIndex++;
					}
				}
			} else {
				notFound = false;
			}

			// TODO remove cacheValid from if
			if ((cacheValid || !notFound)) {
				const std::vector<QString>& logNames = wellHeads[headIndex].bore[boreIndex].logs.getTiny();
				const std::vector<QString>& logPaths = wellHeads[headIndex].bore[boreIndex].logs.getFull();


				std::map<long, BnniTrainingSet::BnniWellHeader>::const_iterator kindIt = m_trainingSet.wellHeaders().begin();
				std::map<long, BnniTrainingSet::BnniWellHeader>::const_iterator kindItEnd = m_trainingSet.wellHeaders().end();

				BnniTrainingSet::WellParameter wellParameter = wellIt->second;
				bool changeNeeded = false;

				while (kindIt!=kindItEnd) {
					if (BnniTrainingSet::isLogEmpty(kindIt->first, wellIt->second.logsPathAndName)) {
						QString itFilterName = kindIt->second.filterStr.toLower();
						bool isLogValid = false;
						int iLog = 0;
						while (!isLogValid && iLog<logPaths.size()) {//iLog<wellBoreList.log_tinyname.size()) {
							if (kindIt->second.filterType==BnniTrainingSet::WellFilter::WellName) {
								isLogValid = itFilterName.compare(logNames[iLog].toLower())==0;
							} else {
								QString kind = getKind(logPaths[iLog]);
								isLogValid = itFilterName.compare(kind.toLower())==0;
							}
							if (isLogValid) {
								wellParameter.logsPathAndName[kindIt->first] = std::pair<QString, QString>(logPaths[iLog],
										logNames[iLog]);
								changeNeeded = true;
							}
							iLog++;
						}
					}
					kindIt++;
				}

				if (changeNeeded) {
					m_trainingSet.changeWellBore(wellIt->first, wellParameter);

					std::map<long, WellKindCell*>::iterator it = m_wellKindCells[wellIt->first].begin();
					while (it!=m_wellKindCells[wellIt->first].end()) {
						it->second->updateName();
						it++;
					}
				}
			}
		}
		wellIt++;
	}
}

void TrainingSetParameterWidget::toggleHorizonInterval(int state) {
	bool enabled = state==Qt::Checked;
	if (enabled) {
		m_horizonIntervalId = m_trainingSet.createNewInterval();
		m_topHorizonLabel->setText("");
		{
			QSignalBlocker tb(m_topDeltaSpinBox);
			m_topDeltaSpinBox->setValue(0);
		}
		m_bottomHorizonLabel->setText("");
		{
			QSignalBlocker bb(m_bottomDeltaSpinBox);
			m_bottomDeltaSpinBox->setValue(0);
		}
		m_horizonHolder->show();
	} else {
		m_horizonHolder->hide();
		m_trainingSet.deleteInterval(m_horizonIntervalId);
		m_horizonIntervalId = -1;
	}
}

void TrainingSetParameterWidget::selectTopHorizon() {
	if (m_horizonIntervalId>=0) {
		std::vector<QString> names = m_projectManagerWidget->getHorizonAllNames();
		std::vector<QString> paths = m_projectManagerWidget->getHorizonAllPath();
//		QStringList horizonList(names.begin(), names.end());

		FileSelectorDialog dialog(&names, "Select top horizon");
		int code = dialog.exec();
		int newHorizonIndex = dialog.getSelectedIndex();
		if (code==QDialog::Accepted && newHorizonIndex>=0 && newHorizonIndex<names.size()) {
			// modify object
			BnniTrainingSet::HorizonIntervalParameter param = m_trainingSet.intervals().at(m_horizonIntervalId);
			param.topHorizon.name = names[newHorizonIndex];
			param.topHorizon.path = paths[newHorizonIndex];

			m_trainingSet.changeInterval(m_horizonIntervalId, param);
			m_topHorizonLabel->setText(param.topHorizon.name);
		}
	}
}

void TrainingSetParameterWidget::selectBottomHorizon() {
	if (m_horizonIntervalId>=0) {
		std::vector<QString> names = m_projectManagerWidget->getHorizonAllNames();
		std::vector<QString> paths = m_projectManagerWidget->getHorizonAllPath();
//		QStringList horizonList(names.begin(), names.end());

		FileSelectorDialog dialog(&names, "Select bottom horizon");
		int code = dialog.exec();
		int newHorizonIndex = dialog.getSelectedIndex();
		if (code==QDialog::Accepted && newHorizonIndex>=0 && newHorizonIndex<names.size()) {
			// modify object
			BnniTrainingSet::HorizonIntervalParameter param = m_trainingSet.intervals().at(m_horizonIntervalId);
			param.bottomHorizon.name = names[newHorizonIndex];
			param.bottomHorizon.path = paths[newHorizonIndex];

			m_trainingSet.changeInterval(m_horizonIntervalId, param);
			m_bottomHorizonLabel->setText(param.bottomHorizon.name);
		}
	}
}

void TrainingSetParameterWidget::changeTopDelta(double delta) {
	if (m_horizonIntervalId>=0) {
		BnniTrainingSet::HorizonIntervalParameter interval = m_trainingSet.intervals().at(m_horizonIntervalId);
		interval.topHorizon.delta = delta;
		m_trainingSet.changeInterval(m_horizonIntervalId, interval);
	}
}

void TrainingSetParameterWidget::changeBottomDelta(double delta) {
	if (m_horizonIntervalId>=0) {
		BnniTrainingSet::HorizonIntervalParameter interval = m_trainingSet.intervals().at(m_horizonIntervalId);
		interval.bottomHorizon.delta = delta;
		m_trainingSet.changeInterval(m_horizonIntervalId, interval);
	}
}

void TrainingSetParameterWidget::toggleDepthUnit() {
	if (*m_depthUnit==MtLengthUnit::METRE) {
		m_depthUnit = &MtLengthUnit::FEET;
		m_depthUnitButton->setIcon(QIcon(":/slicer/icons/regle_ft128_blanc.png"));
	} else {
		m_depthUnit = &MtLengthUnit::METRE;
		m_depthUnitButton->setIcon(QIcon(":/slicer/icons/regle_m128_blanc.png"));
	}

	m_depthUnitButton->setText(m_depthUnit->getName());
	if (m_trainingSet.seismicUnit()==SampleUnit::DEPTH) {
		m_topDeltaSpinBox->setSuffix(m_depthUnit->getSymbol());
		m_bottomDeltaSpinBox->setSuffix(m_depthUnit->getSymbol());
	}
}

void TrainingSetParameterWidget::trainingSetSeismicUnitChanged() {
	if (m_trainingSet.seismicUnit()==SampleUnit::DEPTH) {
		m_topDeltaSpinBox->setSuffix(m_depthUnit->getSymbol());
		m_bottomDeltaSpinBox->setSuffix(m_depthUnit->getSymbol());
	} else if (m_trainingSet.seismicUnit()==SampleUnit::TIME) {
		m_topDeltaSpinBox->setSuffix("ms");
		m_bottomDeltaSpinBox->setSuffix("ms");
	} else {
		m_topDeltaSpinBox->setSuffix("");
		m_bottomDeltaSpinBox->setSuffix("");
	}
}

WellHeaderCell::WellHeaderCell(BnniTrainingSet* trainingSet, long wellId,
			QWidget *parent, Qt::WindowFlags f) :
				QWidget(parent, f) {
	m_trainingSet = trainingSet;
	m_wellId = wellId;

	QHBoxLayout* mainLayout = new QHBoxLayout;
	setLayout(mainLayout);
	m_wellLabel = new QLabel(m_trainingSet->wellBores().at(m_wellId).wellName);
	mainLayout->addWidget(m_wellLabel);
	m_menuButton = new QPushButton("...");
	mainLayout->addWidget(m_menuButton);

	connect(m_menuButton, &QPushButton::clicked, this, &WellHeaderCell::openMenu);
}

WellHeaderCell::~WellHeaderCell() {

}

long WellHeaderCell::wellId() const {
	return m_wellId;
}

void WellHeaderCell::updateName() {
	m_wellLabel->setText(m_trainingSet->wellBores().at(m_wellId).wellName);
}

void WellHeaderCell::openMenu() {
	QMenu menu;

	menu.addAction("Change well", this, &WellHeaderCell::askChangeWellSlot);
	menu.addAction("Remove", this, &WellHeaderCell::askDeleteSlot);

	menu.exec(QCursor::pos());
}

void WellHeaderCell::askChangeWellSlot() {
	emit askChangeWell();
}

void WellHeaderCell::askDeleteSlot() {
	emit askDelete();
}

KindHeaderCell::KindHeaderCell(BnniTrainingSet* trainingSet, long kindId, QWidget *parent,
			Qt::WindowFlags f) : QWidget(parent) {
	m_trainingSet = trainingSet;
	m_kindId = kindId;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QHBoxLayout* nameLayout = new QHBoxLayout;
	mainLayout->addLayout(nameLayout);

	m_typeComboBox = new QComboBox;
	m_typeComboBox->addItem("Name");
	m_typeComboBox->addItem("Kind");
	if (m_trainingSet->wellHeaders().at(m_kindId).filterType==BnniTrainingSet::WellName) {
		m_typeComboBox->setCurrentIndex(m_NAME_INDEX);
	} else {
		m_typeComboBox->setCurrentIndex(m_KIND_INDEX);
	}
	nameLayout->addWidget(m_typeComboBox);
	m_nameLineEdit = new QLineEdit(m_trainingSet->wellHeaders().at(m_kindId).filterStr);
	nameLayout->addWidget(m_nameLineEdit);
	m_menuButton = new QPushButton("...");
	nameLayout->addWidget(m_menuButton);

	// dynamic
	QHBoxLayout* dynamicLayout = new QHBoxLayout;
	mainLayout->addLayout(dynamicLayout);

	dynamicLayout->addWidget(new QLabel("min:"));

	m_minSpinBox = new QDoubleSpinBox;
	m_minSpinBox->setMinimum(std::numeric_limits<float>::lowest());
	m_minSpinBox->setMaximum(m_trainingSet->wellHeaders().at(m_kindId).max);
	m_minSpinBox->setValue(m_trainingSet->wellHeaders().at(m_kindId).min);
	dynamicLayout->addWidget(m_minSpinBox);

	dynamicLayout->addWidget(new QLabel("max:"));

	m_maxSpinBox = new QDoubleSpinBox;
	m_maxSpinBox->setMinimum(m_trainingSet->wellHeaders().at(m_kindId).min);
	m_maxSpinBox->setMaximum(std::numeric_limits<float>::max());
	m_maxSpinBox->setValue(m_trainingSet->wellHeaders().at(m_kindId).max);
	dynamicLayout->addWidget(m_maxSpinBox);

	connect(m_typeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &KindHeaderCell::changeFilterType);
	connect(m_nameLineEdit, &QLineEdit::editingFinished, this, &KindHeaderCell::changeKind);
	connect(m_menuButton, &QPushButton::clicked, this, &KindHeaderCell::openMenu);
	connect(m_minSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &KindHeaderCell::changeMin);
	connect(m_maxSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &KindHeaderCell::changeMax);
}

KindHeaderCell::~KindHeaderCell() {

}

long KindHeaderCell::kindId() const {
	return m_kindId;
}

void KindHeaderCell::changeFilterType(int idx) {
	QSignalBlocker b1(m_nameLineEdit);

	BnniTrainingSet::BnniWellHeader header = m_trainingSet->wellHeaders().at(m_kindId);
	header.filterStr = m_nameLineEdit->text();
	if (m_NAME_INDEX==idx) {
		header.filterType = BnniTrainingSet::WellName;
	} else {
		header.filterType = BnniTrainingSet::WellKind;
	}
	m_trainingSet->changeKind(m_kindId, header);
}

void KindHeaderCell::changeKind() {
	// TODO improve by trying to detect logs
	BnniTrainingSet::BnniWellHeader header = m_trainingSet->wellHeaders().at(m_kindId);
	header.filterStr = m_nameLineEdit->text();
	m_trainingSet->changeKind(m_kindId, header);
}

void KindHeaderCell::openMenu() {
	QMenu menu;

//	menu.addAction("Recompute dynamic", this, &WellHeaderCell::recomputeDynamic);
	menu.addAction("Remove", this, &KindHeaderCell::askDeleteSlot);

	menu.exec(QCursor::pos());
}

void KindHeaderCell::askDeleteSlot() {
	emit askDelete();
}

void KindHeaderCell::changeMin(double min) {
	m_maxSpinBox->setMinimum(min);
	BnniTrainingSet::BnniWellHeader header = m_trainingSet->wellHeaders().at(m_kindId);
	header.min = min;
	m_trainingSet->changeKind(m_kindId, header);
}

void KindHeaderCell::changeMax(double max) {
	m_minSpinBox->setMaximum(max);
	BnniTrainingSet::BnniWellHeader header = m_trainingSet->wellHeaders().at(m_kindId);
	header.max = max;
	m_trainingSet->changeKind(m_kindId, header);
}

TfpHeaderCell::TfpHeaderCell(BnniTrainingSet* trainingSet, QWidget *parent,
			Qt::WindowFlags f) : QWidget(parent) {
	m_trainingSet = trainingSet;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_nameLineEdit = new QLineEdit(m_trainingSet->tfpFilter());
	mainLayout->addWidget(m_nameLineEdit);

	connect(m_nameLineEdit, &QLineEdit::editingFinished, this, &TfpHeaderCell::changeName);
}

TfpHeaderCell::~TfpHeaderCell() {

}

void TfpHeaderCell::changeName() {
	QString tfpName = m_nameLineEdit->text();
	m_trainingSet->setTfpFilter(tfpName);
}

WellKindCell::WellKindCell(BnniTrainingSet* trainingSet, long wellId, long kindId,
			QWidget *parent, Qt::WindowFlags f) : QWidget(parent, f) {
	m_trainingSet = trainingSet;
	m_wellId = wellId;
	m_kindId = kindId;

	QHBoxLayout* mainLayout = new QHBoxLayout;
	setLayout(mainLayout);

	m_nameLabel = new QLabel();
	mainLayout->addWidget(m_nameLabel);

	m_menuButton = new QPushButton("...");
	mainLayout->addWidget(m_menuButton);

	updateName();

	connect(m_menuButton, &QPushButton::clicked, this, &WellKindCell::openMenu);
}

WellKindCell::~WellKindCell() {

}

long WellKindCell::wellId() const {
	return m_wellId;
}

long WellKindCell::kindId() const {
	return m_kindId;
}

void WellKindCell::updateName() {
	QString logName = m_trainingSet->wellBores().at(m_wellId).logsPathAndName.at(m_kindId).second;
	m_nameLabel->setText(logName);
}

void WellKindCell::openMenu() {
	QMenu menu;

	menu.addAction("Change log", this, &WellKindCell::askChangeLogSlot);

	menu.exec(QCursor::pos());
}

void WellKindCell::askChangeLogSlot() {
	emit askChangeLog();
}

WellTfpCell::WellTfpCell(BnniTrainingSet* trainingSet, long wellId,
			QWidget *parent, Qt::WindowFlags f) : QWidget(parent, f) {
	m_trainingSet = trainingSet;
	m_wellId = wellId;

	QHBoxLayout* mainLayout = new QHBoxLayout;
	setLayout(mainLayout);

	m_nameLabel = new QLabel();
	mainLayout->addWidget(m_nameLabel);

	m_menuButton = new QPushButton("...");
	mainLayout->addWidget(m_menuButton);

	updateName();

	connect(m_menuButton, &QPushButton::clicked, this, &WellTfpCell::askChangeTfpSlot);
}

WellTfpCell::~WellTfpCell() {

}

long WellTfpCell::wellId() const {
	return m_wellId;
}

void WellTfpCell::updateName() {
	QString tfpName;
	const BnniTrainingSet::WellParameter& wellData = m_trainingSet->wellBores().at(m_wellId);
	tfpName = wellData.tfpName;
	m_nameLabel->setText(tfpName);
}

void WellTfpCell::askChangeTfpSlot() {
	emit askChangeTfp();
}

LogSelectorTreeDialog::LogSelectorTreeDialog(const std::vector<WELLHEADDATA>& wellHead,
		QWidget *parent, Qt::WindowFlags f) : QDialog(parent, f), m_wellHead(wellHead) {
	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_treeWidget = new QTreeWidget;
	m_treeWidget->setColumnCount(1);
	mainLayout->addWidget(m_treeWidget);

	QList<QTreeWidgetItem*> wellHeadItems;
	for (int headIndex=0; headIndex<wellHead.size(); headIndex++) {
		QStringList strings;
		strings << wellHead[headIndex].tinyName;
		QTreeWidgetItem* headItem = new QTreeWidgetItem(static_cast<QTreeWidgetItem*>(nullptr), strings);
		headItem->setFlags(headItem->flags() & ~Qt::ItemIsSelectable);
		headItem->setData(0, Qt::UserRole, headIndex);

		for (int boreIndex=0; boreIndex<wellHead[headIndex].bore.size(); boreIndex++) {
			const WELLBOREDATA& wellBore = wellHead[headIndex].bore[boreIndex];

			QStringList boreStrings;
			boreStrings << wellBore.tinyName;
			QTreeWidgetItem* boreItem = new QTreeWidgetItem(headItem, boreStrings);
			boreItem->setData(0, Qt::UserRole, boreIndex);
		}

		wellHeadItems.push_back(headItem);
	}
	m_treeWidget->insertTopLevelItems(0, wellHeadItems);

	m_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok |
			QDialogButtonBox::Cancel);
	mainLayout->addWidget(m_buttonBox);

	connect(m_treeWidget, &QTreeWidget::currentItemChanged, this, &LogSelectorTreeDialog::treeSelectionChanged);

	connect(m_buttonBox, &QDialogButtonBox::accepted, this, &LogSelectorTreeDialog::tryAccept);
	connect(m_buttonBox, &QDialogButtonBox::rejected, this, &LogSelectorTreeDialog::reject);

	updateAcceptButtons();
}

LogSelectorTreeDialog::~LogSelectorTreeDialog() {

}

int LogSelectorTreeDialog::wellHeadIdx() const {
	return m_wellHeadIdx;
}

int LogSelectorTreeDialog::wellBoreIdx() const {
	return m_wellBoreIdx;
}

void LogSelectorTreeDialog::tryAccept() {
	if (m_wellHeadIdx>=0 && m_wellBoreIdx>=0) {// use indexes to see if a data is selected
		accept();
	}
}

void LogSelectorTreeDialog::updateAcceptButtons() {
	if (m_wellHeadIdx>=0 && m_wellBoreIdx>=0) {// use indexes to see if a data is selected
		m_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
	} else {
		m_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
	}
}

void LogSelectorTreeDialog::treeSelectionChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous) {
	if (current==nullptr || current->childCount()!=0) {
		m_wellHeadIdx = -1;
		m_wellBoreIdx = -1;
	} else {
		bool headValid, boreValid;
		m_wellHeadIdx = current->parent()->data(0, Qt::UserRole).toInt(&headValid);
		m_wellBoreIdx = current->data(0, Qt::UserRole).toInt(&boreValid);

		if (!headValid || !boreValid) {
			m_wellHeadIdx = -1;
			m_wellBoreIdx = -1;
		}
	}

	updateAcceptButtons();
}

SeismicCell::SeismicCell(BnniTrainingSet* trainingSet, long seismicId,
		QWidget *parent, Qt::WindowFlags f) : QWidget(parent, f) {
	m_trainingSet = trainingSet;
	m_seismicId = seismicId;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QHBoxLayout* nameLayout = new QHBoxLayout;
	mainLayout->addLayout(nameLayout);
	m_seismicLabel = new QLabel;
	nameLayout->addWidget(m_seismicLabel);

	m_menuButton = new QPushButton("...");
	nameLayout->addWidget(m_menuButton);

	QHBoxLayout* dynamicLayout = new QHBoxLayout;
	mainLayout->addLayout(dynamicLayout);

	dynamicLayout->addWidget(new QLabel("min:"));

	m_minSpinBox = new QDoubleSpinBox;
	m_minSpinBox->setMinimum(std::numeric_limits<float>::lowest());
	m_minSpinBox->setMaximum(m_trainingSet->seismics().at(m_seismicId).max);
	m_minSpinBox->setValue(m_trainingSet->seismics().at(m_seismicId).min);
	dynamicLayout->addWidget(m_minSpinBox);

	dynamicLayout->addWidget(new QLabel("max:"));

	m_maxSpinBox = new QDoubleSpinBox;
	m_maxSpinBox->setMinimum(m_trainingSet->seismics().at(m_seismicId).min);
	m_maxSpinBox->setMaximum(std::numeric_limits<float>::max());
	m_maxSpinBox->setValue(m_trainingSet->seismics().at(m_seismicId).max);
	dynamicLayout->addWidget(m_maxSpinBox);

	updateName();

	connect(m_menuButton, &QPushButton::clicked, this, &SeismicCell::openMenu);
	connect(m_minSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SeismicCell::changeMinInternal);
	connect(m_maxSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SeismicCell::changeMaxInternal);
}

SeismicCell::~SeismicCell() {

}

long SeismicCell::seismicId() const {
	return m_seismicId;
}

void SeismicCell::updateName() {
	m_seismicLabel->setText(m_trainingSet->seismics().at(m_seismicId).name);
}

void SeismicCell::openMenu() {
	QMenu menu;

	menu.addAction("Change seismic", this, &SeismicCell::askChangeSeismicSlot);

	menu.addAction("Remove", this, &SeismicCell::askDeleteSlot);

	menu.exec(QCursor::pos());
}

void SeismicCell::askDeleteSlot() {
	emit askDelete();
}

void SeismicCell::askChangeSeismicSlot() {
	emit askChangeSeismic();
}

void SeismicCell::changeMin(double min) {
	m_maxSpinBox->setMinimum(min);
	QSignalBlocker b(m_minSpinBox);
	m_minSpinBox->setValue(min);
}

void SeismicCell::changeMax(double max) {
	m_minSpinBox->setMaximum(max);
	QSignalBlocker b(m_maxSpinBox);
	m_maxSpinBox->setValue(max);
}

void SeismicCell::changeMinInternal(double min) {
	m_maxSpinBox->setMinimum(min);
	BnniTrainingSet::SeismicParameter param = m_trainingSet->seismics().at(m_seismicId);
	param.min = min;
	m_trainingSet->changeSeismic(m_seismicId, param);
}

void SeismicCell::changeMaxInternal(double max) {
	m_minSpinBox->setMaximum(max);
	BnniTrainingSet::SeismicParameter param = m_trainingSet->seismics().at(m_seismicId);
	param.max = max;
	m_trainingSet->changeSeismic(m_seismicId, param);
}

AutoCorrControlWidget::AutoCorrControlWidget(const std::vector<QString>& seismicNames,
		const std::vector<QString>& seismicPaths, QWidget* parent, Qt::WindowFlags f) :
			QWidget(parent, f) {
	if (seismicPaths.size()==seismicNames.size()) {
		m_seismicPaths = seismicPaths;
		m_seismicNames = seismicNames;
	} else {
		qDebug() << "AutoCorrControlWidget : invalid names and paths, size does not match";
	}

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_seismicComboBox = new QComboBox;
	for (int i=0; i<m_seismicNames.size(); i++) {
		m_seismicComboBox->addItem(m_seismicNames[i], QVariant(i));
	}
	if (m_seismicNames.size()>0) {
		m_currentSeismic = 0;
	}
	mainLayout->addWidget(m_seismicComboBox);

	m_seriesAutoCorr = new QLineSeries();
	m_seriesAutoCorr->setName("Auto Correlation");

	m_chartAutoCorr = new QChart();
	m_chartAutoCorr->setAnimationOptions(QChart::NoAnimation);

	m_chartAutoCorr->legend()->hide();
	m_chartAutoCorr->addSeries(m_seriesAutoCorr);
	m_chartAutoCorr->createDefaultAxes();
	m_chartAutoCorr->setTitle("Auto correlation chart");
	m_chartViewAutoCorr = new CustomChartView(m_chartAutoCorr);
	m_chartViewAutoCorr->setRenderHint(QPainter::Antialiasing);
	m_chartAutoCorr->setTitleBrush(QBrush(Qt::white));
	m_chartAutoCorr->axisX()->setLabelsBrush(QBrush(Qt::white));
	m_chartAutoCorr->axisX()->setTitleBrush(QBrush(Qt::white));
	m_chartAutoCorr->axisY()->setLabelsBrush(QBrush(Qt::white));
	m_chartAutoCorr->axisY()->setTitleBrush(QBrush(Qt::white));
	m_chartAutoCorr->setBackgroundBrush(QBrush(QRgb(0x31363b)));

	m_chartAutoCorr->createDefaultAxes();
	m_chartAutoCorr->axisX()->setLabelsBrush(QBrush(Qt::white));
	m_chartAutoCorr->axisX()->setTitleBrush(QBrush(Qt::white));
	m_chartAutoCorr->axisY()->setLabelsBrush(QBrush(Qt::white));
	m_chartAutoCorr->axisY()->setTitleBrush(QBrush(Qt::white));
	m_chartAutoCorr->axisY()->setRange(-110, 110);

	mainLayout->addWidget(m_chartViewAutoCorr);

	connect(m_seismicComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AutoCorrControlWidget::changeCurrentSeismic);

    //connect(m_seriesAutoCorr, &QLineSeries::clicked, m_chartViewAutoCorr, &CustomChartView::keepCallout);
    connect(m_seriesAutoCorr, &QLineSeries::hovered, m_chartViewAutoCorr, &CustomChartView::tooltip);

	computeAutoCorrelation();
}

AutoCorrControlWidget::~AutoCorrControlWidget() {

}

void AutoCorrControlWidget::changeCurrentSeismic(int index) {
	bool valid;
	int seismicIndex = m_seismicComboBox->itemData(index, Qt::UserRole).toInt(&valid);
	if (valid && m_currentSeismic!=seismicIndex) {
		m_currentSeismic = seismicIndex;
		computeAutoCorrelation();
	}
}

std::vector<double> autoCorrelation(double* src, int entries) {
    fftw_complex *obs;
    obs = (fftw_complex *) fftw_malloc( sizeof(fftw_complex)*entries);

    fftw_plan FFT, IFFT;
    FFT= fftw_plan_dft_1d(entries, obs,obs,FFTW_FORWARD,FFTW_ESTIMATE);
    IFFT= fftw_plan_dft_1d(entries, obs,obs,FFTW_BACKWARD,FFTW_ESTIMATE);

    for(int k=0;k<entries;k++){
      //READ IN THE DATA FROM THE FILE INTO THE FOURIER ARRAY.
      obs[k][0] = src[k];
      obs[k][1]=0.; //SET IMAGIANRY PART OF THE ARRAY TO ZERO.
    }


     //NEED TO SUBTRACT THE MEAN

      double mean=0.;
      for(int k=0;k<entries;k++) mean+=obs[k][0]/entries;
      //fprintf(stderr, "The mean of the data is: %e\n", mean);
      for(int k=0;k<entries;k++) {obs[k][0]-=mean; /*fprintf(stderr, "%d %e\n", k, obs[k][0]);*/}

     // CARRY OUT THE COMPUTATION OF THE AUTOCORRELATION VIA FOURIER TRANSFORM

      fftw_execute(FFT);

      for(int k=0;k<entries;k++) {
        double abssqr= obs[k][0]*obs[k][0] + obs[k][1]*obs[k][1];
        obs[k][0]=abssqr/entries;
        //THE FFTW LIBRARY DOES NOT NORMALIZE THE FOURIER TRANSFORM, SO THAT THE
        //INVERSE TRANSOFRM LEADS TO AN ADDITIONAL FACTOR OF N (number of entries
        //in the array) WHICH WE DIVIDE OUT ALREADY HERE 1/entries
        obs[k][1]=0.;
      }

      fftw_execute(IFFT);

      std::vector<double> out(entries);
      for(int k=0;k<entries;k++) out[k] = obs[k][0]/obs[0][0];

      fftw_destroy_plan(FFT);
      fftw_destroy_plan(IFFT);
      fftw_free(obs);

      return out;
}

template<typename DataType>
struct ComputeAutoCorrelationKernel {
	static std::vector<double> run(const QString& path, long dimX, long dimY, long dimZ, long headerSize, float oldSampleRate, float newSampleRate) {
		FILE* fp = fopen(path.toStdString().c_str(), "r");

		long readDimY = std::min((long) 100, dimY);
		long readOffsetY = (dimY - readDimY) / 2;
		long autoCorrDimX = dimX/2;
		long offsetX = (dimX - autoCorrDimX) / 2;
		long sectionSize = readDimY * dimX;
		std::vector<double> sectionBuffer; // in place because sizeof(DataType) < sizeof(double)
		sectionBuffer.resize(sectionSize);

		// resample variables
		int resampledDimX = std::floor((dimX - 1) * oldSampleRate / newSampleRate) + 1;
		std::vector<double> outputBuf;
		outputBuf.resize(resampledDimX);
		long resampledAutoCorrDimX = resampledDimX/2;
		long resampledOffsetX = (resampledDimX - resampledAutoCorrDimX) / 2;

		std::vector<double> cumulAutoCorr;
		cumulAutoCorr.resize(resampledAutoCorrDimX, 0);
		for (int part=0; part<=4; part++) {
			long z = std::min(part * dimZ / 4, dimZ-1);

			long offset = headerSize + sizeof(DataType) * dimX * (dimY * z + readOffsetY);

			fseek(fp, offset, SEEK_SET);
			fread(sectionBuffer.data(), sectionBuffer.size(), sizeof(DataType), fp);
			DataType* readBuf = static_cast<DataType*>(static_cast<void*>(sectionBuffer.data()));
			char tmp;
			for (long i=sectionSize-1; i>=0; i--) {
				char* beginPtr = static_cast<char*>(static_cast<void*>(readBuf+i));
				char* lastPtr = static_cast<char*>(static_cast<void*>(readBuf+i+1))-1;
				while (beginPtr<lastPtr) {
					tmp = *beginPtr;
					*beginPtr = *lastPtr;
					*lastPtr = tmp;
					beginPtr++;
					lastPtr--;
				}

				sectionBuffer[i] = readBuf[i];
			}


			for (long y=0; y<readDimY; y++) {
				int i = 1;
				while (i<autoCorrDimX && sectionBuffer[y*dimX+offsetX]==sectionBuffer[y*dimX+offsetX+i]) {
					i++;
				}
				bool traceValid = i<autoCorrDimX;
				if (traceValid) {
					resampleSpline(newSampleRate, oldSampleRate, sectionBuffer.data()+y*dimX, dimX, outputBuf.data(), resampledDimX);

					std::vector<double> autoCorr = autoCorrelation(outputBuf.data()+resampledOffsetX, resampledAutoCorrDimX);
					for (long x=0; x<resampledAutoCorrDimX; x++) {
						double tmp = cumulAutoCorr[x];
						cumulAutoCorr[x] += autoCorr[x];
					}
				}
			}
		}
		fclose(fp);

		return cumulAutoCorr;
	}
};

void AutoCorrControlWidget::computeAutoCorrelation() {
	if (m_currentSeismic<0 || m_sampleRate<=0) {
		return;
	}
	inri::Xt xt(m_seismicPaths[m_currentSeismic].toStdString());
	if (!xt.is_valid()) {
		return;
	}

	int dimX = xt.nSamples();
	int dimY = xt.nRecords();
	int dimZ = xt.nSlices();
	int headerSize = xt.header_size();
	float oldSampleRate = xt.stepSamples();

	float ratio = 1; //oldSampleRate/m_sampleRate;

	ImageFormats::QSampleType type = Seismic3DAbstractDataset::translateType(xt.type());

	SampleTypeBinder binder(type);
	std::vector<double> autoCorr = binder.bind<ComputeAutoCorrelationKernel>(
			m_seismicPaths[m_currentSeismic], dimX, dimY, dimZ, headerSize, oldSampleRate, m_sampleRate);

	double maxbis = autoCorr[0];//(autoCorr.size()-1) /2];
	m_seriesAutoCorr->clear();
	for (int x=0; x<autoCorr.size(); x++) {
		int i;
		if (x<(autoCorr.size()-1)/2) {
			i = (autoCorr.size()-1)/2 + 1 + x;
		} else {
			i = x - (autoCorr.size()-1)/2;
		}
		qDebug() << (x - ((int)autoCorr.size()-1)/2)*ratio << autoCorr[i]/maxbis*100.0;
		m_seriesAutoCorr->append((x-((int)autoCorr.size()-1)/2)*ratio, autoCorr[i]/maxbis*100.0);
	}
	m_chartAutoCorr->axisX()->setRange(-((int)autoCorr.size()-1)/2*ratio, ((int)autoCorr.size()-1)/2*ratio);
}

void AutoCorrControlWidget::setSampleRate(float newSampleRate) {
	m_sampleRate = newSampleRate;
	computeAutoCorrelation();
}
