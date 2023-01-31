#include "geneticalgorithmlauncher.h"
#include "densegeneticparamswidget.h"
#include "globalconfig.h"
#include "xgboostgeneticparamswidget.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QInputDialog>
#include <QMessageBox>
#include <QSizeGrip>

#include <QTemporaryFile>
#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QDebug>

#include <fstream>
#include <limits>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

typedef rapidjson::GenericDocument<rapidjson::ASCII<> > WDocument;
typedef rapidjson::GenericValue<rapidjson::ASCII<> > WValue;

QString GeneticAlgorithmLauncher::DENSE = "Dense";
QString GeneticAlgorithmLauncher::XGBOOST = "XgBoost";

GeneticAlgorithmLauncher::GeneticAlgorithmLauncher(QWidget* parent, Qt::WindowFlags f) : QWidget(parent, f) {
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle("BNNI Genetic");

    GlobalConfig& config = GlobalConfig::getConfig();
    m_plotDirPath = config.geneticPlotDirPath();
    m_csvDirPath = config.geneticShiftDirPath();

    QVBoxLayout* mainSizeGripLayout = new QVBoxLayout;
    mainSizeGripLayout->setContentsMargins(0, 0, 0, 0);
    setLayout(mainSizeGripLayout);

    QWidget* mainHolder = new QWidget;
    QVBoxLayout* mainLayout = new QVBoxLayout;
    mainHolder->setLayout(mainLayout);
    mainSizeGripLayout->addWidget(mainHolder);

    QFormLayout* formLayout = new QFormLayout;
    mainLayout->addLayout(formLayout);

    m_newTraingSetNameLineEdit = new QLineEdit(m_newTraingSetName);
    connect(m_newTraingSetNameLineEdit, &QLineEdit::editingFinished, this, &GeneticAlgorithmLauncher::newTrainingSetNameChanged);
    formLayout->addRow(new QLabel("New training set name"), m_newTraingSetNameLineEdit);

    m_nPipelineIterationSpinBox = new QSpinBox;
    m_nPipelineIterationSpinBox->setMinimum(1);
    m_nPipelineIterationSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_nPipelineIterationSpinBox->setValue(m_nPipelineIteration);

    connect(m_nPipelineIterationSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::nPipelineIterationChanged);

    formLayout->addRow(new QLabel("Number of pipeline iterations"), m_nPipelineIterationSpinBox);

    m_plotDirPathLineEdit = new QLineEdit(m_plotDirPath);
    connect(m_plotDirPathLineEdit, &QLineEdit::editingFinished, this, &GeneticAlgorithmLauncher::plotDirPathChanged);
    formLayout->addRow(new QLabel("Plots save dir"), m_plotDirPathLineEdit);

    m_csvDirPathLineEdit = new QLineEdit(m_csvDirPath);
    connect(m_csvDirPathLineEdit, &QLineEdit::editingFinished, this, &GeneticAlgorithmLauncher::csvDirPathChanged);
    formLayout->addRow(new QLabel("Shifts save dir"), m_csvDirPathLineEdit);

    QGroupBox* geneticGroupBox = new QGroupBox("Genetic parameters");
    QFormLayout* geneticFormLayout = new QFormLayout;
    geneticGroupBox->setLayout(geneticFormLayout);
    mainLayout->addWidget(geneticGroupBox);

    m_nIterationSpinBox = new QSpinBox;
    m_nIterationSpinBox->setMinimum(1);
    m_nIterationSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_nIterationSpinBox->setValue(m_nIteration);

    connect(m_nIterationSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::nIterationChanged);

    geneticFormLayout->addRow(new QLabel("Number of iterations on the population"), m_nIterationSpinBox);

    m_crossoverRatioSpinBox = new QDoubleSpinBox;
    m_crossoverRatioSpinBox->setMinimum(std::numeric_limits<float>::min());
    m_crossoverRatioSpinBox->setMaximum(1);
    m_crossoverRatioSpinBox->setValue(m_crossoverRatio);

    connect(m_crossoverRatioSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::crossoverRatioChanged);

    geneticFormLayout->addRow(new QLabel("Crossover ratio"), m_crossoverRatioSpinBox);

    m_numberOfChangesPerMutationSpinBox = new QSpinBox;
    m_numberOfChangesPerMutationSpinBox->setMinimum(1);
    m_numberOfChangesPerMutationSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_numberOfChangesPerMutationSpinBox->setValue(m_numberOfChangesPerMutation);

    connect(m_numberOfChangesPerMutationSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::numberOfChangesPerMutationChanged);

    geneticFormLayout->addRow(new QLabel("Number of changes per mutation"), m_numberOfChangesPerMutationSpinBox);

    m_rangeXSpinBox = new QSpinBox;
    m_rangeXSpinBox->setMinimum(0);
    m_rangeXSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_rangeXSpinBox->setValue(m_rangeX);

    connect(m_rangeXSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::rangeXChanged);

    geneticFormLayout->addRow(new QLabel("Half window X"), m_rangeXSpinBox);

    m_rangeYSpinBox = new QSpinBox;
    m_rangeYSpinBox->setMinimum(0);
    m_rangeYSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_rangeYSpinBox->setValue(m_rangeY);

    connect(m_rangeYSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::rangeYChanged);

    geneticFormLayout->addRow(new QLabel("Half window Y"), m_rangeYSpinBox);

    m_rangeDepthSpinBox = new QSpinBox;
    m_rangeDepthSpinBox->setMinimum(0);
    m_rangeDepthSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_rangeDepthSpinBox->setValue(m_rangeDepth);

    connect(m_rangeDepthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::rangeDepthChanged);

    geneticFormLayout->addRow(new QLabel("Half window depth"), m_rangeDepthSpinBox);

    m_populationSizeSpinBox = new QSpinBox;
    m_populationSizeSpinBox->setMinimum(1);
    m_populationSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_populationSizeSpinBox->setValue(m_populationSize);

    connect(m_populationSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::populationSizeChanged);

    geneticFormLayout->addRow(new QLabel("Population size"), m_populationSizeSpinBox);

    m_localSearchDepthSpinBox = new QSpinBox;
    m_localSearchDepthSpinBox->setMinimum(1);
    m_localSearchDepthSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_localSearchDepthSpinBox->setValue(m_localSearchDepth);

    connect(m_localSearchDepthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::localSearchDepthChanged);

    geneticFormLayout->addRow(new QLabel("Local search depth"), m_localSearchDepthSpinBox);

    m_fixTrainWellsCheckBox = new QCheckBox;
    m_fixTrainWellsCheckBox->setCheckState((m_fixTrainWells) ? Qt::Checked : Qt::Unchecked);
    connect(m_fixTrainWellsCheckBox, &QCheckBox::stateChanged, this, &GeneticAlgorithmLauncher::fixTrainWellsChanged);
    geneticFormLayout->addRow(new QLabel("Fix train wells"), m_fixTrainWellsCheckBox);

    m_usePostProcessingFilteringCheckBox = new QCheckBox;
    m_usePostProcessingFilteringCheckBox->setCheckState((m_usePostProcessingFiltering) ? Qt::Checked : Qt::Unchecked);
    connect(m_usePostProcessingFilteringCheckBox, &QCheckBox::stateChanged, this, &GeneticAlgorithmLauncher::usePostProcessingFilteringChanged);
    geneticFormLayout->addRow(new QLabel("Use post processing filtering"), m_usePostProcessingFilteringCheckBox);

    m_postProcessingFrequencySpinBox = new QDoubleSpinBox;
    m_postProcessingFrequencySpinBox->setMinimum(std::numeric_limits<float>::min());
    m_postProcessingFrequencySpinBox->setMaximum(std::numeric_limits<float>::max());
    m_postProcessingFrequencySpinBox->setValue(m_postProcessingFrequency);
    connect(m_postProcessingFrequencySpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::postProcessingFrequencyChanged);
    geneticFormLayout->addRow(new QLabel("Post processing filter frequency"), m_postProcessingFrequencySpinBox);

    m_modelComboBox = new QComboBox;
    m_modelComboBox->addItem(DENSE);
    m_modelComboBox->addItem(XGBOOST);
    mainLayout->addWidget(m_modelComboBox);

    m_xgboostWidget = new XgBoostGeneticParamsWidget(m_xgboostParams, "Xgboost parameters");
    mainLayout->addWidget(m_xgboostWidget);

    m_denseWidget = new DenseGeneticParamsWidget(m_denseParams, "Dense parameters");
    mainLayout->addWidget(m_denseWidget);

    if (m_useXgboost) {
        m_denseWidget->hide();
        m_modelComboBox->setCurrentText(XGBOOST);
    } else {
        m_xgboostWidget->hide();
        m_modelComboBox->setCurrentText(DENSE);
    }

    connect(m_modelComboBox, &QComboBox::currentTextChanged, this, &GeneticAlgorithmLauncher::changeModel);

    QGroupBox* dataGroupBox = new QGroupBox("Data parameters");
    QFormLayout* dataFormLayout = new QFormLayout;
    dataGroupBox->setLayout(dataFormLayout);
    mainLayout->addWidget(dataGroupBox);

    m_halfWindowSizeSpinBox = new QSpinBox;
    m_halfWindowSizeSpinBox->setMinimum(1);
    m_halfWindowSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_halfWindowSizeSpinBox->setValue(m_halfWindowSize);

    connect(m_halfWindowSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::halfWindowSizeChanged);

    dataFormLayout->addRow(new QLabel("Seismic half window"), m_halfWindowSizeSpinBox);

    m_splitWellToVerticalPartsCheckBox = new QCheckBox;
    m_splitWellToVerticalPartsCheckBox->setCheckState((m_splitWellToVerticalParts) ? Qt::Checked : Qt::Unchecked);
    connect(m_splitWellToVerticalPartsCheckBox, &QCheckBox::stateChanged, this, &GeneticAlgorithmLauncher::splitWellToVerticalPartsChanged);
    dataFormLayout->addRow(new QLabel("Split wells into multiple vertical parts"), m_splitWellToVerticalPartsCheckBox);

    m_minimumSubWellSizeSpinBox = new QSpinBox;
    m_minimumSubWellSizeSpinBox->setMinimum(1);
    m_minimumSubWellSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_minimumSubWellSizeSpinBox->setValue(m_minimumSubWellSize);

    connect(m_minimumSubWellSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::minimumSubWellSizeChanged);

    dataFormLayout->addRow(new QLabel("Minimum sub well part"), m_minimumSubWellSizeSpinBox);

    m_maximumSubWellSizeSpinBox = new QSpinBox;
    m_maximumSubWellSizeSpinBox->setMinimum(1);
    m_maximumSubWellSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_maximumSubWellSizeSpinBox->setValue(m_maximumSubWellSize);

    connect(m_maximumSubWellSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GeneticAlgorithmLauncher::maximumSubWellSizeChanged);

    dataFormLayout->addRow(new QLabel("Maximum sub well part"), m_maximumSubWellSizeSpinBox);



    QPushButton* runButton = new QPushButton("Run Genetic Algorithm");
    connect(runButton, &QPushButton::clicked, this, &GeneticAlgorithmLauncher::run);
    mainLayout->addWidget(runButton);

    QPushButton* runCrossButton = new QPushButton("Run Cross Genetic Algorithm");
    connect(runCrossButton, &QPushButton::clicked, this, &GeneticAlgorithmLauncher::runCross);
    mainLayout->addWidget(runCrossButton);

    QSizeGrip* sizeGrip = new QSizeGrip(this);
    sizeGrip->setContentsMargins(0, 0, 0, 0);
    mainSizeGripLayout->addWidget(sizeGrip, 0, Qt::AlignRight);

    m_process = new QProcess;
    m_process->setProcessChannelMode(QProcess::ForwardedChannels);
}

void GeneticAlgorithmLauncher::setProgramLocation(const QString& dirPath) {
    m_programLocation = dirPath;

    if (m_processRunning) {
        QMessageBox::warning(this, tr("Program location changed while a process is running"), tr("Program location changed while a process is running. Beware, the active process use the old program."));
    }
}

void GeneticAlgorithmLauncher::setTrainingSet(const QString& trainingSetJsonFile) {
	m_trainingSetJsonFile = trainingSetJsonFile;

	m_newTraingSetNameLineEdit->setText("genetic_" + QFileInfo(trainingSetJsonFile).dir().dirName());

    if (m_processRunning) {
        QMessageBox::warning(this, tr("Training set changed while a process is running"), tr("Training set location changed while a process is running. Beware, the active process use the old program."));
    }
}

void GeneticAlgorithmLauncher::run() {
	if (!m_fixTrainWells) {
		QString goOnItem = "Proceed";
		QString stopItem = "Stop";
		QStringList items;
		items << goOnItem << stopItem;
		QString text = QInputDialog::getItem(this, tr("Warning : train wells not fixed"), tr("Train well are not fixed, this will allow the genetic algorithm to modify the train wells position. Do you wish to proceed ?"), items);
		if (text.compare(stopItem)==0) {
			return;
		}
	}

	runProgram("launchWithEnv_genetic.sh");
}

void GeneticAlgorithmLauncher::runCross() {
	if (m_fixTrainWells) {
		QString goOnItem = "Proceed";
		QString stopItem = "Stop";
		QStringList items;
		items << goOnItem << stopItem;
		QString text = QInputDialog::getItem(this, tr("Warning : train wells fixed"), tr("Fix train well option selected but it will be ignored by the cross algorithm. Do you wish to proceed ?"), items);
		if (text.compare(stopItem)==0) {
			return;
		}
	}

	runProgram("launchWithEnv_crossGenetic.sh");
}

void GeneticAlgorithmLauncher::runProgram(const QString& programName) {
    if (m_processRunning) {
        return;
    }

    m_cachedProgramLocation = m_programLocation;
    m_cachedTrainingSetJsonFile = m_trainingSetJsonFile;
    m_cacheGeneticProgramName = programName;

    GlobalConfig& config = GlobalConfig::getConfig();
    QDir tempDir = QDir(config.tempDirPath());

    m_geneticInputJsonFile = getTemporaryFilePath(tempDir.absoluteFilePath("BNNI_trainingset_XXXXXXXX.json"));
    m_geneticInputBufferFile = getTemporaryFilePath(tempDir.absoluteFilePath("BNNI_buffer_XXXXXXXX.npy"));
    m_removeWellsFile = getTemporaryFilePath(tempDir.absoluteFilePath("BNNI_remove_wells_XXXXXXXX.txt"));

    fillRemoveWellsFile(m_removeWellsFile);

    QString program = QDir(m_cachedProgramLocation).absoluteFilePath("launchWithEnv_formatCroppedDataForGenetic.sh");
    QStringList arguments;
    arguments << "--input" << m_cachedTrainingSetJsonFile << "--output_json" << m_geneticInputJsonFile <<
    		"--output_buffer" << m_geneticInputBufferFile << "--x_half_window" << QString::number(m_rangeDepth + m_halfWindowSize) <<
			"--y_half_window" << QString::number(m_rangeX) << "--z_half_window" << QString::number(m_rangeY) <<
			"--ignore_wells_file" << m_removeWellsFile;

    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &GeneticAlgorithmLauncher::geneticInputGenerated);
    connect(m_process, QOverload<QProcess::ProcessError>::of(&QProcess::errorOccurred), this, &GeneticAlgorithmLauncher::geneticInputGotError);

    qDebug() << "Generate genetic input : " <<program << arguments;

    m_processRunning = true;
    m_process->start(program, arguments);
}

QString GeneticAlgorithmLauncher::getTemporaryFilePath(const QString& templatePath) {
    QTemporaryFile tempFile(templatePath);
    tempFile.setAutoRemove(false);
    tempFile.open();
    tempFile.close();

    return tempFile.fileName();
}

void GeneticAlgorithmLauncher::geneticInputGenerated(int exitCode, QProcess::ExitStatus exitStatus) {
	disconnect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &GeneticAlgorithmLauncher::geneticInputGenerated);
	disconnect(m_process, QOverload<QProcess::ProcessError>::of(&QProcess::errorOccurred), this, &GeneticAlgorithmLauncher::geneticInputGotError);

	createGeneticConfig();

	QString program = QDir(m_cachedProgramLocation).absoluteFilePath(m_cacheGeneticProgramName);
	QStringList arguments;
	arguments << "--config" << m_geneticConfigFile;

	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &GeneticAlgorithmLauncher::geneticFinished);
	connect(m_process, QOverload<QProcess::ProcessError>::of(&QProcess::errorOccurred), this, &GeneticAlgorithmLauncher::geneticGotError);

	qDebug() << "Generate genetic input : " <<program << arguments;

	m_cachedTime = QDateTime::currentDateTime();
	qDebug()<<"time "<<m_cachedTime;
	m_process->start(program, arguments);
}

void GeneticAlgorithmLauncher::geneticInputGotError(QProcess::ProcessError error) {
	disconnect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &GeneticAlgorithmLauncher::geneticInputGenerated);
	disconnect(m_process, QOverload<QProcess::ProcessError>::of(&QProcess::errorOccurred), this, &GeneticAlgorithmLauncher::geneticInputGotError);

	cleanup();
	m_processRunning = false;

	QMessageBox::warning(this, "Failed to generate input", "The needed files to launch the genetic algorithm could not be generated.");
}

void GeneticAlgorithmLauncher::geneticFinished(int exitCode, QProcess::ExitStatus exitStatus) {
	disconnect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &GeneticAlgorithmLauncher::geneticFinished);
	disconnect(m_process, QOverload<QProcess::ProcessError>::of(&QProcess::errorOccurred), this, &GeneticAlgorithmLauncher::geneticGotError);

	QDir oldTraingSetDir = QFileInfo(m_trainingSetJsonFile).dir();
	oldTraingSetDir.cdUp();
	QString newTrainingSetDirPath = oldTraingSetDir.absoluteFilePath(m_newTraingSetName);
	newTrainingSetDirPath = newTrainingSetDirPath + "/trainingset.json";

	QString csvPath = getCsvPath();
	if (csvPath.isNull() || csvPath.isEmpty() || !QFile::exists(csvPath)) {
		m_processRunning = false;
		cleanup();
		QMessageBox::warning(this, "Failed to find shift file", "Could not find the shift file.");
		return;
	}

	QString program = QDir(m_cachedProgramLocation).absoluteFilePath("launchWithEnv_createTrainingSetFromShifts.sh");
	QStringList arguments;
	arguments << "--shifts" << csvPath  << "--output_json" << newTrainingSetDirPath << "--input_buffer" <<
			m_geneticInputBufferFile << "--input_json" << m_geneticInputJsonFile;

	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &GeneticAlgorithmLauncher::newTraininSetFinished);
	connect(m_process, QOverload<QProcess::ProcessError>::of(&QProcess::errorOccurred), this, &GeneticAlgorithmLauncher::newTrainingSetGotError);

	qDebug() << "Generate genetic input : " <<program << arguments;

	QDir outputDir = QFileInfo(newTrainingSetDirPath).dir();
	if (!outputDir.exists()) {
		QString name = outputDir.dirName();
		outputDir.cdUp();
		outputDir.mkdir(name);
	}

	m_process->start(program, arguments);
}

void GeneticAlgorithmLauncher::geneticGotError(QProcess::ProcessError error) {
	disconnect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &GeneticAlgorithmLauncher::geneticFinished);
	disconnect(m_process, QOverload<QProcess::ProcessError>::of(&QProcess::errorOccurred), this, &GeneticAlgorithmLauncher::geneticGotError);

	cleanup();
	m_processRunning = false;

	QMessageBox::warning(this, "Failed to run genetic", "The genetic algorithm did not run correctly.");
}

void GeneticAlgorithmLauncher::newTraininSetFinished(int exitCode, QProcess::ExitStatus exitStatus) {
	disconnect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &GeneticAlgorithmLauncher::newTraininSetFinished);
	disconnect(m_process, QOverload<QProcess::ProcessError>::of(&QProcess::errorOccurred), this, &GeneticAlgorithmLauncher::newTrainingSetGotError);

	cleanup();
	m_processRunning = false;
	QMessageBox::information(this, "success", "Success");
}

void GeneticAlgorithmLauncher::newTrainingSetGotError(QProcess::ProcessError error) {
	disconnect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &GeneticAlgorithmLauncher::newTraininSetFinished);
	disconnect(m_process, QOverload<QProcess::ProcessError>::of(&QProcess::errorOccurred), this, &GeneticAlgorithmLauncher::newTrainingSetGotError);

	cleanup();
	m_processRunning = false;
	QMessageBox::warning(this, "Failed to generate trainingset", "The shift file could not be converted into a trainingset");
}

void GeneticAlgorithmLauncher::createGeneticConfig() {
	GlobalConfig& config = GlobalConfig::getConfig();
	QDir tempDir = QDir(config.tempDirPath());
	m_geneticConfigFile = getTemporaryFilePath(tempDir.absoluteFilePath("BNNI_config_XXXXXXXX.yaml"));

	std::vector<int> trainWellIdxs = getTrainWellIdxs();

	QFile configFile(m_geneticConfigFile);
    if (!m_geneticConfigFile.isNull() && !m_geneticConfigFile.isEmpty() && configFile.open(QIODevice::WriteOnly)) {
        QTextStream stream(&configFile);

        stream << "data_config:" << "\n";
        stream << "  cube_data_path: " << m_geneticInputBufferFile << "\n";
        stream << "  json_logs_data_path: " << m_geneticInputJsonFile << "\n";
        stream << "  half_window_size: " << QString::number(m_halfWindowSize) << "\n";
        stream << "  n_inputs: " << QString::number(m_nInputs) << "\n";
        stream << "  split_well_to_vertical_parts: " << ((m_splitWellToVerticalParts) ? "True" : "False") << "\n";
        stream << "  minimum_sub_well_size : " << QString::number(m_minimumSubWellSize) << "\n";
        stream << "  maximum_sub_well_size : " << QString::number(m_maximumSubWellSize) << "\n";
        stream << "  training_wells_indexes : " << "\n";
        for (int i=0; i<trainWellIdxs.size(); i++) {
            stream << "    - " << QString::number(trainWellIdxs[i]) << "\n";
    	}
        stream << "model_config:" << "\n";
        if (m_useXgboost) {
		    stream << "  blocks:" << "\n";
		    stream << "  - block_type: xgboost" << "\n";
		    stream << "    xgboost:" << "\n";
		    stream << "      max_depth: " << QString::number(m_xgboostParams.maxDepth()) << "\n";
		    stream << "      n_estimators: " << QString::number(m_xgboostParams.nEstimators()) << "\n";
		    stream << "      eta: " << QString::number(m_xgboostParams.learningRate()) << "\n";
		    stream << "      subsample: " << QString::number(m_xgboostParams.subsample()) << "\n";
		    stream << "      colsample_bytree: " << QString::number(m_xgboostParams.colsampleByTree()) << "\n";
		    stream << "  model_name: Xgboost" << "\n";
        } else {
            stream << "  blocks:" << "\n";

            Activation defaultActivation = m_denseParams.activation();
            QString defaultActivationName;
            switch(defaultActivation) {
            case Activation::sigmoid:
                defaultActivationName = "sigmoid";
                break;
            case Activation::relu:
                defaultActivationName = "relu";
                break;
            case Activation::selu:
                defaultActivationName = "selu";
                break;
            default:
                defaultActivationName = "linear";
                break;
            }
            double dropout = m_denseParams.useDropout() ? m_denseParams.dropout() : 0.0;
            QString defaultNormalisationName = m_denseParams.useNormalisation() ? "true" : "false";

            QVector<unsigned int> layerSizes = m_denseParams.layerSizes();
            for (int i=0; i<layerSizes.size(); i++) {
                QString activationName;
                QString normalisationName;
                if (i==layerSizes.size()-1) {
                    activationName = "linear";
                    normalisationName = "false";
                } else {
                    activationName = defaultActivationName;
                    normalisationName = defaultNormalisationName;
                }
                stream << "  - block_type: dense" << "\n";
                stream << "    dense:" << "\n";
                stream << "      activation_name: " << activationName << "\n";
                stream << "      dropout: " << QString::number(dropout) << "\n";
                stream << "      layer_width: " << QString::number(layerSizes[i]) << "\n";
                stream << "      normalisation: " << normalisationName << "\n";
            }
            stream << "  model_name: Basic_ANN" << "\n";
        }
        stream << "genetic_config:" << "\n";
        stream << "  num_iterations: " << QString::number(m_nIteration) << "\n";
        stream << "  crossover_ratio: " << QString::number(m_crossoverRatio) << "\n";
        stream << "  number_of_changes_per_mutation: " << QString::number(m_numberOfChangesPerMutation) << "\n";
        stream << "  range_x: " << QString::number(m_rangeX) << "\n";
        stream << "  range_y: " << QString::number(m_rangeY) << "\n";
        stream << "  range_z: " << QString::number(m_rangeDepth) << "\n";
        stream << "  population_size: " << QString::number(m_populationSize) << "\n";
        stream << "  local_search_depth: " << QString::number(m_localSearchDepth) << "\n";
        stream << "  fix_train_wells: " << ((m_fixTrainWells) ? "True" : "False") << "\n";
        stream << "  use_postprocessing_filter: " << ((m_usePostProcessingFiltering) ? "True" : "False") << "\n";
        stream << "  postprocessing_filter_freq: " << QString::number(m_postProcessingFrequency) << "\n";
        stream << "plots_path: " << m_plotDirPath << "\n";
        stream << "output_path: " << m_csvDirPath << "\n";
        stream << "pipeline_itteration_number: " << QString::number(m_nPipelineIteration) << "\n";
    }
}

void GeneticAlgorithmLauncher::nPipelineIterationChanged(int val) {
	m_nPipelineIteration = val;
}

void GeneticAlgorithmLauncher::plotDirPathChanged() {
	m_plotDirPath = m_plotDirPathLineEdit->text();
}

void GeneticAlgorithmLauncher::csvDirPathChanged() {
	m_csvDirPath = m_csvDirPathLineEdit->text();
}

void GeneticAlgorithmLauncher::usePostProcessingFilteringChanged(int state) {
	m_usePostProcessingFiltering = state==Qt::Checked;
}

void GeneticAlgorithmLauncher::postProcessingFrequencyChanged(double val) {
	m_postProcessingFrequency = val;
}

void GeneticAlgorithmLauncher::nIterationChanged(int val) {
	m_nIteration = val;
}

void GeneticAlgorithmLauncher::crossoverRatioChanged(double val) {
	m_crossoverRatio = val;
}

void GeneticAlgorithmLauncher::numberOfChangesPerMutationChanged(int val) {
	m_numberOfChangesPerMutation = val;
}

void GeneticAlgorithmLauncher::rangeXChanged(int val) {
	m_rangeX = val;
}

void GeneticAlgorithmLauncher::rangeYChanged(int val) {
	m_rangeY = val;
}

void GeneticAlgorithmLauncher::rangeDepthChanged(int val) {
	m_rangeDepth = val;
}

void GeneticAlgorithmLauncher::populationSizeChanged(int val) {
	m_populationSize = val;
}

void GeneticAlgorithmLauncher::localSearchDepthChanged(int val) {
	m_localSearchDepth = val;
}

void GeneticAlgorithmLauncher::fixTrainWellsChanged(int val) {
	m_fixTrainWells = val==Qt::Checked;
}

void GeneticAlgorithmLauncher::halfWindowSizeChanged(int val) {
	m_halfWindowSize = val;
}

void GeneticAlgorithmLauncher::splitWellToVerticalPartsChanged(int val) {
	m_splitWellToVerticalParts = val==Qt::Checked;
}

void GeneticAlgorithmLauncher::minimumSubWellSizeChanged(int val) {
	m_minimumSubWellSize = val;
}

void GeneticAlgorithmLauncher::maximumSubWellSizeChanged(int val) {
	m_maximumSubWellSize = val;
}

std::vector<int> GeneticAlgorithmLauncher::getTrainWellIdxs() {
	std::vector<int> trainIdxs;

	std::ifstream ifs(m_geneticInputJsonFile.toStdString().c_str());
	rapidjson::IStreamWrapper isw(ifs);
	WDocument document;
	document.ParseStream(isw);

	bool isValid = true;
	if (!document.IsObject()) {
		qWarning() << tr("Unexpected format, could not get root object");
		isValid = false;
	}
	long wellIdx = 0;
	if (isValid && document.HasMember("samples") && document["samples"].IsObject()) {
		auto it=document["samples"].MemberBegin();
		while (it!=document["samples"].MemberEnd()) {
			bool wellValidForCount = isWellValidForGenetic(*it, isValid);

			if (isValid && wellValidForCount) {
				// find if the well is a train well
				QString wellName = QString::fromStdString(it->name.GetString());
				std::vector<int>::iterator it = std::find_if(m_trainIdx.begin(), m_trainIdx.end(), [this, wellName](const int& trainIdx) {
					QString trainWellName = m_wells[trainIdx].name;

					trainWellName = "Sismage2 MainIDL:DmWell/ExistingWellBoreFactory:1.0" + trainWellName;
					return trainWellName.compare(wellName)==0;
				});

				bool useForTraining = it!=m_trainIdx.end();
				if (useForTraining) {
					trainIdxs.push_back(wellIdx);
				}

				wellIdx += 1;
			}

			it++;
		}
	}

	if (!isValid) {
		trainIdxs.clear();
	}

	return trainIdxs;
}

bool GeneticAlgorithmLauncher::isWellValidForGenetic(const WMember& well, bool& isValid) {
	isValid = well.value.IsArray();
	bool wellValidForCount = false;
	if (!isValid) {
		return false;
	}
	long wellSize = well.value.Size();

	// this part mimic how the genetic program test the wells to give the right number for training wells
	// see core/seismic_inversion/data/data_loader_well_logs.py InferenceLogDataLoader:extract_well_log_positions
	if (!m_splitWellToVerticalParts) {
		wellValidForCount = wellSize>=m_minimumSubWellSize;
	} else if (wellSize>=m_minimumSubWellSize && wellSize>0) { // check that is is not too small
		// Search if a vertical sub part of the well has at least a size of m_minimumSubWellSize

		// check if the sample si valid
		isValid = well.value[0].IsArray() && well.value[0].Size()==4 && well.value[0][0].IsArray() && well.value[0][0].Size()==3 &&
				(well.value[0][0][0].IsDouble() || well.value[0][0][0].IsInt()) && (well.value[0][0][1].IsDouble() || well.value[0][0][1].IsInt());

		// define the top position of the vertical part
		long idx = 1;
		long topZ;
		if (well.value[0][0][0].IsDouble()) {
			topZ = well.value[0][0][0].GetDouble();
		} else {
			topZ = well.value[0][0][0].GetInt();
		}
		long topY;
		if (well.value[0][0][1].IsDouble()) {
			topY = well.value[0][0][1].GetDouble();
		} else {
			topY = well.value[0][0][1].GetInt();
		}
		long topIndex = 0;
		while (isValid && !wellValidForCount && idx<wellSize) {
			isValid = well.value[idx].IsArray() && well.value[idx].Size()==4 && well.value[idx][0].IsArray() && well.value[idx][0].Size()==3 &&
					(well.value[idx][0][0].IsDouble() || well.value[idx][0][0].IsInt()) && (well.value[idx][0][1].IsDouble() || well.value[idx][0][1].IsInt());

			// test is the sample is still in the same vertical part of the well
			long currentZ;
			if (isValid && well.value[idx][0][0].IsDouble()) {
				currentZ = well.value[idx][0][0].GetDouble();
			} else if (isValid) {
				currentZ = well.value[idx][0][0].GetInt();
			}
			long currentY;
			if (isValid && well.value[idx][0][1].IsDouble()) {
				currentY = well.value[idx][0][1].GetDouble();
			} else if (isValid) {
				currentY = well.value[idx][0][1].GetInt();
			}
			if (isValid && (currentY!=topY || currentZ!=topZ)) {
				// sample is in another vertical part, test the size and switch to the new vertical part
				wellValidForCount = idx-topIndex>=m_minimumSubWellSize;
				// define the top position of the vertical part
				topY = currentY;
				topZ = currentZ;
				topIndex = idx;
			}
			if (!wellValidForCount) {
				idx++;
			}
		}
		if (isValid && !wellValidForCount) {
			wellValidForCount = idx-topIndex>=m_minimumSubWellSize;
		}
	}
	return wellValidForCount;
}

void GeneticAlgorithmLauncher::setWells(const QVector<Well>& wells, const std::vector<int>& trainIdx,
			const std::vector<int>& validationIdx, const std::vector<int>& blindIdx) {
	m_wells = wells;
	m_trainIdx = trainIdx;
	m_validationIdx = validationIdx;
	m_blindIdx = blindIdx;
}

void GeneticAlgorithmLauncher::setHalfWindow(int halfWindow) {
	if (m_halfWindowSize!=halfWindow && halfWindow>0) {
		m_halfWindowSize = halfWindow;
		QSignalBlocker b(m_halfWindowSizeSpinBox);
		m_halfWindowSizeSpinBox->setValue(m_halfWindowSize);
	}
}

void GeneticAlgorithmLauncher::setLayerSizes(QVector<unsigned int> array) {
	m_denseParams.setLayerSizes(array);
}

void GeneticAlgorithmLauncher::setNumInputSeismics(int val) {
	m_nInputs = val;
}

void GeneticAlgorithmLauncher::newTrainingSetNameChanged() {
	m_newTraingSetName = m_newTraingSetNameLineEdit->text();
}

QString GeneticAlgorithmLauncher::getCsvPath() {
	QString csvPath;

	qDebug()<<"time "<<m_cachedTime;

	int i=0;
	bool found = false;
	while (!found && i<10) {
		QDateTime time(m_cachedTime);
		time = time.addSecs(60*i);
		qDebug()<<"time "<<time;
		QString suffix = time.toString("_dd_MM_yyyy_hh_mm_AP").toUpper();

		csvPath = m_csvDirPath + suffix + "/output.csv";
		found = QFile::exists(csvPath);
		if (!found) {
			csvPath = "";
		}
		i++;
	}

	return csvPath;
}

void GeneticAlgorithmLauncher::cleanup() {
    if (QFile::exists(m_geneticInputJsonFile)) {
    	QFile::remove(m_geneticInputJsonFile);
    }

    if (QFile::exists(m_geneticInputBufferFile)) {
    	QFile::remove(m_geneticInputBufferFile);
    }

    if (QFile::exists(m_geneticConfigFile)) {
    	QFile::remove(m_geneticConfigFile);
    }

    if (QFile::exists(m_removeWellsFile)) {
        QFile::remove(m_removeWellsFile);
    }
}

void GeneticAlgorithmLauncher::fillRemoveWellsFile(const QString& filePath) {
	std::vector<int> wellsToRemove;
	wellsToRemove.reserve(m_wells.size());
	for (int wellIdx = 0; wellIdx<m_wells.size(); wellIdx++) {
		std::vector<int>::iterator itTrain = std::find(m_trainIdx.begin(), m_trainIdx.end(), wellIdx);
		if (itTrain!=m_trainIdx.end()) {
			continue;
		}
		std::vector<int>::iterator itBlind = std::find(m_blindIdx.begin(), m_blindIdx.end(), wellIdx);
		if (itBlind!=m_blindIdx.end()) {
			continue;
		}

		std::vector<int>::iterator itValidation = std::find(m_validationIdx.begin(), m_validationIdx.end(), wellIdx);
		if (itValidation==m_validationIdx.end()) {
			wellsToRemove.push_back(wellIdx);
		}
	}

	QFile wellsFile(m_removeWellsFile);
	if (!m_removeWellsFile.isNull() && !m_removeWellsFile.isEmpty() && wellsFile.open(QIODevice::WriteOnly)) {
		QTextStream stream(&wellsFile);

		for (int i=0; i<wellsToRemove.size(); i++) {
			stream << "Sismage2 MainIDL:DmWell/ExistingWellBoreFactory:1.0" + m_wells[wellsToRemove[i]].name << "\n";
		}
	}
}

void GeneticAlgorithmLauncher::changeModel(QString modelName) {
    m_useXgboost = modelName.compare(XGBOOST)==0;
    if (m_useXgboost) {
        m_denseWidget->hide();
        m_xgboostWidget->show();
    } else {
        m_xgboostWidget->hide();
        m_denseWidget->show();
    }
}

