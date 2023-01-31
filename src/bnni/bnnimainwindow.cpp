#include "bnnimainwindow.h"
#include "ui_bnnimainwindow.h"

/*#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>
#include <QJsonValueRef>*/
#include <QString>
#include <QFile>
#include <QFileDialog>
#include <QDebug>
#include <QIODevice>
#include <QMessageBox>
#include <QToolButton>
#include <QAction>
#include <QListWidgetItem>
#include <QProcess>
#include <QFormLayout>
#include <QSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QPalette>
#include <QComboBox>
#include <QDateTime>
#include <QInputDialog>
#include <QSettings>

#include <stdio.h>
#include <fstream>
#include <cmath>

//#include <rapidjson/filestream.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <nlohmann/json.hpp>
#include <string>
#include <cmath>

#include "wellviewer.h"
#include "bnniconfig.h"
#include "bnniconfigcreationdialog.h"
#include "collapsablescrollarea.h"
#include "functionselector.h"
#include "generalizationsectionwidget.h"
//#include "generalizationcartewidget.h"
//#include "demobnni.h"
#include "geneticalgorithmlauncher.h"
#include "networkparameterform.h"
#include "networkparametersmodel.h"
#include "advancedparameterform.h"
#include "surveyselectiondialog.h"
#include "computelogdialog.h"
#include "algorithm.h"
#include "bnniubjsondecoder.h"
#include "configreader.h"

#include "Xt.h"

const QLatin1String LAST_PROJECT_PATH_IN_SETTINGS("BnniMainWindow/lastProjectPath");
const QLatin1String LAST_TRAININGSET_PATH_IN_SETTINGS("BnniMainWindow/lastTrainingSetPath");
const QLatin1String LAST_CONFIG_PATH_IN_SETTINGS("BnniMainWindow/lastExperimentPath");

typedef rapidjson::GenericDocument<rapidjson::ASCII<> > WDocument;
typedef rapidjson::GenericValue<rapidjson::ASCII<> > WValue;

float BnniMainWindow::DEFAULT_SAMPLE_RATE = 0.5f;

BnniMainWindow::BnniMainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::BnniMainWindow)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);

    ui->unusedTotrain->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(Qt::red).name()));
    ui->trainToUnused->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(Qt::red).name()));
    ui->unusedToVal->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(255, 100, 0).name()));
    ui->valToUnused->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(255, 100, 0).name()));
    ui->unusedToBlind->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(Qt::green).name()));
    ui->blindToUnused->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(Qt::green).name()));


    ui->valToTrain->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(Qt::red).name()));
    ui->trainToVal->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(255, 100, 0).name()));
    ui->blindToVal->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(255, 100, 0).name()));
    ui->valToBlind->setStyleSheet(QString("QToolButton{ background: %1; }").arg(QColor(Qt::green).name()));

    ui->trainLabel->setStyleSheet(QString("QLabel{ color: %1; }").arg(QColor(Qt::red).name()));
    ui->valLabel->setStyleSheet(QString("QLabel{ color: %1; }").arg(QColor(255, 100, 0).name()));
    ui->blindLabel->setStyleSheet(QString("QLabel{ color: %1; }").arg(QColor(Qt::green).name()));
    // connect json button
    /*connect(ui->loadJson, &QPushButton::clicked, [this]() {
        this->loadJson();
    });*/

    m_bnniConfig = new BnniConfig(this);

    connect(ui->loadProject, &QPushButton::clicked, [this]() {
        /*QString searchDir = "/data";
        if (!surveyDir.isNull() && !surveyDir.isEmpty()) {
            QFileInfo fileInfo(surveyDir);
            if (fileInfo.isDir()) {
                searchDir = fileInfo.dir().absolutePath();
            }
        }*/

        //QString output = QFileDialog::getExistingDirectory(this, QString(), searchDir);
        SurveySelectionDialog dialog(this);
        int code = dialog.exec();
        QString _project = dialog.getProject();
        QString _projectDir = dialog.getDirProject();

        if (!_project.isNull() && !_project.isEmpty()) {
            setProject(_projectDir + "/" + _project);
        }
    });

    connect(ui->jsonComboBox, SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &BnniMainWindow::loadTrainingSet);

    connect(ui->configComboBox, SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &BnniMainWindow::changeConfig);

    connect(ui->createConfigButton, &QPushButton::clicked, this, &BnniMainWindow::createNewConfig);

    // connect run and stop button
    connect(ui->launcher, &QPushButton::clicked, this, &BnniMainWindow::launchProcess);
    connect(ui->stopProcess, &QPushButton::clicked, this, &BnniMainWindow::stopProcess);

    connect(ui->replayButton, &QPushButton::clicked, this, &BnniMainWindow::replayLaunchProcess);

    // connect list widgets
    connect(ui->trainToUnused, &QToolButton::clicked, this, &BnniMainWindow::moveTrainToUnused);
    connect(ui->trainToVal, &QToolButton::clicked, this, &BnniMainWindow::moveTrainToVal);
    connect(ui->valToTrain, &QToolButton::clicked, this, &BnniMainWindow::moveValToTrain);
    connect(ui->valToBlind, &QToolButton::clicked, this, &BnniMainWindow::moveValToBlind);
    connect(ui->valToUnused, &QToolButton::clicked, this, &BnniMainWindow::moveValToUnused);
    connect(ui->blindToVal, &QToolButton::clicked, this, &BnniMainWindow::moveBlindToVal);
    connect(ui->blindToUnused, &QToolButton::clicked, this, &BnniMainWindow::moveBlindToUnused);
    connect(ui->unusedTotrain, &QToolButton::clicked, this, &BnniMainWindow::moveUnusedToTrain);
    connect(ui->unusedToVal, &QToolButton::clicked, this, &BnniMainWindow::moveUnusedToVal);
    connect(ui->unusedToBlind, &QToolButton::clicked, this, &BnniMainWindow::moveUnusedToBlind);

    connect(ui->seismicTable, &QTableWidget::itemSelectionChanged, this, &BnniMainWindow::validateArguments);
    connect(ui->seismicTable, &QTableWidget::itemSelectionChanged, this, [this]() {
        QVector<std::pair<QString, QString>> array;

        for (QModelIndex& e : ui->seismicTable->selectionModel()->selectedRows()) {
            QString name = ui->seismicTable->item(e.row(), 0)->data(Qt::UserRole).toString();
            QString survey_name = name.split("\t").last().split("/").first();
            name = name.split("/").last();
            array.append(std::pair<QString, QString>(name, this->projectDir+"/DATA/3D/"+survey_name+"/DATA/SEISMIC/seismic3d."+name));
        }
        updateHiddenLayersSize();
        updatedSeismicSelection(array);
    });
    connect(ui->logTable, &QTableWidget::itemSelectionChanged, this, &BnniMainWindow::validateArguments);
    connect(ui->logTable, &QTableWidget::itemSelectionChanged, this, [this]() {
        QVector<QString> array;

        for (QModelIndex& e : ui->logTable->selectionModel()->selectedRows()) {
            QString name = ui->logTable->item(e.row(), 0)->data(Qt::UserRole).toString();
            array.append(name);
        }
        updatedKindSelection(array);
    });

    connect(ui->actionGeneralizeSection, &QAction::triggered, this, &BnniMainWindow::openGeneralizationSectionWindow);
    connect(ui->actionGeneralizeCarte, &QAction::triggered, this, &BnniMainWindow::openGeneralizationCarteWindow);
    connect(ui->actionDemo_BNNI, &QAction::triggered, this, &BnniMainWindow::openDemoBNNI);
    connect(ui->actionGenetic_Algorithm, &QAction::triggered, this, &BnniMainWindow::openGeneticAlgorithmLauncher);

    /*ui->trainList->addLinkedListWidget(ui->validationList);
    ui->trainList->addLinkedListWidget(ui->wellList);
    ui->validationList->addLinkedListWidget(ui->trainList);
    ui->validationList->addLinkedListWidget(ui->wellList);
    ui->wellList->addLinkedListWidget(ui->trainList);
    ui->wellList->addLinkedListWidget(ui->validationList);*/

    process = new QProcess(this);
    connect(process, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(postProcessing(int, QProcess::ExitStatus)));
    connect(process, SIGNAL(errorOccurred(QProcess::ProcessError)), this, SLOT(errorProcessing(QProcess::ProcessError)));

    process->setProcessChannelMode(QProcess::ForwardedChannels);

    replayProcess = new QProcess(this);
    connect(replayProcess, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(replayPostProcessing(int, QProcess::ExitStatus)));
    connect(replayProcess, SIGNAL(errorOccurred(QProcess::ProcessError)), this, SLOT(replayErrorProcessing(QProcess::ProcessError)));

    replayProcess->setProcessChannelMode(QProcess::ForwardedChannels);

    generalizationProcess = new QProcess(this);
    generalizationProcess->setProcessChannelMode(QProcess::ForwardedChannels);

    connect(ui->actionOpen_Well_Viewers, &QAction::triggered, this, &BnniMainWindow::openWellViewer);

    connect(ui->logTable, &QTableWidget::itemSelectionChanged, this, &BnniMainWindow::changeLogsSelection);

    CollapsableScrollArea* optionsArea = new CollapsableScrollArea("Options");

    ui->collapseHolder->addWidget(optionsArea);
    parameterForm = new NetworkParameterForm(m_bnniConfig->getNetworkModel());

    optionsArea->setContentLayout(*parameterForm);

    //ui->configLineEdit->setText(this->configFilename);
    //ui->configLineEdit->setEnabled(true);
    QSizePolicy policy = ui->configComboBox->sizePolicy();
    policy.setHorizontalPolicy(QSizePolicy::MinimumExpanding);
    ui->configComboBox->setSizePolicy(policy);

    policy = ui->jsonComboBox->sizePolicy();
    policy.setHorizontalPolicy(QSizePolicy::MinimumExpanding);
    ui->jsonComboBox->setSizePolicy(policy);
    //connect(ui->configLineEdit, &QLineEdit::textChanged, this, &BnniMainWindow::updateConfigFilename);
    //connect(ui->searchConfigButton, &QPushButton::clicked, this, [this]() {
    //    this->searchConfigFilename();
    //});
    connect(ui->saveConfigButton, &QPushButton::clicked, this, &BnniMainWindow::saveConfig);
    connect(ui->loadConfigButton, &QPushButton::clicked, this, &BnniMainWindow::loadConfig);

    ui->logTable->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->logTable, &QTableWidget::customContextMenuRequested, this, &BnniMainWindow::logTableContexMenu);

    READY = QColor(0, 255, 0);
    INTERRUPT = QColor(255, 0, 0);
    DISABLED = QColor(204,204,204);

    trainSampleRate = DEFAULT_SAMPLE_RATE;
    predictSampleRate = DEFAULT_SAMPLE_RATE;

    validateArguments();

    loadCurrentSettings();

    ui->seismicTable->setMinimumHeight(100);
    ui->logTable->setMinimumHeight(100);
}

BnniMainWindow::~BnniMainWindow()
{
    if (!configFilename.isNull() && !configFilename.isEmpty() && \
            !jsonfile.isNull() && !jsonfile.isEmpty()) {
        saveConfig();
    }

    delete ui;
    if (wellViewer) {
        wellViewer->deleteLater();
    }
}

void BnniMainWindow::createNewConfig() {
    if (jsonfile.isNull() || jsonfile.isEmpty()) {
        QMessageBox::warning(this, tr("No trainingset"),
                tr("There is no training set selected. Please select a traing set before creating an experiment."));
        return;
    }

    QStringList configs = QFileInfo(jsonfile).dir().entryList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);

    QString ckptExtension = (m_bnniConfig->getNetworkModel()->getNetwork()==NeuralNetwork::Xgboost) ? "ubj" : "index";
    QFileInfoList checkpointsFiles = QDir(m_bnniConfig->getWorkDir()).entryInfoList(QStringList() << "*."+ckptExtension, QDir::Files, QDir::Time);
    QStringList checkpoints;
    for (int i=0; i<checkpointsFiles.size(); i++) {
        checkpoints.append(checkpointsFiles[i].baseName());
    }

    BnniConfigCreationDialog dialog(configs, checkpoints, this);
    int res = dialog.exec();
    QString output = dialog.newName();
    if (res==QDialog::Accepted && !output.isNull() && ! output.isEmpty()) {
        int i=0;
        while (i<ui->configComboBox->count() && output.compare(ui->configComboBox->itemData(i, Qt::UserRole).toString())!=0) {
            i++;
        }
        if (i==ui->configComboBox->count()) {
            QDir dir = QFileInfo(jsonfile).dir();
            if(dir.mkdir(output)) {
                //QFile::copy(configFilename, getConfigFilePath(projectDir, ui->jsonComboBox->currentText(), output));
                if (dialog.checkpointValid()) {
                    QFileInfoList checkPoints = QFileInfo(configFilename).dir().entryInfoList(QStringList() << dialog.checkpoint()+"*", QDir::Files);
                    for (QFileInfo checkPoint : checkPoints) {
                        QFile::copy(checkPoint.absoluteFilePath(),
                                getConfigDirPath(projectDir, ui->jsonComboBox->currentText(), output)+"/"+CHECKPOINT_REFERENCE+"."+checkPoint.completeSuffix());
                    }
                }

                ui->configComboBox->addItem(output, QVariant(output));
                ui->configComboBox->setCurrentText(output);
                m_bnniConfig->getNetworkModel()->setReferenceCheckpoint(CHECKPOINT_REFERENCE);
                saveConfig();
            } else {
                QMessageBox::warning(this, tr("Experiment could not be create"), tr("The experiment could not be created. This may result from right issues."));
            }
        } else {
            QMessageBox::warning(this, tr("Experiment already exist"), tr("The experiment already exist."));
        }
    }
}

void BnniMainWindow::changeConfig(int index) {
    if (!ui->configComboBox->itemText(index).isEmpty()) {
        m_bnniConfig->setWorkDir(getConfigDirPath(projectDir, ui->jsonComboBox->currentText(), ui->configComboBox->itemData(index, Qt::UserRole).toString()));
        configFilename = m_bnniConfig->getWorkDir() + "/config.txt";
        ui->loadConfigButton->click();
        // save last config
        QSettings settings;
        settings.setValue(LAST_CONFIG_PATH_IN_SETTINGS, configFilename);
    } else {
        m_bnniConfig->setWorkDir("");
        configFilename = "";
    }
}

// Use external lib
bool BnniMainWindow::loadJson(QString filename, bool preserveConfig) {
    //
    /*if (filename.isEmpty() || filename.isNull()) {
        return false;
        /*QString searchDir = "/data";
        if (!surveyDir.isNull() && !surveyDir.isEmpty()) {
            QFileInfo fileInfo(surveyDir);
            if (fileInfo.isDir()) {
                searchDir = QFileInfo(fileInfo.dir().path()).dir().absolutePath()+"/NEURONS/neurons2/LogInversion2Problem3";
            }
        }
        filename = QFileDialog::getExistingDirectory(this, tr("Choose training set"), searchDir);

        if (filename.isEmpty() || filename.isNull()) {
            return false;
        }
        filename = filename + "/trainingset.json";* /
    }*/
    //ui->jsonEdit->setText(QFileInfo(filename).dir().dirName());
    jsonfile = "";

    seismics.clear();
    logs.clear();
    wells.clear();

    if (!preserveConfig) {
        ui->configComboBox->clear();
        ui->configComboBox->addItem("");
    }

    updateSeismic();
    updateLogs();
    updateWells();

    QString survey_name = "";

    std::ifstream ifs(filename.toStdString().c_str());
    rapidjson::IStreamWrapper isw(ifs);
    WDocument document;
    document.ParseStream(isw);
    qDebug() <<document.HasParseError();

    bool isValid = true;

    //qDebug() << "Is the document null ? :" << document.IsNull() << document.HasParseError() << document.GetParseError() <<
    //            document.GetErrorOffset() << rapidjson::GetParseError_En(document.GetParseError());
    if (isValid && !document.IsObject()) {
        qWarning() << tr("Unexpected format, could not get root object");
        isValid = false;
    }

    if (isValid && !(document.HasMember("seismicParameters") && document["seismicParameters"].IsObject())) {
        qWarning() << tr("Unexpected format, could not get seismicParameters");
        isValid = false;
    } else if(isValid) {
        WValue& parseValue = document["seismicParameters"];

        if (isValid && !(parseValue.IsObject() && parseValue.HasMember("datasets"))) {
            qWarning() << tr("Unexpected format, could not find correct 'seismicParameters'");
            isValid = false;
        } else if (isValid) {
            WValue& array = parseValue["datasets"];

            if (isValid && !array.IsArray()) {
                qWarning() << tr("Unexpected format, could not find 'datasets' in 'seismicParameters'");
                isValid = false;
            } else if(isValid){
                // fill seismic parameters
                for (unsigned int i=0; i<array.Size();i++) {
                    WValue& _e = array[i];
                    if (isValid && !(_e.IsObject() && _e.HasMember("dataset") && _e["dataset"].IsString())) {
                        qWarning() << tr("Unexpected format, could not find correct 'dataset' in 'seismicParameters'");
                        isValid = false;
                        break;
                    }

                    Parameter param;
                    param.name = _e["dataset"].GetString();
                    QString tmp_survey = param.name.split("\t").last().split("/").first();
                    if (i==0) {
                        survey_name = tmp_survey;
                        qDebug() << "Found survey : " << survey_name;
                    }

                    if (isValid && !(_e.HasMember("dynamic") && _e["dynamic"].IsArray() &&_e["dynamic"].Size()==2 &&
                                     _e["dynamic"][0].IsDouble() && _e["dynamic"][1].IsDouble())) {
                        qWarning() << tr("Unexpected format, could not find correct 'dynamic' in 'datasets' array from 'seismicParameters'");
                        isValid = false;
                        break;
                    }
                    param.min = _e["dynamic"][0].GetDouble();
                    param.max = _e["dynamic"][1].GetDouble();
                    param.InputMin = param.min;
                    param.InputMax = param.max;
                    param.OutputMin = -0.5;
                    param.OutputMax = 0.5;
                    if (_e.HasMember("samplingRate") && _e["samplingRate"].IsDouble()) {
                        trainSampleRate = _e["samplingRate"].GetDouble();
                    } else {
                        trainSampleRate = DEFAULT_SAMPLE_RATE;
                    }
                    predictSampleRate = trainSampleRate;

                    seismics.append(param);

                    WValue& halfWindow = parseValue["halfWindowHeight"];
                    if (isValid && !halfWindow.IsInt()) {
                        qWarning() << tr("Unexpected format, could not find 'halfWindowHeight' in 'seismicParameters'");
                        isValid = false;

                    } else if(isValid){
                        this->setHalfWindow(halfWindow.GetInt());
                    }
                }
            }


        }


    }

    if (isValid && !(document.HasMember("logsParameters") && document["logsParameters"].IsObject() &&
                     document["logsParameters"].HasMember("logColumns") && document["logsParameters"]["logColumns"].IsArray() &&
                     document["logsParameters"].HasMember("logsDynamics") && document["logsParameters"]["logsDynamics"].IsArray() &&
                     document["logsParameters"]["logColumns"].Size() == document["logsParameters"]["logsDynamics"].Size())) {
        qWarning() << tr("Unexpected format, could not find correct 'logsParameters'");
        isValid = false;
    } else if (isValid) {
        // fill log parameters
        for (unsigned int i=0; i<document["logsParameters"]["logColumns"].Size(); i++) {
            LogParameter param;

            WValue& column = document["logsParameters"]["logColumns"][i];
            WValue& dynamic = document["logsParameters"]["logsDynamics"][i];

            if (isValid && !(column.IsObject() && (column.HasMember("kind") && column["kind"].IsString() ||
                                                   column.HasMember("name") && column["name"].IsString()) &&
                             dynamic.IsArray()) && dynamic.Size()==2 && dynamic[0].IsDouble() && dynamic[1].IsDouble()) {
                qWarning() << tr("Unexpected format, could not find correct 'logsParameters'");
                break;
            }

            if (column.HasMember("kind")) {
                param.name = column["kind"].GetString();
            } else {
                param.name = column["name"].GetString();
            }
            param.min = dynamic[0].GetDouble();
            param.max = dynamic[1].GetDouble();

            param.InputMin = param.min;
            param.InputMax = param.max;
            param.OutputMin = 0.25;
            param.OutputMax = 0.75;
            param.preprocessing = LogNone;

            logs.append(param);
        }
    }

    if (isValid && !(document.HasMember("logsParameters") && document["logsParameters"].IsObject() &&
                     document["logsParameters"].HasMember("wellbores") && document["logsParameters"]["wellbores"].IsArray())) {
        qWarning() << tr("Unexpected format, could not find correct 'wellbores'");
        isValid = false;
    } else if(isValid) {
        for (unsigned int i=0; i<document["logsParameters"]["wellbores"].Size(); i++) {
            WValue& w = document["logsParameters"]["wellbores"][i];
            if (isValid && !(w.IsString())) {
                qWarning() << tr("Unexpected format, could not find correct 'wellbores'");
                break;
            }

            Well well;
            well.name = w.GetString();
            wells.append(well);
        }
    }


    //qDebug() << QString::fromStdString(std::string(document["samples"].MemberBegin()->name.GetString()));
    //qDebug() << QString::compare(wells[0].name.replace("\t",""), QString::fromStdString(std::string(document["samples"].MemberBegin()->name.GetString())));
    if (isValid && document.HasMember("samples") && document["samples"].IsObject()) {
         for (int i=0; i<wells.size(); i++) {
             char name[1000];
             strcpy(name, wells[i].name.replace("\t","").toUtf8().toStdString().c_str());
             qDebug() << name;
             qDebug() << wells[i].name.replace("\t","");
             qDebug() << wells[i].name.replace("\t","").toUtf8().toStdString().c_str();

             if (!(document["samples"].HasMember(name) &&
                   document["samples"][name].IsArray())) {

                 qWarning() << tr("Unexpected format, could not find correct expected well in 'samples'") << name;
                 //isValid = false;
                 continue;
             }

             if (isValid) {
                 for (int k=0; k<document["samples"][name].Size(); k++) {
                     WValue& sample = document["samples"][name][k];
                     if (!(sample.IsArray() && sample.Size()==4 && sample[0].IsArray() && sample[1].IsArray())) {
                         qWarning() << tr("Unexpected format, could not find correct 'samples'");
                         isValid = false;
                         break;
                     }
                     LogSample example;
                     for (int j=0; j<sample[0].Size(); j++) {
                         WValue& val = sample[0][j];
                         if (!(val.IsDouble() || val.IsInt())) {
                             qWarning() << tr("Unexpected format, could not find correct 'samples'");
                             isValid = false;
                             break;
                         }
                         if (val.IsDouble()) {
                             example.seismicVals.append(val.GetDouble());
                         } else {
                        	 example.seismicVals.append(val.GetInt());
                         }
                     }
                     if (isValid) {
                         for (int j=0; j<sample[1].Size(); j++) {
                             WValue& val = sample[1][j];
                             if (!(val.IsDouble())) {
                                 qWarning() << tr("Unexpected format, could not find correct 'samples'");
                                 isValid = false;
                                 break;
                             }
                             example.logVals.append(val.GetDouble());
                         }
                     }
                     if (isValid) {
                         isValid = sample[2].IsDouble();
                         if (isValid) {
                             example.depth = sample[2].GetDouble();
                         }
                     }
                     if (isValid) {
                         isValid = sample[3].IsArray() && sample[3].Size()==3 && sample[3][0].IsDouble() &&
                                 sample[3][1].IsDouble() && sample[3][2].IsDouble();
                         if (isValid) {
                             example.x = sample[3][0].GetDouble();
                             example.y = sample[3][1].GetDouble();
                             example.z = sample[3][2].GetDouble();
                         }
                     }
                     if (!isValid) {
                         break;
                     } else {
                         wells[i].samples.append(example);
                     }

                 }
             }
             /*if (isValid) {
                 for (int j=0; j<document["samples"][name][0].Size(); j++) {
                     WValue& val = document["samples"][name][0][j];
                     if (!(val.IsDouble())) {
                         qWarning() << tr("Unexpected format, could not find correct 'samples'");
                         isValid = false;
                         break;
                     }
                     example.seismicVals.append(val.GetDouble());
                 }
             }
             if (isValid) {
                 for (int j=0; j<document["samples"][name][1].Size(); j++) {
                     WValue& val = document["samples"][name][1][j];
                     if (!(val.IsDouble())) {
                         qWarning() << tr("Unexpected format, could not find correct 'samples'");
                         isValid = false;
                         break;
                     }
                     example.logVals.append(val.GetDouble());
                 }
             }
             if (isValid) {
                 wells[i].samples.append(example);
             }*/
             if (isValid) {
                 wells[i].ranges.resize(1);
                 wells[i].ranges[0].min = 0;
                 wells[i].ranges[0].max = wells[i].samples.length()-1;

                 WellViewer::changeRangeToRemoveWellPlateaus(wells[i]);
                 WellViewer::removeWellsWithConstantSeismic(wells[i], seismics.size(), halfSignalSize);

                 // Check if well is vertical
                 bool isVertical = true;
                 int indexLoop = 1;
                 double xDir = 0; // well is vertical if it goes forward in the same direction for all axises
                 double yDir = 0;

                 while (isVertical && indexLoop<wells[i].samples.count()) {
                	 isVertical = wells[i].samples[indexLoop].z - wells[i].samples[indexLoop-1].z > 0;

                	 if (isVertical && xDir==0) {
                		 xDir = wells[i].samples[indexLoop].x - wells[i].samples[indexLoop-1].x;
                	 }
                	 isVertical = isVertical && (wells[i].samples[indexLoop].x - wells[i].samples[indexLoop-1].x)*xDir >= 0;

                	 if (isVertical && yDir==0) {
                		 yDir = wells[i].samples[indexLoop].y - wells[i].samples[indexLoop-1].y;
                	 }
                	 isVertical = isVertical && (wells[i].samples[indexLoop].y - wells[i].samples[indexLoop-1].y)*yDir >= 0;

                	 indexLoop ++;
                 }
                 // limit the angle
//                 float val = wells[i].samples.last().z - wells[i].samples[0].z;
//                 isVertical = isVertical && val!=0 && std::sqrt(std::pow(wells[i].samples.last().x - wells[i].samples.first().x, 2) +
//                		 	 	 	 	 	 	 std::pow(wells[i].samples.last().y - wells[i].samples.first().y, 2)) / val < 0.1;

                 wells[i].isVertical = isVertical;
             }
         }
    }

    cleanWells();

    if (isValid) {
        //ui->jsonEdit->setText(filename);
        jsonfile = filename;

        dataMap.clear();
        scales.clear();
        clearDynamicCache();
        computeWellDynamic();
        cache_log_preprocessing.clear();
        for (int i=0; i<logs.size(); i++) {
            cache_log_preprocessing.append(LogNone);
        }


        updateSeismic();
        updateLogs();
        updateWells();
        updatedJson(filename);
        surveyName = survey_name;

        if (!preserveConfig) {
            QDir dir = QFileInfo(jsonfile).dir();
            qDebug() << dir.absolutePath();
            QFileInfoList infolist = dir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
            for (QFileInfo& e : infolist) {
                QString suffix = getConfigSuffixFromFile(e.absoluteFilePath()+"/config.txt");
                ui->configComboBox->addItem(e.baseName() + suffix, QVariant(e.baseName()));
            }
        }
    }
    // JSON research strategy has been transfered to function loadTrainingSet
    /*else {
        if (document.HasParseError() && !filename.isNull() && !filename.isEmpty()) {
            //QMessageBox::StandardButton answer = QMessageBox::question(this, tr("Invalid format"), tr("The format of given file is invalid.\n Do you want to try a repair of the file ?"));

            //if (answer == QMessageBox::Yes) {

                /*QString searchDir = "/home";
                if (!surveyDir.isNull() && !surveyDir.isEmpty()) {
                    searchDir = surveyDir;
                    searchDir += "/" + QFileInfo(QFileInfo(filename).dir().absolutePath()).fileName() + ".json";
                }
                QString new_file = QFileDialog::getSaveFileName(this, tr("Create new json file"), searchDir, tr("JSON Files (*.json)"));
                * /
                QFileInfo fileinfo(filename);
                QString new_file = fileinfo.dir().absoluteFilePath("BNNI_trainingset.json");
                QFileInfo new_fileinfo(new_file);
                if (!new_file.isNull() && !new_file.isEmpty() && (!new_fileinfo.exists() || new_fileinfo.lastModified()<fileinfo.lastModified())) {

                    QFile::copy(":/python/convertJson.py" , "/tmp/BNNI_interface_python_script.py");
                    if (!QDir("/tmp/jsoncomment").exists()) {
                        QDir("/tmp").mkdir("jsoncomment");
                    }

                    if (!QDir("/tmp/jsoncomment/package").exists()) {
                        QDir("/tmp/jsoncomment").mkdir("package");
                    }

                    QFile::copy(":/python/jsoncomment/__init__.py" , "/tmp/jsoncomment/__init__.py");
                    QFile::copy(":/python/jsoncomment/COPYING" , "/tmp/jsoncomment/COPYING");
                    QFile::copy(":/python/jsoncomment/README.md" , "/tmp/jsoncomment/README.md");
                    QFile::copy(":/python/jsoncomment/README.rst" , "/tmp/jsoncomment/README.rst");
                    QFile::copy(":/python/jsoncomment/package/__init__.py" , "/tmp/jsoncomment/package/__init__.py");
                    QFile::copy(":/python/jsoncomment/package/comments.py" , "/tmp/jsoncomment/package/comments.py");
                    QFile::copy(":/python/jsoncomment/package/wrapper.py" , "/tmp/jsoncomment/package/wrapper.py");

                    QFile myfile("/tmp/BNNI_interface_python_script.py");
                    myfile.setPermissions(myfile.permissions() | QFile::ExeGroup | QFile::ExeOther | QFile::ExeOther | QFile::ExeUser);

                    QProcess process;
                    QStringList arguments;
                    arguments << filename << new_file;
                    process.setWorkingDirectory("/tmp");
                    process.start("./BNNI_interface_python_script.py", arguments);
                    process.waitForFinished(-1);
                    //if (process.exitStatus()==QProcess::NormalExit) {
                    if (process.exitCode()==0) {
                        qDebug() << process.readAll();
                        this->loadJson(new_file);
                    } else {
                        QMessageBox::critical(this, tr("Conversion failure"), tr("The conversion process encountered an error, please check files ")+filename+tr(" and ")+new_file+tr(" : \n")+ process.readAllStandardError());
                    }
                } else if (new_fileinfo.exists()){
                    this->loadJson(new_file);
                }


            //}
        }
    }*/
    return isValid;
}

bool BnniMainWindow::loadUbjson(QString filename, bool preserveConfig) {
    jsonfile = "";

    seismics.clear();
    logs.clear();
    wells.clear();

    if (!preserveConfig) {
        ui->configComboBox->clear();
        ui->configComboBox->addItem("");
    }

    updateSeismic();
    updateLogs();
    updateWells();

    QString survey_name = "";

    bool result = BnniUbjsonDecoder::load(filename, this, trainSampleRate, seismics, logs, wells, survey_name);
    cleanWells();
    if (result) {
        //ui->jsonEdit->setText(filename);
        jsonfile = filename;

        dataMap.clear();
        scales.clear();
        clearDynamicCache();
        computeWellDynamic();
        cache_log_preprocessing.clear();
        for (int i=0; i<logs.size(); i++) {
            cache_log_preprocessing.append(LogNone);
        }


        updateSeismic();
        updateLogs();
        updateWells();
        updatedJson(filename);
        surveyName = survey_name;

        if (!preserveConfig) {
            QDir dir = QFileInfo(jsonfile).dir();
            qDebug() << dir.absolutePath();
            QFileInfoList infolist = dir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
            for (QFileInfo& e : infolist) {
                QString suffix = getConfigSuffixFromFile(e.absoluteFilePath()+"/config.txt");
                ui->configComboBox->addItem(e.baseName() + suffix, QVariant(e.baseName()));
            }
        }
        jsonfile = filename;
    }

    return result;
}

void BnniMainWindow::saveCurrentStateToJsonFile() {
	QFileInfo jsonFileInfo(jsonfile);

	if (!jsonFileInfo.exists() || !jsonFileInfo.isFile() || !jsonFileInfo.isReadable() ||
			!jsonFileInfo.isWritable() || jsonFileInfo.suffix().compare("json")!=0) {
		qDebug() << "saveCurrentStateToJsonFile impossible to save to file " << jsonfile;
		return;
	}


	std::ifstream ifs(jsonfile.toStdString().c_str());
	rapidjson::IStreamWrapper isw(ifs);
	WDocument document;
	document.ParseStream(isw);
	qDebug() <<document.HasParseError();

	WDocument::AllocatorType& allocator = document.GetAllocator();

	bool isValid = true;

	if (!document.IsObject()) {
		qWarning() << tr("Unexpected format, could not get root object");
		isValid = false;
	}

	// Will only update new logs
	if (isValid) {
		/*
		 * Tasks done below :
		 *
		 * data["logsParameters"]["logsDynamics"] append dynamic as list
		 * data["logsParameters"]["logsWeights"] add float 1.0
         * data["logsParameters"]["logsCount"] increase to match size
		 * data["logsParameters"]["logColumns"] add object {'name': 'log_name'}
		 *
		 * Update for all wells and all samples
		 * data["samples"][well_name][sampleIdx][1] append log value to list
		 */

		if (document.HasMember("logsParameters") && document["logsParameters"].IsObject()) {
			WValue& logsParameters = document["logsParameters"];

			// set dynamic
			if (logsParameters.HasMember("logsDynamics") && logsParameters["logsDynamics"].IsArray()) {
				WValue& logsDynamics = logsParameters["logsDynamics"];

				int i=0;
				while (isValid && i<logs.count()) {
					WValue minVal;
					minVal.SetDouble((double) logs[i].min);
					WValue maxVal;
					maxVal.SetDouble((double) logs[i].max);

					if (i==logsDynamics.Size()) {
						WValue logDynamic;
						logDynamic.SetArray();
						logDynamic.PushBack(minVal, allocator);
						logDynamic.PushBack(maxVal, allocator);

						logsDynamics.PushBack(logDynamic, allocator);
					} else {
						WValue& logDynamic = logsDynamics[i];
						if (logDynamic.IsArray() && logDynamic.Size()==2 && logDynamic[0].IsDouble() && logDynamic[1].IsDouble()) {
							logDynamic[0] = minVal;
							logDynamic[1] = maxVal;
						} else {
							isValid = false;
						}
					}
					i++;
				}
			} else {
				isValid = false;
			}

			// set log weights
			isValid = isValid && logsParameters.HasMember("logsWeights") && logsParameters["logsWeights"].IsArray();
			if (isValid) {
				WValue& logsWeights = logsParameters["logsWeights"];

				while (logsWeights.Size()<logs.count()) {
					logsWeights.PushBack(WValue().SetDouble(1.0), allocator);
				}
			}

			// set log count
			isValid = isValid && logsParameters.HasMember("logsCount") && logsParameters["logsCount"].IsInt();
			if (isValid) {
				logsParameters["logsCount"] = WValue().SetInt(logs.count());
			}

			// set logs name
			isValid = isValid && logsParameters.HasMember("logColumns") && logsParameters["logColumns"].IsArray();
			if(isValid) {
				WValue& logColumns = logsParameters["logColumns"];
				int i=0;
				while (isValid && i<logs.count()) {
					WValue name(logs[i].name.toStdString().c_str(), allocator);
					if (i==logColumns.Size()) {
						WValue logColumn;
						logColumn.SetObject();
						logColumn.AddMember("name", name, allocator);

						logColumns.PushBack(logColumn, allocator);
					} else {
						WValue& logColumn = logColumns[i];
						if (logColumn.IsObject() && ((logColumn.HasMember("kind") && logColumn["kind"].IsString()) || (logColumn.HasMember("name") && logColumn["name"].IsString()))) {
							if (logColumn.HasMember("kind")) {
								logColumn["kind"] = name;
							} else {
								logColumn["name"] = name;
							}
						} else {
							isValid = false;
						}
					}
					i++;
				}
			}
		} else {
			isValid = false;
		}

		// set logs values
		isValid = isValid && document.HasMember("samples") && document["samples"].IsObject();
		if (isValid) {
			//get keys
			WValue::MemberIterator it = document["samples"].MemberBegin();

			while (isValid && it!=document["samples"].MemberEnd()) {
				qDebug() << (*it).name.GetString();

				QString wellName = QString((*it).name.GetString());

				int wellIdx = 0;
				while (wellIdx<wells.count() &&
						wellName.compare(wells[wellIdx].name.replace("\t","").toUtf8())!=0) {
					wellIdx++;
				}

				if (wellIdx<wells.count()) {
					const Well& well = wells[wellIdx];
					WValue& wellValue = (*it).value;

					isValid = isValid && wellValue.IsArray() && wellValue.Size()==well.samples.count();
					int i = 0;
					while (isValid && i<wellValue.Size()) {
						isValid = isValid && wellValue[i].IsArray() && wellValue[i].Size()==4 && wellValue[i][1].IsArray();
						int j = 0;
						while (isValid && j<logs.count()) {
							WValue logVal;
							logVal.SetDouble(well.samples[i].logVals[j]);
							if(j==wellValue[i][1].Size()) {
								wellValue[i][1].PushBack(logVal, allocator);
							} else {
								wellValue[i][1][j] = logVal;
							}
							j++;
						}
						i++;
					}

				} else {
					isValid = false;
					QMessageBox::critical(this, tr("Well mismatch between training set file and application"), tr("A well in the training set file is not present in the application. This should never happen !!!"));
				}

				it++;
			}
		}
	}

	if (isValid) {
		// Write document were it was
		qDebug() << "saveCurrentStateToJsonFile end reached without issue";

		std::ofstream ofs(jsonfile.toStdString().c_str());
		rapidjson::OStreamWrapper osw(ofs);

		rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
		document.Accept(writer);

	}
}

bool compareQVector(const QVector<IntRange>& a, const QVector<IntRange>& b) {
    bool equal = a.size()==b.size();
    int i = 0;
    while (equal && i<a.size()) {
        equal = a[i].min==b[i].min && a[i].max==b[i].max;
        i++;
    }
    return equal;
}

void BnniMainWindow::computeWellDynamic() {
    if (cache_wellIndexRanges.size()!=wells.size()) {
        clearDynamicCache();
    }
    for(int k=0; k<wells.size(); k++) {
        Well& well = wells[k];

        if (cache_wellIndexRanges.size()==k ||
                compareQVector(cache_wellIndexRanges[k], well.ranges)) {
            well.dynamic.clear();
            for (int rangeIdx=0; rangeIdx<well.ranges.size(); rangeIdx++) {
                IntRange& minMaxRange = well.ranges[rangeIdx];
                for (int i=0; i<well.samples[0].logVals.size(); i++) {
                    Range range;
                    for ( int j=minMaxRange.min; j<=minMaxRange.max; j++) {
                        float val = well.samples[j].logVals[i];
                        if (val<range.min) {
                            range.min = val;
                        }
                        if (val>range.max) {
                            range.max = val;
                        }
                    }
                    well.dynamic.append(range);
                }
            }
            if (cache_wellIndexRanges.size()==k) {
                cache_wellIndexRanges.append(well.ranges);
            } else {
                cache_wellIndexRanges[k] = well.ranges;
            }
        }
    }
}

Range BnniMainWindow::getLogDynamic(int k) {
    Range range;
    for (Well& well : wells) {
    	// to avoid crash in there is no range/no dynamic
    	if (k<well.dynamic.size()) {
			Range val = well.dynamic[k];
			if (val.min < range.min) {
				range.min = val.min;
			}

			if (val.max > range.max) {
				range.max = val.max;
			}
    	}
    }
    return range;
}

void BnniMainWindow::clearDynamicCache() {
    cache_wellIndexRanges.clear();
}

void BnniMainWindow::updateMinMaxIndex() {
    computeWellDynamic();
    for (int i=0; i<logs.size(); i++) {
        Range logDataRange = getLogDynamic(i);

        float min_val = logDataRange.min;
        float max_val = logDataRange.max;
        if (logs[i].preprocessing==LogLn) {
            if (min_val <= 0) {
                min_val = std::numeric_limits<float>::lowest();
            } else {
                min_val = log(min_val);
            }
            if (max_val <= 0) {
                max_val = std::numeric_limits<float>::lowest();
            } else {
                max_val = log(max_val);
            }
        }
        logsForm[i].lineEditDataMin->setText(locale().toString(min_val));
        logsForm[i].lineEditDataMax->setText(locale().toString(max_val));
    }
}

void BnniMainWindow::updateSeismic() {
    ui->seismicTable->clear();
    seismicsForm.clear();
    QStringList horizontalHeaderList;
    horizontalHeaderList << "Name" << "Input min" << "Input max" << "Output min" << "Output max";
    ui->seismicTable->setHorizontalHeaderLabels(horizontalHeaderList);

    ui->seismicTable->setRowCount(seismics.size());
    ui->seismicTable->setColumnCount(5);
    for (int i=0; i<seismics.size(); i++) {
        QTableWidgetItem* item = new QTableWidgetItem(seismics[i].name.split("/").last().split(".xt").first());
        item->setData(Qt::UserRole, QVariant(seismics[i].name));
        item->setFlags(item->flags() & ~Qt::ItemIsEditable);
        ui->seismicTable->setItem(i, 0, item);

        // Setup seismic form
        SeismicForm form;

        form.lineEditInputMin = new QLineEdit(locale().toString(seismics[i].InputMin));
        QDoubleValidator* validator = new QDoubleValidator(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), std::numeric_limits<int>::max());
        form.lineEditInputMin->setValidator(validator);
        connect(form.lineEditInputMin, &QLineEdit::editingFinished, this, [this, i, form]() {
            seismics[i].InputMin = locale().toFloat(form.lineEditInputMin->text());
        });
        ui->seismicTable->setCellWidget(i, 1, form.lineEditInputMin);

        form.lineEditInputMax = new QLineEdit(locale().toString(seismics[i].InputMax));
        validator = new QDoubleValidator(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), std::numeric_limits<int>::max());
        form.lineEditInputMax->setValidator(validator);
        connect(form.lineEditInputMax, &QLineEdit::editingFinished, this, [this, i, form]() {
            seismics[i].InputMax = locale().toFloat(form.lineEditInputMax->text());
        });
        ui->seismicTable->setCellWidget(i, 2, form.lineEditInputMax);

        form.lineEditOutputMin = new QLineEdit(locale().toString(seismics[i].OutputMin));
        validator = new QDoubleValidator(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), std::numeric_limits<int>::max());
        form.lineEditOutputMin->setValidator(validator);
        connect(form.lineEditOutputMin, &QLineEdit::editingFinished, this, [this, i, form]() {
            seismics[i].OutputMin = locale().toFloat(form.lineEditOutputMin->text());
        });
        ui->seismicTable->setCellWidget(i, 3, form.lineEditOutputMin);

        form.lineEditOutputMax = new QLineEdit(locale().toString(seismics[i].OutputMax));
        validator = new QDoubleValidator(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), std::numeric_limits<int>::max());
        form.lineEditOutputMax->setValidator(validator);
        connect(form.lineEditOutputMax, &QLineEdit::editingFinished, this, [this, i, form]() {
            seismics[i].OutputMax = locale().toFloat(form.lineEditOutputMax->text());
        });
        ui->seismicTable->setCellWidget(i, 4, form.lineEditOutputMax);

        seismicsForm.append(form);
    }

    updateSeismicDataMap();
}

void BnniMainWindow::updateLogs() {
    ui->logTable->clear();
    logsForm.clear();
    QStringList horizontalHeaderList;
    horizontalHeaderList << "Name" << "Data min" << "Data max" << "Input min" << "Input max" << "Output min" << "Output max" << "Preprocessing";
    ui->logTable->setHorizontalHeaderLabels(horizontalHeaderList);

    ui->logTable->setRowCount(logs.size());
    ui->logTable->setColumnCount(8);
    for (int i=0; i<logs.size(); i++) {
        QTableWidgetItem* item = new QTableWidgetItem(logs[i].name);
        item->setData(Qt::UserRole, QVariant(logs[i].name));
        item->setFlags(item->flags() & ~Qt::ItemIsEditable);
        ui->logTable->setItem(i, 0, item);

        // Setup seismic form
        LogForm form;
        Range logDataRange = getLogDynamic(i);
        form.lineEditDataMin = new QLineEdit(locale().toString(logDataRange.min));
        form.lineEditDataMin->setReadOnly(true);
        ui->logTable->setCellWidget(i, 1, form.lineEditDataMin);

        form.lineEditDataMax = new QLineEdit(locale().toString(logDataRange.max));
        form.lineEditDataMax->setReadOnly(true);
        ui->logTable->setCellWidget(i, 2, form.lineEditDataMax);

        form.lineEditInputMin = new QLineEdit(locale().toString(logs[i].InputMin));
        QDoubleValidator* validator = new QDoubleValidator(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), std::numeric_limits<int>::max());
        form.lineEditInputMin->setValidator(validator);
        connect(form.lineEditInputMin, &QLineEdit::editingFinished, this, [this, i, form]() {
            logs[i].InputMin = locale().toFloat(form.lineEditInputMin->text());
        });
        ui->logTable->setCellWidget(i, 3, form.lineEditInputMin);

        form.lineEditInputMax = new QLineEdit(locale().toString(logs[i].InputMax));
        validator = new QDoubleValidator(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), std::numeric_limits<int>::max());
        form.lineEditInputMax->setValidator(validator);
        connect(form.lineEditInputMax, &QLineEdit::editingFinished, this, [this, i, form]() {
            logs[i].InputMax = locale().toFloat(form.lineEditInputMax->text());
        });
        ui->logTable->setCellWidget(i, 4, form.lineEditInputMax);

        form.lineEditOutputMin = new QLineEdit(locale().toString(logs[i].OutputMin));
        validator = new QDoubleValidator(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), std::numeric_limits<int>::max());
        form.lineEditOutputMin->setValidator(validator);
        connect(form.lineEditOutputMin, &QLineEdit::editingFinished, this, [this, i, form]() {
            logs[i].OutputMin = locale().toFloat(form.lineEditOutputMin->text());
        });
        ui->logTable->setCellWidget(i, 5, form.lineEditOutputMin);

        form.lineEditOutputMax = new QLineEdit(locale().toString(logs[i].OutputMax));
        validator = new QDoubleValidator(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max(), std::numeric_limits<int>::max());
        form.lineEditOutputMax->setValidator(validator);
        connect(form.lineEditOutputMax, &QLineEdit::editingFinished, this, [this, i, form]() {
            logs[i].OutputMax = locale().toFloat(form.lineEditOutputMax->text());
        });
        ui->logTable->setCellWidget(i, 6, form.lineEditOutputMax);

        form.comboboxPreprocessing = new QComboBox;
        form.comboboxPreprocessing->addItem("None", QVariant(LogPreprocessing::LogNone));
        form.comboboxPreprocessing->addItem("Log", QVariant(LogPreprocessing::LogLn));
        ui->logTable->setCellWidget(i, 7, form.comboboxPreprocessing);
        if (logs[i].preprocessing==LogLn) {
            form.comboboxPreprocessing->setCurrentIndex(1);
        } else { // LogNone is default
            form.comboboxPreprocessing->setCurrentIndex(0);
        }
        connect(form.comboboxPreprocessing,SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), [this, i, form](int index) {
            bool test;
            int val = form.comboboxPreprocessing->itemData(index).toInt(&test);
            if (test) {
                if (val==LogLn) {
                    Range range = getLogDynamic(i);
                    if (range.min<0) {
                        QMessageBox::StandardButton btn = QMessageBox::warning(this, tr("Data range error"), tr("Data Range not adapted to preprocessing : ln"), QMessageBox::Ignore | QMessageBox::Abort);
                        if (btn==QMessageBox::Abort) {
                            LogPreprocessing cache_val = cache_log_preprocessing[i];
                            if (val==cache_val) {
                               val = LogNone;
                            } else {
                                val = cache_val;
                            }
                            form.comboboxPreprocessing->setCurrentIndex(val);
                        }
                    }
                }
                cache_log_preprocessing[i] = static_cast<LogPreprocessing>(val);
                logs[i].preprocessing = static_cast<LogPreprocessing>(val);

                updateMinMaxIndex();
            }
        });
                // this, &NetworkParameterForm::updateLogPreprocessing);


        logsForm.append(form);
    }
}

void BnniMainWindow::updateWells() {
    ui->wellList->clear();
    ui->trainList->clear();
    ui->validationList->clear();
    ui->blindList->clear();

    for (int i=0; i<wells.size(); i++) {
        if (wells[i].active) {
            //replace("\t","")
            //QString name = wells[i].name.split("\t").last().split("||").last();
            QString name = wells[i].name.replace("\t","").replace("Sismage2 MainIDL:DmWell/ExistingWellBoreFactory:1.0", "");
            QListWidgetItem* item = new QListWidgetItem(name);
            item->setData(Qt::UserRole, QVariant(wells[i].name));
            if (wells[i].isVertical) {
            	//item->setStyleSheet(QString("QListWidgetItem{ color: %1; }").arg(QColor(Qt::green).name()));
            	item->setForeground(QColor(Qt::green));
            }
            ui->wellList->addItem(item);
        }
    }
    if (wellViewer) {
        wellViewer->updateWells(&wells);
    }
}

void BnniMainWindow::launchProcess() {
    if (process->state()!=QProcess::NotRunning) {
        qWarning() << "Cannot launch a new process an old one is running";
        return;
    }

    if (!testWarningForArguments()) {
        return;
    }

    QString filename = configFilename;
    this->saveConfig();

    QDir configDir = QFileInfo(configFilename).dir();
    QFileInfoList list = configDir.entryInfoList(QStringList() << "*.index" << "*.meta" << "checkpoint" << "*.data-*-of-*" << "*.ubj", QDir::Files);
    for (QFileInfo e : list) {
        if (e.baseName().compare(CHECKPOINT_REFERENCE)!=0) {
            configDir.remove(e.fileName());
        }
    }

    QString program = QDir(programLocation).absoluteFilePath(programFileName + "_train.sh");
    QStringList arguments;
    arguments << "--gui" << "--config" << filename;
    /*arguments << "/data/PLI/armand/pour_antoine/fc2cnn/trainRNN.py" << "--json" << ui->jsonEdit->text() << "--gpu" << "1" <<
                 "--save" << "tmp" << "--channels";
    QList<QListWidgetItem*> list = ui->seismicList->selectedItems();
    for (int i=0; i<list.size(); i++) {
        arguments << "'"+list[i]->text()+"'";
    }
    arguments << "--output_channels";
    list = ui->logList->selectedItems();
    for (int i=0; i<list.size(); i++) {
        arguments << "'"+list[i]->text()+"'";
    }

    if (ui->trainList->count()>0) {
        arguments << "--train_wells";
        for (int i=0; i<ui->trainList->count(); i++) {
            arguments << ui->trainList->item(i)->text().replace("|", "\\|");
        }
    }

    if (ui->validationList->count()>0) {
        arguments << "--blind_wells";
        for (int i=0; i<ui->validationList->count(); i++) {
            arguments << ui->validationList->item(i)->text().replace("|", "\\|");
        }
    }*/

    process->setWorkingDirectory(m_bnniConfig->getWorkDir());
    process->start(program, arguments);
    qDebug() << arguments.join(" ");

    applyRunningProcessColors();
}

void BnniMainWindow::stopProcess() {
    if (process) {
        qWarning() << "Stop process, process state :" << process->state();
        process->terminate();

        validateArguments();
    } else {
        qWarning() << "No process to stop";
    }
}

void BnniMainWindow::errorProcessing(QProcess::ProcessError error) {
    qWarning() << "An Error occured in process :" << error;
    QMessageBox::critical(this, tr("Process error"), tr("Process end with error : ") +QString::number(error));

    validateArguments();
}

void BnniMainWindow::postProcessing(int exitCode, QProcess::ExitStatus exitStatus) {
    qWarning() << "End of process :" << exitCode;
    QString message = tr("Process end with exit code : ")+ QString::number(exitCode) + tr(", and exit status: ")+ QString::number(exitStatus);
    if (exitCode==0) {
        message += "\n" + tr("Training Finished");
    }
    QMessageBox::information(this, tr("Process ended"), message);

    validateArguments();
}

void BnniMainWindow::replayLaunchProcess() {
    if (replayProcess->state()!=QProcess::NotRunning) {
        qWarning() << "Cannot launch a new process an old one is running";
        return;
    }

    QString filename = configFilename;
    this->saveConfig();

    QString program = QDir(programLocation).absoluteFilePath(programFileName + "_viewer.sh");
    QStringList arguments;
    arguments << "--config" << filename;

    replayProcess->setWorkingDirectory(m_bnniConfig->getWorkDir());
    replayProcess->start(program, arguments);
    qDebug() << arguments.join(" ");
}

void BnniMainWindow::replayErrorProcessing(QProcess::ProcessError error) {
    qWarning() << "An Error occured in replay process :" << error;
    QMessageBox::critical(this, tr("Replay Process error"), tr("Replay Process end with error : ") +QString::number(error));
}

void BnniMainWindow::replayPostProcessing(int exitCode, QProcess::ExitStatus exitStatus) {
    qWarning() << "End of replay process :" << exitCode;
    QString message = tr("Replay Process end with exit code : ")+ QString::number(exitCode) + tr(", and exit status: ")+ QString::number(exitStatus);
    if (exitCode==0) {
        message += "\n" + tr("Replay Finished");
    }
    QMessageBox::information(this, tr("Relay Process ended"), message);
}

void BnniMainWindow::moveSelectedItems(QListWidget* src, QListWidget* dst) {
    QList<QListWidgetItem*> items = src->selectedItems();
    for (int i=0; i<items.length(); i++) {
        QListWidgetItem* item = new QListWidgetItem(*items[i]);
        dst->addItem(item);
        delete items[i];
    }
}

void BnniMainWindow::moveTrainToUnused() {
    moveSelectedItems(ui->trainList, ui->wellList);
    validateArguments();
}

void BnniMainWindow::moveTrainToVal() {
    moveSelectedItems(ui->trainList, ui->validationList);
    validateArguments();
}

void BnniMainWindow::moveValToTrain() {
    moveSelectedItems(ui->validationList, ui->trainList);
    validateArguments();
}

void BnniMainWindow::moveValToBlind() {
    moveSelectedItems(ui->validationList, ui->blindList);
    validateArguments();
}

void BnniMainWindow::moveValToUnused() {
    moveSelectedItems(ui->validationList, ui->wellList);
    validateArguments();
}

void BnniMainWindow::moveBlindToVal() {
    moveSelectedItems(ui->blindList, ui->validationList);
    validateArguments();
}

void BnniMainWindow::moveBlindToUnused() {
    moveSelectedItems(ui->blindList, ui->wellList);
    validateArguments();
}

void BnniMainWindow::moveUnusedToVal() {
    moveSelectedItems(ui->wellList, ui->validationList);
    validateArguments();
}

void BnniMainWindow::moveUnusedToBlind() {
	moveSelectedItems(ui->wellList, ui->blindList);
    validateArguments();
}

void BnniMainWindow::moveUnusedToTrain() {
    moveSelectedItems(ui->wellList, ui->trainList);
    validateArguments();
}

void BnniMainWindow::openWellViewer() {
    if (!wellViewer) {
        wellViewer = new WellViewer();
        wellViewer->updateWells(&wells);
        wellViewer->setAttribute(Qt::WA_DeleteOnClose);
        wellViewer->setVisible(true);
        wellViewer->resize(1000,700);
        connect(wellViewer, &WellViewer::toggleWell, this, &BnniMainWindow::toggleWells);
        connect(wellViewer, &WellViewer::destroyed, this, &BnniMainWindow::closeWellViewer);
        connect(wellViewer, &WellViewer::minMaxIndexChanged, this, &BnniMainWindow::updateMinMaxIndex);
        changeLogsSelection();
    }
}

void BnniMainWindow::closeWellViewer() {
    wellViewer = nullptr;
}

void BnniMainWindow::toggleWells(Well &well, bool active) {
    if (active) {
        QString name = well.name.split("\t").last();
        QListWidgetItem* item = new QListWidgetItem(name);
        item->setData(Qt::UserRole, QVariant(well.name.split("\t").last()));
        ui->wellList->addItem(item);
    } else {
        bool test = true;
        unsigned int i = 0;
        unsigned int l = ui->wellList->count() + ui->trainList->count() + ui->validationList->count() + ui->blindList->count();
        while (i<l && test) {
            if (i<ui->wellList->count()) {
                test = !(ui->wellList->item(i)->data(Qt::UserRole).toString() == well.name.split("\t").last());
            } else if (i<ui->wellList->count()+ui->trainList->count()) {
                test = !(ui->trainList->item(i-ui->wellList->count())->data(Qt::UserRole).toString() == well.name.split("\t").last());
            } else if (i<ui->wellList->count()+ui->trainList->count()+ui->validationList->count()) {
                test = !(ui->validationList->item(i-ui->wellList->count()-ui->trainList->count())->data(Qt::UserRole).toString() == well.name.split("\t").last());
            } else {
            	test = !(ui->blindList->item(i-ui->wellList->count()-ui->trainList->count()-ui->validationList->count())->data(Qt::UserRole).toString() == well.name.split("\t").last());
            }
            if (test) {
                i++;
            }
        }
        if (!test) {
            if (i<ui->wellList->count()) {
                delete ui->wellList->item(i);
            } else if (i<ui->wellList->count()+ui->trainList->count()) {
                delete ui->trainList->item(i-ui->wellList->count());
            } else if (i<ui->wellList->count()+ui->trainList->count()+ui->validationList->count()) {
                delete ui->validationList->item(i-ui->wellList->count()-ui->trainList->count());
            } else {
            	delete ui->blindList->item(i-ui->wellList->count()-ui->trainList->count()-ui->validationList->count());
            }
        }
    }
}

void BnniMainWindow::cleanWells() {
    int i=0;
    bool test;
    while(i<wells.size()) {
        test = wells[i].samples.size()>0;

        if (test) {
            i++;
        } else {
            wells.remove(i);
        }
    }
}

void BnniMainWindow::changeLogsSelection() {
    if (!wellViewer) {
        return;
    }
    QVector<unsigned int> vector;
    QModelIndexList list = ui->logTable->selectionModel()->selectedRows();
    for (int i=0; i<list.count(); i++) {
        vector.push_back(list.at(i).row());
    }
    wellViewer->selectLogs(vector, logs);
}

// Use QJsonDocument to parse, limited to QString max size
/*void BnniMainWindow::loadJson() {
    QString filename = QFileDialog::getOpenFileName(this, tr("Open BBoxes file"), "/home", tr("JSON Files (*.json)"));
    QFile file(filename);
    qDebug() << "Filename :" <<filename;
    if(!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::information(0,"error",file.errorString());
    }

    QString val = file.readAll();

    qDebug() << val;
    this->json = QJsonDocument::fromJson(val.toUtf8());

    file.close();
    QJsonObject root = json.object();

    QJsonValue parseValue = root.value("seismicParameters");
    if (!parseValue.isObject()) {
        qWarning() << tr("Unexpected format, could not find 'seismicParameters'");
        return;
    }
    QJsonObject object = parseValue.toObject();

    parseValue = object.value("datasets");
    if (!parseValue.isArray()) {
        qWarning() << tr("Unexpected format, could not find 'datasets' in 'seismicParameters'");
        return;
    }
    QJsonArray array = parseValue.toArray();

    // fill seismic parameters
    for (int i=0; i<array.size();i++) {
        QJsonValue _e = array.at(i);
        if (_e.isObject()) {
            qWarning() << tr("Unexpected format, could not find correct 'datasets' in 'seismicParameters'");
            return;
        }
        QJsonObject e = _e.toObject();

        Parameter param;

        parseValue = e.value("dataset");
        if (!parseValue.isString()) {
            qWarning() << tr("Unexpected format, could not find 'dataset' in 'datasets' array from 'seismicParameters'");
            return;
        }
        param.name = parseValue.toString();

        parseValue = e.value("dynamic");
        if (!parseValue.isArray()) {
            qWarning() << tr("Unexpected format, could not find 'dynamic' in 'datasets' array from 'seismicParameters'");
            return;
        }
        QJsonArray subArray = parseValue.toArray();


        if (subArray.size()!=2) {
            qWarning() << tr("Unexpected format,unexpected lenght for 'dynamic' in 'datasets' array from 'seismicParameters'");
        }

        parseValue = subArray.at(0);
        if (!parseValue.isDouble()) {
            qWarning() << tr("Unexpected format, could not find the minimun of 'dynamic' in 'datasets' array from 'seismicParameters'");
            return;
        }
        param.min = parseValue.toDouble();

        parseValue = subArray.at(1);
        if (!parseValue.isDouble()) {
            qWarning() << tr("Unexpected format, could not find the minimun of 'dynamic' in 'datasets' array from 'seismicParameters'");
            return;
        }
        param.max = parseValue.toDouble();

        seismics.append(param);
    }

    parseValue = root.value("logsParameters");
    if (!parseValue.isObject()) {
        qWarning() << tr("Unexpected format, could not find 'logsParameters'");
        return;
    }
    object = parseValue.toObject();

    parseValue = object.value("logColumns");
    if (!parseValue.isArray()) {
        qWarning() << tr("Unexpected format, could not find 'logColumns' in 'logsParameters'");
        return;
    }
    QJsonArray array_names = parseValue.toArray();

    parseValue = object.value("logsDynamics");
    if (!parseValue.isArray()) {
        qWarning() << tr("Unexpected format, could not find 'logsDynamics' in 'logsParameters'");
        return;
    }
    QJsonArray array_dynamics = parseValue.toArray();

    if (array_dynamics.size() != array_names.size()) {
        qWarning() << tr("Unexpected format, 'logsDynamics' and 'logColumns' do not match size");
        return;
    }

    // fill seismic parameters
    for (int i=0; i<array_names.size(); i++) {
        Parameter param;

        parseValue = array_names.at(i);
        if (!parseValue.isObject()) {
            qWarning() << tr("Unexpected format, could not find 'logsDynamics' in 'logsParameters'");
            return;
        }
        object = parseValue.toObject();

        parseValue = object.value("kind");
        if (!parseValue.isString()) {
            qWarning() << tr("Unexpected format, could not find 'kind' in 'logColumns' array from 'logsParameters'");
            return;
        }
        param.name = parseValue.toString();

        parseValue = array_dynamics.at(i);
        if (!parseValue.isArray()) {
            qWarning() << tr("Unexpected format, could not find 'dynamic' in 'logColumns' array from 'logsParameters'");
            return;
        }
        QJsonArray subArray = parseValue.toArray();

        if (subArray.size()!=2) {
            qWarning() << tr("Unexpected format,unexpected lenght for 'dynamic' in 'logColumns' array from 'logsParameters'");
        }

        parseValue = subArray.at(0);
        if (!parseValue.isDouble()) {
            qWarning() << tr("Unexpected format, could not find the minimun of 'logColumns' in 'datasets' array from 'logsParameters'");
            return;
        }
        param.min = parseValue.toDouble();

        parseValue = subArray.at(1);
        if (!parseValue.isDouble()) {
            qWarning() << tr("Unexpected format, could not find the minimun of 'logColumns' in 'datasets' array from 'logsParameters'");
            return;
        }
        param.max = parseValue.toDouble();

        logs.append(param);
    }

}*/
//data["seismicParameters"]["datasets"][i]["dataset"]
//data["logsParameters"]['logColumns'][i]["kind"]



void BnniMainWindow::applyRunningProcessColors() {
    QPalette palette = ui->launcher->palette();
    palette.setColor(QPalette::Base, DISABLED);
    ui->launcher->setAutoFillBackground(true);
    ui->launcher->setPalette(palette);

    palette = ui->stopProcess->palette();
    palette.setColor(QPalette::Base, INTERRUPT);
    ui->stopProcess->setAutoFillBackground(true);
    ui->stopProcess->setPalette(palette);
}

void BnniMainWindow::applyStandbyProcessColors() {
    QPalette palette = ui->launcher->palette();
    palette.setColor(QPalette::Base, READY);
    ui->launcher->setAutoFillBackground(true);
    ui->launcher->setPalette(palette);

    palette = ui->stopProcess->palette();
    palette.setColor(QPalette::Base, DISABLED);
    ui->stopProcess->setAutoFillBackground(true);
    ui->stopProcess->setPalette(palette);
}

void BnniMainWindow::applyNotLaunchableProcess() {
    QPalette palette = ui->launcher->palette();
    palette.setColor(QPalette::Base, DISABLED);
    ui->launcher->setAutoFillBackground(true);
    ui->launcher->setPalette(palette);

    palette = ui->stopProcess->palette();
    palette.setColor(QPalette::Base, DISABLED);
    ui->stopProcess->setAutoFillBackground(true);
    ui->stopProcess->setPalette(palette);
}

void BnniMainWindow::validateArguments() {
    QFileInfo jsonInfo(jsonfile);
    QFileInfo configInfo(configFilename);
    NetworkParametersModel* networkModel = m_bnniConfig->getNetworkModel();
    bool test = networkModel != nullptr;
    if (test) {
        test = jsonInfo.exists() && jsonInfo.isFile() && !jsonfile.isNull() && !jsonfile.isEmpty() &&
            ui->seismicTable->selectionModel()->selectedRows().count()!=0 && ui->logTable->selectionModel()->selectedRows().count()!=0 &&
            ui->trainList->count()!=0 && ui->blindList->count()!=0 && !configInfo.isDir() && !configFilename.isNull() &&
            !configFilename.isEmpty() && configInfo.absoluteDir().exists() && networkModel->validateArguments();
    }

    if (test) {
        applyStandbyProcessColors();
    } else {
        applyNotLaunchableProcess();
    }
}

bool BnniMainWindow::testWarningForArguments() {
    QString title = "Bad argument";

    QFileInfo jsonInfo(jsonfile);
    QFileInfo configInfo(configFilename);
    NetworkParametersModel* networkModel = m_bnniConfig->getNetworkModel();
    bool notValid = networkModel == nullptr;
    QStringList badArguments;
    if (notValid) {
        badArguments << "Form not defined";
    }
    if (ui->seismicTable->selectionModel()->selectedRows().count()==0) {
        badArguments << "No seismic selected";
        notValid = true;
    }
    if (jsonfile.isNull() || jsonfile.isEmpty() || !jsonInfo.exists() || !jsonInfo.isFile()) {
        badArguments << "Training set not valid";
        notValid = true;
    }
    if (ui->logTable->selectionModel()->selectedRows().count()==0) {
        badArguments << "No log selected";
        notValid = true;
    }
    if (ui->trainList->count()==0) {
        badArguments << "No training well selected";
        notValid = true;
    }
    if (ui->blindList->count()==0) {
        badArguments << "No blind well selected";
        notValid = true;
    }
    if (configFilename.isNull() || configFilename.isEmpty() || configInfo.isDir() || !configInfo.absoluteDir().exists()) {
        badArguments << "Experiment does not exists";
        notValid = true;
    }
    if (!networkModel->validateArguments()) {
        badArguments << networkModel->warningForArguments();
        notValid = true;
    }
    if (notValid) {
        QMessageBox::warning(this, title, badArguments.join("\n"));
    }
    return !notValid;
}

void BnniMainWindow::openGeneralizationSectionWindow() {
	if (ui->logTable->selectionModel()->selectedRows().size()==0) {
		QMessageBox::warning(this, "Generalize section cannot open", "No log selected");
		return;
	}

    GeneralizationSectionWidget* widget = new GeneralizationSectionWidget();
    widget->setAttribute(Qt::WA_DeleteOnClose);
    widget->setVisible(true);
    widget->resize(1000,700);

    widget->setIJKDirectory(findIJKDirectory());

    widget->setSurvey(projectDir + "/DATA/3D/" + surveyName);
    widget->setSuffix(saveSuffixGeneralization);
    widget->setYStep(ystep);
    widget->setWorkDir(m_bnniConfig->getWorkDir());
    widget->setNetwork(m_bnniConfig->getNetworkModel()->getNetwork());
    widget->setTrainSampleRate(trainSampleRate);
    widget->setPredictSampleRate(predictSampleRate);
    connect(this, &BnniMainWindow::reset, widget, &GeneralizationSectionWidget::reset);
    connect(this, &BnniMainWindow::updatedKindSelection, widget, &GeneralizationSectionWidget::setWellKind);
    connect(this, &BnniMainWindow::updatedSeismicSelection, widget, &GeneralizationSectionWidget::setSeismicNames);
    connect(m_bnniConfig->getNetworkModel(), &NetworkParametersModel::networkChanged, widget, &GeneralizationSectionWidget::setNetwork);
    connect(widget, &GeneralizationSectionWidget::generalize, this, &BnniMainWindow::sliceGeneralization);
    connect(this, &BnniMainWindow::sliceGeneralizationFinishedSignal, widget, &GeneralizationSectionWidget::generalizationFinished);
    connect(this, &BnniMainWindow::sliceGeneralizationRefusedSignal, widget, &GeneralizationSectionWidget::generalizationFailed);
    connect(this, &BnniMainWindow::sliceGeneralizationErrorSignal, widget, &GeneralizationSectionWidget::generalizationRefused);
    connect(widget, &GeneralizationSectionWidget::generalizeRandom, this, &BnniMainWindow::randomGeneralization);

    connect(widget, &GeneralizationSectionWidget::ystepChanged, this, [this](int val) {
        this->ystep = val;
    });
    connect(widget, &GeneralizationSectionWidget::saveSuffixChanged, this, [this](QString txt) {
        this->saveSuffixGeneralization = txt;
    });
    connect(widget, &GeneralizationSectionWidget::predictSampleRateChanged, this, [this](float val) {
        this->predictSampleRate = val;
    });
    connect(widget, &GeneralizationSectionWidget::scalesChanged, this, [this](QVector<float> array) {
        this->scales.clear();

        QVector<QString> logArray;
        for (QModelIndex& e : ui->logTable->selectionModel()->selectedRows()) {
            QString name = ui->logTable->item(e.row(), 0)->data(Qt::UserRole).toString();
            logArray.append(name);
        }
        if (logArray.size() != array.size()) {
            qDebug() << "BnniMainWindow::openGeneralizationWindow lambda scalesChanged : array not of the same size !!!!";
            return;
        }

        for (int i=0; i<array.size(); i++) {
            scales.insert(logArray[i], array[i]);
        }
    });

    QVector<QString> logArray;
    QVector<float> scaleArray;
    QVector<std::pair<float, float>> logDynamic;
    QVector<int> selectedLogsIndexes;

    // selected logs
    for (QModelIndex& e : ui->logTable->selectionModel()->selectedRows()) {
        QString name = ui->logTable->item(e.row(), 0)->data(Qt::UserRole).toString();
        //logArray.append(name);
        scaleArray.append(scales.value(name, 1.0));

        int i=0;

        QVector<LogParameter>::iterator iter = logs.begin();
        while(iter!=logs.end() && iter->name.compare(name)!=0) {
            iter++;
            i++;
        }
        if (iter!=logs.end()) {
            //LogParameter& param = *iter;
            //logDynamic.append(std::pair<int,int>(param.InputMin, param.InputMax));
            selectedLogsIndexes.append(i);
        } else {
            qDebug() << "Could not find name in logs";
        }
    }

    // all logs
    for (int i=0; i<logs.count(); i++) {
    	LogParameter& param = logs[i];
    	logDynamic.append(std::pair<float,float>(param.InputMin, param.InputMax));
    	logArray.append(param.name);
    }

    widget->setSelectedLogsIndexes(selectedLogsIndexes);// link selected logs to the position in "all logs"
    widget->setWellKind(logArray); // all logs name
    widget->setLogDynamic(logDynamic); // all logs dynamic
    widget->setScales(scaleArray); // linked to selected logs


    QVector<std::pair<QString, QString>> array;
    QVector<std::pair<float, float>> seismicDynamic;
    dataMap.clear();
    for (QModelIndex& e : ui->seismicTable->selectionModel()->selectedRows()) {
        QString _name = ui->seismicTable->item(e.row(), 0)->data(Qt::UserRole).toString();
        QString name = _name.split("/").last();
        QString survey_name = _name.split("\t").last().split("/").first();
        array.append(std::pair<QString, QString>(name, this->projectDir+"/DATA/3D/"+survey_name+"/DATA/SEISMIC/seismic3d."+name));
        dataMap.insert(_name, array.last().second);

        QVector<Parameter>::iterator iter = seismics.begin();
        while(iter!=seismics.end() && iter->name.compare(_name)!=0) {
            iter++;
        }
        if (iter!=seismics.end()) {
            Parameter& param = *iter;
            seismicDynamic.append(std::pair<int,int>(param.InputMin, param.InputMax));
        } else {
            qDebug() << "Could not find name in seismics";
        }
    }
    widget->setSeismicDynamic(seismicDynamic);
    widget->setSeismicNames(array);
    widget->setWells(&(this->wells));
}

void BnniMainWindow::openGeneralizationCarteWindow() {
//    GeneralizationCarteWidget* widget = new GeneralizationCarteWidget();
//    widget->setAttribute(Qt::WA_DeleteOnClose);
//    widget->setVisible(true);
//    widget->resize(1000,700);
//
//    widget->setSurvey(projectDir + "/DATA/3D/" + surveyName);
//    widget->setSuffix(saveSuffixGeneralization);
//    widget->setWorkDir(workdir);
//    connect(this, &BnniMainWindow::reset, widget, &GeneralizationCarteWidget::reset);
//    connect(this, &BnniMainWindow::updatedKindSelection, widget, &GeneralizationCarteWidget::setWellKind);
//    //connect(this, &BnniMainWindow::updatedSeismicSelection, widget, &GeneralizationCarteWidget::setSeismicNames);
//    connect(widget, &GeneralizationCarteWidget::generalize, this, &BnniMainWindow::carteGeneralization);
//    connect(this, &BnniMainWindow::carteGeneralizationFinishedSignal, widget, &GeneralizationCarteWidget::generalizationFinished);
//    connect(this, &BnniMainWindow::carteGeneralizationRefusedSignal, widget, &GeneralizationCarteWidget::generalizationFailed);
//    connect(this, &BnniMainWindow::carteGeneralizationErrorSignal, widget, &GeneralizationCarteWidget::generalizationRefused);
//
//    connect(widget, &GeneralizationCarteWidget::saveSuffixChanged, this, [this](QString txt) {
//        this->saveSuffixGeneralization = txt;
//    });
//    connect(widget, &GeneralizationCarteWidget::scalesChanged, this, [this](QVector<float> array) {
//        this->scales.clear();
//
//        QVector<QString> logArray;
//        for (QModelIndex& e : ui->logTable->selectionModel()->selectedRows()) {
//            QString name = ui->logTable->item(e.row(), 0)->data(Qt::UserRole).toString();
//            logArray.append(name);
//        }
//        if (logArray.size() != array.size()) {
//            qDebug() << "BnniMainWindow::openGeneralizationWindow lambda scalesChanged : array not of the same size !!!!";
//            return;
//        }
//
//        for (int i=0; i<array.size(); i++) {
//            scales.insert(logArray[i], array[i]);
//        }
//    });
//
//    QVector<QString> logArray;
//    QVector<float> scaleArray;
//    QVector<std::pair<float, float>> logDynamic;
//    for (QModelIndex& e : ui->logTable->selectionModel()->selectedRows()) {
//        QString name = ui->logTable->item(e.row(), 0)->data(Qt::UserRole).toString();
//        logArray.append(name);
//        scaleArray.append(scales.value(name, 1.0));
//
//        QVector<LogParameter>::iterator iter = logs.begin();
//        while(iter!=logs.end() && iter->name.compare(name)!=0) {
//            iter++;
//        }
//        if (iter!=logs.end()) {
//            LogParameter& param = *iter;
//            logDynamic.append(std::pair<int,int>(param.InputMin, param.InputMax));
//        } else {
//            qDebug() << "Could not find name in seismics";
//        }
//    }
//    widget->setWellKind(logArray);
//    widget->setScales(scaleArray);
//    widget->setLogDynamic(logDynamic);
//
//
//    /*QVector<std::pair<QString, QString>> array;
//    QVector<std::pair<float, float>> seismicDynamic;
//    dataMap.clear();
//    for (QModelIndex& e : ui->seismicTable->selectionModel()->selectedRows()) {
//        QString _name = ui->seismicTable->item(e.row(), 0)->data(Qt::UserRole).toString();
//        QString name = _name.split("/").last();
//        array.append(std::pair<QString, QString>(name, this->surveyDir+"/DATA/SEISMIC/seismic3d."+name));
//        dataMap.insert(_name, array.last().second);
//
//        QVector<Parameter>::iterator iter = seismics.begin();
//        while(iter!=seismics.end() && iter->name.compare(_name)!=0) {
//            iter++;
//        }
//        if (iter!=seismics.end()) {
//            Parameter& param = *iter;
//            seismicDynamic.append(std::pair<int,int>(param.InputMin, param.InputMax));
//        } else {
//            qDebug() << "Could not find name in seismics";
//        }
//    }
//    widget->setSeismicDynamic(seismicDynamic);
//    widget->setSeismicNames(array);*/
//
//    widget->setDataInfo(getDataInfo());
}

void BnniMainWindow::openDemoBNNI() {
//    DemoBNNI* widget = new DemoBNNI();
//    widget->setAttribute(Qt::WA_DeleteOnClose);
//    widget->setVisible(true);
//    widget->resize(1000,700);
}

void BnniMainWindow::openGeneticAlgorithmLauncher() {
	std::vector<int> trainIdxs;
	std::vector<int> validationIdxs;
	std::vector<int> blindIdxs;
	for (int i=0; i<wells.size(); i++) {
		int j=0;
		bool found = false;
		while (!found && j<ui->trainList->count()) {
			QString name = ui->trainList->item(j)->data(Qt::UserRole).toString();

			found = QString::compare(wells[i].name, name)==0;
			if (!found) {
				j++;
			}
		}
		if (ui->trainList->count()!=j) {
			trainIdxs.push_back(i);
		} else {
			int idxValidation = 0;
			bool foundValidation = false;

			while (!foundValidation && idxValidation<ui->validationList->count()) {
				QString name = ui->validationList->item(idxValidation)->data(Qt::UserRole).toString();

				foundValidation = QString::compare(wells[i].name, name)==0;
				if (!foundValidation) {
					idxValidation++;
				}
			}
			if (ui->validationList->count()!=idxValidation) {
				validationIdxs.push_back(i);
			} else {
				int idxBlind = 0;
				bool foundBlind = false;

				while (!foundBlind && idxBlind<ui->blindList->count()) {
					QString name = ui->blindList->item(idxBlind)->data(Qt::UserRole).toString();

					foundBlind = QString::compare(wells[i].name, name)==0;
					if (!foundBlind) {
						idxBlind++;
					}
				}
				if (ui->blindList->count()!=idxBlind) {
					blindIdxs.push_back(i);
				}
			}
		}
	}

    QVector<unsigned int> layerSizes = m_bnniConfig->getNetworkModel()->getHiddenLayers();
    layerSizes.push_back(1);

    GeneticAlgorithmLauncher* geneticWidget = new GeneticAlgorithmLauncher();
    geneticWidget->setAttribute(Qt::WA_DeleteOnClose);
    geneticWidget->setVisible(true);
    geneticWidget->resize(1000,700);

    geneticWidget->setProgramLocation(programLocation);
    geneticWidget->setTrainingSet(jsonfile);
    geneticWidget->setWells(wells, trainIdxs, validationIdxs, blindIdxs);
    geneticWidget->setHalfWindow(halfSignalSize);
    geneticWidget->setLayerSizes(layerSizes);
    geneticWidget->setNumInputSeismics(seismics.size());
}

void logger(QString key, bool ok) {
    if (!ok) {
        qDebug() << "BnniMainWindow::loadConfig fail to load key : " << key;
    }
}

void BnniMainWindow::loadConfig() {
    QString filename = configFilename;
    QFile file(filename);

    if (!filename.isNull() && !filename.isEmpty() && file.open(QIODevice::ReadOnly)) {
        QTextStream stream(&file);
        QString line;
        QString key;
        QStringList values;
        QMap<QString, QStringList> map;
        //QString criticalKey = "json";
        //bool criticalKeyIsHere = false;
        //bool surveyIsHere = false;

        for (int i=0; i<ui->seismicTable->rowCount(); i++) {
            ui->seismicTable->item(i, 0)->setSelected(false);
        }

        for (int i=0; i<ui->logTable->rowCount(); i++) {
            ui->logTable->item(i, 0)->setSelected(false);
        }

        while (stream.readLineInto(&line)) {
            values = line.split("\n")[0].split(separator);
            key = values[0];
            values.pop_front();

            // Ignore line if there is no
            if (values.size()==0) {
                continue;
            }

            map[key] = values;
            /*if (key.compare(criticalKey)==0) {
                QFileInfo jsonInfo(values[0]);
                criticalKeyIsHere = jsonInfo.exists() && jsonInfo.isFile();
            } else if (key.compare(QString("survey"))==0) {
                surveyIsHere = true;
            }*/
        }

        qDebug() << "DEBUG Keys";
        for (QString key : map.keys()) {
            qDebug() << key;
        }
        qDebug() << "end keys";
        bool hiddenLayersFound = false;
        bool referenceCheckpointSet = false;
        QStringList hiddenLayersValues;
        for(QString key : map.keys()) {
            if (key.compare("json")==0) {
                continue;
            }
            values = map[key];
            bool ok = false;
            /*if (key.compare("program_file")==0) {
                ok = true; programFilePath = values[0];
            } else*/
            if (key.compare("work_dir")==0) {
                ok = true; m_bnniConfig->setWorkDir(values[0]);
            } else if (key.compare("gpu")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadNGpus(values[0]);
            } else if (key.compare("hiddenLayers")==0) {
            	hiddenLayersFound = true;
            	hiddenLayersValues = values;
                ok = m_bnniConfig->getNetworkModel()->loadHiddenLayers(values);
            } else if (key.compare("learningRate")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadLearningRate(values[0]);
            } else if (key.compare("momentum")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadMomentum(values[0]);
            } else if (key.compare("restore")==0) {
                ok =  m_bnniConfig->getNetworkModel()->loadSavePrefix(values[0]);
            } else if (key.compare("train_initialisation")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadReferenceCheckpoint(values[0]);
                referenceCheckpointSet = true;
            } else if (key.compare("epochs")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadNumEpochs(values[0]);
            } else if (key.compare("batch_size")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadBatchSize(values[0]);
            } else if (key.compare("dropout")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadDropout(values[0]);
            } else if (key.compare("precision")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadPrecision(values[0]);
            } else if (key.compare("optimizer")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadOptimizer(values[0]);
            } else if (key.compare("network")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadNetwork(values[0]);
            } else if (key.compare("activation")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadLayerActivation(values[0]);
            } else if (key.compare("use_bias")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadUseBias(values[0]);
            } else if (key.compare("seismic_preprocessing")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadSeismicPreprocessing(values[0]);
            } else if (key.compare("batch_norm")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadBatchNorm(values[0]);
            } else if (key.compare("channels")==0) {
                ok = this->setChannels(values);
            } else if (key.compare("unselected_channels")==0) {
                ok = this->setUnselectedSeismic(values);
            } else if (key.compare("output_channels")==0) {
                ok = this->setOutputChannels(values);
            } else if (key.compare("unselected_output_channels")==0) {
                ok = this->setUnselectedOutputChannels(values);
            } else if (key.compare("train_wells")==0) {
                ok = this->setTrainWells(values);
            } else if (key.compare("validation_wells")==0) {
                ok = this->setValidationWells(values);
            } else if (key.compare("blind_wells")==0) {
                ok = this->setBlindWells(values);
            } else if (key.compare("unused_wells")==0) {
                ok = this->setUnusedWells(values);
            } /*else if (key.compare("xmin")==0) {
                ok = this->setXMin(values[0]);
            } else if (key.compare("xmax")==0) {
                ok = this->setXMax(values[0]);
            } else if (key.compare("ymin")==0) {
                ok = this->setYMin(values[0]);
            } else if (key.compare("ymax")==0) {
                ok = this->setYMax(values[0]);
            } else if (key.compare("zmin")==0) {
                ok = this->setZMin(values[0]);
            } else if (key.compare("zmax")==0) {
                ok = this->setZMax(values[0]);
            }*/ else if (key.compare("ystep")==0) {
                ok = this->setYStep(values[0]);
            } else if (key.compare("generalizationsuffix")==0) {
                ok = this->setSaveSuffixGeneralization(values[0]);
            } else if (key.compare("data")==0) {
                ok = this->setDataMap(values);
            } else if (key.compare("unselected_data")==0) {
                ok = this->setUnselectedDataMap(values);
            } else if (key.compare("scales")==0) {
                ok = this->setScalesMap(values);
            } else if (key.compare("unselected_scales")==0) {
                ok = this->setUnselectedScalesMap(values);
            } else if (key.compare("survey")==0) {
                continue;
            } else if (key.compare("hat_parameter")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadHatParameter(values[0]);
            } else if (key.compare("epoch_save_step")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadEpochSaveStep(values[0]);
            } else if (key.compare("subsample")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadSubSample(values[0]);
            } else if (key.compare("colsample_bytree")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadColSampleByTree(values[0]);
            } else if (key.compare("xgboost_max_depth")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadMaxDepth(values[0]);
            } else if (key.compare("n_estimator")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadNEstimator(values[0]);
            } else if (key.compare("well_postprocessing")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadWellPostprocessing(values[0]);
            } else if (key.compare("well_post_filtering_freq")==0) {
                ok = m_bnniConfig->getNetworkModel()->loadWellFilterFrequency(values[0]);
            } else if (key.compare("predict_sample_rate")==0) {
                float tmpVal = values[0].toFloat(&ok);
                if (ok) {
                    predictSampleRate = tmpVal;
                }
            }
            logger(key, ok);
        }
        if (hiddenLayersFound) {
            // redo hidden layers to override default behavior from seismic selection
            m_bnniConfig->getNetworkModel()->loadHiddenLayers(hiddenLayersValues);
        }
        if (!referenceCheckpointSet) {
            m_bnniConfig->getNetworkModel()->loadReferenceCheckpoint("");
        }
        computeWellDynamic();


    }
    file.close();

    validateArguments();
}

void BnniMainWindow::saveConfig() {
    QString filename = configFilename;
    QFile file(filename);
    if (!filename.isNull() && !filename.isEmpty() && file.open(QIODevice::WriteOnly)) {
        QTextStream stream(&file);
        //stream << "program_file" << separator << programFilePath << endl;
        stream << "work_dir" << separator << m_bnniConfig->getWorkDir() << Qt::endl;
        stream << "gpu" << separator << QString::number(m_bnniConfig->getNetworkModel()->getNGpus()) << Qt::endl;
        stream << "hiddenLayers";
        for (int i=0; i<m_bnniConfig->getNetworkModel()->getHiddenLayers().size(); i++) {
            stream << separator << m_bnniConfig->getNetworkModel()->getHiddenLayers()[i];
        }
        stream << Qt::endl;
        stream << "learningRate" << separator << QString::number(m_bnniConfig->getNetworkModel()->getLearningRate()) << Qt::endl;
        stream << "momentum" << separator << QString::number(m_bnniConfig->getNetworkModel()->getMomentum()) << Qt::endl;
        stream << "restore" << separator << m_bnniConfig->getNetworkModel()->getSavePrefix() << Qt::endl;
        if (m_bnniConfig->getNetworkModel()->hasReferenceCheckpoint()) {
            stream << "train_initialisation" << separator << m_bnniConfig->getNetworkModel()->getReferenceCheckpoint() << Qt::endl;
        }
        QString writtenJsonFile = jsonfile;
        if (QFileInfo(writtenJsonFile).suffix().compare("json")!=0) {
            writtenJsonFile = QFileInfo(writtenJsonFile).dir().absoluteFilePath(
                    QFileInfo(writtenJsonFile).completeBaseName() + ".json");
            stream << "ubjson" << separator << jsonfile << Qt::endl;
        }
        stream << "json" << separator << writtenJsonFile << Qt::endl;

        QStringList channels;
        QStringList unselected_channels;

        QStringList mapData;
        QStringList mapDataUnselected;
        for (int i=0; i<ui->seismicTable->rowCount(); i++) {
            QStringList buf;
            QStringList mapBuf;
            QString seismic_name = ui->seismicTable->item(i, 0)->data(Qt::UserRole).toString();
            buf << separator << seismic_name;
            mapBuf << separator;

            // find the correct seismic and add
            int j=0;
            while (j<seismics.size() && seismic_name.compare(seismics[j].name)!=0) {
                j++;
            }
            if (j<seismics.size()) {
                buf << subSeparator << QString::number(seismics[j].InputMin) << subSeparator << QString::number(seismics[j].InputMax);
                buf << subSeparator << QString::number(seismics[j].OutputMin) << subSeparator << QString::number(seismics[j].OutputMax);
                if (dataMap.contains(seismic_name)) {
                    QMap<QString, QString>::const_iterator iter = dataMap.find(seismic_name);
                    mapBuf << iter.value();
                }
            }

            if (ui->seismicTable->item(i, 0)->isSelected()) {
                channels << buf;
                mapData << mapBuf;
            } else {
                unselected_channels << buf;
                mapDataUnselected << mapBuf;
            }
        }
        stream << "channels" << channels.join("") << Qt::endl;
        stream << "unselected_channels" << unselected_channels.join("") << Qt::endl;

        stream << "data" << mapData.join("") << Qt::endl;
        stream << "unselected_data" << mapDataUnselected.join("") << Qt::endl;


        QStringList output_channels;
        QStringList unselected_output_channels;

        QStringList scales_list;
        QStringList unselected_scales_list;
        for (int i=0; i<ui->logTable->rowCount(); i++) {
            QStringList buf;
            QStringList mapBuf;
            QString log_name = ui->logTable->item(i, 0)->text();
            buf << separator << log_name;
            mapBuf << separator;

            // find the correct seismic and add
            int j=0;
            while (j<logs.size() && log_name.compare(logs[j].name)!=0) {
                j++;
            }
            if (j<logs.size()) {
                buf << subSeparator << QString::number(logs[j].InputMin) << subSeparator << QString::number(logs[j].InputMax);
                buf << subSeparator << QString::number(logs[j].OutputMin) << subSeparator << QString::number(logs[j].OutputMax);
                buf << subSeparator;

                if (logs[j].preprocessing == LogNone) {
                    buf << "normal";
                } else {
                    buf << "log";
                }

                if (scales.contains(log_name)) {
                    QMap<QString, float>::const_iterator iter = scales.find(log_name);
                    mapBuf << QString::number(iter.value());
                }
            }

            if (ui->logTable->item(i, 0)->isSelected()) {
                output_channels << buf;
                scales_list << mapBuf;
            } else {
                unselected_output_channels << buf;
                unselected_scales_list << mapBuf;
            }
        }
        stream << "output_channels" << output_channels.join("") << Qt::endl;
        stream << "unselected_output_channels" << unselected_output_channels.join("") << Qt::endl;

        stream << "scales" << scales_list.join("") << Qt::endl;
        stream << "unselected_scales" << unselected_scales_list.join("") << Qt::endl;


        if (ui->trainList->count()>0) {
            stream << "train_wells";
            for (int i=0; i<ui->trainList->count(); i++) {
                QString name = ui->trainList->item(i)->data(Qt::UserRole).toString();
                stream << separator << name;//QString(name).replace("|", "\\|");

                int k = 0;

                qDebug() << "Find well associated to" << name;
                while (wells.size()>k && QString::compare(wells[k].name, name)!=0) {
                    qDebug() << wells[k].name;
                    k++;
                }
                if (wells.size()==k) {
                    continue;
                }
                for (int rangeIdx=0; rangeIdx<wells[k].ranges.size(); rangeIdx++) {
                    stream << subSeparator << wells[k].ranges[rangeIdx].min;
                    stream << subSeparator << wells[k].ranges[rangeIdx].max;
                }
            }
            stream << Qt::endl;
        }
        //if (ui->validationList->count()>0) {
	{
            stream << "validation_wells";
            for (int i=0; i<ui->validationList->count(); i++) {
                QString name = ui->validationList->item(i)->data(Qt::UserRole).toString();
                stream << separator << name;//.replace("|", "\\|");

                int k = 0;

                qDebug() << "Find well associated to" << name;
                while (wells.size()>k && QString::compare(wells[k].name, name)!=0) {
                    qDebug() << wells[k].name;
                    k++;
                }
                if (wells.size()==k) {
                    continue;
                }
                for (int rangeIdx=0; rangeIdx<wells[k].ranges.size(); rangeIdx++) {
                    stream << subSeparator << wells[k].ranges[rangeIdx].min;
                    stream << subSeparator << wells[k].ranges[rangeIdx].max;
                }
            }
            stream << Qt::endl;
        }
        //if (ui->blindList->count()>0) {
	{
            stream << "blind_wells";
            for (int i=0; i<ui->blindList->count(); i++) {
                QString name = ui->blindList->item(i)->data(Qt::UserRole).toString();
                stream << separator << name;//.replace("|", "\\|");

                int k = 0;

                qDebug() << "Find well associated to" << name;
                while (wells.size()>k && QString::compare(wells[k].name, name)!=0) {
                    qDebug() << wells[k].name;
                    k++;
                }
                if (wells.size()==k) {
                    continue;
                }
                for (int rangeIdx=0; rangeIdx<wells[k].ranges.size(); rangeIdx++) {
                    stream << subSeparator << wells[k].ranges[rangeIdx].min;
                    stream << subSeparator << wells[k].ranges[rangeIdx].max;
                }
            }
            stream << Qt::endl;
        }
        if (ui->wellList->count()>0) {
            stream << "unused_wells";
            for (int i=0; i<ui->wellList->count(); i++) {
                QString name = ui->wellList->item(i)->data(Qt::UserRole).toString();
                stream << separator << name;//.replace("|", "\\|");

                int k = 0;

                qDebug() << "Find well associated to" << name;
                while (wells.size()>k && QString::compare(wells[k].name, name)!=0) {
                    qDebug() << wells[k].name;
                    k++;
                }
                if (wells.size()==k) {
                    continue;
                }
                for (int rangeIdx=0; rangeIdx<wells[k].ranges.size(); rangeIdx++) {
                    stream << subSeparator << wells[k].ranges[rangeIdx].min;
                    stream << subSeparator << wells[k].ranges[rangeIdx].max;
                }
            }
            stream << Qt::endl;
        }

        stream << "epochs" << separator << m_bnniConfig->getNetworkModel()->getNumEpochs() << Qt::endl;
        stream << "batch_size" << separator << m_bnniConfig->getNetworkModel()->getBatchSize() << Qt::endl;
        stream << "dropout" << separator << m_bnniConfig->getNetworkModel()->getDropout() << Qt::endl;
        stream << "use_bias" << separator << (m_bnniConfig->getNetworkModel()->getUseBias()? "true" : "false") << Qt::endl;


        stream << "precision" << separator;
        if (m_bnniConfig->getNetworkModel()->getPrecision() == PrecisionType::float16) {
            stream << "fp16";
        } else {
            stream << "fp32";
        }
        stream << Qt::endl;

        stream << "optimizer" << separator;
        if (m_bnniConfig->getNetworkModel()->getOptimizer() == Optimizer::gradientDescent) {
            stream << "gd";
        } else if (m_bnniConfig->getNetworkModel()->getOptimizer() == Optimizer::momentum) {
            stream << "momentum";
        } else {
            stream << "adam";
        }
        stream << Qt::endl;

        stream << "network" << separator;
        if (m_bnniConfig->getNetworkModel()->getNetwork() == NeuralNetwork::Dense) {
            stream << "Dense";
        } else if (m_bnniConfig->getNetworkModel()->getNetwork() == NeuralNetwork::Dnn) {
            stream << "Dnn";
        } else if (m_bnniConfig->getNetworkModel()->getNetwork() == NeuralNetwork::Xgboost) {
            stream << "Xgboost";
        }
        stream << Qt::endl;

        stream << "activation" << separator;
        if (m_bnniConfig->getNetworkModel()->getLayerActivation() == Activation::linear) {
            stream << "linear";
        } else if (m_bnniConfig->getNetworkModel()->getLayerActivation() == Activation::sigmoid) {
            stream << "sigmoid";
        } else if (m_bnniConfig->getNetworkModel()->getLayerActivation() == Activation::relu) {
            stream << "relu";
        } else if (m_bnniConfig->getNetworkModel()->getLayerActivation() == Activation::selu) {
            stream << "selu";
        } else if (m_bnniConfig->getNetworkModel()->getLayerActivation() == Activation::leaky_relu) {
            stream << "leaky_relu";
        }
        stream << Qt::endl;

        stream << "seismic_preprocessing" << separator;
        if (m_bnniConfig->getNetworkModel()->getSeismicPreprocessing() == SeismicPreprocessing::SeismicNone) {
            stream << "normal";
        } else {
            stream << "hat";
        }
        stream << Qt::endl;

        stream << "hat_parameter" << separator << m_bnniConfig->getNetworkModel()->getHatParameter() << Qt::endl;

        stream << "well_postprocessing" << separator;
        if (m_bnniConfig->getNetworkModel()->getWellPostprocessing() == WellPostprocessing::WellNone) {
            stream << "normal";
        } else {
            stream << "filter";
        }
        stream << Qt::endl;

        stream << "well_post_filtering_freq" << separator << m_bnniConfig->getNetworkModel()->getWellFilterFrequency() << Qt::endl;

        stream << "batch_norm" << separator;
        if (m_bnniConfig->getNetworkModel()->getBatchNorm()) {
            stream << "true";
        } else {
            stream << "false";
        }

        stream << Qt::endl;

        stream << "xmin" << separator << xmin << Qt::endl;
        stream << "xmax" << separator << xmax << Qt::endl;
        stream << "ymin" << separator << ymin << Qt::endl;
        stream << "ymax" << separator << ymax << Qt::endl;
        stream << "zmin" << separator << zmin << Qt::endl;
        stream << "zmax" << separator << zmax << Qt::endl;
        stream << "ystep" << separator << ystep << Qt::endl;
        stream << "train_sample_rate" << separator << trainSampleRate << Qt::endl;
        stream << "predict_sample_rate" << separator << predictSampleRate << Qt::endl;

        stream << "generalizationdir" << separator << projectDir + "/DATA/3D/" + surveyName + "/DATA/SEISMIC/" << Qt::endl;
        stream << "generalizationsuffix" << separator << saveSuffixGeneralization << Qt::endl;
        stream << "half_window" << separator << halfSignalSize << Qt::endl;
        stream << "epoch_save_step" << separator << m_bnniConfig->getNetworkModel()->getEpochSaveStep() << Qt::endl;
        stream << "subsample" << separator << m_bnniConfig->getNetworkModel()->getSubSample() << Qt::endl;
        stream << "colsample_bytree" << separator << m_bnniConfig->getNetworkModel()->getColSampleByTree() << Qt::endl;
        stream << "xgboost_max_depth" << separator << m_bnniConfig->getNetworkModel()->getMaxDepth() << Qt::endl;
        stream << "n_estimator" << separator << m_bnniConfig->getNetworkModel()->getNEstimator() << Qt::endl;


    }
    file.close();

    // update config combobox name
    QString suffix = getConfigSuffixFromFile(configFilename);
    QString configDisplayName = ui->configComboBox->currentData(Qt::UserRole).toString() + suffix;
    ui->configComboBox->setItemText(ui->configComboBox->currentIndex(), configDisplayName);
}

/**
 * @brief BnniMainWindow::setChannels
 * @param list of Channels (seismic names)
 * @return if there was an unrecognize well or not
 */
bool BnniMainWindow::setChannels(QStringList list) {
    bool out = true;
    for (int j=0; j<list.count(); j++) {
        QStringList values = list[j].split(subSeparator);
        QString name = values.first().split("/").last().split(".xt").first();
        int i=0;
        while (i<ui->seismicTable->rowCount() && name.compare(ui->seismicTable->item(i, 0)->text())!=0) {
            i++;
        }

        if (i<ui->seismicTable->rowCount()) {
            ui->seismicTable->selectRow(i);

            if (values.size()==5) {
                int j=0;
                while (j<seismics.size() && values.first().compare(seismics[j].name)!=0) {
                    j++;
                }
                if (j < seismics.size()) {
                    bool ok;
                    float val = values[1].toFloat(&ok);
                    if (ok) {
                        seismicsForm[j].lineEditInputMin->setText(locale().toString(val));
                        seismicsForm[j].lineEditInputMin->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setChannels failed to read" << values[1] << "as a float" ;
                    }
                    val = values[2].toFloat(&ok);
                    if (ok) {
                        seismicsForm[j].lineEditInputMax->setText(locale().toString(val));
                        seismicsForm[j].lineEditInputMax->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setChannels failed to read" << values[2] << "as a float" ;
                    }
                    val = values[3].toFloat(&ok);
                    if (ok) {
                        seismicsForm[j].lineEditOutputMin->setText(locale().toString(val));
                        seismicsForm[j].lineEditOutputMin->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setChannels failed to read" << values[3] << "as a float" ;
                    }
                    val = values[4].toFloat(&ok);
                    if (ok) {
                        seismicsForm[j].lineEditOutputMax->setText(locale().toString(val));
                        seismicsForm[j].lineEditOutputMax->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setChannels failed to read" << values[4] << "as a float" ;
                    }
                }
            }
        } else {
            qDebug() << "BnniMainWindow::setChannels Could not find value : " << list[j];
            out = false;
        }
    }
    return out;
}

/**
 * @brief BnniMainWindow::setUnselectedSeismic
 * @param list of Channels (seismic names)
 * @return if there was an unrecognize well or not
 */
bool BnniMainWindow::setUnselectedSeismic(QStringList list) {
    bool out = true;
    for (int j=0; j<list.count(); j++) {
        QStringList values = list[j].split(subSeparator);
        QString name = values.first().split("/").last().split(".xt").first();
        int i=0;
        while (i<ui->seismicTable->rowCount() && name.compare(ui->seismicTable->item(i, 0)->text())!=0) {
            i++;
        }

        if (i<ui->seismicTable->rowCount()) {

            if (values.size()==5) {
                int j=0;
                while (j<seismics.size() && values.first().compare(seismics[j].name)!=0) {
                    j++;
                }
                if (j < seismics.size()) {
                    bool ok;
                    float val = values[1].toFloat(&ok);
                    if (ok) {
                        seismicsForm[j].lineEditInputMin->setText(locale().toString(val));
                        seismicsForm[j].lineEditInputMin->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setUnselectedChannels failed to read" << values[1] << "as a float" ;
                    }
                    val = values[2].toFloat(&ok);
                    if (ok) {
                        seismicsForm[j].lineEditInputMax->setText(locale().toString(val));
                        seismicsForm[j].lineEditInputMax->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setUnselectedChannels failed to read" << values[2] << "as a float" ;
                    }
                    val = values[3].toFloat(&ok);
                    if (ok) {
                        seismicsForm[j].lineEditOutputMin->setText(locale().toString(val));
                        seismicsForm[j].lineEditOutputMin->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setUnselectedChannels failed to read" << values[3] << "as a float" ;
                    }
                    val = values[4].toFloat(&ok);
                    if (ok) {
                        seismicsForm[j].lineEditOutputMax->setText(locale().toString(val));
                        seismicsForm[j].lineEditOutputMax->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setUnselectedChannels failed to read" << values[4] << "as a float" ;
                    }
                }
            }
        } else {
            qDebug() << "BnniMainWindow::setChannels Could not find value : " << list[j];
            out = false;
        }
    }
    return out;
}

/**
 * @brief BnniMainWindow::setOutputChannels
 * @param list of Output Channels (log names/kinds)
 * @return if there was an unrecognize well or not
 */
bool BnniMainWindow::setOutputChannels(QStringList list) {
    bool out = true;
    for (int j=0; j<list.count(); j++) {
        QStringList values = list[j].split(subSeparator);
        QString name = values.first().split("/").last().split(".xt").first();
        int i=0;
        while (i<ui->logTable->rowCount() && name.compare(ui->logTable->item(i, 0)->text())!=0) {
            i++;
        }

        if (i<ui->logTable->rowCount()) {
            ui->logTable->selectRow(i);

            if (values.size()==6) {
                int j=0;
                while (j<logs.size() && values.first().compare(logs[j].name)!=0) {
                    j++;
                }
                if (j < logs.size()) {
                    bool ok;
                    float val = values[1].toFloat(&ok);
                    if (ok) {
                        logsForm[j].lineEditInputMin->setText(locale().toString(val));
                        logsForm[j].lineEditInputMin->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setOutputChannels failed to read" << values[1] << "as a float" ;
                    }
                    val = values[2].toFloat(&ok);
                    if (ok) {
                        logsForm[j].lineEditInputMax->setText(locale().toString(val));
                        logsForm[j].lineEditInputMax->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setOutputChannels failed to read" << values[2] << "as a float" ;
                    }
                    val = values[3].toFloat(&ok);
                    if (ok) {
                        logsForm[j].lineEditOutputMin->setText(locale().toString(val));
                        logsForm[j].lineEditOutputMin->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setOutputChannels failed to read" << values[3] << "as a float" ;
                    }
                    val = values[4].toFloat(&ok);
                    if (ok) {
                        logsForm[j].lineEditOutputMax->setText(locale().toString(val));
                        logsForm[j].lineEditOutputMax->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setOutputChannels failed to read" << values[4] << "as a float" ;
                    }

                    if (values[5].toLower().compare("log")==0) {
                        logsForm[j].comboboxPreprocessing->setCurrentIndex(1);
                    } else { // Default is LogNone
                        logsForm[j].comboboxPreprocessing->setCurrentIndex(0);
                    }
                }
            }
        } else {
            qDebug() << "BnniMainWindow::setOutputChannels Could not find value : " << list[j];
            out = false;
        }
    }
    return out;
}

/*
 * bool NetworkParameterForm::setLogPreprocessing(QString txt) {
    bool out = true;
    if (txt.compare("log")==0) {
        logPreprocessingComboBox->setCurrentIndex(LogPreprocessing::LogLn);
    } else if(txt.compare("normal")==0) {
        logPreprocessingComboBox->setCurrentIndex(LogPreprocessing::LogNone);
    } else {
        out = false;
    }
    return out;
}
 */
bool BnniMainWindow::setUnselectedOutputChannels(QStringList list) {
    bool out = true;
    for (int j=0; j<list.count(); j++) {
        QStringList values = list[j].split(subSeparator);
        QString name = values.first().split("/").last().split(".xt").first();
        int i=0;
        while (i<ui->logTable->rowCount() && name.compare(ui->logTable->item(i, 0)->text())!=0) {
            i++;
        }

        if (i<ui->logTable->rowCount()) {
            if (values.size()==6) {
                int j=0;
                while (j<logs.size() && values.first().compare(logs[j].name)!=0) {
                    j++;
                }
                if (j < logs.size()) {
                    bool ok;
                    float val = values[1].toFloat(&ok);
                    if (ok) {
                        logsForm[j].lineEditInputMin->setText(locale().toString(val));
                        logsForm[j].lineEditInputMin->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setUnselectedOutputChannels failed to read" << values[1] << "as a float" ;
                    }
                    val = values[2].toFloat(&ok);
                    if (ok) {
                        logsForm[j].lineEditInputMax->setText(locale().toString(val));
                        logsForm[j].lineEditInputMax->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setUnselectedOutputChannels failed to read" << values[2] << "as a float" ;
                    }
                    val = values[3].toFloat(&ok);
                    if (ok) {
                        logsForm[j].lineEditOutputMin->setText(locale().toString(val));
                        logsForm[j].lineEditOutputMin->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setUnselectedOutputChannels failed to read" << values[3] << "as a float" ;
                    }
                    val = values[4].toFloat(&ok);
                    if (ok) {
                        logsForm[j].lineEditOutputMax->setText(locale().toString(val));
                        logsForm[j].lineEditOutputMax->editingFinished();
                    } else {
                        qDebug() << "BnniMainWindow::setUnselectedOutputChannels failed to read" << values[4] << "as a float" ;
                    }

                    if (values[5].toLower().compare("log")==0) {
                        logsForm[j].comboboxPreprocessing->setCurrentIndex(1);
                    } else { // Default is LogNone
                        logsForm[j].comboboxPreprocessing->setCurrentIndex(0);
                    }
                }
            }
        } else {
            qDebug() << "BnniMainWindow::setUnselectedOutputChannels Could not find value : " << list[j];
            out = false;
        }
    }
    return out;
}

/**
 * @brief BnniMainWindow::setWellMinMax
 * @param list of string
 *     Allowed formats are [name] or [name, indexMin, indexMax]
 * @return true if format is respected and setWellMinMax(QString, int , int) return true
 *
 * Function is a wrapper for bool BnniMainWindow::setWellMinMax(QString name, int minIndex, int maxIndex)
 */
bool BnniMainWindow::setWellMinMax(QStringList list) {
    bool ok = true;
    if ((list.count()%2)==1 && list.count()!=1) {
        // extract values and send them to setWellMinMax(QString, QVector<int>)
        QString name = list[0];
        QVector<int> vals;
        vals.resize(list.size()-1);
        int i = 1;
        while (ok && i<list.size()) {
            int index = locale().toInt(list[i], &ok);
            vals[i-1] = index;
            i++;
        }
        if (ok) {
            ok = setWellMinMax(name, vals);
        }
    } else if (list.count()!=1) {
        // return false for all case list.count not 1 and not 3
        ok = false;
    }
    // if list.count() == 1 -> always return true
    return ok;
}

/**
 * @brief BnniMainWindow::setWellMinMax
 * @param name -> wellname
 * @param vals -> list of index organized as pairs
 * @return true if 0 <= minIndex <= maxIndex < well length and well found with the name
 */
bool BnniMainWindow::setWellMinMax(QString name, QVector<int> vals) {
    bool check = vals.size()%2==0 && vals.size()>0 && vals[0]>=0;
    if (check) {
        int i=1;
        while (check && i<vals.size()) {
            check = vals[i-1] < vals[i];
            i++;
        }
    }
    if (!check) {
        qDebug() << "BnniMainWindow::setWellMinMax(QString, QVector<int>) :first check: indexes are not set correctly";
        return false;
    }

    int k = 0;
    bool ok = true;
    while (wells.size()>k && QString::compare(wells[k].name, name)!=0) {
        qDebug() << wells[k].name;
        k++;
    }
    if (wells.size()>k && vals[vals.size()-1]<wells[k].samples.length()) {
        wells[k].ranges.resize(vals.size()/2);
        for (int rangeIdx=0; rangeIdx<wells[k].ranges.size(); rangeIdx++) {
            wells[k].ranges[rangeIdx].min = vals[rangeIdx*2];
            wells[k].ranges[rangeIdx].max = vals[rangeIdx*2+1];
        }
    } else {
        ok = false;
        if (vals[vals.size()-1]>=wells[k].samples.length()) {
            qDebug() << "BnniMainWindow::setWellMinMax(QString, QVector<int>) :second check: indexes are not set correctly";
        }
    }
    return ok;
}

/**
 * @brief BnniMainWindow::setTrainWells
 * @param list
 * @return if there was an unrecognize well or not
 */
bool BnniMainWindow::setTrainWells(QStringList list) {
    bool out = true;

    for (int j=0; j<list.count(); j++) {
        QStringList _list = list[j].split(subSeparator);
        if ((_list.count()%2)==1) {
            QString name = _list[0].replace("\t","").replace("Sismage2 MainIDL:DmWell/ExistingWellBoreFactory:1.0", "");
            int i=0;
            while (i<ui->wellList->count() && name.compare(ui->wellList->item(i)->text())!=0) {
                i++;
            }
            if (i<ui->wellList->count()) {
                QListWidgetItem* item = new QListWidgetItem(*(ui->wellList->item(i)));
                ui->trainList->addItem(item);
                delete ui->wellList->item(i);
                out = setWellMinMax(_list);
            } else {
                i = 0;

                while (i<ui->validationList->count() && name.compare(ui->validationList->item(i)->text())!=0) {
                    i++;
                }
                if (i<ui->validationList->count()) {
                    QListWidgetItem* item = new QListWidgetItem(*(ui->validationList->item(i)));
                    ui->trainList->addItem(item);
                    delete ui->validationList->item(i);
                    out = setWellMinMax(_list);
                } else {
                    i = 0;

                    while (i<ui->blindList->count() && name.compare(ui->blindList->item(i)->text())!=0) {
                        i++;
                    }
                    if (i<ui->blindList->count()) {
                        QListWidgetItem* item = new QListWidgetItem(*(ui->blindList->item(i)));
                        ui->trainList->addItem(item);
                        delete ui->blindList->item(i);
                        out = setWellMinMax(_list);
                    } else {
						i = 0;

						while (i<ui->trainList->count() && name.compare(ui->trainList->item(i)->text())!=0) {
							i++;
						}
						if (i<ui->trainList->count()) {
							out = setWellMinMax(_list);
						} else {
							out = false;
							qDebug() << "BnniMainWindow::setTrainWells Could not find value : " << name;
						}
                    }
                }
            }
        } else {
            out = false;
            qDebug() << "BnniMainWindow::setTrainWells Could not find correct syntax in : " << list[j];
        }
    }
    return out;
}

/**
 * @brief BnniMainWindow::setValidationWells
 * @param list
 * @return if there was an unrecognize well or not
 */
bool BnniMainWindow::setValidationWells(QStringList list) {
    bool out = true;

    for (int j=0; j<list.count(); j++) {
        QStringList _list = list[j].split(subSeparator);
        if ((_list.count()%2)==1) {
            QString name = _list[0].replace("\t","").replace("Sismage2 MainIDL:DmWell/ExistingWellBoreFactory:1.0", "");
            int i=0;
            while (i<ui->wellList->count() && name.compare(ui->wellList->item(i)->text())!=0) {
                i++;
            }
            if (i<ui->wellList->count()) {
                QListWidgetItem* item = new QListWidgetItem(*(ui->wellList->item(i)));
                ui->validationList->addItem(item);
                delete ui->wellList->item(i);
                out = out && setWellMinMax(_list);
            } else {
                i = 0;

                while (i<ui->trainList->count() && name.compare(ui->trainList->item(i)->text())!=0) {
                    i++;
                }
                if (i<ui->trainList->count()) {
                    QListWidgetItem* item = new QListWidgetItem(*(ui->trainList->item(i)));
                    ui->validationList->addItem(item);
                    delete ui->trainList->item(i);
                    out = out && setWellMinMax(_list);
                } else {
                    i = 0;

                    while (i<ui->blindList->count() && name.compare(ui->blindList->item(i)->text())!=0) {
                        i++;
                    }
                    if (i<ui->blindList->count()) {
                        QListWidgetItem* item = new QListWidgetItem(*(ui->blindList->item(i)));
                        ui->validationList->addItem(item);
                        delete ui->blindList->item(i);
                        out = out && setWellMinMax(_list);
                    } else {
						i = 0;

						while (i<ui->validationList->count() && name.compare(ui->validationList->item(i)->text())!=0) {
							i++;
						}
						if (i<ui->validationList->count()) {
							out = out && setWellMinMax(_list);
						} else {
							out = false;
							qDebug() << "BnniMainWindow::setValidationWells Could not find value : " << name;
						}
                    }
                }
            }
        } else {
            out = false;
            qDebug() << "BnniMainWindow::setValidationWells Could not find correct syntax in : " << list[j];
        }
    }
    return out;
}

/**
 * @brief BnniMainWindow::setValidationWells
 * @param list
 * @return if there was an unrecognize well or not
 */
bool BnniMainWindow::setBlindWells(QStringList list) {
    bool out = true;

    for (int j=0; j<list.count(); j++) {
        QStringList _list = list[j].split(subSeparator);
        if ((_list.count()%2)==1) {
            QString name = _list[0].replace("\t","").replace("Sismage2 MainIDL:DmWell/ExistingWellBoreFactory:1.0", "");
            int i=0;
            while (i<ui->wellList->count() && name.compare(ui->wellList->item(i)->text())!=0) {
                i++;
            }
            if (i<ui->wellList->count()) {
                QListWidgetItem* item = new QListWidgetItem(*(ui->wellList->item(i)));
                ui->blindList->addItem(item);
                delete ui->wellList->item(i);
                out = out && setWellMinMax(_list);
            } else {
                i = 0;

                while (i<ui->trainList->count() && name.compare(ui->trainList->item(i)->text())!=0) {
                    i++;
                }
                if (i<ui->trainList->count()) {
                    QListWidgetItem* item = new QListWidgetItem(*(ui->trainList->item(i)));
                    ui->blindList->addItem(item);
                    delete ui->trainList->item(i);
                    out = out && setWellMinMax(_list);
                } else {
                	i = 0;

					while (i<ui->validationList->count() && name.compare(ui->validationList->item(i)->text())!=0) {
						i++;
					}
					if (i<ui->validationList->count()) {
						QListWidgetItem* item = new QListWidgetItem(*(ui->validationList->item(i)));
						ui->blindList->addItem(item);
						delete ui->validationList->item(i);
						out = out && setWellMinMax(_list);
					} else {
						i = 0;

						while (i<ui->blindList->count() && name.compare(ui->blindList->item(i)->text())!=0) {
							i++;
						}
						if (i<ui->blindList->count()) {
							out = out && setWellMinMax(_list);
						} else {
							out = false;
							qDebug() << "BnniMainWindow::setBlindWells Could not find value : " << name;
						}
					}
                }
            }
        } else {
            out = false;
            qDebug() << "BnniMainWindow::setBlindWells Could not find correct syntax in : " << list[j];
        }
    }
    return out;
}

/**
 * @brief BnniMainWindow::setUnusedWells
 * @param list
 * @return if there was an unrecognize well or not
 */
bool BnniMainWindow::setUnusedWells(QStringList list) {
    bool out = true;

    for (int j=0; j<list.count(); j++) {
        QStringList _list = list[j].split(subSeparator);
        if ((_list.count()%2)==1) {
            QString name = _list[0].replace("\t","").replace("Sismage2 MainIDL:DmWell/ExistingWellBoreFactory:1.0", "");
            int i=0;
            while (i<ui->trainList->count() && name.compare(ui->trainList->item(i)->text())!=0) {
                i++;
            }
            if (i<ui->trainList->count()) {
                QListWidgetItem* item = new QListWidgetItem(*(ui->trainList->item(i)));
                ui->wellList->addItem(item);
                delete ui->trainList->item(i);
                out = setWellMinMax(_list);
            } else {
                i = 0;

                while (i<ui->validationList->count() && name.compare(ui->validationList->item(i)->text())!=0) {
                    i++;
                }
                if (i<ui->validationList->count()) {
                    QListWidgetItem* item = new QListWidgetItem(*(ui->validationList->item(i)));
                    ui->wellList->addItem(item);
                    delete ui->validationList->item(i);
                    out = setWellMinMax(_list);
                } else {
                	i = 0;

					while (i<ui->blindList->count() && name.compare(ui->blindList->item(i)->text())!=0) {
						i++;
					}
					if (i<ui->blindList->count()) {
						QListWidgetItem* item = new QListWidgetItem(*(ui->blindList->item(i)));
						ui->wellList->addItem(item);
						delete ui->blindList->item(i);
						out = out && setWellMinMax(_list);
					} else {
						i = 0;

						while (i<ui->wellList->count() && name.compare(ui->wellList->item(i)->text())!=0) {
							i++;
						}
						if (i<ui->wellList->count()) {
							out = setWellMinMax(_list);
						} else {
							out = false;
							qDebug() << "BnniMainWindow::setUnusedWells Could not find value : " << name;
						}
					}
                }
            }
        } else {
            out = false;
            qDebug() << "BnniMainWindow::setUnusedWells Could not find correct syntax in : " << list[j];
        }
    }
    return out;
}

bool BnniMainWindow::setXMin(QString value) {
    bool out;
    int val = value.toInt(&out);
    if (out) {
        xmin = val;
    }
    return out;
}

bool BnniMainWindow::setXMax(QString value) {
    bool out;
    int val = value.toInt(&out);
    if (out) {
        xmax = val;
    }
    return out;
}

bool BnniMainWindow::setYMin(QString value) {
    bool out;
    int val = value.toInt(&out);
    if (out) {
        ymin = val;
    }
    return out;
}

bool BnniMainWindow::setYMax(QString value) {
    bool out;
    int val = value.toInt(&out);
    if (out) {
        ymax = val;
    }
    return out;
}

bool BnniMainWindow::setZMin(QString value) {
    bool out;
    int val = value.toInt(&out);
    if (out) {
        zmin = val;
    }
    return out;
}

bool BnniMainWindow::setZMax(QString value) {
    bool out;
    int val = value.toInt(&out);
    if (out) {
        zmax = val;
    }
    return out;
}

bool BnniMainWindow::setYStep(QString value) {
    bool out;
    int val = value.toInt(&out);
    if (out) {
        ystep= val;
    }
    return out;
}

bool BnniMainWindow::setSaveSuffixGeneralization(QString value) {
    //QFileInfo file(value);
    //bool out = file.isWritable();
    bool out = true;
    if (out) {
        saveSuffixGeneralization = value;
    }
    return out;
}

bool BnniMainWindow::setDataMap(QStringList values) {
    QMap<QString, QString> map;
    int j=0;
    for(int i=0; i<ui->seismicTable->rowCount(); i++) {
        if (ui->seismicTable->item(i, 0)->isSelected()) {
            qDebug()<< ui->seismicTable->item(i, 0)->text();
            map.insert(ui->seismicTable->item(i, 0)->data(Qt::UserRole).toString(), values[j]);
            j++;
        }
    }
    bool out = j==values.size();
    if (out) {
        for (QString key : map.keys()) {
            dataMap.insert(key, map.find(key).value());
        }
    }
    return out;
}

bool BnniMainWindow::setUnselectedDataMap(QStringList values) {
    QMap<QString, QString> map;
    int j=0;
    for(int i=0; i<ui->seismicTable->rowCount(); i++) {
        if (!ui->seismicTable->item(i, 0)->isSelected()) {
            qDebug()<< ui->seismicTable->item(i, 0)->text();
            map.insert(ui->seismicTable->item(i, 0)->data(Qt::UserRole).toString(), values[j]);
            j++;
        }
    }
    bool out = j==values.size();
    if (out) {
        for (QString key : map.keys()) {
            dataMap.insert(key, map.find(key).value());
        }
    }
    return out;
}

bool BnniMainWindow::setScalesMap(QStringList values) {
    QMap<QString, float> map;
    int i=0, j=0;
    bool ok=true;
    while (ok && i<ui->logTable->rowCount()) {
        if (ui->logTable->item(i, 0)->isSelected()) {
            map.insert(ui->logTable->item(i, 0)->data(Qt::UserRole).toString(), values[j].toInt(&ok));
            j++;
        }
        i++;
    }
    bool out = ok && j==values.size();
    if (out) {
        for (QString key : map.keys()) {
            scales.insert(key, map.find(key).value());
        }
    }
    return out;
}

bool BnniMainWindow::setUnselectedScalesMap(QStringList values) {
    QMap<QString, float> map;
    int i=0, j=0;
    bool ok=true;
    while (ok && i<ui->logTable->rowCount()) {
        if (!ui->logTable->item(i, 0)->isSelected()) {
            map.insert(ui->logTable->item(i, 0)->data(Qt::UserRole).toString(), values[j].toInt(&ok));
            j++;
        }
        i++;
    }
    bool out = ok && j==values.size();
    if (out) {
        for (QString key : map.keys()) {
            scales.insert(key, map.find(key).value());
        }
    }
    return out;
}

/*void BnniMainWindow::updateConfigFilename(QString configFilename) {
    this->configFilename = configFilename;
    if (debug) {
        qDebug() << "NetworkParameterForm::updateConfigFilename" << this->configFilename;
    }
    validateArguments();
}*/

/*void BnniMainWindow::searchConfigFilename(QString dir_path) {
    QString val;
    if (dir_path.isNull() || dir_path.isEmpty()) {
        QFileInfo file(configFilename);
        QString searchPath = file.dir().absolutePath();

        if (!jsonfile.isNull() && !jsonfile.isEmpty()) {
            QDir dir = QFileInfo(jsonfile).dir();

            searchPath = dir.absolutePath();
        }
        val = QFileDialog::getExistingDirectory(this, tr("Select config directory"), searchPath);
    } else {
        val = dir_path;
    }
    if (!val.isEmpty() && !val.isNull()) {
        parameterForm->setWorkDir(val);
        val = val + "/config.txt";
        ui->configLineEdit->setText(val);
    }
}*/

void BnniMainWindow::setHalfWindow(int halfWindow) {
    halfSignalSize = halfWindow;
    if (halfWindow>=0) {
//        QStringList list;
//        int i=0;
//        int val = (2*halfWindow+1)*seismics.size();
//        while (i<3 && val>0) {
//            list << locale().toString(val);
//            val /= 2;
//            i++;
//        }
//        parameterForm->setHiddenLayers(list);
    	updateHiddenLayersSize();
    }
}

void BnniMainWindow::updateHiddenLayersSize() {
	int N = 0;
	for (int j=0; j<ui->seismicTable->rowCount(); j++) {
		if (ui->seismicTable->item(j, 0)->isSelected()) {
			N++;
		}
	}
	if (N>0 && halfSignalSize>=0) {
		QVector<unsigned int> list;
		int i=0;
		int val = (2*halfSignalSize+1)*N;
		while (i<3 && val>0) {
			list << val;
			val /= 2;
			i++;
		}
		m_bnniConfig->getNetworkModel()->setHiddenLayers(list);
	}
}

void BnniMainWindow::sliceGeneralization(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, QString checkpoint, QString suffix) {
    if (generalizationProcess->state()!=QProcess::NotRunning) {
        qWarning() << "Cannot launch a new generalization process an old one is running";
        emit sliceGeneralizationRefusedSignal(checkpoint);
        return;
    }
    this->xmin = xmin;
    this->xmax = xmax;
    this->ymin = ymin;
    this->ymax = ymax;
    this->zmin = zmin;
    this->zmax = zmax;

    QString filename = configFilename;
    this->saveConfig();


    QString program;
    if (zmin==zmax) {
        program = QDir(programLocation).absoluteFilePath("spliter_y.sh");
    } else {
        program = QDir(programLocation).absoluteFilePath("spliter_z.sh");
    }
    QStringList arguments;
    arguments << QString::number(zmin) << QString::number(zmax) << QString::number(ymin) << QString::number(ymax);
    arguments << QDir(programLocation).absoluteFilePath(programFileName + "_predict.sh");
    arguments << "--config" << filename << "--restore" << checkpoint << "--work_dir" << "" << "--xmin" << QString::number(xmin) << "--xmax" << QString::number(xmax) <<
                 "--generalizationsuffix" << suffix;
    generalizationCheckpoint = checkpoint;
    generalizationProcess->setWorkingDirectory(programLocation);

    connect(generalizationProcess, SELECT<int, QProcess::ExitStatus>::OVERLOAD_OF(&QProcess::finished), this, &BnniMainWindow::sliceGeneralizationFinished);
    connect(generalizationProcess, SELECT<QProcess::ProcessError>::OVERLOAD_OF(&QProcess::errorOccurred), this, &BnniMainWindow::sliceGeneralizationError);

    generalizationProcess->start(program, arguments);
    qDebug() << program << arguments.join(" ");

}

void BnniMainWindow::randomGeneralization(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, QString checkpoint, QString suffix, QStringList data, QString generalizationDir) {
    if (generalizationProcess->state()!=QProcess::NotRunning) {
        qWarning() << "Cannot launch a new generalization process an old one is running";
        emit sliceGeneralizationRefusedSignal(checkpoint);
        return;
    }
    this->xmin = xmin;
    this->xmax = xmax;
    this->ymin = ymin;
    this->ymax = ymax;
    this->zmin = zmin;
    this->zmax = zmax;

    QString filename = configFilename;
    this->saveConfig();


    QString program;
    if (zmin==zmax) {
        program = QDir(programLocation).absoluteFilePath("spliter_y.sh");
    } else {
        program = QDir(programLocation).absoluteFilePath("spliter_z.sh");
    }
    QStringList arguments;
    arguments << QString::number(zmin) << QString::number(zmax) << QString::number(ymin) << QString::number(ymax);
    arguments << QDir(programLocation).absoluteFilePath(programFileName + "_predict.sh");
    arguments << "--config" << filename << "--restore" << checkpoint << "--work_dir" << "" << "--xmin" << QString::number(xmin) << "--xmax" << QString::number(xmax) <<
                 "--generalizationsuffix" << suffix << "--generalizationdir" << generalizationDir << "--data" << data;
    generalizationCheckpoint = checkpoint;
    generalizationProcess->setWorkingDirectory(m_bnniConfig->getWorkDir());

    connect(generalizationProcess, SELECT<int, QProcess::ExitStatus>::OVERLOAD_OF(&QProcess::finished), this, &BnniMainWindow::sliceGeneralizationFinished);
    connect(generalizationProcess, SELECT<QProcess::ProcessError>::OVERLOAD_OF(&QProcess::errorOccurred), this, &BnniMainWindow::sliceGeneralizationError);

    generalizationProcess->start(program, arguments);
    qDebug() << program << arguments.join(" ");
}

void BnniMainWindow::sliceGeneralizationFinished(int code, QProcess::ExitStatus status) {

    disconnect(generalizationProcess, SELECT<int, QProcess::ExitStatus>::OVERLOAD_OF(&QProcess::finished), this, &BnniMainWindow::sliceGeneralizationFinished);
    disconnect(generalizationProcess, SELECT<QProcess::ProcessError>::OVERLOAD_OF(&QProcess::errorOccurred), this, &BnniMainWindow::sliceGeneralizationError);

    if (code==0) {
        emit sliceGeneralizationFinishedSignal(generalizationCheckpoint);
    } else {
        emit sliceGeneralizationErrorSignal(generalizationCheckpoint);
    }
}

void BnniMainWindow::sliceGeneralizationError(QProcess::ProcessError err) {

}

void BnniMainWindow::carteGeneralization(QString horizon_top, int shift_top, QString horizon_bottom, int shift_bottom, QString checkpoint) {
    if (generalizationProcess->state()!=QProcess::NotRunning) {
        qWarning() << "Cannot launch a new generalization process an old one is running";
        emit carteGeneralizationRefusedSignal(checkpoint);
        return;
    }

    DataInfo info = getDataInfo();

    QString filename = configFilename;
    this->saveConfig();
    QString program = QDir(programLocation).absoluteFilePath(programFileName + "_predict.sh");
    QStringList arguments;
    arguments << "0" << QString::number(info.dim.getI()-1) << "0" << QString::number(info.dim.getJ()-1) << program;
    arguments << "--config" << filename << "--restore" << checkpoint << "--work_dir" << "" << \
                 "--horizon_top" << horizon_top << "--shift_top" << QString::number(shift_top) << "--mode" << "carte" << \
                 "--horizon_bottom" << horizon_bottom << "--shift_bottom" << QString::number(shift_bottom) << \
                 "--step" << QString::number(info.step.getI()) << "--origin" << QString::number(info.origin.getI()) << \
                 "--generalizationdir" << projectDir + "/DATA/3D/" + surveyName + "/DATA/HORIZONS";
    generalizationCheckpoint = checkpoint;
    generalizationProcess->setWorkingDirectory(m_bnniConfig->getWorkDir());

    connect(generalizationProcess, SELECT<int, QProcess::ExitStatus>::OVERLOAD_OF(&QProcess::finished), this, &BnniMainWindow::carteGeneralizationFinished);
    connect(generalizationProcess, SELECT<QProcess::ProcessError>::OVERLOAD_OF(&QProcess::errorOccurred), this, &BnniMainWindow::carteGeneralizationError);

    QString _program = QDir(programLocation).absoluteFilePath("spliter.sh");
    generalizationProcess->start(_program, arguments);
    qDebug() << arguments.join(" ");

}

void BnniMainWindow::carteGeneralizationFinished(int code, QProcess::ExitStatus status) {

    disconnect(generalizationProcess, SELECT<int, QProcess::ExitStatus>::OVERLOAD_OF(&QProcess::finished), this, &BnniMainWindow::carteGeneralizationFinished);
    disconnect(generalizationProcess, SELECT<QProcess::ProcessError>::OVERLOAD_OF(&QProcess::errorOccurred), this, &BnniMainWindow::carteGeneralizationError);

    if (code==0) {
        emit carteGeneralizationFinishedSignal(generalizationCheckpoint);
    } else {
        emit carteGeneralizationErrorSignal(generalizationCheckpoint);
    }
}

void BnniMainWindow::carteGeneralizationError(QProcess::ProcessError err) {

}

DataInfo BnniMainWindow::getDataInfo() {
    DataInfo info;
    if (seismics.count()!=0 && dataMap.size()!=0) {
        QString filename = dataMap.values().first();
//        murat::io::InputOutputCube<float>* cube = murat::io::openCube<float>(filename.toStdString().c_str());
        inri::Xt xt(filename.toStdString().c_str());
        if (xt.is_valid()) {
            info.dim = CubeDimension(xt.nSamples(), xt.nRecords(), xt.nSlices());
            info.step = CubeStep(xt.stepSamples(), xt.stepRecords(), xt.stepSlices());
            info.origin = CubeOrigin(xt.startSamples(), xt.startRecord(), xt.startSlice());
        }
//        delete cube;
    } else if(seismics.count()!=0 && dataMap.size()==0) {
        QString filename = seismics[0].name.split("/").last();
        QString surveyName = seismics[0].name.split("\t").last().split("/").first();
        filename = this->projectDir+"/DATA/3D/"+surveyName+"/DATA/SEISMIC/seismic3d."+filename;
//        murat::io::InputOutputCube<float>* cube = murat::io::openCube<float>(filename.toStdString().c_str());
        inri::Xt xt(filename.toStdString().c_str());
        if (xt.is_valid()) {
            info.dim = CubeDimension(xt.nSamples(), xt.nRecords(), xt.nSlices());
            info.step = CubeStep(xt.stepSamples(), xt.stepRecords(), xt.stepSlices());
            info.origin = CubeOrigin(xt.startSamples(), xt.startRecord(), xt.startSlice());
        }
//        delete cube;
    } else {
        qDebug() << "getDataInfo called but there is no seismic to use to get informations.";
    }
    return info;
}

void BnniMainWindow::setProject(QString _projectDir) {
    projectDir = _projectDir;
    QString _project = QDir(_projectDir).dirName();
    ui->projectLineEdit->setText(_project);

    QDir trainingsetDir(_projectDir+"/DATA/NEURONS/neurons2/LogInversion2Problem3");
    QFileInfoList dirs = trainingsetDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);

    ui->jsonComboBox->clear();
    ui->jsonComboBox->addItem("");
    for (int i=0; i<dirs.length(); i++) {
        ui->jsonComboBox->addItem(dirs[i].baseName());
    }

    // save last project
    QSettings settings;
    settings.setValue(LAST_PROJECT_PATH_IN_SETTINGS, _projectDir);
    unsetTrainingSetSetting();
}

void BnniMainWindow::setSurvey(QString output) {
    if (QDir(projectDir + "/DATA/3D/" + output).exists()) {
        surveyName = output;

        //ui->surveyLineEdit->setText(output);
    }
}

void BnniMainWindow::updateSeismicDataMap() {
    QVector<std::pair<QString, QString>> array;
    QVector<std::pair<float, float>> seismicDynamic;
    dataMap.clear();
    for (QModelIndex& e : ui->seismicTable->selectionModel()->selectedRows()) {
        QString _name = ui->seismicTable->item(e.row(), 0)->data(Qt::UserRole).toString();
        QString name = _name.split("/").last();
        QString survey_name = _name.split("\t").last().split("/").first();
        array.append(std::pair<QString, QString>(name, this->projectDir + "/DATA/3D/"+survey_name+"/DATA/SEISMIC/seismic3d."+name));
        dataMap.insert(_name, array.last().second);

        QVector<Parameter>::iterator iter = seismics.begin();
        while(iter!=seismics.end() && iter->name.compare(_name)!=0) {
            iter++;
        }
        if (iter!=seismics.end()) {
            Parameter& param = *iter;
            seismicDynamic.append(std::pair<int,int>(param.InputMin, param.InputMax));
        } else {
            qDebug() << "Could not find name in seismics";
        }
    }
}

void BnniMainWindow::setProgramLocation(QString dirPath) {
    programLocation = dirPath;

    if (process->state()!=QProcess::NotRunning || generalizationProcess->state()!=QProcess::NotRunning) {
        QMessageBox::warning(this, tr("Program location change while a process is running"), tr("Program location changed while a process is running. Beware, the active process use the old program."));
    }
}

void BnniMainWindow::setInterfaceProgramLocation(QString dirPath) {
    interfaceProgramLocation = dirPath;
}

void BnniMainWindow::logTableContexMenu(const QPoint& pos) {
	QMenu menu;
	menu.addAction("Compute log", [this]() {


		QStringList logNames;
		for (const LogParameter& log : this->logs) {
			logNames.append(log.name);
		}

		ComputeLogDialog dialog(logNames, this);
		int code = dialog.exec();
		if (code == QDialog::Accepted) {
			QString logName = dialog.getLogName();
			QString dtName = dialog.getLogDt();
			QString attributName = dialog.getLogAttribut();
			int waveletSize = dialog.getWaveletSize();
			float frequency = dialog.getFrequency();

			int indexDt = 0;
			while (indexDt<logs.count() && logs[indexDt].name.compare(dtName)!=0) {
				indexDt++;
			}

			int indexAttribut = 0;
			while (indexAttribut<logs.count() && logs[indexAttribut].name.compare(attributName)!=0) {
				indexAttribut++;
			}
			if (indexDt<logs.count() && indexAttribut<logs.count()) {
				this->computeReflectivityLog(logName, indexDt, indexAttribut, frequency, waveletSize);
			} else {
				QMessageBox::warning(this, tr("Fail to match log from dialog"), tr("Dialog log selections outputs mismatch stored logs. Computation will not be run."));
			}
		}
	});


	QPoint globalPos = ui->logTable->mapToGlobal(pos);
	menu.exec(globalPos);
}

void BnniMainWindow::computeReflectivityLog(QString name, int indexDt, int indexAttribut, float frequency, int waveletSize) {
	LogParameter newLog;
	newLog.name = name;
	newLog.OutputMin = 0.25;
	newLog.OutputMax = 0.75;

	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::min();

	// get pasech
	float pasech = getDataInfo().step.getI();

	std::vector<float> dt, attribut, output;

	for (int idxWell=0; idxWell<wells.count(); idxWell++) {
		Well& well = wells[idxWell] ;
		int N = well.samples.count();

		// fill dt & attribut;
		dt.resize(N);
		attribut.resize(N);
		output.resize(N);

		for (int i=0; i<N; i++) {
			dt[i] = well.samples[i].logVals[indexDt];
			attribut[i] = well.samples[i].logVals[indexAttribut];
		}

		// compute output
		reflectivity(dt.data(), attribut.data(), pasech, 6, 7, 0, frequency, output.data(), N);

		// compute min max
		for (int i=0; i<output.size(); i++) {
			if (output[i]<min) {
				min = output[i];
			}
			if (output[i]>max) {
				max = output[i];
			}
		}

		// add to well
		for (int i=0; i<N; i++) {
			well.samples[i].logVals.append(output[i]);
		}
	}
	float amp = std::max(fabs(max), fabs(min));
	min = -amp;
	max = amp;

	newLog.min = min;
	newLog.max = max;
	newLog.InputMin = min;
	newLog.InputMax = max;


	logs.append(newLog);

	cache_wellIndexRanges.clear();

	computeWellDynamic();
	updateLogs();
	updateWells();

	saveCurrentStateToJsonFile();
}

void BnniMainWindow::loadTrainingSet(int index) {
    if (!ui->jsonComboBox->itemText(index).isEmpty()) {
    	bool loadingResult;
        QString jsonOri = getTrainingSetDirPath(projectDir, ui->jsonComboBox->itemText(index))+"/trainingset.json";
        QString jsonBNNI = getTrainingSetDirPath(projectDir, ui->jsonComboBox->itemText(index))+"/BNNI_trainingset.json";
        QString ubjson = getTrainingSetDirPath(projectDir, ui->jsonComboBox->itemText(index))+"/trainingset.ubjson";
    	QFileInfo oriFileInfo(jsonOri);
    	QFileInfo bnniFileInfo(jsonBNNI);
    	QFileInfo ubjsonFileInfo(ubjson);

    	bool loadedUbjson = ubjsonFileInfo.exists() && ubjsonFileInfo.isFile() && ubjsonFileInfo.isReadable() && loadUbjson(ubjson);

    	loadingResult = bnniFileInfo.exists() && bnniFileInfo.isFile() &&
    			bnniFileInfo.isReadable() && oriFileInfo.lastModified()<bnniFileInfo.lastModified();
    	loadingResult = loadingResult || loadedUbjson;
    	if (!loadedUbjson && loadingResult) {
    		qDebug() << "loadTrainingSet : try file " << jsonBNNI;
    		loadingResult = loadJson(jsonBNNI);
    	}
    	if (!loadedUbjson && !loadingResult) {
    		qDebug() << "loadTrainingSet : invalid file " << jsonBNNI;
    	}

    	if (!loadedUbjson && !loadingResult && oriFileInfo.exists() && oriFileInfo.isFile() &&
    			oriFileInfo.isReadable()) {
    		qDebug() << "loadTrainingSet : try file " << jsonOri;
    		loadingResult = loadJson(jsonOri);
    		if (loadingResult) {
    			qDebug() << "loadTrainingSet : copy" << jsonOri;
    			if (bnniFileInfo.exists() && bnniFileInfo.isFile()) {
    				QFile::remove(jsonBNNI);
    			}
    			QFile::copy(jsonOri, jsonBNNI);
    			this->jsonfile = jsonBNNI;
    		} else if ((bnniFileInfo.exists() && bnniFileInfo.isFile() && bnniFileInfo.isReadable() && bnniFileInfo.isWritable()) ||
    					!bnniFileInfo.exists()) {

    			qDebug() << "loadTrainingSet : convert" << jsonOri;

    			// Correct json format of jsonOri
//                QFile::copy(":/bnni/convertJson.py" , "/tmp/BNNI_interface_python_script.py");
//                if (!QDir("/tmp/jsoncomment").exists()) {
//                    QDir("/tmp").mkdir("jsoncomment");
//                }
//
//                if (!QDir("/tmp/jsoncomment/package").exists()) {
//                    QDir("/tmp/jsoncomment").mkdir("package");
//                }

//                QFile::copy(":/python/jsoncomment/__init__.py" , "/tmp/jsoncomment/__init__.py");
//                QFile::copy(":/python/jsoncomment/COPYING" , "/tmp/jsoncomment/COPYING");
//                QFile::copy(":/python/jsoncomment/README.md" , "/tmp/jsoncomment/README.md");
//                QFile::copy(":/python/jsoncomment/README.rst" , "/tmp/jsoncomment/README.rst");
//                QFile::copy(":/python/jsoncomment/package/__init__.py" , "/tmp/jsoncomment/package/__init__.py");
//                QFile::copy(":/python/jsoncomment/package/comments.py" , "/tmp/jsoncomment/package/comments.py");
//                QFile::copy(":/python/jsoncomment/package/wrapper.py" , "/tmp/jsoncomment/package/wrapper.py");

//                QFile myfile("/tmp/BNNI_interface_python_script.py");
//                myfile.setPermissions(myfile.permissions() | QFile::ExeGroup | QFile::ExeOther | QFile::ExeUser);

                QProcess process;
                QStringList arguments;
                arguments << jsonOri << jsonBNNI;
//                process.setWorkingDirectory("/tmp");
                process.start(interfaceProgramLocation + "/launch_convertJson.sh", arguments);
                process.waitForFinished(-1);
                //if (process.exitStatus()==QProcess::NormalExit) {
                if (process.exitCode()==0) {
                	qDebug() << "loadTrainingSet : try converted" << jsonBNNI;
                    qDebug() << process.readAll();
                    loadingResult = loadJson(jsonBNNI);
                } else {
                	loadingResult = false;
                    QMessageBox::critical(this, tr("Conversion failure"),
                    		tr("The conversion process encountered an error, please check files ")+jsonOri+tr(" and ")+jsonBNNI+tr(" : \n")+ process.readAllStandardError());
                }

    		}
    	}

    	if (!loadingResult) {
    		// Load empty json file to clear all caches
    		this->loadJson("");
    		QMessageBox::critical(this, tr("Fail to load training set"),
    				tr("The training set could not be loaded, please check permissions and file consistency."));
    	} else {
    		// save last trainingset
    		QSettings settings;
    		settings.setValue(LAST_TRAININGSET_PATH_IN_SETTINGS, jsonOri);
    		unsetConfigSetting();

    		// select all seismics and first log
    		for (int i=0; i<ui->seismicTable->rowCount(); i++) {
    			ui->seismicTable->selectRow(i);
    		}

    		if (ui->logTable->rowCount()>0) {
    			ui->logTable->selectRow(0);
    		}
    	}

    } else {
        this->loadJson("");
    }
}

QString BnniMainWindow::findIJKDirectory() {
	QString IJK_dir = "";

	bool goal_reached = false;
	int i = 0;
	int N = seismics.size()+1;

	QString basePath = projectDir + "/DATA/3D/" + surveyName + "/ImportExport/IJK/";
	while (!goal_reached && i<N) {
		QString addOnPath = "";
		if (i>0) {
			// addOnPath use seismic name
			QString _name = seismics[i-1].name;
			QString seismicFilePath = _name.split("/").last();
			QString survey_name = _name.split("\t").last().split("/").first();
			seismicFilePath = this->projectDir+"/DATA/3D/"+survey_name+"/DATA/SEISMIC/seismic3d."+seismicFilePath;
			//qDebug() << "Seismic file" << seismicFilePath;
			QString descFilePath = seismicFilePath.section(".", 0, -2) + ".desc";
			QFileInfo descFileInfo = QFileInfo(descFilePath);
			if (descFileInfo.exists() && descFileInfo.isFile()) { // if desc is here we use it to find sismage name
				QFile descFile(descFilePath);
				if (descFile.open(QIODevice::ReadOnly)) {
					QTextStream stream(&descFile);
					QString line;
					bool test = false;

					while (!test && stream.readLineInto(&line)) {
						//qDebug() << "DEBUG line : " << line << "-section-" << line.section("=", 0, 0) << (line.section("=", 0, 0).compare("name")==0) << (line.section("=", 0, 0).compare(QString("name"))==0);
						test = line.section("=", 0, 0).compare("name")==0;
						if(test) {
							addOnPath = line.section("=", 1);
						}
					}
				}
			} else { // else we use the name from seismic filename
				addOnPath = seismicFilePath.section("/", -1).section(".", 1, 1);
			}
		} // else i=0 => no addOnPath

		QString path = basePath + addOnPath + "/WELL_TRAJECTORIES";

		goal_reached = QDir(path).exists();
		if (goal_reached) {
			IJK_dir = basePath + addOnPath + "/";
		}

		i++;
	}
	return IJK_dir;
}

void BnniMainWindow::loadCurrentSettings() {
	// load them all first because they are unset during the loading
	QSettings settings;
	QString projectPath = settings.value(LAST_PROJECT_PATH_IN_SETTINGS, "").toString();
	QString trainingSetPath = settings.value(LAST_TRAININGSET_PATH_IN_SETTINGS, "").toString();
	QString configPath = settings.value(LAST_CONFIG_PATH_IN_SETTINGS, "").toString();

	// remove excessive /
	while (projectPath.size()>1 && projectPath.last(1).compare("/")==0) {
		projectPath.truncate(projectPath.size()-1);
	}

	QFileInfo projectInfo(projectPath);
	bool projectValid = !projectPath.isNull() && !projectPath.isEmpty() && projectInfo.exists() && projectInfo.isDir();
	QFileInfo trainingSetInfo(trainingSetPath);
	bool trainingSetValid = projectValid && !trainingSetPath.isNull() && !trainingSetPath.isEmpty() && trainingSetInfo.exists() && trainingSetInfo.isFile();
	QFileInfo configInfo(configPath);
	bool configValid = trainingSetValid && !configPath.isNull() && !configPath.isEmpty() && configInfo.exists() && configInfo.isFile();

	if (trainingSetValid) {
		QString text;
		QString projectName = projectInfo.fileName();
		QString trainingSetName = trainingSetInfo.dir().dirName();
		if (configValid) {
			QString configName = configInfo.dir().dirName();
			text = "Do you want to load project: "+projectName+", trainingset: "+trainingSetName+", config: "+configName;
		} else {
			text = "Do you want to load project: "+projectName+", trainingset: "+trainingSetName;
		}

		QMessageBox msgBox(this);
		msgBox.setWindowTitle("Previous session detected.");
		msgBox.setText(text);
		msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
		msgBox.setDefaultButton(QMessageBox::Ok);
		int ret = msgBox.exec();

		if (ret!=QMessageBox::StandardButton::Ok) {
			trainingSetValid = false;
			configValid = false;
		}
	}

	if (projectValid) {
		setProject(projectPath);
	}
	if (trainingSetValid) {
		// search the training set with the right name
		QString trainingSetName = QFileInfo(trainingSetPath).dir().dirName();

		int i=0;
		bool notFound = true;
		while (notFound && i<ui->jsonComboBox->count()) {
			notFound = ui->jsonComboBox->itemText(i).compare(trainingSetName)!=0;

			if (notFound) {
				i++;
			}
		}
		if (!notFound) {
			ui->jsonComboBox->setCurrentIndex(i);
		}
	}
	if (configValid) {
		// search the training set with the right name
		QString configName = QFileInfo(configPath).dir().dirName();

		int i=0;
		bool notFound = true;
		while (notFound && i<ui->configComboBox->count()) {
			notFound = ui->configComboBox->itemData(i, Qt::UserRole).toString().compare(configName)!=0;

			if (notFound) {
				i++;
			}
		}
		if (!notFound) {
			ui->configComboBox->setCurrentIndex(i);
		}
	}
}

void BnniMainWindow::unsetTrainingSetSetting() {
	QSettings settings;
	settings.remove(LAST_TRAININGSET_PATH_IN_SETTINGS);
	unsetConfigSetting();
}

void BnniMainWindow::unsetConfigSetting() {
	QSettings settings;
	settings.remove(LAST_CONFIG_PATH_IN_SETTINGS);
}

QString BnniMainWindow::getConfigSuffixFromFile(const QString& configFilePath) {
    bool networkFound;
    NeuralNetwork configNetwork = ConfigReader::getNetworkFromFile(configFilePath, &networkFound);

    QString suffix = "";
    if (networkFound && configNetwork==NeuralNetwork::Xgboost) {
        suffix = " (trees)";
    } else if (networkFound) {
        suffix = " (dense)";
    }
    return suffix;
}

QString BnniMainWindow::getTrainingSetDirPath(const QString& projectPath, const QString& trainingSetName) {
    return projectPath + "/DATA/NEURONS/neurons2/LogInversion2Problem3/" + trainingSetName;
}

QString BnniMainWindow::getConfigDirPath(const QString& projectPath, const QString& trainingSetName, const QString& configName) {
    return getTrainingSetDirPath(projectPath, trainingSetName) + "/" + configName;
}

QString BnniMainWindow::getConfigFilePath(const QString& projectPath, const QString& trainingSetName, const QString& configName) {
    return getConfigDirPath(projectPath, trainingSetName, configName) + "/config.txt";
}
