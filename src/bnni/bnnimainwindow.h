#ifndef SRC_BNNI_BNNIMAINWINDOW_H
#define SRC_BNNI_BNNIMAINWINDOW_H

#include <QMainWindow>
#include <QListWidget>
#include <QProcess>
#include <QColor>
#include <utility>

#include <rapidjson/document.h>
#include "structures.h"

class WellViewer;
class BnniConfig;
class CollapsableScrollArea;
class QLineEdit;
class NetworkParameterForm;
class AdvancedParameterForm;

namespace Ui {
class BnniMainWindow;
}

class BnniMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit BnniMainWindow(QWidget *parent = 0);
    ~BnniMainWindow();

    DataInfo getDataInfo();
    void setProject(QString);
    void setSurvey(QString);
    void setProgramLocation(QString);
    void setInterfaceProgramLocation(QString);

    void setHalfWindow(int halfWindow);

    static QString getTrainingSetDirPath(const QString& projectPath, const QString& trainingSetName);
    static QString getConfigDirPath(const QString& projectPath, const QString& trainingSetName, const QString& configName);
    static QString getConfigFilePath(const QString& projectPath, const QString& trainingSetName, const QString& configName);

    static float DEFAULT_SAMPLE_RATE;

private:
    Ui::BnniMainWindow *ui;
    //rapidjson::Document document;

    QVector<LogParameter> logs;
    QVector<Parameter> seismics;
    QVector<SeismicForm> seismicsForm; // This does not own the pointers in it, the ui does. So keep it synchronize
    QVector<LogForm> logsForm; // This does not own the pointers in it, the ui does. So keep it synchronize
    float trainSampleRate; // to load from json and add to config to avoid reading json
    float predictSampleRate;

    // Generalization data
    QMap<QString, QString> dataMap; // Map key:seismic name, value:xt file
    QMap<QString, float> scales; // Map key:log name, value:scale to apply
    int xmin = 0, xmax = 0, ymin = 0, ymax = 0, zmin = 0, zmax = 0; // 3d bounding box
    int ystep=1; // batch size
    //QString savePrefixGeneralization; // prefix to save output files (include the path)
    QString saveSuffixGeneralization = "mu";
    QString surveyName = "";
    QString projectDir = "";

    QVector<Well > wells;

    void moveSelectedItems(QListWidget*, QListWidget*);

    QProcess* process = NULL;
    QProcess* replayProcess = NULL;
    QProcess* generalizationProcess = NULL;
    QString generalizationCheckpoint = "";

    QString separator=",";
    QString subSeparator=";";

    BnniConfig* m_bnniConfig = nullptr;
    QString configFilename = "/tmp/config_nn.txt";
    QString jsonfile = "";
    int halfSignalSize = 0;
    QString programLocation="";
    QString interfaceProgramLocation="";
    QString programFileName = "launchWithEnv";

    QColor READY;
    QColor INTERRUPT;
    QColor DISABLED;

    // Private graphics items
    WellViewer* wellViewer = nullptr;
    NetworkParameterForm* parameterForm = nullptr;

    QVector<QVector<IntRange>> cache_wellIndexRanges;
    QVector<LogPreprocessing> cache_log_preprocessing;

    // Private functions
    void cleanWells();
    QWidget* buildAdvancedOptions();
    void applyRunningProcessColors();
    void applyStandbyProcessColors();
    void applyNotLaunchableProcess();

    // util functions for config load
    bool setChannels(QStringList list);
    bool setUnselectedSeismic(QStringList list);
    bool setOutputChannels(QStringList list);
    bool setUnselectedOutputChannels(QStringList list);
    bool setTrainWells(QStringList list);
    bool setValidationWells(QStringList list);
    bool setBlindWells(QStringList list);
    bool setUnusedWells(QStringList list);

    bool setWellMinMax(QStringList list);
    bool setWellMinMax(QString name, QVector<int> vals);

    bool setXMin(QString);
    bool setXMax(QString);
    bool setYMin(QString);
    bool setYMax(QString);
    bool setZMin(QString);
    bool setZMax(QString);
    bool setYStep(QString);

    //bool setSavePrefixGeneralization(QString);
    bool setSaveSuffixGeneralization(QString);
    bool setDataMap(QStringList values);
    bool setUnselectedDataMap(QStringList values);

    bool setScalesMap(QStringList values);
    bool setUnselectedScalesMap(QStringList values);

    void clearDynamicCache();
    Range getLogDynamic(int k);
    void computeWellDynamic();

    void computeReflectivityLog(QString name, int indexDt, int indexAttribut, float frequency, int waveletSize);
    void saveCurrentStateToJsonFile();

    QString findIJKDirectory();

    void loadCurrentSettings();
    void unsetTrainingSetSetting();
    void unsetConfigSetting();

    QString getConfigSuffixFromFile(const QString& configFilePath);

    bool debug = false;

private slots:
    void changeConfig(int index);
    void createNewConfig();
    bool loadJson(QString filename="", bool preserveConfig=false);
    bool loadUbjson(QString filename="", bool preserveConfig=false);
    void updateSeismic();
    void updateSeismicDataMap();
    void updateLogs();
    void updateWells();
    void moveTrainToUnused();
    void moveTrainToVal();
    void moveValToTrain();
    void moveValToBlind();
    void moveValToUnused();
    void moveBlindToVal();
    void moveBlindToUnused();
    void moveUnusedToBlind();
    void moveUnusedToVal();
    void moveUnusedToTrain();

    void launchProcess();
    void postProcessing(int, QProcess::ExitStatus);
    void errorProcessing(QProcess::ProcessError);
    void stopProcess();

    void replayLaunchProcess();
    void replayPostProcessing(int, QProcess::ExitStatus);
    void replayErrorProcessing(QProcess::ProcessError);

    void openWellViewer();
    void closeWellViewer();
    void toggleWells(Well& well, bool active);
    void changeLogsSelection();

    void openDemoBNNI();

    void openGeneralizationSectionWindow();
    void openGeneralizationCarteWindow();
    void openGeneticAlgorithmLauncher();
    void validateArguments();
    bool testWarningForArguments();

    void loadConfig();
    //void updateConfigFilename(QString configFilename);
    //void searchConfigFilename(QString dir_path="");
    void saveConfig();

    void updateMinMaxIndex();
    void sliceGeneralization(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, QString checkpoint, QString suffix);
    void sliceGeneralizationFinished(int, QProcess::ExitStatus);
    void sliceGeneralizationError(QProcess::ProcessError);

    void randomGeneralization(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, QString checkpoint, QString suffix, QStringList data, QString generalizationDir);

    void carteGeneralization(QString horizon_top, int shift_top, QString horizon_bottom, int shift_bottom, QString checkpoint);
    void carteGeneralizationFinished(int, QProcess::ExitStatus);
    void carteGeneralizationError(QProcess::ProcessError);

    void logTableContexMenu(const QPoint& pos);

    void loadTrainingSet(int index);

    void updateHiddenLayersSize();

signals:
    void updatedJson(QString s);
    void updatedKindSelection(QVector<QString> array);
    void updatedSeismicSelection(QVector<std::pair<QString, QString>> array);
    void reset();
    void sliceGeneralizationRefusedSignal(QString);
    void sliceGeneralizationFinishedSignal(QString);
    void sliceGeneralizationErrorSignal(QString);
    void carteGeneralizationRefusedSignal(QString);
    void carteGeneralizationFinishedSignal(QString);
    void carteGeneralizationErrorSignal(QString);
};

#endif // MAINWINDOW_H
