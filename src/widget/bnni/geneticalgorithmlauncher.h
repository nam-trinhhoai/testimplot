#ifndef SRC_WIDGET_BNNI_GENETICALGORITHMLAUNCHER_H
#define SRC_WIDGET_BNNI_GENETICALGORITHMLAUNCHER_H

#include "densegeneticparams.h"
#include "structures.h"
#include "xgboostgeneticparams.h"

#include <QWidget>
#include <QProcess>
#include <QDateTime>
#include <vector>
#include <rapidjson/document.h>

class QSpinBox;
class QDoubleSpinBox;
class QLineEdit;
class QCheckBox;
class QComboBox;

class DenseGeneticParamsWidget;
class XgBoostGeneticParamsWidget;

typedef rapidjson::GenericMember<rapidjson::ASCII<>, rapidjson::MemoryPoolAllocator<> > WMember;

class GeneticAlgorithmLauncher : public QWidget {
    Q_OBJECT
public:
    GeneticAlgorithmLauncher(QWidget* parent=nullptr, Qt::WindowFlags f=Qt::WindowFlags());

    void setProgramLocation(const QString& programLocation);
    void setTrainingSet(const QString& trainingSetJsonFile);
    void setWells(const QVector<Well>& wells, const std::vector<int>& trainIdx,
            const std::vector<int>& validationIdx, const std::vector<int>& blindIdx);
    void setHalfWindow(int halfWindow);
    void setLayerSizes(QVector<unsigned int> array);
    void setNumInputSeismics(int val);

    static QString getTemporaryFilePath(const QString& templatePath);

private slots:
    void nPipelineIterationChanged(int val);
    void plotDirPathChanged();
    void csvDirPathChanged();
    void usePostProcessingFilteringChanged(int state);
    void postProcessingFrequencyChanged(double val);
    void nIterationChanged(int val);
    void crossoverRatioChanged(double val);
    void numberOfChangesPerMutationChanged(int val);
    void rangeXChanged(int val);
    void rangeYChanged(int val);
    void rangeDepthChanged(int val);
    void populationSizeChanged(int val);
    void localSearchDepthChanged(int val);
    void fixTrainWellsChanged(int val);
    void halfWindowSizeChanged(int val);
    void splitWellToVerticalPartsChanged(int val);
    void minimumSubWellSizeChanged(int val);
    void maximumSubWellSizeChanged(int val);
    void newTrainingSetNameChanged();
    void changeModel(QString modelName);
    void run();
    void runCross();
    void runProgram(const QString& programName);
    void createGeneticConfig();

    void geneticInputGenerated(int exitCode, QProcess::ExitStatus exitStatus);
    void geneticInputGotError(QProcess::ProcessError error);
    void geneticFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void geneticGotError(QProcess::ProcessError error);
    void newTraininSetFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void newTrainingSetGotError(QProcess::ProcessError error);

    QString getCsvPath();

private:
    std::vector<int> getTrainWellIdxs();
    bool isWellValidForGenetic(const WMember& well, bool& isValid);
    void cleanup();
    void fillRemoveWellsFile(const QString& filePath);

    static QString DENSE;
    static QString XGBOOST;

    // internal variables
    int m_nPipelineIteration = 5;
    QString m_plotDirPath;
    QString m_csvDirPath;
    bool m_usePostProcessingFiltering = true;
    double m_postProcessingFrequency = 100.0;
    int m_nIteration = 100;
    double m_crossoverRatio = 0.2;
    int m_numberOfChangesPerMutation = 1;
    int m_rangeX = 5;
    int m_rangeY = 5;
    int m_rangeDepth = 30;
    int m_populationSize = 4;
    int m_localSearchDepth = 3;
    bool m_fixTrainWells = true;
    bool m_useXgboost = true;
    XgBoostGeneticParams m_xgboostParams;
    DenseGeneticParams m_denseParams;
    int m_halfWindowSize = 20;
    bool m_splitWellToVerticalParts = true;
    int m_minimumSubWellSize = 30;
    int m_maximumSubWellSize = 100;
    QString m_newTraingSetName = "";

    QVector<Well> m_wells;
    std::vector<int> m_trainIdx;
    std::vector<int> m_blindIdx;
    std::vector<int> m_validationIdx;
    int m_nInputs = 1;


    QString m_trainingSetJsonFile;
    QString m_geneticInputJsonFile;
    QString m_geneticInputBufferFile;
    QString m_programLocation;
    QString m_geneticConfigFile;
    QString m_removeWellsFile;

    QProcess* m_process = nullptr;
    bool m_processRunning = false;
    QString m_cachedProgramLocation; // copy of m_programLocation when the first program was launched, to avoid change in the middle of the computation
    QString m_cachedTrainingSetJsonFile;
    QString m_cacheGeneticProgramName;
    QDateTime m_cachedTime;

    // gui variables
    QSpinBox* m_nPipelineIterationSpinBox;
    QLineEdit* m_plotDirPathLineEdit;
    QLineEdit* m_csvDirPathLineEdit;
    QCheckBox* m_usePostProcessingFilteringCheckBox;
    QDoubleSpinBox* m_postProcessingFrequencySpinBox;
    QSpinBox* m_nIterationSpinBox;
    QDoubleSpinBox* m_crossoverRatioSpinBox;
    QSpinBox* m_numberOfChangesPerMutationSpinBox;
    QSpinBox* m_rangeXSpinBox;
    QSpinBox* m_rangeYSpinBox;
    QSpinBox* m_rangeDepthSpinBox;
    QSpinBox* m_populationSizeSpinBox;
    QSpinBox* m_localSearchDepthSpinBox;
    QCheckBox* m_fixTrainWellsCheckBox;
    QComboBox* m_modelComboBox;
    DenseGeneticParamsWidget* m_denseWidget;
    XgBoostGeneticParamsWidget* m_xgboostWidget;
    QSpinBox* m_halfWindowSizeSpinBox;
    QCheckBox* m_splitWellToVerticalPartsCheckBox;
    QSpinBox* m_minimumSubWellSizeSpinBox;
    QSpinBox* m_maximumSubWellSizeSpinBox;
    QLineEdit* m_newTraingSetNameLineEdit;

};

#endif
