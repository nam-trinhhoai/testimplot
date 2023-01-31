#ifndef DENSEPARAMETERFORM_H
#define DENSEPARAMETERFORM_H

#include <QFormLayout>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QVector>
#include <structures.h>

class DenseParametersModel;

class QLabel;
class QLineEdit;
class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;

class DenseParameterForm : public QGridLayout
{
public:
    DenseParameterForm(DenseParametersModel* model, QWidget* parent = 0);

private:
    QVector<unsigned int> decodeHiddenLayer(const QString& txt);

    DenseParametersModel* m_model;

    bool debug = false;

    QLineEdit* learningRateLineEdit = nullptr;
    QLineEdit* momentumLineEdit = nullptr;
    QSpinBox* epochSpinBox = nullptr;
    QDoubleSpinBox* dropoutSpinBox = nullptr;
    QCheckBox* batchNormCheckBox = nullptr;
    QComboBox* optimizerComboBox = nullptr;
    QComboBox* seismicPreprocessingComboBox = nullptr;
    QComboBox* layerActivationComboBox = nullptr;
    QLineEdit* hiddenLayersEdit = nullptr;
    QSpinBox* hatParameterSpinBox = nullptr;
    QSpinBox* epochSaveStepSpinBox = nullptr;
    QSpinBox* gpuSpinBox = nullptr;
    QSpinBox* batchSpinBox = nullptr;
    QLineEdit* savePrefixLineEdit = nullptr;
    QComboBox* precisionComboBox = nullptr;
    QComboBox* m_wellPostprocessingComboBox;
    QDoubleSpinBox* m_wellFilterFrequencySpinBox;
    QCheckBox* m_useBiasCheckBox;
    QLabel* m_referenceCheckpointLabel;

    QString separator=",";

private slots:
    void updateHiddenLayers(QString txt);
    void updateHiddenLayersFromModel(QVector<unsigned int> val);
    void updateLearningRate(QString txt);
    void updateLearningRateFromModel(double val);
    void updateMomentum(QString txt);
    void updateMomentumFromModel(double val);

    void updateNumEpochs(int val);
    void updateNumEpochsFromModel(unsigned int val);
    void updateOptimizer(int index);
    void updateOptimizerFromModel(int val);
    void updateDropout(float val);
    void updateDropoutFromModel(float val);
    void updateBatchNorm(int state);
    void updateBatchNormFromModel(bool val);
    void updateSeismicPreprocessing(int index);
    void updateSeismicPreprocessingFromModel(int val);
    void updateLayerActivation(int index);
    void updateLayerActivationFromModel(int val);
    void updateHatParameter(int val);
    void updateHatParameterFromModel(unsigned int val);
    void updateEpochSaveStep(int val);
    void updateEpochSaveStepFromModel(unsigned int val);
    void updateWellPostprocessing(int index);
    void updateWellPostprocessingFromModel(int val);
    void updateWellFilterFrequency(double val);
    void updateWellFilterFrequencyFromModel(float val);
    void updateReferenceCheckpointFromModel(QString txt);

    void updateGpuOption(int nGpu);
    void updateGpuOptionFromModel(unsigned int nGpu);
    void updateSavePrefix(QString savePrefix);
    void updateSavePrefixFromModel(QString savePrefix);
    void updatePrecision(int index);
    void updatePrecisionFromModel(int val);
    void updateBatchSize(int val);
    void updateBatchSizeFromModel(unsigned int val);
    void updateUseBias(int state);
    void updateUseBiasFromModel(bool val);
};

#endif // PARAMETERFORM_H
