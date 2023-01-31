#ifndef TREEPARAMETERFORM_H
#define TREEPARAMETERFORM_H

#include "structures.h"

#include <QGridLayout>

class TreeParametersModel;

class QSpinBox;
class QDoubleSpinBox;
class QComboBox;
class QLabel;
class QLineEdit;

class TreeParameterForm : public QGridLayout
{
	Q_OBJECT
public:
    TreeParameterForm(TreeParametersModel* model, QWidget* parent = 0);

private:
    bool m_debug = false; // Debug mode

    TreeParametersModel* m_model;

    QLineEdit* m_learningRateLineEdit;
    QSpinBox* m_saveStepSpinBox;
    QSpinBox* m_nGpusSpinBox;
    QLineEdit* m_savePrefixLineEdit;
    QComboBox* m_seismicPreprocessingComboBox;
    QSpinBox* m_hatParameterSpinBox;
    QComboBox* m_wellPostprocessingComboBox;
    QDoubleSpinBox* m_wellFilterFrequencySpinBox;
    QSpinBox* m_maxDepthSpinBox;
    QDoubleSpinBox* m_subSampleSpinBox;
    QDoubleSpinBox* m_colSampleByTreeSpinBox;
    QSpinBox* m_nEstimatorSpinBox;
    QLabel* m_referenceCheckpointLabel;

private slots:
    void updateLearningRate(QString txt);
    void updateLearningRateFromModel(double val);
    void updateEpochSaveStep(int val);
    void updateEpochSaveStepFromModel(unsigned int val);
    void updateGpuOption(int nGpu);
    void updateGpuOptionFromModel(unsigned int nGpu);
    void updateSavePrefix(QString savePrefix);
    void updateSavePrefixFromModel(QString savePrefix);
    void updateSeismicPreprocessing(int index);
    void updateSeismicPreprocessingFromModel(int val);
    void updateHatParameter(int val);
    void updateHatParameterFromModel(unsigned int val);
    void updateWellPostprocessing(int index);
    void updateWellPostprocessingFromModel(int val);
    void updateWellFilterFrequency(double val);
    void updateWellFilterFrequencyFromModel(float val);

    void updateMaxDepth(int val);
    void updateMaxDepthFromModel(unsigned int val);
    void updateSubSample(double val);
    void updateSubSampleFromModel(double val);
    void updateColSampleByTree(double val);
    void updateColSampleByTreeFromModel(double val);
    void updateNEstimator(int val);
    void updateNEstimatorFromModel(unsigned int val);
    void updateReferenceCheckpointFromModel(QString txt);
};

#endif
