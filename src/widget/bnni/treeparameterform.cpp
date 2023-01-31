#include "treeparameterform.h"
#include "functionselector.h"
#include "treeparametersmodel.h"

#include <QDebug>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLineEdit>

#include <limits>

TreeParameterForm::TreeParameterForm(TreeParametersModel* model, QWidget* parent) : QGridLayout(parent) {
    m_model = model;

    m_seismicPreprocessingComboBox = new QComboBox;
    m_seismicPreprocessingComboBox->addItem("None", QVariant(SeismicPreprocessing::SeismicNone));
    m_seismicPreprocessingComboBox->addItem("Hat", QVariant(SeismicPreprocessing::SeismicHat));
    this->addWidget(new QLabel("Seismic Preprocessing"), 0, 0);
    this->addWidget(m_seismicPreprocessingComboBox, 0, 1);
    m_seismicPreprocessingComboBox->setCurrentIndex(m_model->getSeismicPreprocessing());
    connect(m_seismicPreprocessingComboBox,SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &TreeParameterForm::updateSeismicPreprocessing);
    connect(m_model, &TreeParametersModel::seismicPreprocessingChanged, this, &TreeParameterForm::updateSeismicPreprocessingFromModel);

    m_hatParameterSpinBox = new QSpinBox;
    m_hatParameterSpinBox->setMinimum(1);
    m_hatParameterSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_hatParameterSpinBox->setSingleStep(1);
    m_hatParameterSpinBox->setValue(m_model->getHatParameter());
    m_hatParameterSpinBox->setDisplayIntegerBase(10);
    this->addWidget(new QLabel("Hat Parameter"), 0, 2);
    this->addWidget(m_hatParameterSpinBox, 0, 3);

    connect(m_hatParameterSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &TreeParameterForm::updateHatParameter);
    connect(m_model, &TreeParametersModel::hatParameterChanged, this, &TreeParameterForm::updateHatParameterFromModel);

    m_wellPostprocessingComboBox = new QComboBox;
    m_wellPostprocessingComboBox->addItem("None", QVariant(WellPostprocessing::WellNone));
    m_wellPostprocessingComboBox->addItem("Filter", QVariant(WellPostprocessing::WellFilter));
    this->addWidget(new QLabel("Well Postprocessing"), 1, 0);
    this->addWidget(m_wellPostprocessingComboBox, 1, 1);
    m_wellPostprocessingComboBox->setCurrentIndex(m_model->getWellPostprocessing());
    connect(m_wellPostprocessingComboBox,SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &TreeParameterForm::updateWellPostprocessing);
    connect(m_model, &TreeParametersModel::wellPostprocessingChanged, this, &TreeParameterForm::updateWellPostprocessingFromModel);

    m_wellFilterFrequencySpinBox = new QDoubleSpinBox;
    m_wellFilterFrequencySpinBox->setMinimum(std::numeric_limits<float>::min());
    m_wellFilterFrequencySpinBox->setMaximum(std::numeric_limits<float>::max());
    m_wellFilterFrequencySpinBox->setValue(m_model->getWellFilterFrequency());
    this->addWidget(new QLabel("Well Filter frequency"), 1, 2);
    this->addWidget(m_wellFilterFrequencySpinBox, 1, 3);

    connect(m_wellFilterFrequencySpinBox,SELECT<double>::OVERLOAD_OF(&QDoubleSpinBox::valueChanged), this, &TreeParameterForm::updateWellFilterFrequency);
    connect(m_model, &TreeParametersModel::wellFilterFrequencyChanged, this, &TreeParameterForm::updateWellFilterFrequencyFromModel);

    m_maxDepthSpinBox = new QSpinBox;
    m_maxDepthSpinBox->setMinimum(1);
    m_maxDepthSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_maxDepthSpinBox->setSingleStep(1);
    m_maxDepthSpinBox->setValue(m_model->getMaxDepth());
    this->addWidget(new QLabel("Max depth"), 2, 0);
    this->addWidget(m_maxDepthSpinBox, 2, 1);

    connect(m_maxDepthSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &TreeParameterForm::updateMaxDepth);
    connect(m_model, &TreeParametersModel::maxDepthChanged, this, &TreeParameterForm::updateMaxDepthFromModel);

    m_subSampleSpinBox = new QDoubleSpinBox;
    m_subSampleSpinBox->setMinimum(0);
    m_subSampleSpinBox->setMaximum(1);
    m_subSampleSpinBox->setSingleStep(0.1);
    m_subSampleSpinBox->setValue(m_model->getSubSample());
    this->addWidget(new QLabel("SubSample"), 2, 2);
    this->addWidget(m_subSampleSpinBox, 2, 3);

    connect(m_subSampleSpinBox,SELECT<double>::OVERLOAD_OF(&QDoubleSpinBox::valueChanged), this, &TreeParameterForm::updateSubSample);
    connect(m_model, &TreeParametersModel::subSampleChanged, this, &TreeParameterForm::updateSubSampleFromModel);

    m_colSampleByTreeSpinBox = new QDoubleSpinBox;
    m_colSampleByTreeSpinBox->setMinimum(0);
    m_colSampleByTreeSpinBox->setMaximum(1);
    m_colSampleByTreeSpinBox->setSingleStep(0.1);
    m_colSampleByTreeSpinBox->setValue(m_model->getColSampleByTree());
    this->addWidget(new QLabel("Column Sample by Tree"), 2, 4);
    this->addWidget(m_colSampleByTreeSpinBox, 2, 5);

    connect(m_colSampleByTreeSpinBox,SELECT<double>::OVERLOAD_OF(&QDoubleSpinBox::valueChanged), this, &TreeParameterForm::updateColSampleByTree);
    connect(m_model, &TreeParametersModel::colSampleByTreeChanged, this, &TreeParameterForm::updateColSampleByTreeFromModel);

    m_nEstimatorSpinBox = new QSpinBox;
    m_nEstimatorSpinBox->setMinimum(1);
    m_nEstimatorSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_nEstimatorSpinBox->setSingleStep(1);
    m_nEstimatorSpinBox->setValue(m_model->getNEstimator());
    m_nEstimatorSpinBox->setDisplayIntegerBase(10);
    this->addWidget(new QLabel("Number of Trees"), 3, 0);
    this->addWidget(m_nEstimatorSpinBox, 3, 1);

    connect(m_nEstimatorSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &TreeParameterForm::updateNEstimator);
    connect(m_model, &TreeParametersModel::nEstimatorChanged, this, &TreeParameterForm::updateNEstimatorFromModel);

    m_saveStepSpinBox = new QSpinBox;
    m_saveStepSpinBox->setMinimum(1);
    m_saveStepSpinBox->setMaximum(std::numeric_limits<int>::max());
    m_saveStepSpinBox->setSingleStep(1);
    m_saveStepSpinBox->setValue(this->m_model->getEpochSaveStep());
    m_saveStepSpinBox->setDisplayIntegerBase(10);
    this->addWidget(new QLabel("Epoch saving step"), 3, 2);
    this->addWidget(m_saveStepSpinBox, 3, 3);

    connect(m_saveStepSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &TreeParameterForm::updateEpochSaveStep);
    connect(m_model, &TreeParametersModel::epochSaveStepChanged, this, &TreeParameterForm::updateEpochSaveStepFromModel);

    m_learningRateLineEdit = new QLineEdit(QString::number(m_model->getLearningRate()));
    m_learningRateLineEdit->setEnabled(true);
    this->addWidget(new QLabel("Learning rate"), 3, 4);
    this->addWidget(m_learningRateLineEdit, 3, 5);

    connect(m_learningRateLineEdit, &QLineEdit::textChanged, this, &TreeParameterForm::updateLearningRate);
    connect(m_model, &TreeParametersModel::learningRateChanged, this, &TreeParameterForm::updateLearningRateFromModel);

    m_nGpusSpinBox = new QSpinBox;
    m_nGpusSpinBox->setMinimum(0);
    m_nGpusSpinBox->setMaximum(99);
    m_nGpusSpinBox->setSingleStep(1);
    m_nGpusSpinBox->setValue(m_model->getNGpus());
    m_nGpusSpinBox->setDisplayIntegerBase(10);
    this->addWidget(new QLabel("Number of GPUs to use"), 4, 0);
    this->addWidget(m_nGpusSpinBox, 4, 1);

    connect(m_nGpusSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &TreeParameterForm::updateGpuOption);
    connect(m_model, &TreeParametersModel::nGpusChanged, this, &TreeParameterForm::updateGpuOptionFromModel);

    m_savePrefixLineEdit = new QLineEdit(m_model->getSavePrefix());
    m_savePrefixLineEdit->setEnabled(true);
    this->addWidget(new QLabel("Save prefix"), 4, 2);
    this->addWidget(m_savePrefixLineEdit, 4, 3);

    connect(m_savePrefixLineEdit, &QLineEdit::textChanged, this, &TreeParameterForm::updateSavePrefix);
    connect(m_model, &TreeParametersModel::savePrefixChanged, this, &TreeParameterForm::updateSavePrefixFromModel);

    QHBoxLayout* referenceLayout = new QHBoxLayout;
    QPushButton* referenceButton = new QPushButton("edit");
    QSizePolicy policy = referenceButton->sizePolicy();
    policy.setHorizontalPolicy(QSizePolicy::Fixed);
    referenceButton->setSizePolicy(policy);
    referenceLayout->addWidget(referenceButton);
    m_referenceCheckpointLabel = new QLabel;
    m_referenceCheckpointLabel->setText(m_model->getReferenceCheckpoint());
    referenceLayout->addWidget(m_referenceCheckpointLabel);

    this->addWidget(new QLabel("Reference checkpoint"), 4, 4);
    this->addLayout(referenceLayout, 4, 5);
    connect(m_model, &TreeParametersModel::referenceCheckpointChanged, this, &TreeParameterForm::updateReferenceCheckpointFromModel);
    connect(referenceButton, &QPushButton::clicked, m_model, &TreeParametersModel::editReferenceCheckpoint);
}

void TreeParameterForm::updateLearningRate(QString txt) {
    bool test;
    double learningRate = txt.toDouble(&test);
    if (test) {
        m_model->setLearningRate(learningRate);
    }
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateLearningRate" << m_model->getLearningRate();
    }
}

void TreeParameterForm::updateLearningRateFromModel(double val) {
    QSignalBlocker b(m_learningRateLineEdit);

    m_learningRateLineEdit->setText(QString::number(val));
}

void TreeParameterForm::updateEpochSaveStep(int val) {
    m_model->setEpochSaveStep(val);
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateEpochSaveStep" << m_model->getEpochSaveStep();
    }
}

void TreeParameterForm::updateEpochSaveStepFromModel(unsigned int val) {
    QSignalBlocker b(m_saveStepSpinBox);

    m_saveStepSpinBox->setValue(val);
}

void TreeParameterForm::updateGpuOption(int nGpu) {
    m_model->setNGpus(nGpu);
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateGpuOption" << m_model->getNGpus();
    }
    m_model->validateArguments();
}

void TreeParameterForm::updateGpuOptionFromModel(unsigned int val) {
    QSignalBlocker b(m_nGpusSpinBox);

    m_nGpusSpinBox->setValue(val);
}

void TreeParameterForm::updateSavePrefix(QString savePrefix) {
    m_model->setSavePrefix(savePrefix);
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateSavePrefix" << m_model->getSavePrefix();
    }
    m_model->validateArguments();
}

void TreeParameterForm::updateSavePrefixFromModel(QString txt) {
    QSignalBlocker b(m_savePrefixLineEdit);

    m_savePrefixLineEdit->setText(txt);
}

void TreeParameterForm::updateSeismicPreprocessing(int index) {
    bool test;
    int val = m_seismicPreprocessingComboBox->itemData(index).toInt(&test);
    if (test) {
        m_model->setSeismicPreprocessing(static_cast<SeismicPreprocessing>(val));
    }
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateSeismicPreprocessing" << m_model->getSeismicPreprocessing();
    }
}

void TreeParameterForm::updateSeismicPreprocessingFromModel(int val) {
    QSignalBlocker b(m_seismicPreprocessingComboBox);

    m_seismicPreprocessingComboBox->setCurrentIndex(val);
}

void TreeParameterForm::updateHatParameter(int val) {
    m_model->setHatParameter(val);
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateHatParameter" << m_model->getHatParameter();
    }
}

void TreeParameterForm::updateHatParameterFromModel(unsigned int val) {
    QSignalBlocker b(m_hatParameterSpinBox);

    m_hatParameterSpinBox->setValue(val);
}

void TreeParameterForm::updateWellPostprocessing(int index) {
    bool test;
    int val = m_wellPostprocessingComboBox->itemData(index).toInt(&test);
    if (test) {
        m_model->setWellPostprocessing(static_cast<WellPostprocessing>(val));
    }
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateWellPostprocessing" << m_model->getWellPostprocessing();
    }
}

void TreeParameterForm::updateWellPostprocessingFromModel(int val) {
    QSignalBlocker b(m_wellPostprocessingComboBox);

    m_wellPostprocessingComboBox->setCurrentIndex(val);
}

void TreeParameterForm::updateWellFilterFrequency(double val) {
    m_model->setWellFilterFrequency(val);
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateWellFilterFrequency" << m_model->getWellFilterFrequency();
    }
}

void TreeParameterForm::updateWellFilterFrequencyFromModel(float val) {
    QSignalBlocker b(m_wellFilterFrequencySpinBox);

    m_wellFilterFrequencySpinBox->setValue(val);
}

void TreeParameterForm::updateMaxDepth(int val) {
    m_model->setMaxDepth(val);
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateMaxDepth" << m_model->getMaxDepth();
    }
    m_model->validateArguments();
}

void TreeParameterForm::updateMaxDepthFromModel(unsigned int val) {
    QSignalBlocker b(m_maxDepthSpinBox);

    m_maxDepthSpinBox->setValue(val);
}

void TreeParameterForm::updateSubSample(double val) {
    m_model->setSubSample(val);
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateSubSample" << m_model->getSubSample();
    }
    m_model->validateArguments();
}

void TreeParameterForm::updateSubSampleFromModel(double val) {
    QSignalBlocker b(m_subSampleSpinBox);

    m_subSampleSpinBox->setValue(val);
}

void TreeParameterForm::updateColSampleByTree(double val) {
    m_model->setColSampleByTree(val);
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateColSampleByTree" << m_model->getColSampleByTree();
    }
    m_model->validateArguments();
}

void TreeParameterForm::updateColSampleByTreeFromModel(double val) {
    QSignalBlocker b(m_colSampleByTreeSpinBox);

    m_colSampleByTreeSpinBox->setValue(val);
}

void TreeParameterForm::updateNEstimator(int val) {
    m_model->setNEstimator(val);
    if (m_debug) {
        qDebug() << "TreeParameterForm::updateNEstimator" << m_model->getNEstimator();
    }
}

void TreeParameterForm::updateNEstimatorFromModel(unsigned int val) {
    QSignalBlocker b(m_nEstimatorSpinBox);

    m_nEstimatorSpinBox->setValue(val);
}

void TreeParameterForm::updateReferenceCheckpointFromModel(QString txt) {
    QSignalBlocker b(m_referenceCheckpointLabel);

    m_referenceCheckpointLabel->setText(txt);
}
