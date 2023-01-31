#include "denseparameterform.h"
#include "denseparametersmodel.h"
#include "functionselector.h"

// Qt headers
#include <QHBoxLayout>
#include <QSizePolicy>
#include <QString>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QFileInfo>
#include <QFileDialog>
#include <QDebug>

// standard libraries headers
#include <limits>

DenseParameterForm::DenseParameterForm(DenseParametersModel* model, QWidget* parent) : QGridLayout(parent)
{
    m_model = model;

    seismicPreprocessingComboBox = new QComboBox;
    seismicPreprocessingComboBox->addItem("None", QVariant(SeismicPreprocessing::SeismicNone));
    seismicPreprocessingComboBox->addItem("Hat", QVariant(SeismicPreprocessing::SeismicHat));
    this->addWidget(new QLabel("Seismic Preprocessing"), 0, 0);
    this->addWidget(seismicPreprocessingComboBox, 0, 1);
    seismicPreprocessingComboBox->setCurrentIndex(m_model->getSeismicPreprocessing());
    connect(seismicPreprocessingComboBox,SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &DenseParameterForm::updateSeismicPreprocessing);
    connect(m_model, &DenseParametersModel::seismicPreprocessingChanged, this, &DenseParameterForm::updateSeismicPreprocessingFromModel);

    hatParameterSpinBox = new QSpinBox;
    hatParameterSpinBox->setMinimum(1);
    hatParameterSpinBox->setMaximum(std::numeric_limits<int>::max());
    hatParameterSpinBox->setSingleStep(1);
    hatParameterSpinBox->setValue(m_model->getHatParameter());
    hatParameterSpinBox->setDisplayIntegerBase(10);
    this->addWidget(new QLabel("Hat Parameter"), 0, 2);
    this->addWidget(hatParameterSpinBox, 0, 3);
    connect(hatParameterSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &DenseParameterForm::updateHatParameter);
    connect(m_model, &DenseParametersModel::hatParameterChanged, this, &DenseParameterForm::updateHatParameterFromModel);

    m_wellPostprocessingComboBox = new QComboBox;
    m_wellPostprocessingComboBox->addItem("None", QVariant(WellPostprocessing::WellNone));
    m_wellPostprocessingComboBox->addItem("Filter", QVariant(WellPostprocessing::WellFilter));
    this->addWidget(new QLabel("Well Postprocessing"), 1, 0);
    this->addWidget(m_wellPostprocessingComboBox, 1, 1);
    m_wellPostprocessingComboBox->setCurrentIndex(m_model->getWellPostprocessing());
    connect(m_wellPostprocessingComboBox,SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &DenseParameterForm::updateWellPostprocessing);
    connect(m_model, &DenseParametersModel::wellPostprocessingChanged, this, &DenseParameterForm::updateWellPostprocessingFromModel);

    m_wellFilterFrequencySpinBox = new QDoubleSpinBox;
    m_wellFilterFrequencySpinBox->setMinimum(std::numeric_limits<float>::min());
    m_wellFilterFrequencySpinBox->setMaximum(std::numeric_limits<float>::max());
    m_wellFilterFrequencySpinBox->setValue(m_model->getWellFilterFrequency());
    this->addWidget(new QLabel("Well Filter frequency"), 1, 2);
    this->addWidget(m_wellFilterFrequencySpinBox, 1, 3);

    connect(m_wellFilterFrequencySpinBox,SELECT<double>::OVERLOAD_OF(&QDoubleSpinBox::valueChanged), this, &DenseParameterForm::updateWellFilterFrequency);
    connect(m_model, &DenseParametersModel::wellFilterFrequencyChanged, this, &DenseParameterForm::updateWellFilterFrequencyFromModel);


    QString hiddenLayersString = "";
    for (int i=0; i<m_model->getHiddenLayers().size(); i++) {
        hiddenLayersString += QString::number(m_model->getHiddenLayers()[i]);
    }
    hiddenLayersEdit = new QLineEdit(hiddenLayersString);
    hiddenLayersEdit->setEnabled(true);
    this->addWidget(new QLabel("Hidden Layers sizes"), 2, 0);
    this->addWidget(hiddenLayersEdit, 2, 1);
    connect(hiddenLayersEdit, &QLineEdit::textChanged, this, &DenseParameterForm::updateHiddenLayers);
    connect(m_model, &DenseParametersModel::hiddenLayersChanged, this, &DenseParameterForm::updateHiddenLayersFromModel);

    layerActivationComboBox = new QComboBox;
    layerActivationComboBox->addItem("Linear", QVariant(Activation::linear));
    layerActivationComboBox->addItem("Sigmoid", QVariant(Activation::sigmoid));
    layerActivationComboBox->addItem("RELU", QVariant(Activation::relu));
    layerActivationComboBox->addItem("SELU", QVariant(Activation::selu));
    layerActivationComboBox->addItem("Leaky RELU", QVariant(Activation::leaky_relu));
    this->addWidget(new QLabel("Layer Activation"), 2, 2);
    this->addWidget(layerActivationComboBox, 2, 3);
    layerActivationComboBox->setCurrentIndex(m_model->getLayerActivation());
    connect(layerActivationComboBox,SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &DenseParameterForm::updateLayerActivation);
    connect(m_model, &DenseParametersModel::layerActivationChanged, this, &DenseParameterForm::updateLayerActivationFromModel);

    m_useBiasCheckBox = new QCheckBox;
    m_useBiasCheckBox->setCheckState(m_model->getUseBias() ? Qt::Checked : Qt::Unchecked);
    this->addWidget(new QLabel("Use bias"), 2, 4);
    this->addWidget(m_useBiasCheckBox, 2, 5);
    connect(m_useBiasCheckBox, &QCheckBox::stateChanged, this, &DenseParameterForm::updateUseBias);
    connect(m_model, &DenseParametersModel::useBiasChanged, this, &DenseParameterForm::updateUseBiasFromModel);

    optimizerComboBox = new QComboBox;
    optimizerComboBox->addItem("Gradient Descent", QVariant(Optimizer::gradientDescent));
    optimizerComboBox->addItem("Momentum", QVariant(Optimizer::momentum));
    optimizerComboBox->addItem("Adam", QVariant(Optimizer::adam));
    this->addWidget(new QLabel("Optimizer"), 3, 0);
    this->addWidget(optimizerComboBox, 3, 1);
    optimizerComboBox->setCurrentIndex(m_model->getOptimizer());
    connect(optimizerComboBox,SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &DenseParameterForm::updateOptimizer);
    connect(m_model, &DenseParametersModel::optimizerChanged, this, &DenseParameterForm::updateOptimizerFromModel);

    learningRateLineEdit = new QLineEdit(QString::number(m_model->getLearningRate()));
    learningRateLineEdit->setEnabled(true);
    this->addWidget(new QLabel("Learning rate"), 3, 2);
    this->addWidget(learningRateLineEdit, 3, 3);
    connect(learningRateLineEdit, &QLineEdit::textChanged, this, &DenseParameterForm::updateLearningRate);
    connect(m_model, &DenseParametersModel::learningRateChanged, this, &DenseParameterForm::updateLearningRateFromModel);

    momentumLineEdit = new QLineEdit(QString::number(m_model->getMomentum()));
    momentumLineEdit->setEnabled(true);
    this->addWidget(new QLabel("Momentum"), 3, 4);
    this->addWidget(momentumLineEdit, 3, 5);
    connect(momentumLineEdit, &QLineEdit::textChanged, this, &DenseParameterForm::updateMomentum);
    connect(m_model, &DenseParametersModel::momentumChanged, this, &DenseParameterForm::updateMomentumFromModel);

    batchSpinBox = new QSpinBox;
    batchSpinBox->setMinimum(1);
    batchSpinBox->setMaximum(std::numeric_limits<int>::max());
    batchSpinBox->setSingleStep(1);
    batchSpinBox->setValue(m_model->getBatchSize());
    batchSpinBox->setDisplayIntegerBase(10);
    this->addWidget(new QLabel("Batch Size"), 4, 0);
    this->addWidget(batchSpinBox, 4, 1);
    connect(batchSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &DenseParameterForm::updateBatchSize);
    connect(m_model, &DenseParametersModel::batchSizeChanged, this, &DenseParameterForm::updateBatchSizeFromModel);

    dropoutSpinBox = new QDoubleSpinBox;
    dropoutSpinBox->setMinimum(0);
    dropoutSpinBox->setMaximum(1);
    dropoutSpinBox->setSingleStep(0.1);
    dropoutSpinBox->setValue(m_model->getDropout());
    this->addWidget(new QLabel("Dropout"), 4, 2);
    this->addWidget(dropoutSpinBox, 4, 3);
    connect(dropoutSpinBox,SELECT<double>::OVERLOAD_OF(&QDoubleSpinBox::valueChanged), this, &DenseParameterForm::updateDropout);
    connect(m_model, &DenseParametersModel::dropoutChanged, this, &DenseParameterForm::updateDropoutFromModel);

    batchNormCheckBox = new QCheckBox;
    batchNormCheckBox->setChecked(m_model->getBatchNorm());
    this->addWidget(new QLabel("Use Batch normalization"), 4, 4);
    this->addWidget(batchNormCheckBox, 4, 5);
    connect(batchNormCheckBox, &QCheckBox::stateChanged, this, &DenseParameterForm::updateBatchNorm);
    connect(m_model, &DenseParametersModel::batchNormChanged, this, &DenseParameterForm::updateBatchNormFromModel);


    epochSpinBox = new QSpinBox;
    epochSpinBox->setMinimum(1);
    epochSpinBox->setMaximum(std::numeric_limits<int>::max());
    epochSpinBox->setSingleStep(1);
    epochSpinBox->setValue(m_model->getNumEpochs());
    epochSpinBox->setDisplayIntegerBase(10);
    this->addWidget(new QLabel("Number of Epochs"), 5, 0);
    this->addWidget(epochSpinBox, 5, 1);
    connect(epochSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &DenseParameterForm::updateNumEpochs);
    connect(m_model, &DenseParametersModel::numEpochsChanged, this, &DenseParameterForm::updateNumEpochsFromModel);

    epochSaveStepSpinBox = new QSpinBox;
    epochSaveStepSpinBox->setMinimum(1);
    epochSaveStepSpinBox->setMaximum(std::numeric_limits<int>::max());
    epochSaveStepSpinBox->setSingleStep(1);
    epochSaveStepSpinBox->setValue(m_model->getEpochSaveStep());
    epochSaveStepSpinBox->setDisplayIntegerBase(10);
    this->addWidget(new QLabel("Epoch saving step"), 5, 2);
    this->addWidget(epochSaveStepSpinBox, 5, 3);
    connect(epochSaveStepSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &DenseParameterForm::updateEpochSaveStep);
    connect(m_model, &DenseParametersModel::epochSaveStepChanged, this, &DenseParameterForm::updateEpochSaveStepFromModel);

    QHBoxLayout* referenceLayout = new QHBoxLayout;
    QPushButton* referenceButton = new QPushButton("edit");
    QSizePolicy policy = referenceButton->sizePolicy();
    policy.setHorizontalPolicy(QSizePolicy::Fixed);
    referenceButton->setSizePolicy(policy);
    referenceLayout->addWidget(referenceButton);
    m_referenceCheckpointLabel = new QLabel;
    m_referenceCheckpointLabel->setText(m_model->getReferenceCheckpoint());
    referenceLayout->addWidget(m_referenceCheckpointLabel);

    this->addWidget(new QLabel("Reference checkpoint"), 5, 4);
    this->addLayout(referenceLayout, 5, 5);
    connect(m_model, &DenseParametersModel::referenceCheckpointChanged, this, &DenseParameterForm::updateReferenceCheckpointFromModel);
    connect(referenceButton, &QPushButton::clicked, m_model, &DenseParametersModel::editReferenceCheckpoint);

    gpuSpinBox = new QSpinBox;
    gpuSpinBox->setMinimum(0);
    gpuSpinBox->setMaximum(99);
    gpuSpinBox->setSingleStep(1);
    gpuSpinBox->setValue(m_model->getNGpus());
    gpuSpinBox->setDisplayIntegerBase(10);
    this->addWidget(new QLabel("Number of GPUs to use"), 6, 0);
    this->addWidget(gpuSpinBox, 6, 1);
    connect(gpuSpinBox,SELECT<int>::OVERLOAD_OF(&QSpinBox::valueChanged), this, &DenseParameterForm::updateGpuOption);
    connect(m_model, &DenseParametersModel::nGpusChanged, this, &DenseParameterForm::updateGpuOptionFromModel);

    savePrefixLineEdit = new QLineEdit(m_model->getSavePrefix());
    savePrefixLineEdit->setEnabled(true);
    this->addWidget(new QLabel("Save prefix"), 6, 2);
    this->addWidget(savePrefixLineEdit, 6, 3);
    connect(savePrefixLineEdit, &QLineEdit::textChanged, this, &DenseParameterForm::updateSavePrefix);
    connect(m_model, &DenseParametersModel::savePrefixChanged, this, &DenseParameterForm::updateSavePrefixFromModel);

    precisionComboBox = new QComboBox;
    precisionComboBox->addItem("float16", QVariant(PrecisionType::float16));
    precisionComboBox->addItem("float32", QVariant(PrecisionType::float32));
    this->addWidget(new QLabel("Precision"), 6, 4);
    this->addWidget(precisionComboBox, 6, 5);
    precisionComboBox->setCurrentIndex(m_model->getPrecision());
    connect(precisionComboBox,SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &DenseParameterForm::updatePrecision);
    connect(m_model, &DenseParametersModel::precisionChanged, this, &DenseParameterForm::updatePrecisionFromModel);
}

QVector<unsigned int> DenseParameterForm::decodeHiddenLayer(const QString& txt) {
    QStringList list = txt.split(",");
    bool test = true;
    int i=0;
    QVector<unsigned int> hiddenLayers;
    while (i<list.size() && test) {
        int val = list.at(i).toInt(&test);
        if (test) {
            hiddenLayers.append(val);
            i++;
        }
    }
    return hiddenLayers;
}

void DenseParameterForm::updateHiddenLayers(QString txt) {
    QVector<unsigned int> hiddenLayers = decodeHiddenLayer(txt);
    m_model->setHiddenLayers(hiddenLayers);
    if (debug) {
        qDebug() << "DenseParameterForm::updateHiddenLayers" << m_model->getHiddenLayers();
    }
}

void DenseParameterForm::updateHiddenLayersFromModel(QVector<unsigned int> array) {
    QVector<unsigned int> oldHiddenLayers = decodeHiddenLayer(hiddenLayersEdit->text());
    if (oldHiddenLayers==array) {
        return;
    }

    QSignalBlocker b(hiddenLayersEdit);

    QString hiddenLayersString = "";
    for (int i=0; i<array.size(); i++) {
        if (i>0) {
            hiddenLayersString += separator;
        }
        hiddenLayersString += QString::number(array[i]);
    }
    hiddenLayersEdit->setText(hiddenLayersString);
}

void DenseParameterForm::updateLearningRate(QString txt) {
    bool test;
    double learningRate = txt.toDouble(&test);
    if (test) {
        m_model->setLearningRate(learningRate);
    }
    if (debug) {
        qDebug() << "DenseParameterForm::updateLearningRate" << m_model->getLearningRate();
    }
}

void DenseParameterForm::updateLearningRateFromModel(double val) {
    QSignalBlocker b(learningRateLineEdit);

    learningRateLineEdit->setText(QString::number(val));
}

void DenseParameterForm::updateMomentum(QString txt) {
    bool test;
    double momentumParameter = txt.toDouble(&test);
    if (test) {
        m_model->setMomentum(momentumParameter);
    }
    if (debug) {
        qDebug() << "DenseParameterForm::updateMomentum" << m_model->getMomentum();
    }
}

void DenseParameterForm::updateMomentumFromModel(double val) {
    QSignalBlocker b(momentumLineEdit);

    momentumLineEdit->setText(QString::number(val));
}

void DenseParameterForm::updateNumEpochs(int val) {
    m_model->setNumEpochs(val);
    if (debug) {
        qDebug() << "DenseParameterForm::updateNumEpochs" << m_model->getNumEpochs();
    }
}

void DenseParameterForm::updateNumEpochsFromModel(unsigned int val) {
    QSignalBlocker b(epochSpinBox);

    epochSpinBox->setValue(val);
}

void DenseParameterForm::updateHatParameter(int val) {
    m_model->setHatParameter(val);
    if (debug) {
        qDebug() << "DenseParameterForm::updateHatParameter" << m_model->getHatParameter();
    }
}

void DenseParameterForm::updateHatParameterFromModel(unsigned int val) {
    QSignalBlocker b(hatParameterSpinBox);

    hatParameterSpinBox->setValue(val);
}

void DenseParameterForm::updateOptimizer(int index) {
    bool test;
    int val = optimizerComboBox->itemData(index).toInt(&test);
    if (test) {
        m_model->setOptimizer(static_cast<Optimizer>(val));
    }
    if (debug) {
        qDebug() << "DenseParameterForm:updateOptimizer:" << m_model->getOptimizer();
    }
}

void DenseParameterForm::updateOptimizerFromModel(int val) {
    QSignalBlocker b(optimizerComboBox);

    optimizerComboBox->setCurrentIndex(val);
}

void DenseParameterForm::updateLayerActivation(int index) {
    bool test;
    int val = layerActivationComboBox->itemData(index).toInt(&test);
    if (test) {
        m_model->setLayerActivation(static_cast<Activation>(val));
    }
    if (debug) {
        qDebug() << "DenseParameterForm::updateLayerActivation" << m_model->getLayerActivation();
    }
}

void DenseParameterForm::updateLayerActivationFromModel(int val) {
    QSignalBlocker b(layerActivationComboBox);

    layerActivationComboBox->setCurrentIndex(val);
}

void DenseParameterForm::updateDropout(float val) {
    m_model->setDropout(val);
    if (debug) {
        qDebug() << "DenseParameterForm::setDropout" << m_model->getDropout();
    }
    m_model->validateArguments();
}

void DenseParameterForm::updateDropoutFromModel(float val) {
    QSignalBlocker b(dropoutSpinBox);

    dropoutSpinBox->setValue(val);
}

void DenseParameterForm::updateBatchNorm(int state) {
    m_model->setBatchNorm(state == Qt::Checked);
    if (debug) {
        qDebug() << "DenseParameterForm::setBatchNorm" << m_model->getBatchNorm();
    }
}

void DenseParameterForm::updateBatchNormFromModel(bool val) {
    QSignalBlocker b(batchNormCheckBox);

    batchNormCheckBox->setCheckState(val ? Qt::Checked : Qt::Unchecked);
}


void DenseParameterForm::updateSeismicPreprocessing(int index) {
    bool test;
    int val = seismicPreprocessingComboBox->itemData(index).toInt(&test);
    if (test) {
        m_model->setSeismicPreprocessing(static_cast<SeismicPreprocessing>(val));
    }
    if (debug) {
        qDebug() << "DenseParameterForm::updateSeismicPreprocessing" << m_model->getSeismicPreprocessing();
    }
}

void DenseParameterForm::updateSeismicPreprocessingFromModel(int val) {
    QSignalBlocker b(seismicPreprocessingComboBox);

    seismicPreprocessingComboBox->setCurrentIndex(val);
}

void DenseParameterForm::updateEpochSaveStep(int val) {
    m_model->setEpochSaveStep(val);
    if (debug) {
        qDebug() << "DenseParameterForm::updateEpochSaveStep" << m_model->getEpochSaveStep();
    }
}

void DenseParameterForm::updateEpochSaveStepFromModel(unsigned int val) {
    QSignalBlocker b(epochSaveStepSpinBox);

    epochSaveStepSpinBox->setValue(val);
}

void DenseParameterForm::updateGpuOption(int nGpu) {
    m_model->setNGpus(nGpu);
    if (debug) {
        qDebug() << "DenseParameterForm::updateGpuOption" << m_model->getNGpus();
    }
    m_model->validateArguments();
}

void DenseParameterForm::updateGpuOptionFromModel(unsigned int val) {
    QSignalBlocker b(gpuSpinBox);

    gpuSpinBox->setValue(val);
}

void DenseParameterForm::updateSavePrefix(QString savePrefix) {
    m_model->setSavePrefix(savePrefix);
    if (debug) {
        qDebug() << "DenseParameterForm::updateSavePrefix" << m_model->getSavePrefix();
    }
    m_model->validateArguments();
}

void DenseParameterForm::updateSavePrefixFromModel(QString txt) {
    QSignalBlocker b(savePrefixLineEdit);

    savePrefixLineEdit->setText(txt);
}

void DenseParameterForm::updateBatchSize(int val) {
    m_model->setBatchSize(val);
    if (debug) {
        qDebug() << "DenseParameterForm::updateBatchSize" << m_model->getBatchSize();
    }
}

void DenseParameterForm::updateBatchSizeFromModel(unsigned int val) {
    QSignalBlocker b(batchSpinBox);

    batchSpinBox->setValue(val);
}

void DenseParameterForm::updatePrecision(int index) {
    bool test;
    int val = precisionComboBox->itemData(index).toInt(&test);
    if (test) {
        m_model->setPrecision(static_cast<PrecisionType>(val));
    }
    if (debug) {
        qDebug() << "DenseParameterForm::updatePrecision" << m_model->getPrecision();
    }
}

void DenseParameterForm::updatePrecisionFromModel(int val) {
    QSignalBlocker b(precisionComboBox);

    precisionComboBox->setCurrentIndex(val);
}

void DenseParameterForm::updateWellPostprocessing(int index) {
    bool test;
    int val = m_wellPostprocessingComboBox->itemData(index).toInt(&test);
    if (test) {
        m_model->setWellPostprocessing(static_cast<WellPostprocessing>(val));
    }
    if (debug) {
        qDebug() << "DenseParameterForm::updateWellPostprocessing" << m_model->getWellPostprocessing();
    }
}

void DenseParameterForm::updateWellPostprocessingFromModel(int val) {
    QSignalBlocker b(m_wellPostprocessingComboBox);

    m_wellPostprocessingComboBox->setCurrentIndex(val);
}

void DenseParameterForm::updateWellFilterFrequency(double val) {
    m_model->setWellFilterFrequency(val);
    if (debug) {
        qDebug() << "DenseParameterForm::updateWellFilterFrequency" << m_model->getWellFilterFrequency();
    }
}

void DenseParameterForm::updateWellFilterFrequencyFromModel(float val) {
    QSignalBlocker b(m_wellFilterFrequencySpinBox);

    m_wellFilterFrequencySpinBox->setValue(val);
}

void DenseParameterForm::updateUseBias(int state) {
    m_model->setUseBias(state == Qt::Checked);
    if (debug) {
        qDebug() << "DenseParameterForm::updateUseBias" << m_model->getUseBias();
    }
}

void DenseParameterForm::updateUseBiasFromModel(bool val) {
    QSignalBlocker b(m_useBiasCheckBox);

    m_useBiasCheckBox->setCheckState(val ? Qt::Checked : Qt::Unchecked);
}

void DenseParameterForm::updateReferenceCheckpointFromModel(QString txt) {
    QSignalBlocker b(m_referenceCheckpointLabel);

    m_referenceCheckpointLabel->setText(txt);

    bool enableNetworkParams = txt.isNull() || txt.isEmpty();
    seismicPreprocessingComboBox->setEnabled(enableNetworkParams);
    hiddenLayersEdit->setEnabled(enableNetworkParams);
    precisionComboBox->setEnabled(enableNetworkParams);
    m_useBiasCheckBox->setEnabled(enableNetworkParams);
}

