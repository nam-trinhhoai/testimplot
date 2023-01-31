#include "networkparameterform.h"
#include "networkparametersmodel.h"
#include "denseparameterform.h"
#include "treeparameterform.h"
#include "functionselector.h"

#include <QWidget>
#include <QHBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <QDebug>

NetworkParameterForm::NetworkParameterForm(NetworkParametersModel* networkModel, QWidget* parent) : QVBoxLayout(parent) {
    m_networkModel = networkModel;

    QHBoxLayout* networkLayout = new QHBoxLayout;
    this->addLayout(networkLayout);

    networkLayout->addWidget(new QLabel("Network : "), 0);
    m_networkComboBox = new QComboBox;
    m_networkComboBox->addItem("Dense", QVariant(NeuralNetwork::Dense));
    m_networkComboBox->addItem("DNN", QVariant(NeuralNetwork::Dnn));
    m_networkComboBox->addItem("Xgboost", QVariant(NeuralNetwork::Xgboost));
    networkLayout->addWidget(m_networkComboBox, 1);

    m_denseParamHolder = new QWidget;
    m_denseParameterForm = new DenseParameterForm(m_networkModel->denseModel());
    m_denseParamHolder->setLayout(m_denseParameterForm);
    this->addWidget(m_denseParamHolder);

    m_treeParamHolder = new QWidget;
    m_treeParameterForm = new TreeParameterForm(m_networkModel->treeModel());
    m_treeParamHolder->setLayout(m_treeParameterForm);
    this->addWidget(m_treeParamHolder);

    if (m_networkModel->getNetwork()==NeuralNetwork::Dense) {
        m_treeParamHolder->hide();
    } else {
        m_denseParamHolder->hide();
    }

    connect(m_networkComboBox,SELECT<int>::OVERLOAD_OF(&QComboBox::currentIndexChanged), this, &NetworkParameterForm::updateNet);
    connect(m_networkModel, &NetworkParametersModel::networkChanged, this, &NetworkParameterForm::updateNetChangedFromConfig);

    m_networkComboBox->setCurrentIndex(m_networkModel->getNetwork());
}

NeuralNetwork NetworkParameterForm::getNet() {
    return m_networkModel->getNetwork();
}

void NetworkParameterForm::updateNet(int index) {
    bool test;
    int val = m_networkComboBox->itemData(index).toInt(&test);
    bool netChanged = false;
    if (test) {
        NeuralNetwork newNet = static_cast<NeuralNetwork>(val);
        netChanged = newNet!=m_networkModel->getNetwork();
        m_networkModel->setNetwork(newNet);
    }
    if (m_debug) {
        qDebug() << "NetworkParameterForm::updateNet" << m_networkModel->getNetwork();
    }

    m_networkModel->validateArguments();
}

void NetworkParameterForm::updateNetChangedFromConfig(NeuralNetwork val) {
    QSignalBlocker b(m_networkComboBox);
    m_networkComboBox->setCurrentIndex(val);

    if (val==NeuralNetwork::Xgboost) {
        m_denseParamHolder->hide();
        m_treeParamHolder->show();
    } else {
        m_denseParamHolder->show();
        m_treeParamHolder->hide();
    }
    m_networkModel->validateArguments();
}
