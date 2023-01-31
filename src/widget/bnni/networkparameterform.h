#ifndef NETWORKPARAMETERFORM_H
#define NETWORKPARAMETERFORM_H

#include "structures.h"

#include <QVBoxLayout>

class NetworkParametersModel;
class DenseParameterForm;
class TreeParameterForm;
class QComboBox;

class NetworkParameterForm : public QVBoxLayout
{
    Q_OBJECT
public:
    NetworkParameterForm(NetworkParametersModel* networkModel, QWidget* parent = 0);

    NeuralNetwork getNet();

private:
    QComboBox* m_networkComboBox;

    QWidget* m_denseParamHolder;
    DenseParameterForm* m_denseParameterForm;
    QWidget* m_treeParamHolder;
    TreeParameterForm* m_treeParameterForm;

    NetworkParametersModel* m_networkModel = nullptr;
    bool m_debug = false; // Debug mode

private slots:
    void updateNet(int index);
    void updateNetChangedFromConfig(NeuralNetwork val);
};

#endif
