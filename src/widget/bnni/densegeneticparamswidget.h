#ifndef SRC_WIDGET_BNNI_DENSEGENETICPARAMSWIDGET_H
#define SRC_WIDGET_BNNI_DENSEGENETICPARAMSWIDGET_H

#include "densegeneticparams.h"

#include <QGroupBox>
#include <QPointer>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;

class DenseGeneticParamsWidget : public QGroupBox {
	Q_OBJECT
public:
	DenseGeneticParamsWidget(DenseGeneticParams& params, const QString& title, QWidget* parent=0);
	~DenseGeneticParamsWidget();

	QVector<unsigned int> layerSizes() const;

private  slots:
	void hiddenLayersChanged();
	void dropoutStateChanged(int state);
	void dropoutChanged(double val);
	void normalisationStateChanged(int state);
	void activationChanged(int index);

	void dataLayerSizeChanged(QVector<unsigned int> array);
	void dataUseDropoutChanged(bool val);
	void dataUseNormalisationChanged(bool val);
	void dataActivationChanged(Activation val);

private:
	QPointer<DenseGeneticParams> m_params;

	QLineEdit* m_layersLineEdit;
	QCheckBox* m_dropoutCheckBox;
	QDoubleSpinBox* m_dropoutSpinBox;
	QCheckBox* m_normalisationCheckBox;
	QComboBox* m_activationComboBox;
};

#endif // SRC_WIDGET_BNNI_DENSEGENETICPARAMSWIDGET_H
