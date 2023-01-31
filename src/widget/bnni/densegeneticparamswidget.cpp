#include "densegeneticparamswidget.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLineEdit>

DenseGeneticParamsWidget::DenseGeneticParamsWidget(DenseGeneticParams& params,
		const QString& title, QWidget* parent) : QGroupBox(title, parent), m_params(&params) {
	QFormLayout* denseFormLayout = new QFormLayout;
	setLayout(denseFormLayout);

	m_layersLineEdit = new QLineEdit;
	if (!m_params.isNull()) {
		QString hiddenLayersString = "";
		QVector<unsigned int> layers = m_params->layerSizes();
		for (int i=0; i<layers.size()-1; i++) {
			hiddenLayersString += QString::number(layers[i]);
		}
		m_layersLineEdit->setText(hiddenLayersString);

		connect(m_params.data(), &DenseGeneticParams::layerSizesChanged, this, &DenseGeneticParamsWidget::dataLayerSizeChanged);
	}
	denseFormLayout->addRow("Hidden layers", m_layersLineEdit);
	connect(m_layersLineEdit, &QLineEdit::editingFinished, this, &DenseGeneticParamsWidget::hiddenLayersChanged);

	m_activationComboBox = new QComboBox;
	m_activationComboBox->addItem("Linear", QVariant(Activation::linear));
	m_activationComboBox->addItem("Sigmoid", QVariant(Activation::sigmoid));
	m_activationComboBox->addItem("RELU", QVariant(Activation::relu));
	m_activationComboBox->addItem("SELU", QVariant(Activation::selu));
	if (!m_params.isNull()) {
		m_activationComboBox->setCurrentIndex(m_params->activation());
		connect(m_params.data(), &DenseGeneticParams::activationChanged, this, &DenseGeneticParamsWidget::dataActivationChanged);
	}
	denseFormLayout->addRow("Activation", m_activationComboBox);
	connect(m_activationComboBox,QOverload<int>::of(&QComboBox::currentIndexChanged), this, &DenseGeneticParamsWidget::activationChanged);

	m_dropoutCheckBox = new QCheckBox;
	if (!m_params.isNull()) {
		m_dropoutCheckBox->setCheckState(m_params->useDropout() ? Qt::Checked : Qt::Unchecked);
		connect(m_params.data(), &DenseGeneticParams::useDropoutChanged, this, &DenseGeneticParamsWidget::dataUseDropoutChanged);
	}
	denseFormLayout->addRow("Use dropout", m_dropoutCheckBox);
	connect(m_dropoutCheckBox, &QCheckBox::stateChanged, this, &DenseGeneticParamsWidget::dropoutStateChanged);

	m_dropoutSpinBox = new QDoubleSpinBox;
	m_dropoutSpinBox->setMinimum(0);
	m_dropoutSpinBox->setMaximum(1);
	m_dropoutSpinBox->setSingleStep(0.1);
	if (!m_params.isNull()) {
		m_dropoutSpinBox->setValue(m_params->dropout());
		connect(m_params.data(), &DenseGeneticParams::dropoutChanged, m_dropoutSpinBox, &QDoubleSpinBox::setValue);
	}
	denseFormLayout->addRow("Dropout", m_dropoutSpinBox);
	connect(m_dropoutSpinBox,QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &DenseGeneticParamsWidget::dropoutChanged);

	m_normalisationCheckBox = new QCheckBox;
	if (!m_params.isNull()) {
		m_normalisationCheckBox->setCheckState(m_params->useNormalisation() ? Qt::Checked : Qt::Unchecked);
		connect(m_params.data(), &DenseGeneticParams::useNormalisationChanged, this, &DenseGeneticParamsWidget::dataUseNormalisationChanged);
	}
	denseFormLayout->addRow("Use normalisation", m_normalisationCheckBox);
	connect(m_normalisationCheckBox, &QCheckBox::stateChanged, this, &DenseGeneticParamsWidget::normalisationStateChanged);
}

DenseGeneticParamsWidget::~DenseGeneticParamsWidget() {
	if (!m_params.isNull()) {
		disconnect(m_params.data(), &DenseGeneticParams::layerSizesChanged, this, &DenseGeneticParamsWidget::dataLayerSizeChanged);
		disconnect(m_params.data(), &DenseGeneticParams::activationChanged, this, &DenseGeneticParamsWidget::dataActivationChanged);
		disconnect(m_params.data(), &DenseGeneticParams::useDropoutChanged, this, &DenseGeneticParamsWidget::dataUseDropoutChanged);
		disconnect(m_params.data(), &DenseGeneticParams::dropoutChanged, m_dropoutSpinBox, &QDoubleSpinBox::setValue);
		disconnect(m_params.data(), &DenseGeneticParams::useNormalisationChanged, this, &DenseGeneticParamsWidget::dataUseNormalisationChanged);
	}
}

QVector<unsigned int> DenseGeneticParamsWidget::layerSizes() const {
	QStringList list = m_layersLineEdit->text().split(",");
	bool test = true;
	int i=0;
	QVector<unsigned int> layers;
	while (i<list.size() && test) {
		int val = list.at(i).toInt(&test);
		if (test) {
			layers.append(val);
			i++;
		}
	}
	layers.append(1);

	return layers;
}

void DenseGeneticParamsWidget::hiddenLayersChanged() {
	if (m_params.isNull()) {
		return;
	}

	QVector<unsigned int> layers = layerSizes();
	m_params->setLayerSizes(layers);
}

void DenseGeneticParamsWidget::dropoutStateChanged(int state) {
	if (!m_params.isNull()) {
		m_params->toggleDropout(state==Qt::Checked);
	}
}

void DenseGeneticParamsWidget::dropoutChanged(double val) {
	if (!m_params.isNull()) {
		m_params->setDropout(val);
	}
}

void DenseGeneticParamsWidget::normalisationStateChanged(int state) {
	if (!m_params.isNull()) {
		m_params->toggleNormalisation(state==Qt::Checked);
	}
}

void DenseGeneticParamsWidget::activationChanged(int index) {
	if (m_params.isNull()) {
		return;
	}

	Activation activation;
	switch (index) {
	case 1:
		activation = Activation::sigmoid;
		break;
	case 2:
		activation = Activation::relu;
		break;
	case 3:
		activation = Activation::selu;
		break;
	default:
		activation = Activation::linear;
		break;
	}
	m_params->setActivation(activation);
}

void DenseGeneticParamsWidget::dataLayerSizeChanged(QVector<unsigned int> array) {
	QVector<unsigned int> currentArray = layerSizes();
	if (currentArray==array) {
		return;
	}

	QString hiddenLayersString = "";
	for (int i=0; i<array.size()-1; i++) {
		if (i>0) {
			hiddenLayersString += ",";
		}
		hiddenLayersString += QString::number(array[i]);
	}
	m_layersLineEdit->setText(hiddenLayersString);
}

void DenseGeneticParamsWidget::dataUseDropoutChanged(bool val) {
	m_dropoutCheckBox->setCheckState(val ? Qt::Checked : Qt::Unchecked);
}

void DenseGeneticParamsWidget::dataUseNormalisationChanged(bool val) {
	m_normalisationCheckBox->setCheckState(val ? Qt::Checked : Qt::Unchecked);
}

void DenseGeneticParamsWidget::dataActivationChanged(Activation val) {
	int index;
	switch (val) {
	case Activation::sigmoid:
		index = 1;
		break;
	case Activation::relu:
		index = 2;
		break;
	case Activation::selu:
		index = 3;
		break;
	default:
		index = 0;
		break;
	}
	m_activationComboBox->setCurrentIndex(index);
}

