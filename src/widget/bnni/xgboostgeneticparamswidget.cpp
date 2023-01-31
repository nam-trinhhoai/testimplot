#include "xgboostgeneticparamswidget.h"

#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QSpinBox>

XgBoostGeneticParamsWidget::XgBoostGeneticParamsWidget(XgBoostGeneticParams& params,
		const QString& title, QWidget* parent) : QGroupBox(title, parent), m_params(&params) {
	QFormLayout* xgboostFormLayout = new QFormLayout;
	setLayout(xgboostFormLayout);

	m_maxDepthSpinBox = new QSpinBox;
	m_maxDepthSpinBox->setMinimum(1);
	m_maxDepthSpinBox->setMaximum(std::numeric_limits<int>::max());
	if (!m_params.isNull()) {
		m_maxDepthSpinBox->setValue(m_params->maxDepth());
		connect(m_params.data(), &XgBoostGeneticParams::maxDepthChanged, m_maxDepthSpinBox, &QSpinBox::setValue);
	}

	connect(m_maxDepthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &XgBoostGeneticParamsWidget::maxDepthChanged);

	xgboostFormLayout->addRow(new QLabel("Max tree depth"), m_maxDepthSpinBox);

	m_nEstimatorsSpinBox = new QSpinBox;
	m_nEstimatorsSpinBox->setMinimum(1);
	m_nEstimatorsSpinBox->setMaximum(std::numeric_limits<int>::max());
	if (!m_params.isNull()) {
		m_nEstimatorsSpinBox->setValue(m_params->nEstimators());
		connect(m_params.data(), &XgBoostGeneticParams::nEstimatorsChanged, m_nEstimatorsSpinBox, &QSpinBox::setValue);
	}

	connect(m_nEstimatorsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &XgBoostGeneticParamsWidget::nEstimatorsChanged);

	xgboostFormLayout->addRow(new QLabel("Number of estimators"), m_nEstimatorsSpinBox);

	m_learningRateSpinBox = new QDoubleSpinBox;
	m_learningRateSpinBox->setMinimum(std::numeric_limits<float>::min());
	m_learningRateSpinBox->setMaximum(1);
	m_learningRateSpinBox->setSingleStep(0.01);
	m_learningRateSpinBox->setDecimals(10);
	if (!m_params.isNull()) {
		m_learningRateSpinBox->setValue(m_params->learningRate());
		connect(m_params.data(), &XgBoostGeneticParams::learningRateChanged, m_learningRateSpinBox, &QDoubleSpinBox::setValue);
	}

	connect(m_learningRateSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &XgBoostGeneticParamsWidget::learningRateChanged);

	xgboostFormLayout->addRow(new QLabel("Learning rate"), m_learningRateSpinBox);

	m_subsampleSpinBox = new QDoubleSpinBox;
	m_subsampleSpinBox->setMinimum(std::numeric_limits<float>::min());
	m_subsampleSpinBox->setMaximum(1);
	m_subsampleSpinBox->setSingleStep(0.1);
	if (!m_params.isNull()) {
		m_subsampleSpinBox->setValue(m_params->subsample());
		connect(m_params.data(), &XgBoostGeneticParams::subsampleChanged, m_subsampleSpinBox, &QDoubleSpinBox::setValue);
	}

	connect(m_subsampleSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &XgBoostGeneticParamsWidget::subsampleChanged);

	xgboostFormLayout->addRow(new QLabel("Sub sample"), m_subsampleSpinBox);

	m_colsampleByTreeSpinBox = new QDoubleSpinBox;
	m_colsampleByTreeSpinBox->setMinimum(std::numeric_limits<float>::min());
	m_colsampleByTreeSpinBox->setMaximum(1);
	m_colsampleByTreeSpinBox->setSingleStep(0.1);
	if (!m_params.isNull()) {
		m_colsampleByTreeSpinBox->setValue(m_params->colsampleByTree());
		connect(m_params.data(), &XgBoostGeneticParams::colsampleByTreeChanged, m_colsampleByTreeSpinBox, &QDoubleSpinBox::setValue);
	}

	connect(m_colsampleByTreeSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &XgBoostGeneticParamsWidget::colsampleByTreeChanged);

	xgboostFormLayout->addRow(new QLabel("Column sample by tree"), m_colsampleByTreeSpinBox);
}

XgBoostGeneticParamsWidget::~XgBoostGeneticParamsWidget() {
	if (!m_params.isNull()) {
		disconnect(m_params.data(), &XgBoostGeneticParams::maxDepthChanged, m_maxDepthSpinBox, &QSpinBox::setValue);
		disconnect(m_params.data(), &XgBoostGeneticParams::nEstimatorsChanged, m_nEstimatorsSpinBox, &QSpinBox::setValue);
		disconnect(m_params.data(), &XgBoostGeneticParams::learningRateChanged, m_learningRateSpinBox, &QDoubleSpinBox::setValue);
		disconnect(m_params.data(), &XgBoostGeneticParams::subsampleChanged, m_subsampleSpinBox, &QDoubleSpinBox::setValue);
		disconnect(m_params.data(), &XgBoostGeneticParams::colsampleByTreeChanged, m_colsampleByTreeSpinBox, &QDoubleSpinBox::setValue);
	}
}

void XgBoostGeneticParamsWidget::maxDepthChanged(int val) {
	if (!m_params.isNull() && val!=m_params->maxDepth()) {
		m_params->setMaxDepth(val);
	}
}

void XgBoostGeneticParamsWidget::nEstimatorsChanged(int val) {
	if (!m_params.isNull() && val!=m_params->nEstimators()) {
		m_params->setNEstimators(val);
	}
}

void XgBoostGeneticParamsWidget::learningRateChanged(double val) {
	if (!m_params.isNull() && val!=m_params->learningRate()) {
		m_params->setLearningRate(val);
	}
}

void XgBoostGeneticParamsWidget::subsampleChanged(double val) {
	if (!m_params.isNull() && val!=m_params->subsample()) {
		m_params->setSubsample(val);
	}
}

void XgBoostGeneticParamsWidget::colsampleByTreeChanged(double val) {
	if (!m_params.isNull() && val!=m_params->colsampleByTree()) {
		m_params->setColsampleByTree(val);
	}
}
