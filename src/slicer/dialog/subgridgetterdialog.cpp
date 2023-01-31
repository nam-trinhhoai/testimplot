#include "subgridgetterdialog.h"
#include "GeotimeSystemInfo.h"

#include <QVBoxLayout>
#include <QFormLayout>
#include <QSpinBox>
#include <QDialogButtonBox>
#include <QPushButton>
#include <limits>

SubGridGetterDialog::SubGridGetterDialog(long begin, long end, long step, QWidget *parent, Qt::WindowFlags f) : QDialog(parent, f) {
	if (begin>end) {
		long tmp = end;
		end = begin;
		begin = tmp;
	}
	step = std::abs(step);

	m_oriBegin = begin;
	m_oriEnd = end;
	m_oriStep = step;

	m_outBegin = m_oriBegin;
	m_outEnd = m_oriEnd;
	m_outStep = m_oriStep;

	m_activateMemoryCost = false;
	m_oneStepMemoryCost = 0;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);
	QFormLayout* form = new QFormLayout;
	mainLayout->addLayout(form);

	QSpinBox* beginSpinBox = new QSpinBox;
	beginSpinBox->setMinimum(m_oriBegin);
	beginSpinBox->setMaximum(m_outEnd);
	beginSpinBox->setSingleStep(m_oriStep);
	beginSpinBox->setValue(m_outBegin);
	form->addRow("Begin", beginSpinBox);

	QSpinBox* endSpinBox = new QSpinBox;
	endSpinBox->setMinimum(m_outBegin);
	endSpinBox->setMaximum(m_oriEnd);
	endSpinBox->setSingleStep(m_outStep);
	endSpinBox->setValue(m_outEnd);
	form->addRow("End", endSpinBox);

	QSpinBox* stepSpinBox = new QSpinBox;
	stepSpinBox->setMinimum(m_oriStep);
	stepSpinBox->setMaximum(std::numeric_limits<int>::max());
	stepSpinBox->setSingleStep(m_oriStep);
	stepSpinBox->setValue(m_outStep);
	form->addRow("Step", stepSpinBox);

	m_memoryCostFormLabel = new QLabel("Memory Cost");
	m_memoryCostLabel = new QLabel("");
	form->addRow(m_memoryCostFormLabel, m_memoryCostLabel);
	m_memoryCostFormLabel->hide();
	m_memoryCostLabel->hide();

	m_availableMemFormLabel = new QLabel("Memory Available");
	m_availableMemLabel = new QLabel("");
	form->addRow(m_availableMemFormLabel, m_availableMemLabel);
	m_availableMemFormLabel->hide();
	m_availableMemLabel->hide();

	connect(beginSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this, endSpinBox](int val) {
		m_outBegin = val;
		endSpinBox->setMinimum(m_outBegin);
		updateMemoryCost();
	});

	connect(endSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this, beginSpinBox](int val) {
		m_outEnd = val;
		beginSpinBox->setMaximum(m_outEnd);
		updateMemoryCost();
	});

	connect(stepSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this, endSpinBox](int val) {
		m_outStep = val;
		endSpinBox->setSingleStep(m_outStep);
		updateMemoryCost();
	});

	m_buttonBox = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok);
	mainLayout->addWidget(m_buttonBox);

	connect(m_buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
	connect(m_buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
}

SubGridGetterDialog::~SubGridGetterDialog() {

}

long SubGridGetterDialog::outBegin() const {
	return m_outBegin;
}

long SubGridGetterDialog::outEnd() const {
	return m_outEnd;
}

long SubGridGetterDialog::outStep() const {
	return m_outStep;
}

void SubGridGetterDialog::activateMemoryCost(long long oneStepCost) {
	m_activateMemoryCost = true;
	m_oneStepMemoryCost = oneStepCost;

	updateMemoryCost();

	m_memoryCostFormLabel->show();
	m_memoryCostLabel->show();
	m_availableMemFormLabel->show();
	m_availableMemLabel->show();
}

void SubGridGetterDialog::updateMemoryCost() {
	if (m_activateMemoryCost) {
		long long memoryCost = m_oneStepMemoryCost * ((m_outEnd - m_outBegin) / m_outStep + 1 );
		m_memoryCostLabel->setText(QString::number(memoryCost / 1e9) + " GB");

		double availMem = GeotimeSystemInfo::qt_cpu_free_memory() * 0.8; // in Giga bytes (not Gibi /GiB), use safety margin
		m_availableMemLabel->setText(QString::number(availMem) + " GB");

		if (memoryCost > availMem * 1e9) {
			m_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
		} else {
			m_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
		}
	}
}
