#include "computelogdialog.h"

#include <limits>

#include <QVBoxLayout>
#include <QFormLayout>
#include <QSpinBox>
#include <QLineEdit>
#include <QDoubleValidator>
#include <QComboBox>
#include <QDialogButtonBox>

ComputeLogDialog::ComputeLogDialog(QStringList log_list, QWidget *parent)
	: QDialog(parent) {

	this->log_list = log_list;

	QVBoxLayout* mainLayout = new QVBoxLayout();
	this->setLayout(mainLayout);

	QFormLayout* layout = new QFormLayout();
	mainLayout->addLayout(layout);

	nameLineEdit = new QLineEdit();
	layout->addRow("Log name", nameLineEdit);

	logDtComboBox = new QComboBox();
	logDtComboBox->addItems(this->log_list);
	layout->addRow("Dt log", logDtComboBox);

	logAttributComboBox = new QComboBox();
	logAttributComboBox->addItems(this->log_list);
	layout->addRow("Attribut log", logAttributComboBox);

	this->freqLineEdit = new QLineEdit("1");
	this->freqValidator = new QDoubleValidator(std::numeric_limits<double>::min(),
			std::numeric_limits<double>::max(),
			std::numeric_limits<int>::max());
	this->freqLineEdit->setValidator(this->freqValidator);
	layout->addRow("Frequency", this->freqLineEdit);
	connect(this->freqLineEdit, &QLineEdit::textChanged, this, [this](const QString& txt) {
		bool test;
		float freq = this->locale().toFloat(txt, &test);
		if (test) {
			this->frequency = freq;
		}
	});

	this->waveletSizeSpinBox = new QSpinBox();
	this->waveletSizeSpinBox->setValue(1);
	this->waveletSizeSpinBox->setMinimum(1);
	this->waveletSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	layout->addRow("Wavelet Size", this->waveletSizeSpinBox);
	connect(this->waveletSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int val) {
		this->wavelet_size = val;
	});

	QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &ComputeLogDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &ComputeLogDialog::reject);

	mainLayout->addWidget(buttonBox);
}

ComputeLogDialog::~ComputeLogDialog() {}

float ComputeLogDialog::getFrequency() {
	return frequency;
}

void ComputeLogDialog::setFrequency(float freq) {
	frequency = freq;
	freqLineEdit->setText(locale().toString(freq));
}

int ComputeLogDialog::getWaveletSize() {
	return wavelet_size;
}

void ComputeLogDialog::setWaveletSize(int wavelet_size) {
	this->wavelet_size = wavelet_size;
	waveletSizeSpinBox->setValue(wavelet_size);
}

QString ComputeLogDialog::getLogDt() {
	return logDtComboBox->currentText();
}

QString ComputeLogDialog::getLogAttribut() {
	return logAttributComboBox->currentText();
}

QString ComputeLogDialog::getLogName() {
	return nameLineEdit->text();
}

void ComputeLogDialog::setLogName(QString txt) {
	nameLineEdit->setText(txt);
}
