
#include <QSpinBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QFormLayout>

#include <horizonAttributSpectrumParam.h>

HorizonAttributSpectrumParam::HorizonAttributSpectrumParam()
{
	QVBoxLayout *mainLayout = new QVBoxLayout(this);

	// QGroupBox *gb = new QGroupBox("Spectrum Parameters :");
	QFormLayout* spectrumForm = new QFormLayout;
	// QLabel *label = new QLabel;
	// spectrumForm->addRow("Spectrum", label);

	QFormLayout* qfWindowSize = new QFormLayout;
	m_windowSizeSpinBox = new QSpinBox;
	m_windowSizeSpinBox->setMinimum(1);
	m_windowSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_windowSizeSpinBox->setValue(64);
	m_windowSizeSpinBox->setMaximumWidth(spinBoxWidth);
	m_windowSizeSpinBox->setFixedHeight(25);
	qfWindowSize->addRow("Window Size", m_windowSizeSpinBox);


	QFormLayout* qfHatPower = new QFormLayout;
	m_hatPower = new QLineEdit("5.0");
	m_hatPower->setMaximumWidth(70);
	m_hatPower->setMaximumWidth(75);
	qfHatPower->addRow("Hat power", m_hatPower);

	QHBoxLayout *qhblayout = new QHBoxLayout;
	qhblayout->addLayout(qfWindowSize);
	qhblayout->addLayout(qfHatPower);


	// spectrumForm->addRow("Window Size", m_windowSizeSpinBox);
	spectrumForm->addRow("", qhblayout);

	// m_spectrumParameters->setLayout(spectrumForm);
	mainLayout->addLayout(spectrumForm);
}

HorizonAttributSpectrumParam::~HorizonAttributSpectrumParam()
{

}


void HorizonAttributSpectrumParam::setWSize(int size)
{
	m_windowSizeSpinBox->setValue(size);
}

int HorizonAttributSpectrumParam::getWSize()
{
	return m_windowSizeSpinBox->value();
}

float HorizonAttributSpectrumParam::getHatPower()
{
	return m_hatPower->text().toFloat();
}


