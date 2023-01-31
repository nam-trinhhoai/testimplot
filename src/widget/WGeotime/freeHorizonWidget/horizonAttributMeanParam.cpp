
#include <QSpinBox>
#include <QGroupBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QFormLayout>

#include <horizonAttributMeanParam.h>

HorizonAttributMeanParam::HorizonAttributMeanParam()
{
	QVBoxLayout *mainLayout = new QVBoxLayout(this);

	// QGroupBox *gb = new QGroupBox("Spectrum Parameters :");
	QFormLayout* spectrumForm = new QFormLayout;

	// QLabel *label = new QLabel;
	// spectrumForm->addRow("Mean", label);

	m_windowSizeSpinBox = new QSpinBox;
	m_windowSizeSpinBox->setMinimum(1);
	m_windowSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_windowSizeSpinBox->setValue(64);
	m_windowSizeSpinBox->setMaximumWidth(spinBoxWidth);
	m_windowSizeSpinBox->setFixedHeight(25);
	spectrumForm->addRow("Window size", m_windowSizeSpinBox);

	// m_spectrumParameters->setLayout(spectrumForm);
	mainLayout->addLayout(spectrumForm);

}

HorizonAttributMeanParam::~HorizonAttributMeanParam()
{

}


void HorizonAttributMeanParam::setWSize(int size)
{
	m_windowSizeSpinBox->setValue(size);
}

int HorizonAttributMeanParam::getWSize()
{
	return m_windowSizeSpinBox->value();
}


