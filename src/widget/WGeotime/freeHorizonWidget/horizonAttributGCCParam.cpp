
#include <QSpinBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QLabel>
#include <QFormLayout>

#include <horizonAttributGCCParam.h>

HorizonAttributGCCParam::HorizonAttributGCCParam()
{
	QHBoxLayout *mainLayout = new QHBoxLayout(this);


	// QFormLayout* gccForm = new QFormLayout;
	// QLabel *label = new QLabel;
	// gccForm->addRow("GCC", label);

	QFormLayout* offsetForm = new QFormLayout;
	m_gccOffsetSpinBox = new QSpinBox;
	m_gccOffsetSpinBox->setMinimum(1);
	m_gccOffsetSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_gccOffsetSpinBox->setValue(7);
	m_gccOffsetSpinBox->setMaximumWidth(spinBoxWidth);
	m_gccOffsetSpinBox->setFixedHeight(25);
	offsetForm->addRow("Window size ", m_gccOffsetSpinBox);

	QFormLayout* wForm = new QFormLayout;
	m_wSpinBox = new QSpinBox;
	m_wSpinBox->setMinimum(std::numeric_limits<int>::min());
	m_wSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_wSpinBox->setValue(7);
	m_wSpinBox->setMaximumWidth(spinBoxWidth);
	m_wSpinBox->setFixedHeight(25);
	wForm->addRow("W ", m_wSpinBox);

	QFormLayout* shiftForm = new QFormLayout;
	m_shiftSpinBox = new QSpinBox;
	m_shiftSpinBox->setMinimum(std::numeric_limits<int>::min());
	m_shiftSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_shiftSpinBox->setValue(5);
	m_shiftSpinBox->setMaximumWidth(spinBoxWidth);
	m_shiftSpinBox->setFixedHeight(25);
	shiftForm->addRow("Shift ", m_shiftSpinBox);

	// m_gccParameters->setLayout(gccForm);

	mainLayout->addLayout(offsetForm);
	mainLayout->addLayout(wForm);
	mainLayout->addLayout(shiftForm);

}

HorizonAttributGCCParam::~HorizonAttributGCCParam()
{

}


void HorizonAttributGCCParam::setOffset(int val)
{
	m_gccOffsetSpinBox->setValue(val);
}

int HorizonAttributGCCParam::getOffset()
{
	return m_gccOffsetSpinBox->value();
}

void HorizonAttributGCCParam::setW(int val)
{
	m_wSpinBox->setValue(val);
}

int HorizonAttributGCCParam::getW()
{
	return m_wSpinBox->value();
}

void HorizonAttributGCCParam::setShift(int val)
{
	m_shiftSpinBox->setValue(val);
}

int HorizonAttributGCCParam::getShift()
{
	return m_shiftSpinBox->value();
}









