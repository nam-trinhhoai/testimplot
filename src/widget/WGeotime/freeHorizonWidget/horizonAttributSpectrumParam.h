
#ifndef __HORIZONATTRIBUTSPECTRUMPARAM__
#define __HORIZONATTRIBUTSPECTRUMPARAM__

class QSpinBox;

class HorizonAttributSpectrumParam : public QWidget {
	Q_OBJECT

public:
	HorizonAttributSpectrumParam();
	virtual ~HorizonAttributSpectrumParam();
	void setWSize(int size);
	int getWSize();
	float getHatPower();

private:
	QSpinBox *m_windowSizeSpinBox = nullptr;
	int spinBoxWidth = 75;
	QLineEdit *m_hatPower = nullptr;
};


#endif
