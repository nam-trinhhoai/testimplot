
#ifndef __HORIZONATTRIBUTMEANPARAM__
#define __HORIZONATTRIBUTMEANPARAM__

class QSpinBox;

class HorizonAttributMeanParam : public QWidget {
	Q_OBJECT

public:
	HorizonAttributMeanParam();
	virtual ~HorizonAttributMeanParam();
	void setWSize(int size);
	int getWSize();

private:
	QSpinBox *m_windowSizeSpinBox = nullptr;
	const int spinBoxWidth = 75;
};


#endif
