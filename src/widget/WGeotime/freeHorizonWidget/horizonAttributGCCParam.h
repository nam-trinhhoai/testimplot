
#ifndef __HORIZONATTRIBUTGCCPARAM__
#define __HORIZONATTRIBUTGCCPARAM__

class QSpinBox;

class HorizonAttributGCCParam : public QWidget {
	Q_OBJECT

public:
	HorizonAttributGCCParam();
	virtual ~HorizonAttributGCCParam();
	void setOffset(int val);
	int getOffset();
	void setW(int val);
	int getW();
	void setShift(int val);
	int getShift();

private:
	QSpinBox *m_gccOffsetSpinBox = nullptr;
	QSpinBox *m_wSpinBox = nullptr;
	QSpinBox *m_shiftSpinBox = nullptr;
	const int spinBoxWidth = 75;
};


#endif
