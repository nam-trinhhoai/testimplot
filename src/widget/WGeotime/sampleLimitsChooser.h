
#ifndef __SAMPLELIMITSCHOOSER__
#define __SAMPLELIMITSCHOOSER__

#include <QDialog>

class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QScrollBar;

class SampleLimitsChooser: public QDialog {
Q_OBJECT
public:
	SampleLimitsChooser(QWidget *parent, int dimx, int maxSize, float startSample, float stepSample, int *x1, int *x2);
	virtual ~SampleLimitsChooser();

private:
	int dimx;
	int maxSize;
	float startSample;
	float endSample;
	float stepSample;
	int *x1;
	int *x2;
	int px1;
	int px2;
	int margin = 10;

	QScrollBar *sbT1, *sbT2;

	QLineEdit *lineEditT1 = nullptr;
	QLineEdit *lineEditT2 = nullptr;
	QPushButton *qpbPlusT1 = nullptr;
	QPushButton *qpbMinusT1 = nullptr;
	QPushButton *qpbPlusT2 = nullptr;
	QPushButton *qpbMinusT2 = nullptr;

	QLabel *labelT1Value;
	QLabel *labelT2Value;
	QLabel *labelSizeInfo;

	void labelInfoDisplay();


private slots:
	void accepted();
	void sbT1Change(int val);
	void sbT2Change(int val);
};


#endif
