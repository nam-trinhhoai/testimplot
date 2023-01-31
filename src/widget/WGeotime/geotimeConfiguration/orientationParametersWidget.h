
#ifndef __ORIENTATIONPARAMETERSWIDGET__
#define __ORIENTATIONPARAMETERSWIDGET__

#include <QWidget>
class QLineEdit;
class QComboBox;

class OrientationParametersWidget: public QWidget{
	Q_OBJECT
public:
	OrientationParametersWidget();
	virtual ~OrientationParametersWidget();
	void setGradient(double val);
	void setTensor(double val);
	double getGradient();
	double getTensor();


private:
	QLineEdit *lineedit_sigmagradient = nullptr,
	*lineedit_sigmatensor = nullptr;


};



#endif
