
#ifndef __RGTPATCHPARAMETERSWIDGET__
#define __RGTPATCHPARAMETERSWIDGET__

#include <QWidget>
class QLineEdit;
class QComboBox;

class ProjectManagerWidget;

class RgtPatchParametersWidget: public QWidget{
	Q_OBJECT
public:
	RgtPatchParametersWidget(ProjectManagerWidget *projectManager);
	virtual ~RgtPatchParametersWidget();

	void setScaleInitIter(int val);
	void setScaleInitEpsilon(double val);
	void setScaleInitDecim(int val);
	void setIter(int val);
	void setEpsilon(double val);
	void setDecim(int val);
	int getScaleInitIter();
	double getScaleInitEpsilon();
	int getScaleInitDecim();
	int getIter();
	double getEpsilon();
	int getDecim();
	bool getScaleInitValid();


private:
	QLineEdit *lineEditRgtIdleDipMax = nullptr,
	*lineEditRgtIterScaleInit = nullptr,
	*lineeditTimeSmoothparameterScaleInit = nullptr,
	*lineeditRgtDecimYScaleInit = nullptr,
	*lineEditRgtIter = nullptr,
	*lineeditTimeSmoothparameter = nullptr,
	*lineeditRgtDecimY = nullptr;
	QCheckBox *qcbScaleInit = nullptr;
};



#endif
