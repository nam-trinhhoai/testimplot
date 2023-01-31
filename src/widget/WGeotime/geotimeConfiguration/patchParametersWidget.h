
#ifndef __PATCHPARAMETERSWIDGET__
#define __PATCHPARAMETERSWIDGET__

#include <string>

class QLineEdit;
class QComboBox;

class HorizonSelectWidget;
class ProjectManagerWidget;
class FileSelectWidget;
class QCheckBox;
class WorkingSetManager;


class PatchParametersWidget: public QWidget{
	Q_OBJECT
public:
	PatchParametersWidget(WorkingSetManager *workingSetManager);
	virtual ~PatchParametersWidget();
	void setPatchSize(int size);
	void setPatchPolarity(int pol);
	void setGradientMax(int grad);
	void setPatchRatio(int ratio);
	void setFaultThreshold(int th);

	int getPatchSize();
	int getPatchRatio();
	int getFaultThreshold();
	bool getFaultMaskInput();
	QString getFaultMaskPath();
	QString getPatchPolarity();
	double getDeltaVOverV();
	int getGradMax();
	std::vector<QString> getHorizonPaths();


private:
	QLineEdit *lineEditPatchSize = nullptr,
	*lineeditPatchGradMax = nullptr,
	*lineeditPatchDeltaVoverV = nullptr,
	*lineEditPatchFitThreshold = nullptr,
	*lineEditPatchFaultMaskThreshold = nullptr;
	QComboBox *cbPatchPolarity = nullptr;
	HorizonSelectWidget *m_horizon = nullptr;
	QCheckBox *faultInput = nullptr;
	FileSelectWidget *m_faultFileSelectWidget = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;


};



#endif
