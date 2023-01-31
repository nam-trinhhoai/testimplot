
#ifndef __RGTSTACKINGPARAMETERSWIDGET__
#define __RGTSTACKINGPARAMETERSWIDGET__

#include <QWidget>
class QLineEdit;
class QComboBox;

class HorizonSelectWidget;
class ProjectManagerWidget;
class WorkingSetManager;

class RgtStackingParametersWidget: public QWidget{
	Q_OBJECT
public:
	RgtStackingParametersWidget(WorkingSetManager *workingSetManager);
	virtual ~RgtStackingParametersWidget();

	void setNbIter(int val);
	void setDipThreshold(double val);
	void setDecimation(int val);
	void setSnapping(bool val);
	void setSeedMaxvalid(bool val);
	void setSeedMax(long val);

	int getNbIter();
	double getDipThreshold();
	int getDecimation();
	bool getSnapping();
	bool getSeedMaxvalid();
	long getSeedMax();
	int getXlimit1();
	int getXlimit2();

	std::vector<QString> getHorizonNames();
	std::vector<QString> getHorizonPath();
	WorkingSetManager *m_workingSetManager = nullptr;



private:
	QLineEdit *lineedit_iteration = nullptr,
	*lineedit_dipthreshold = nullptr,
	*lineedit_decimationfactor = nullptr,
	*lineedit_seedthreshold= nullptr,
	*lineedit_xlimit1 = nullptr,
	*lineedit_xlimit2 = nullptr;

	QCheckBox *qcb_snapping = nullptr,
	*qcb_seedthreshold_valid = nullptr;

	HorizonSelectWidget *m_horizon = nullptr;

};



#endif
