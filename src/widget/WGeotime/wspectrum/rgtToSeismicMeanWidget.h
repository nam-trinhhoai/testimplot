

#ifndef __RGTTOSEISMICMEANWIDGET__
#define __RGTTOSEISMICMEANWIDGET__



#include <QThread>
#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QComboBox>
#include <QGroupBox>
#include <QDialog>
#include <QString>
#include <QTimer>

#include <string>
#include <vector>
#include <string>
#include <math.h>

#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <horizonSelectWidget.h>
#include <ihm2.h>
#include <labelLineEditWidget.h>


class SpectrumProcessWidget;

class RgtToSeismicMeanWidget : public QWidget{
	Q_OBJECT

private:
	class PARAM
	{
	public:
		int windowSize = 11;
		int isoStep = 25;
		int layerNumber = 10;
		QString prefix = "mean";
	};

	class PARAM_INIT
	{
	public:
		QString rgb2TmpFullName = "";
		QString seismicFilename = "";
		QString rgtFilename = "";
		QString outMainDirectory = "";
		int dimx = 0;
		int dimy = 0;
		int dimz = 0;
		float tdeb = 0.0f;
		float pasech = 1.0f;
		int isoStep = 25;
		int windowSize = 64;
		int nbLayers = 10;
		float *horizon1 = nullptr;
		float *horizon2 = nullptr;
	};

public:
	RgtToSeismicMeanWidget(ProjectManagerWidget *selectorWidget, QWidget* parent = 0);
	virtual ~RgtToSeismicMeanWidget();
	void setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget);
	void trt_threadRun();

private:
	enum START_STOP_STATUS { STATUS_STOP = 0, STATUS_START };
	QTimer *timer = nullptr;
	Ihm2 *pIhm2 = nullptr;
	// GeotimeProjectManagerWidget *m_selectorWidget;
	ProjectManagerWidget *m_selectorWidget = nullptr;
	SpectrumProcessWidget *m_spectrumProcessWidget = nullptr;
	FileSelectWidget *m_rgtFileSelectWidget;
	LabelLineEditWidget *m_prefixFilename;
	LabelLineEditWidget *m_windowSize;
	LabelLineEditWidget *m_isoStep;
	LabelLineEditWidget *m_layerNumber;
	QComboBox *m_isoValOrHorizon;
	HorizonSelectWidget *m_horizonSelectWidget;
	QProgressBar *m_progressBar;
	QPushButton *m_start;
	QPushButton *m_stop;
	QPushButton *m_kill;
	PARAM param;
	PARAM_INIT paramInit;
	int valStartStop = 0;
	void horizonTypeDisplay();
	bool paramInitCreate();
	void setStartStopStatus(START_STOP_STATUS status);



	private slots:
	void showTime();
	void trt_horizonTypeDisplay(int idx);
	void trt_start();
	void trt_stop();
};



class RgtToSeismicMeanWidgetTHREAD : public QThread
{
	// Q_OBJECT
public:
	RgtToSeismicMeanWidgetTHREAD(RgtToSeismicMeanWidget *p);
private:
	RgtToSeismicMeanWidget *pp;

protected:
	void run();
};




#endif
