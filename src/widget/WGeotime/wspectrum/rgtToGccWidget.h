

#ifndef __RGTTOGCCWIDGET__
#define __RGTTOGCCWIDGET__



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

#include <string>
#include <vector>
#include <string>
#include <math.h>

#include <ihm2.h>
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <horizonSelectWidget.h>
#include <labelLineEditWidget.h>


class SpectrumProcessWidget;

class RgtToGccWidget : public QWidget{
	Q_OBJECT

private:
	class PARAM
	{
	public:
		int windowSize = 11;
		int isoMin = 0;
		int isoMax = 32000;
		int isoStep = 25;
		int layerNumber = 10;
		int w = 7;
		int shift = 5;
		QString prefix = "gcc";
	};

	class PARAM_INIT
	{
	public:
		QString rgb2TmpFullName = "";
		QString seismicFilename = "";
		QString rgtFilename = "";
		int dimx = 0;
		int dimy = 0;
		int dimz = 0;
		float tdeb = 0.0f;
		float pasech = 1.0f;
		int isoMin = 0;
		int isoMax = 32000;
		int isoStep = 25;
		int windowSize = 64;
		int nbLayers = 10;
		float *horizon1 = nullptr;
		float *horizon2 = nullptr;
		// std::string strSeismicFilename;
		// char *seismicFilename0 = nullptr;
		// std::string strRgtFilename;
		// char *rgtFilename0 = nullptr;
		// std::string strOutHorizonsDirectory;
		// std::string strOutRGB2Directory;
		// std::string strOutGccDirectory;
		// char *outHorizonsDirectory0 = nullptr;
		// char *outRGB2Directory0 = nullptr;
		// char *outGccDirectory0 = nullptr;

		// std::string strOutMeanDirectory;
		// char *outMeanDirectory0 = nullptr;

		// std::string strRgb2TmpFullName = "";
		// char *rgb2TmpFullName0;
		// std::string outDirectory = "";
		QString outMainDirectory = "";
	};

public:
	RgtToGccWidget(ProjectManagerWidget *selectorWidget, QWidget* parent = 0);
	virtual ~RgtToGccWidget();
	void setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget);
	void trt_threadRun();

private:
	enum START_STOP_STATUS { STATUS_STOP = 0, STATUS_START };
	// GeotimeProjectManagerWidget *m_selectorWidget;
	ProjectManagerWidget *m_selectorWidget = nullptr;
	SpectrumProcessWidget *m_spectrumProcessWidget = nullptr;
	FileSelectWidget *m_rgtFileSelectWidget;
	LabelLineEditWidget *m_prefixFilename;
	LabelLineEditWidget *m_windowSize;
	LabelLineEditWidget *m_isoMin;
	LabelLineEditWidget *m_isoMax;
	LabelLineEditWidget *m_isoStep;
	LabelLineEditWidget *m_layerNumber;
	LabelLineEditWidget *m_w;
	LabelLineEditWidget *m_shift;
	QComboBox *m_isoValOrHorizon;
	HorizonSelectWidget *m_horizonSelectWidget;
	QProgressBar *m_progressBar;
	QPushButton *m_start;
	QPushButton *m_stop;
	QPushButton *m_kill;
	Ihm2 *pIhm2 = nullptr;
	PARAM param;
	PARAM_INIT paramInit;
	int valStartStop = 0;
	QTimer *timer = nullptr;
	void horizonTypeDisplay();
	bool paramInitCreate();
	void setStartStopStatus(START_STOP_STATUS status);

	private slots:
	void showTime();
	void trt_horizonTypeDisplay(int idx);
	void trt_start();
	void trt_stop();
};



class RgtToGccWidgetTHREAD : public QThread
{
	// Q_OBJECT
public:
	RgtToGccWidgetTHREAD(RgtToGccWidget *p);
private:
	RgtToGccWidget *pp;

protected:
	void run();
};




#endif
