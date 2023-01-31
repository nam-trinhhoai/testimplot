

#ifndef __RGTTORGB16BITSWIDGET__
#define __RGTTORGB16BITSWIDGET__



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
// #include <spectrumProcessWidget.h>
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <horizonSelectWidget.h>
#include <labelLineEditWidget.h>


class SpectrumProcessWidget;
class QTimer;

class RgtToRgb16bitsWidget : public QWidget{
	Q_OBJECT

private:
	class PARAM
	{
	public:
		int windowSize = 64;
		int isoStep = 25;
		int layerNumber = 10;
		QString prefix = "spectrum";
	};

	class PARAM_INIT
	{
	public:
		QString rgb2TmpFullName = "";
		QString seismicFilename = "";
		QString rgtFilename = "";
		QString rgb2tmpfilename = "";
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

		// std::string seismicFilename;
		// char *seismicFilename0 = nullptr;
		// std::string rgtFilename;
		// char *rgtFilename0 = nullptr;
		// std::string outHorizonsDirectory;
		// std::string strOutRGB2Directory;
		// char *outHorizonsDirectory0 = nullptr;
		// char *outRGB2Directory0 = nullptr;
		// std::string strRgb2TmpFullName = "";
		// char *rgb2TmpFullName0;
		// std::string outDirectory = "";
	};

public:
	RgtToRgb16bitsWidget(ProjectManagerWidget *selectorWidget, QWidget* parent = 0);
	virtual ~RgtToRgb16bitsWidget();
	void setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget);
	void trt_threadRun();
	void setDataOutFormat(int val);

private:
	enum START_STOP_STATUS { STATUS_STOP = 0, STATUS_START };
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
	Ihm2 *pIhm2 = nullptr;
	QTimer *timer = nullptr;
	PARAM param;
	PARAM_INIT paramInit;
	int valStartStop = 0;
	void horizonTypeDisplay();
	bool paramInitCreate();
	void setStartStopStatus(START_STOP_STATUS status);
	int dataOutFormat = 0;

	private slots:
	void showTime();
	void trt_horizonTypeDisplay(int idx);
	// void trt_startStop();
	void trt_start();
	void trt_stop();
};



class RgtToRgb16bitsWidgetTHREAD : public QThread
{
	// Q_OBJECT
public:
	RgtToRgb16bitsWidgetTHREAD(RgtToRgb16bitsWidget *p);
private:
	RgtToRgb16bitsWidget *pp;

protected:
	void run();
};




#endif
