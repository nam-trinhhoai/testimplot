

#ifndef __ATTRIBUTTOXTWIDGET__
#define __ATTRIBUTTOXTWIDGET__


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

#include <ihm2.h>
#include <spectrumProcessWidget.h>
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <horizonSelectWidget.h>
#include <labelLineEditWidget.h>
#include <attributToXt.h>

class SpectrumProcessWidget;


class AttributToXtWidget : public QWidget{
	Q_OBJECT

private:
	class PARAM
	{
	public:
		QString directory = "";
		QString outFilename = "";
		QString seismicFilename = "";
		AttributToXt::TYPE0 type = AttributToXt::TYPE0::rgb2;
		QString dataType = "rgb2";
	};

public:
	AttributToXtWidget(ProjectManagerWidget *selectorWidget, QWidget* parent = 0);
	virtual ~AttributToXtWidget();
	void setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget);
	void trt_threadRun();


private:
	enum START_STOP_STATUS { STATUS_STOP = 0, STATUS_START };
	ProjectManagerWidget *m_selectorWidget = nullptr;
	// FileSelectWidget *m_attributDirectoty = nullptr;
	LabelLineEditWidget *m_prefixFilename = nullptr;
	QLineEdit *m_lineEditAttributDirectory = nullptr;
	QComboBox *m_attributType = nullptr;
	QPushButton *m_start, *m_stop;
	QProgressBar *m_progressBar;
	SpectrumProcessWidget *m_spectrumProcessWidget = nullptr;
	int valStartStop = 0;
	float rgb2Torgb1Ratio = 0.0001;
	float rgb2Torgb1Alpha = 1.0;
	Ihm2 *pIhm2 = nullptr;
	QTimer *timer;
	QString m_attributDirectory = "";

	PARAM param;
	std::vector<std::string> getIsoPath(QString path);
	int paramCreate();
	void setStartStopStatus(START_STOP_STATUS status);
	private slots:
	void trt_startStop();
	void trt_start();
	void trt_stop();
	void showTime();
	void trt_attributDirectoryOpen();
};


class AttributToXtWidgetTHREAD : public QThread
{
	// Q_OBJECT
public:
	AttributToXtWidgetTHREAD(AttributToXtWidget *p);
private:
	AttributToXtWidget *pp;

protected:
	void run();
};








#endif
