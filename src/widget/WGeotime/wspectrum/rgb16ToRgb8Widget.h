

#ifndef __RGB16TORGB8WIDGET__
#define __RGB16TORGB8WIDGET__



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
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <horizonSelectWidget.h>
#include <labelLineEditWidget.h>


class SpectrumProcessWidget;

class Rgb16ToRgb8Widget : public QWidget{
	Q_OBJECT

public:
	Rgb16ToRgb8Widget(ProjectManagerWidget *selectorWidget, QWidget* parent = 0);
	virtual ~Rgb16ToRgb8Widget();
	void setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget);
	void trt_threadRun();

private:
	enum START_STOP_STATUS { STATUS_STOP = 0, STATUS_START };
	// GeotimeProjectManagerWidget *m_selectorWidget;
	ProjectManagerWidget *m_selectorWidget = nullptr;
	SpectrumProcessWidget *m_spectrumProcessWidget = nullptr;
	FileSelectWidget *m_rgb2FileSelectWidget;
	LabelLineEditWidget *m_prefixFilename;
	LabelLineEditWidget *m_ratio;
	LabelLineEditWidget *m_alpha;
	QProgressBar *m_progressBar;
	QPushButton *m_start;
	QPushButton *m_stop;
	QPushButton *m_kill;
	Ihm2 *pIhm2 = nullptr;
	int valStartStop = 0;
	QTimer *timer = nullptr;
	void setStartStopStatus(START_STOP_STATUS status);
	QString filenameToPath(QString fullName);

	private slots:
	void showTime();
	void trt_start();
	void trt_stop();
};



class  Rgb16ToRgb8WidgetTHREAD : public QThread
{
	// Q_OBJECT
public:
	 Rgb16ToRgb8WidgetTHREAD( Rgb16ToRgb8Widget *p);
private:
	 Rgb16ToRgb8Widget *pp;

protected:
	void run();
};




#endif
