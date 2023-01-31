

#ifndef __RAWTOAVIWIDGET__
#define __RAWTOAVIWIDGET__



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

#include "processwatcherwidget.h"

#include <ihm2.h>
#include "SectionToVideo.h"
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <horizonSelectWidget.h>
#include <labelLineEditWidget.h>


class SpectrumProcessWidget;
class QDoubleSpinBox;
class QSpinBox;

class RawToAviWidget : public QWidget{
	Q_OBJECT

private:

public:
	RawToAviWidget(ProjectManagerWidget *selectorWidget, QWidget* parent = 0);
	virtual ~RawToAviWidget();
	void setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget);
	void trt_threadRun();
	void rawToAviRunFfmeg(bool onlyFirstImage);

private:
	SpectrumProcessWidget *m_spectrumProcessWidget = nullptr;
	ProjectManagerWidget *m_selectorWidget = nullptr;
	// FileSelectWidget *m_rawToAviRgb1FileSelectWidget = nullptr;
	QLineEdit *m_lineEditAttributDirectory = nullptr;
	FileSelectWidget *m_rawToAviRgb2FileSelectWidget = nullptr;
	QLineEdit *lineedit_aviPrefix;
	QSpinBox* spinboxRgb1TOrigin, *spinboxRgb1TStep, *spinboxFps, *spinboxFirstIso, *spinboxLastIso;
	QDoubleSpinBox* spinboxvideoScale;
	QCheckBox* checkboxRgb1IsReversed;
	QComboBox* directionComboBox;
	QComboBox *m_attributType;
	QSpinBox* sectionPositionSpinBox;
	QSlider* sectionPositionSlider;
	QPushButton* colorHolder;
	double videoScale = 1.0;
	int rgb1TOrigin = 0;
	int rgb1TStep = 25;
	int framePerSecond = 25;
	int firstIso = 0;
	int lastIso = 31975;
	int textSize = 24;
	QColor textColor = Qt::white;
	ProcessWatcherWidget* processwatcher_rawtoaviprocess;
	QString aviTinyName, aviFullName;
	QString rgb2_Avi_FullName;
	QProgressBar *m_progressBar;
	Ihm2 *pIhm2 = nullptr;
	QTimer *timer;
	int valStartStop = 0;
	QString m_attributDirectory = "";

	bool rgb1IsReversed = true;

	std::unique_ptr<QMetaObject::Connection> m_processwatcher_connection;
	std::pair<QString, QString> getPadCropFFMPEG(double wRatio, double hRatio);


	/*
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
	QPushButton *m_startStop;
	PARAM param;
	PARAM_INIT paramInit;
	int valStartStop = 0;
	void horizonTypeDisplay();
	bool paramInitCreate();
	void setStartStopStatus(START_STOP_STATUS status);
	*/
	void computeoptimalscale_rawToAvi();
	std::size_t getFileSize(const char* filePath);
	QString formatColorForFFMPEG(const QColor& color);
	void aviFilenameUpdate();
	bool getPropertiesFromDatasetPath(QString filename, int* size, double* steps, double* origins = nullptr);
	std::tuple<double, bool, double> getAngleFromFilename(QString seismicFullName);
	QSizeF newSizeFromSizeAndAngle(QSizeF oriSize, double angle);
	QString filenameToPath(QString fullName);
	QString getFFMPEGTime(double _time);
	QString formatTimeWithMinCharacters(double time, int minCharNumber=2);
	void setAviTextColor(const QColor& color);
	std::vector<std::string> getIsoPath(QString path);


	private slots:
	void trt_videoScaleChanged(double val);
	void trt_rgb1TOriginChanged(int val);
	void trt_rgb1TStepChanged(int val);
	void trt_rgb1IsReversedChanged(int state);
	void trt_fpsChanged(int val);
	void trt_firstIsoChanged(int val);
	void trt_lastIsoChanged(int val);
	void trt_textSizeChanged(int val);
	void updateRGBD();
	void trt_rawToAviRun();
	void trt_rawToAviDisplay();
	void trt_changeAviTextColor();
	void trt_directionChanged(int newComboBoxIndex);
	void trt_sectionIndexChanged(int sectionIndex);
	void showTime();
	void trt_attributDirectoryOpen();
};

class RawToAviWidgetTHREAD : public QThread
{
	// Q_OBJECT
public:
	RawToAviWidgetTHREAD(RawToAviWidget *p, bool _val);
private:
	RawToAviWidget *pp;
	bool val;

protected:
	void run();
};



#endif
