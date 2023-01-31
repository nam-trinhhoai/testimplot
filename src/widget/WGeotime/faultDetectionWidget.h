
#ifndef __FAULTDETECTIONWIDGET__
#define __FAULTDETECTIONWIDGET__

class QWidget;
class LabelLineEditWidget;
class GeotimeConfigurationWidget;
class QProgressBar;
class QTimer;

class WorkingSetManager;
class ProjectManagerWidget;

#include <faultDetection.h>

class FaultDetectionWidget : public QWidget{
	Q_OBJECT

private:
	class NORMALIZATION
		{
		public:
			int type = 1;
			double alpha1 = 0.1;
			double alpha2 = 0.05;
			float coef = 100.0f;

		};
		class DETECTION
		{
		public:
			int length = 71;
			int width = 1;
			int partitionInto = 9;
		};
		class DIP
		{
		public:
			int maximum = 40;
			int step = 2;
		};
		class FILTER
		{
		public:
			int size = 51;
			double brightPoint = 0.05;
			int variance = 0;
		};
		class PARAMETERS
		{
		public:
			NORMALIZATION normalization;
			DETECTION detection;
			DIP dip;
			FILTER filter;
			int crestDetection = 1;
			FaultDetection::DIRECTION direction = FaultDetection::DIRECTION::_both;
			int labelMinSize = 25;
		};
public:
	FaultDetectionWidget(WorkingSetManager *workingSetManager, QWidget* parent = 0);
	virtual ~FaultDetectionWidget();
	void setGeotimeConfigurationWidget(GeotimeConfigurationWidget *val);
	void trt_threadRun();
	PARAMETERS parameters;

private:
	enum START_STOP_STATUS { STATUS_STOP = 0, STATUS_START };
	QLabel *m_processing = nullptr;
	LabelLineEditWidget *m_faulFileName;
	LabelLineEditWidget *m_crestDetection;
	QPushButton *m_btnStartStop;
	QLineEdit *qleNormalizationType, *qleNormalizationAlpha1, *qleNormalizationAlpha2,
	*qleDetectionLength, *qleDetectionWidth, *qleDetectionPartition,
	*qleDipMaximum, *qleDipStep, *qleFilterSize, *qleFilterBright, *qleFilterVariance,
	*qleLabelMinSize = nullptr;
	QProgressBar *m_progressBar;
	QCheckBox *m_validCrestDetection;
	QTimer *timer = nullptr;
	QComboBox *m_direction = nullptr;
	FileSelectWidget *m_seismicFileSelectWidget = nullptr;
	ProjectManagerWidget *m_projectManager = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;

	GeotimeConfigurationWidget *geotimeConfigurationWidget = nullptr;
	int imhId = 1;
	int imhAbortId = 2;
	int imhEndId = 3;
	int valStartStop = STATUS_STOP;
	void trt_start();
	void trt_stop();
	void setStartStopStatus(START_STOP_STATUS status);
	bool flagEnd = 0;
	unsigned int m_cptProcessing = 0;
	void processingDisplay();

private slots:
	void trt_startStop();
	void showTime();

};


class FaultDetectionWidgetTHREAD : public QThread
{
	// Q_OBJECT
public:
	FaultDetectionWidgetTHREAD(FaultDetectionWidget *p);
private:
	FaultDetectionWidget *pp;

protected:
	void run();
};





#endif
