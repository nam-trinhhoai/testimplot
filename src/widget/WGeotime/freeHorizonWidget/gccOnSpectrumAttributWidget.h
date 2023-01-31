
#ifndef __GCCONSPECTRUMATTRIBUTWIDGET__
#define __GCCONSPECTRUMATTRIBUTWIDGET__

#include <QWidget>
#include <QSpinBox>
#include <QListWidget>
#include <QComboBox>
#include <QGroupBox>
#include <QProgressBar>
#include <QPushButton>
#include <QString>
#include <vector>
#include <QThread>
#include <QCheckBox>
#include <QTimer>
#include <ihm2.h>
#include <QLineEdit>
#include <QSlider>



class DataSelectorDialog;
class GeotimeProjectManagerWidget;
class HorizonAttributSpectrumParam;
class HorizonAttributMeanParam;
class HorizonAttributGCCParam;
class WorkingSetManager;



class GccOnSpectrumAttributWidget : public QWidget {
	Q_OBJECT

public:
	GccOnSpectrumAttributWidget(QString surveyPath, QString dirPath, QString spectrumName, WorkingSetManager *manager, QWidget *parent = 0);
	virtual ~GccOnSpectrumAttributWidget();
	void trt_threadRun();

	// void setProjectManager(GeotimeProjectManagerWidget *p) { m_selectorWidget = p; }
	// void setWorkingSetManager(WorkingSetManager *p) { m_workingSetManager = p; }
	// void setInputHorizons(QString path, QString name);

private:
	QString m_dirPath = "";
	QString m_spectrumName = "";
	QString m_path = "";
	QString m_dataSetPath = "";
	QString m_surveyPath = "";
	QString m_horizonName = "";

	QSpinBox *m_centralFrequency = nullptr;
	QLineEdit *m_fcentral_hz = nullptr;
	QSlider *m_scrollFreq = nullptr;
	QLineEdit *m_amplFreqIdx = nullptr;
	QLineEdit *m_amplFreqHz = nullptr;
	QSpinBox *m_spinW = nullptr;
	QLineEdit *m_ampl = nullptr;

	QSpinBox *m_f1SpinBox = nullptr;
	QSpinBox *m_f2SpinBox = nullptr;

	QLineEdit *m_f1_idx = nullptr;
	QLineEdit *m_f2_idx = nullptr;
	QLineEdit *m_f1_hz = nullptr;
	QLineEdit *m_f2_hz = nullptr;
	QLineEdit *m_f1 = nullptr;
	QLineEdit *m_f2 = nullptr;
	int m_nbFreq = 0;
	float m_fech = 1.0f;
	int m_dimx = 0;
	int m_dimy = 0;
	int m_dimz = 0;
	int m_w = 7;
	int m_shift = 0;
	Ihm2 *pIhm = nullptr;
	int pStatus = 0;
	unsigned int m_cptProcessing = 0;

	QPushButton *m_start = nullptr;
	QProgressBar *m_progress = nullptr;
	void initParams();
	void displayParams();
	void trt_compute();
	// QString getOutAttibutFilename();
	QString getOutAttibutPath();
	QString getOutAttibutName();
	WorkingSetManager *m_manager = nullptr;
	QLabel *m_processing = nullptr;
	void processingDisplay();
	void displayProcessFinish();
	const int spinBoxWidth = 150;
	double idxToFreq(int idx);
	int m_cacheAmpl = 0;
	int m_cacheFc = 0;



private slots:
	void fcChanged(int value);
	void famplChanged(int value);
	void trt_launch();
	void showTime();


private:
	class MyThread0 : public QThread
	{
		// Q_OBJECT
	public:
		MyThread0(GccOnSpectrumAttributWidget *p);
	private:
		GccOnSpectrumAttributWidget *pp;

	protected:
		void run();
	};

};

/*

class HorizonAttributComputeDialogTHREAD : public QThread
{
	// Q_OBJECT
public:
	HorizonAttributComputeDialogTHREAD(HorizonAttributComputeDialog *p);
private:
	HorizonAttributComputeDialog *pp;

protected:
	void run();
};
*/

#endif
