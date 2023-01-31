
#ifndef __HORIZONATTRIBUTCOMPUTEDIALOG__
#define __HORIZONATTRIBUTCOMPUTEDIALOG__

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



class DataSelectorDialog;
class GeotimeProjectManagerWidget;
class HorizonAttributSpectrumParam;
class HorizonAttributMeanParam;
class HorizonAttributGCCParam;
class WorkingSetManager;
class CollapsableScrollArea;
class ProjectManagerWidget;
class HorizonSelectWidget;



class HorizonAttributComputeDialog : public QWidget {
	Q_OBJECT

public:
	HorizonAttributComputeDialog(DataSelectorDialog *dataSelectorDialog, QWidget *parent = 0);
	virtual ~HorizonAttributComputeDialog();
	void trt_threadRun();
	void setProjectManager(GeotimeProjectManagerWidget *p);
	void setProjectManager(ProjectManagerWidget *p);
	void setWorkingSetManager(WorkingSetManager *p);
	void setInputHorizons(QString path, QString name);
	void setTreeUpdate(bool val) { m_treeUpdate = val; }

private:
	DataSelectorDialog *m_dataSelectorDialog = nullptr;
	GeotimeProjectManagerWidget* m_selectorWidget = nullptr;
	ProjectManagerWidget* m_selectorWidget2 = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;
	HorizonAttributSpectrumParam *m_spectrumParam = nullptr;
	HorizonAttributMeanParam *m_meanParam = nullptr;
	HorizonAttributGCCParam *m_gccParam = nullptr;
	CollapsableScrollArea* m_collapseParamSpectrum = nullptr;
	CollapsableScrollArea* m_collapseParamMean = nullptr;
	CollapsableScrollArea* m_collapseParamGcc = nullptr;
	HorizonSelectWidget *m_horizonSelectWidget = nullptr;

	// QComboBox *m_attributType = nullptr;
	QListWidget *m_horizonListWidget = nullptr;
	QListWidget *m_seismicListWidget = nullptr;

	QCheckBox *m_cbSpectrum = nullptr;
	QCheckBox *m_cbMean = nullptr;
	QCheckBox *m_cbGcc = nullptr;

	bool m_treeUpdate = true;

	/*
	QGroupBox *m_spectrumParameters = nullptr;
	QGroupBox *m_gccParameters = nullptr;
	QGroupBox *m_meanParameters = nullptr;
	QSpinBox *m_spectrumWindowSizeSpinBox = nullptr;
	// QGroupBox *m_gccGroupBox = nullptr;
	QSpinBox *m_gccOffsetSpinBox = nullptr;
	QSpinBox *m_wSpinBox = nullptr;
	QSpinBox *m_shiftSpinBox = nullptr;
	QSpinBox *m_meanWindowSizeSpinBox = nullptr;
	*/
	int m_spectrumWindowSize = 64;
	int m_gccOffset = 7;
	int m_w = 7;
	int m_shift = 5;
	int m_meanWindowSize = 7;

	void displaySeismicList();
	void displayHorizonList();

	QProgressBar *m_progressBar = nullptr;
	QPushButton *m_buttonStart = nullptr;
	QPushButton *m_buttonStop = nullptr;

	std::vector<QString> m_horizonNames;
	std::vector<QString> m_horizonPath;

	std::vector<QString> m_seismicNames;
	std::vector<QString> m_seismicPath;

	std::vector<float*> getHorizonData();
	std::vector<std::string> QStringToStdString(std::vector<QString> &in);
	std::vector<std::vector<std::string>> getAttributFilename(QString suffix);
	std::vector<std::vector<std::string>> getAttributSpectrumFilename();
	std::vector<std::vector<std::string>> getAttributGCCFilename();
	std::vector<std::vector<std::string>> getAttributMeanFilename();

	bool checkFileExist(std::vector<std::vector<std::string>>& spectrumFilename,
			std::vector<std::vector<std::string>>& gccFilename,
			std::vector<std::vector<std::string>>& meanFilename);
	bool checkFormat(std::vector<QString> &path);
	// void runSpectrum();
	// void runGCC();
	// void runMean();
	Ihm2 *pIhm2 = nullptr;
	QTimer *timer = nullptr;
	int m_valStartStop = 0;

private slots:
	void trt_seismicAdd();
	void trt_seismicSub();
	void trt_horizonAdd();
	void trt_horizonSub();
	void trt_start();
	void trt_stop();
	void showTime();
	void trt_cbSpectrumChange(int val);
	void trt_cbMeanChange(int val);
	void trt_cbGccChange(int val);
};



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


#endif
