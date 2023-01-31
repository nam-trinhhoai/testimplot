/*
 *
 *
 *  Created on:
 *      Author: l1000501
 */

#ifndef __RGTSTACKINGWIDGET_H_
#define __RGTSTACKINGWIDGET_H_

#include <QThread>
#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QComboBox>
#include <QGroupBox>
#include <QDialog>
#include <QString>
#include <QPushButton>


#include <vector>
#include <math.h>

class QTableView;
class QStandardItemModel;
class RgtStackingParametersWidget;
class ProjectManagerWidget;
class FileSelectWidget;
class OrientationWidget;
class GeotimeSystemInfo;
class WorkingSetManager;
class Ihm2;
class ProcessRelay;


class RgtStackingWidget : public QWidget{
	Q_OBJECT
public:
	RgtStackingWidget(WorkingSetManager *workingSetManager, QWidget* parent = 0);
	virtual ~RgtStackingWidget();

	void setSystemInfo(GeotimeSystemInfo *val);

	void setCompute(bool val);
	void setCpuGpu(int val);
	void setRgtSuffix(QString val);
	void setPropagationSeedOnHorizon(bool val);
	void setIter(int val);
	void setDipThreshold(double val);
	void setDecimation(int val);
	void setSnapping(bool val);
	void setEnableSeedMax(bool val);
	void setSeedMax(long val);

	bool getCompute();
	int getCpuGpu();
	QString getRgtSuffix();
	bool getPropagationSeedOnHorizon();
	int getIter();
	double getDipThreshold();
	int getDecimation();
	bool getSnapping();
	bool getEnableSeedMax();
	long getSeedMax();
	void setProcessRelay(ProcessRelay* relay) { m_processRelay = relay; }
	void setWorkingSetManager(WorkingSetManager *p){ m_workingSetManager = p; }


private:
	int i = 0;
	unsigned int m_cptProcessing = 0;
	QLabel *m_processing = nullptr;
	ProjectManagerWidget *m_projectManagerWidget = nullptr;
	QCheckBox *m_compute = nullptr;
	QLineEdit *m_rgtSuffix = nullptr;
	QComboBox *m_cpuGpu = nullptr;
	QCheckBox *m_propagateSeed = nullptr;
	RgtStackingParametersWidget *m_rgtStackParameters = nullptr;
	FileSelectWidget *m_seismicFileSelectWidget = nullptr;
	OrientationWidget *m_orientationWidget = nullptr;
	QPushButton *m_startStop = nullptr,
			*m_save = nullptr,
			*m_kill = nullptr,
			*m_launchBatch = nullptr,
			*m_preview = nullptr;
	QProgressBar *m_progress = nullptr;
	QPlainTextEdit *m_textInfo = nullptr;
	GeotimeSystemInfo *m_systemInfo = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;
	Ihm2 *pIhm = nullptr;
	int pStatus = 0;
	ProcessRelay* m_processRelay = nullptr;

	// todo
	// int traceLimitX1 = -1, traceLimitX2 = -1;

	int checkFieldsForCompute();
	int checkMemoryForCompute();
	double qt_cuda_needed_memory(int *size, int decim, int rgt_format, int nbsurfaces, bool polarity);
	double qt_ram_needed_memory(int nbthreads, int *size, int decim, int sizeof_stack, int nbsurfaces, bool polarity);
	void trt_compute();
	bool dipCompute();
	int rgtStackingCompute();
	void processingDisplay();
	QString getDipxyPath();
	QString getDipxzPath();
	QString getRgtPath();
	void startStopConfigDisplay();
	bool createDataSet(char *src, char *dst);
	void sizeRectifyWithTraceLimits(int *size, int *sizeX);
	bool checkGpuTextureSize();





private slots:
	void trt_launch();
	void showTime();
	void trt_rgtGraph_Kill();
	void trt_rgtSave();
	void seismicFilenameChanged();




	/*
	QGroupBox* groupBox = nullptr;
	QCheckBox *checkBoxCompute;
	QComboBox *comboProcessingType;
	FileSelectWidget *m_dipxyFileSelectWidget = nullptr;
	FileSelectWidget *m_dipxzFileSelectWidget = nullptr;
	ProjectManagerWidget *m_projectManager = nullptr;
	OrientationParametersWidget *m_parameters = nullptr;
	*/

	/*
	private slots:
	void trt_setEnabled(bool val);
	*/

private:
	class MyThread0 : public QThread
	{
	     // Q_OBJECT
		 public:
		 MyThread0(RgtStackingWidget *p);
		 private:
		 RgtStackingWidget *pp;

	protected:
	     void run();
	};

};




#endif
