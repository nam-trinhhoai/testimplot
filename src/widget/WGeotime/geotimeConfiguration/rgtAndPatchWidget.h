/*
 *
 *
 *  Created on:
 *      Author: l1000501
 */

#ifndef __RGTANDPATCHWIDGET_H_
#define __RGTANDPATCHWIDGET_H_

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
#include <QPushButton>
#include <QProgressBar>
#include <QPlainTextEdit>


#include <vector>
#include <math.h>

class QTableView;
class QStandardItemModel;
class RgtStackingParametersWidget;
class PatchWidget;
class RgtPatchWidget;
class ProjectManagerWidget;
class OrientationWidget;
class FileSelectWidget;
class Ihm2;
class GeotimeSystemInfo;
class ProcessRelay;




class RgtAndPatchWidget : public QWidget{
	Q_OBJECT

private:
	class ReturnParam
	{
	public:
		double timeOrientationAll = 0.0;
		double timeOrientationRead = 0.0;
		double timeOrientationNormal = 0.0;
		double timeOrientationWrite = 0.0;

		double timePatchAll = 0.0;
		double timePatchReadData = 0.0;
		double timePatchPatch = 0.0;
		double timePatchFusion = 0.0;

		double timeRgtAll = 0.0;
	};
public:
	RgtAndPatchWidget(WorkingSetManager *workingSetManager, QWidget* parent = 0);
	virtual ~RgtAndPatchWidget();
	void setSystemInfo(GeotimeSystemInfo *val);

	void setPatchCompute(bool val);
	void setPatchSuffix(QString val);
	void setPatchEnableFaultInput(bool val);
	void setPatchSize(int val);
	void setPatchPolarity(int val);
	void setPatchGradMax(int val);
	void setPatchDvOverV(double val);
	void setPatchRatio(int val);
	void setPatchMaskThreshold(int val);

	bool getPatchCompute();
	QString getPatchSuffix();
	bool getPatchEnableFaultInput();
	int getPatchSize();
	QString getPatchPolarity();
	int getPatchGradMax();
	double getPatchDvOverV();
	int getPatchRatio();
	int getPatchMaskThreshold();
	std::vector<QString> getHorizonPaths();
	QString getRgtPath();

	bool getScaleInitValid();
	int getScaleInitIter();
	double getScaleInitEpsilon();
	int getScaleInitDecim();
	int getIter();
	double getEpsilon();
	int getDecim();

	void setProcessRelay(ProcessRelay* relay);



private:
	int i = 0;
	ProcessRelay *m_processRelay = nullptr;

	unsigned int m_cptProcessing = 0;
	QLabel *m_processing = nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;
	ProjectManagerWidget *m_projectManager = nullptr;
	OrientationWidget *m_orientationWidget = nullptr;
	PatchWidget *m_patchWidget = nullptr;
	RgtPatchWidget *m_rgtPatchWidget = nullptr;
	FileSelectWidget *m_seismicFileSelectWidget = nullptr;

	QPushButton *m_startStop = nullptr,
			*m_save = nullptr,
			*m_scaleStop = nullptr,
			*m_kill = nullptr,
			*m_launchBatch = nullptr,
			*m_preview = nullptr;
	QProgressBar *m_progress = nullptr;
	QPlainTextEdit *m_textInfo = nullptr;
	GeotimeSystemInfo *m_systemInfo = nullptr;


	//
	Ihm2 *pIhm = nullptr;
	int pStatus = 0;

	QString getDipxyPath();
	QString getDipxzPath();


	void trt_compute();
	int dipCompute();
	int patchCompute();
	int rgtPatchCompute();

	QString getPatchPath();
	QString getGraphPath();
	QString getGraphDir();
	QString getPatchDir();

	bool getFaultMaskInput();
	QString getFaultMaskPath();

	int checkFieldsForCompute();
	bool checkMemoryForCompute();
	bool displayWarning(QString msg);
	bool datasetValid(QString path1);
	bool fitDatasetSize(QString path1, QString path2);
	int rgtVolumicDecimationFactorEstimation();
	long getVertexnbreEstimation();
	void processingDisplay();
	void displayProcessFinish();
	QString timeMSToString(double val);
	QString getDatasetFormat(QString filename);




	ReturnParam m_returnParam;

private slots:
	void trt_launch();
	void trt_scaleStop();
	void trt_save();
	void trt_kill();
	void showTime();
	void trt_launchBatch();
	void seismicFilenameChanged();



private:
	class MyThread0 : public QThread
	{
		// Q_OBJECT
	public:
		MyThread0(RgtAndPatchWidget *p);
	private:
		RgtAndPatchWidget *pp;

	protected:
		void run();
	};
};




#endif
