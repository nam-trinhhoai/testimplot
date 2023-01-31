#ifndef __MEANSEISMICSPECTRUMWIDGET__H_
#define __MEANSEISMICSPECTRUMWIDGET__H_

#include <QGroupBox>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QTimer>
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>


class SpectrumComputeWidget;
class MyThreadMeanSeismicSpectrumCompute;

// class FileSelectWidget;



class MeanSeismicSpectrumWidget:public QWidget{
	Q_OBJECT

public:
		MeanSeismicSpectrumWidget(ProjectManagerWidget *projectManagerWidget, QWidget* parent=0);
		virtual ~MeanSeismicSpectrumWidget();
		QGroupBox *getGroupBox();
		// void setParent(SpectrumComputeWidget *spectrumComputeWidget);
		void setProjectManagerWidget(ProjectManagerWidget *projectManagerWidget);
		void setSpectrumComputeWidget(SpectrumComputeWidget *spectrumComputeWidget);
		QString getOutDataFilename(QString mainPath, QString seismicTinyName, int windowSize, int width, int height, int depth);
		QString getMainPath(QString seismicTinyName);
		void trt_StartStop();


private:
		SpectrumComputeWidget *m_spectrumComputeWidget = nullptr;
		ProjectManagerWidget *m_projectManager = nullptr;
		std::vector<QString> horizonTinyName, horizonFullname;
		QComboBox *cbHorizonChoice;
		QGroupBox *mainGroupBox;
		QLineEdit *le_SeismicMeanRgtFilename, *le_outMeanPrefix, *le_outMeanWindowSize, *le_outMeanIsoStep, *le_outMeanStepNbre;
		QProgressBar *qpb_seismicMean;
		QPushButton *qbp_seismicMeanStart;
		// QString rgtTinyName = "";
		// QString rgtFullName = "";
		QTimer *timer = nullptr;
		QLabel *isoStepLabel, *labelLayerNbre;
		QWidget *horizonWidget;
		QListWidget *lwHorizons;
		int GLOBAL_RUN_TYPE = 0;
		void initIhm();
		MyThreadMeanSeismicSpectrumCompute *thread;
		void DisplayHorizonType();

		FileSelectWidget *m_seismicMeanRgtFileSelectWidget = nullptr;


private slots:
		void trt_rgtMeanSeismicOpen();
		void trt_lauchRgtMeanSeismicStart();
		void showTime();
		void trt_horizonchoiceclick(int idx);
		void trt_horizonAdd();
		void trt_horizonSub();

};

class MyThreadMeanSeismicSpectrumCompute : public QThread
{
     // Q_OBJECT
	 public:
	MyThreadMeanSeismicSpectrumCompute(MeanSeismicSpectrumWidget *p);
	 private:
	MeanSeismicSpectrumWidget *pp;

protected:
     void run();
};






#endif
