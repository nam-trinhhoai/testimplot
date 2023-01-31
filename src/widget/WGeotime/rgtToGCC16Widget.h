
#ifndef __RGTTOGCC16WIDGET__
#define __RGTTOGCC16WIDGET__




#include <QWidget>
#include <QGroupBox>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QComboBox>
#include <QTimer>

class GeotimeProjectManagerWidget;
class SpectrumComputeWidget;
class MyThreadRgtToGCC16Widget;
class ProjectManagerWidget;




class RgtToGCC16Widget : public QWidget{
	Q_OBJECT
public:
	RgtToGCC16Widget(ProjectManagerWidget *projectManager, QWidget* parent = 0);
	virtual ~RgtToGCC16Widget();
	QGroupBox *getMainGroupBox();
	void setProjectManagerWidget(ProjectManagerWidget *projectManagerWidget);
	void setSpectrumComputeWidget(SpectrumComputeWidget *spectrumComputeWidget);
	void trt_run();


private:
	QGroupBox *mainGroupBox = nullptr;
	QLineEdit *qleRgtFilename = nullptr;
	QLineEdit *qleGccFilenamePrefix = nullptr;
	QLineEdit *qleWindowSize = nullptr;
	QLineEdit *qleIsoMin = nullptr;
	QLineEdit *qleIsoMax = nullptr;
	QLineEdit *qleIsoStep = nullptr;
	QLineEdit *qleW = nullptr;
	QLineEdit *qleShift = nullptr;
	QString gccFilename = "";
	QTimer *timer = nullptr;
	QComboBox *qcbRgtHorizonChoice = nullptr;

	QProgressBar *qpb_progress = nullptr;
	QPushButton *qpbStart = nullptr;
	ProjectManagerWidget *m_projectManager = nullptr;
	SpectrumComputeWidget *m_spectrumComputeWidget = nullptr;
	int filenamesUpdate();
	int getSizeFromFilename(QString filename, int *size);
	MyThreadRgtToGCC16Widget *thread = nullptr;
	int globalRun = 0;
	// QString rgtTinyName = "";
	// QString rgtFullName = "";
	FileSelectWidget *m_rgtFileSelectWidget = nullptr;

private slots:
	void trt_launchThread();
	void trt_rgtFilenameOpen();
	void showTime();
};

class MyThreadRgtToGCC16Widget : public QThread
{
     // Q_OBJECT
	 public:
	MyThreadRgtToGCC16Widget(RgtToGCC16Widget *p);
	 private:
	RgtToGCC16Widget *pp;

protected:
     void run();
};





#endif
