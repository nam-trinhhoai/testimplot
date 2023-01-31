
#ifndef __RGB2TOXTWIDGET__
#define __RGB2TOXTWIDGET__




#include <QWidget>
#include <QGroupBox>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QComboBox>
#include <QTimer>

#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>

class GeotimeProjectManagerWidget;
class SpectrumComputeWidget;
class MyThreadRGB2TOXTWidget;




class Rgb2ToXtWidget : public QWidget{
	Q_OBJECT
public:
	Rgb2ToXtWidget(ProjectManagerWidget *projectManager, QWidget* parent = 0);
	virtual ~Rgb2ToXtWidget();
	QGroupBox *getMainGroupBox();
	void setProjectManagerWidget(ProjectManagerWidget *projectManagerWidget);
	void setSpectrumComputeWidget(SpectrumComputeWidget *spectrumComputeWidget);
	void trt_run();


private:
	QGroupBox *mainGroupBox = nullptr;
	QProgressBar *qpb_progress = nullptr;
	QPushButton *qpbStart = nullptr;
	// GeotimeProjectManagerWidget *m_projectManagerWidget = nullptr;
	SpectrumComputeWidget *m_spectrumComputeWidget = nullptr;

	MyThreadRGB2TOXTWidget *thread = nullptr;
	int globalRun = 0;
	QString xtFilename = "";
	// QString rgb2Name = "";
	// QString rgb2Filename = "";

	QLineEdit *qleRgb2Filename = nullptr;
	QLineEdit *qleXtFilenamePrefix = nullptr;
	QTimer *timer = nullptr;
	ProjectManagerWidget *m_projectManager = nullptr;
	FileSelectWidget *m_rgb2FileSelectWidget = nullptr;;

private slots:
	void trt_launchThread();
	void trt_rgb2FilenameOpen();
	// void trt_rgtFilenameOpen();
	int filenamesUpdate();
	void showTime();
};

class MyThreadRGB2TOXTWidget : public QThread
{
     // Q_OBJECT
	 public:
	MyThreadRGB2TOXTWidget(Rgb2ToXtWidget *p);
	 private:
	Rgb2ToXtWidget *pp;

protected:
     void run();
};





#endif
