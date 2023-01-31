

#ifndef __VIDEOCREATEWIDGET__
#define __VIDEOCREATEWIDGET__


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
#include <QSizeF>
#include <QProcess>
#include <QHBoxLayout>
#include <QListWidget>


#include <vector>
#include <math.h>
#include <QTemporaryFile>

//#include <ProjectManagerWidget.h>
//#include <fileSelectWidget.h>
//#include <rgtToRgb16bitsWidget.h>
//#include <rgtToSeismicMeanWidget.h>
//#include <rgtToGccWidget.h>
//#include <rawToAviWidget.h>
//#include <aviViewWidget.h>
//#include <rgb16ToRgb8Widget.h>
//#include <GeotimeSystemInfo.h>
//// #include <GeotimeConfiguratorWidget.h>

// class AttributToXtWidget;
#include <sliceutils.h>
class ProjectManagerWidget;
class FileSelectWidget;
class CollapsableScrollArea;
class AviParamWidget;
class Ihm2;
class WorkingSetManager;

#include "processwatcherwidget.h"


class VideoCreateWidget : public QWidget{
	Q_OBJECT
public:
	// enum DATA_OUT_FORMAT { CUBE3D, SINGLE_DATA };
	VideoCreateWidget(ProjectManagerWidget *projectManager);
	virtual ~VideoCreateWidget();
	QString getSeismicPath();
	bool getPropertiesFromDatasetPath(QString filename, int* size, double* steps, double* origins = nullptr);
	void trt_run();


private:
	class PARAM
	{
	public:
		QString outSectionVideoRawFile = "";

		QString seismicName = "";
		QString seismicPath = "";
		QString rgtName = "";
		QString rgtPath = "";
		QString ImportExportPath = "";

		QString IJKPath = "";
		QString	horizonPath = "";
		QString isoValPath = "";
		QString rgtPath0 = "";
		QString rgb2tmpfilename = "";
		QString outMainDirectory = "";
		QString	attribut = "";
		QString attributDirectory = "";

		QString rgb1TmpFilename = "";
		QString dataType = "";
		QString attributFilename = "";

		QString prefix = "";
		int spectrumWindowSize = 64;
		int meanWindowSize = 7;
		int gccWindowSize = 7;

		QString jpgTmpPath;
		int isoStart = 0;
		int isoEnd = 32000-25;
		int isoStep = 25;
		int axisValue = 0;
		int axisValueIndex = 0;

		QString sectionName;
		SliceDirection direction;
	};

	PARAM m_param;
	WorkingSetManager *m_workingSetManager = nullptr;
	ProjectManagerWidget *m_projectmanager = nullptr;
	FileSelectWidget *m_seismicFileSelectWidget = nullptr;;
	FileSelectWidget *m_rgtFileSelectWidget = nullptr;
	AviParamWidget *m_aviParam = nullptr;

	QLabel *labelSeismic = nullptr;
	QLabel *labelIso = nullptr;

	QComboBox *m_attributType = nullptr;
	QComboBox *m_processingType = nullptr;

	QHBoxLayout * m_qhbSeismic = nullptr;

	CollapsableScrollArea* m_spectrumCollapseParam = nullptr;
	CollapsableScrollArea* m_gccCollapseParam = nullptr;
	CollapsableScrollArea* m_meanCollapseParam = nullptr;
	CollapsableScrollArea* m_collapseAviParam = nullptr;

	QLineEdit *m_seismicLineEdit = nullptr;
	QLineEdit *m_isoLineEdit = nullptr;
	QLineEdit *m_preffix = nullptr;
	QLineEdit *m_isoStep = nullptr;
	QLineEdit *m_spectrumWindowSize = nullptr;
	QLineEdit *m_meanWindowSize = nullptr;
	QLineEdit *m_gccWindowSize = nullptr;
	QLineEdit *m_gccW = nullptr;
	QLineEdit *m_gccShift = nullptr;

	QPushButton *m_buttonSeismicOpen = nullptr;
	QPushButton *m_buttonIsoOpen = nullptr;

	QProgressBar *m_progress = nullptr;
	QPlainTextEdit *m_textInfo = nullptr;
	QPushButton *m_start = nullptr;
	QPushButton *m_kill = nullptr;
	QLabel *m_processing = nullptr;

	QListWidget *m_attributList = nullptr;

	std::unique_ptr<QMetaObject::Connection> m_processwatcher_connection;
	ProcessWatcherWidget *processwatcher_rawtoaviprocess = nullptr;


	std::tuple<double, bool, double> getAngleFromFilename(QString seismicFullName);
	std::pair<QString, QString> getPadCropFFMPEG(double wRatio, double hRatio);
	void rawToAviRunFfmeg(bool onlyFirstImage);
	bool rawToAviRunFfmegPreprocessing(bool onlyFirstImage);

	void computeoptimalscale_rawToAvi();
	QSizeF newSizeFromSizeAndAngle(QSizeF oriSize, double angle);
	std::vector<std::string> getIsoPath(QString path);
	QString getFFMPEGTime(double _time);
	QString formatColorForFFMPEG(const QColor& color);
	QString formatTimeWithMinCharacters(double time, int minCharNumber=2);
	void processingDisplay();
	void processReset();
	Ihm2 *pIhm2 = nullptr;
	int pStatus = 0;
	std::unique_ptr<QProcess> m_process;
	unsigned int m_cptProcessing = 0;
	std::vector<QString> m_vTmpPath;
	void eraseTmpFiles();
	QStringList m_options;

	QString m_seismicName = "";
	QString m_seismicPath = "";

	QString m_isoPath = "";
	QString m_isoName = "";

	QString getAviPath();
	bool paramInit();
	bool sectionVideoCreate();
	void ffmpegProcessRun();
	QString getSeismicNameFromIsoPath(QString path0);
	QString getSeimsicNameFromAttributName(QString name0, QString attribut, int w);
	QString getAttributNameFromIsoPath(QString path0);

	std::vector<QString> getAttributNames(QString path0);
	void updateAttributList();
	void setWindowTitle0();
	bool attributAndIsochroneExist(std::vector<std::string> &attributpath, std::vector<std::string> &isochronepath);

	const QString m_comboMenuNew = "new rgt isovalues";
	const QString m_comboMenuExist = "existing rgt isovalues";


private slots:
	void trt_attributTypeChange();
	void trt_launch();
	void trt_rgt_Kill();
	void showTime();
	void processFinished(int exitCode, QProcess::ExitStatus exitStatus);
	void errorOccured(QProcess::ProcessError error);
	void readyRead();
	void trt_seismicOpen();
	void trt_processingTypeChange();
	void trt_isoOpen();
	void projectChanged();
	void surveyChanged();
	void filenameChanged();




	// QString getSeismicName();
	// QString getSeismicPath();
	// GeotimeSystemInfo *m_systemInfo = nullptr;
	// int getDataOutFormat();

//private:
//	ProjectManagerWidget *m_selectorWidget;
//	FileSelectWidget *m_seismicFileSelectWidget;
//	RgtToRgb16bitsWidget *m_rgtToRgb16bitsWidget;
//	RgtToSeismicMeanWidget *m_rgtToSeismicMeanWidget;
//	RgtToGccWidget *m_rgtToGccWidget;
//	AttributToXtWidget *m_attributToXtWidget;
//	RawToAviWidget *m_rawToAviWidget = nullptr;
//	AviViewWidget *m_aviViewWidget = nullptr;
//	Rgb16ToRgb8Widget *m_rgb16ToRgb8Widget = nullptr;
//	int dataOutFormat = 1;


};

class VideoCreateWidget_Thread : public QThread
{
	// Q_OBJECT
public:
	VideoCreateWidget_Thread(VideoCreateWidget *p);
private:
	VideoCreateWidget *pp;

protected:
	void run();
};


#endif




