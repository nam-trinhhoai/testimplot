/*
 *
 *
 *  Created on: 11 September 2020
 *      Author: l1000501
 */

#ifndef MURATAPP_SRC_TOOLS_XCOM_SPECTRUMCOMPUTEWIDGET_H_
#define MURATAPP_SRC_TOOLS_XCOM_SPECTRUMCOMPUTEWIDGET_H_

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


#include <vector>
#include <math.h>
#include <memory>

#include "GeotimeSystemInfo.h"
#include <GeotimeConfiguratorWidget.h>
// #include <MeanSeismicSpectrumWidget.h>
#include <fileSelectWidget.h>


class QTableView;
class QStandardItemModel;
class QDoubleSpinBox;
class QSpinBox;
class QCheckBox;
class ProcessWatcherWidget;
class MeanSeismicSpectrumWidget;
class RgtToGCC16Widget;
class Rgb2ToXtWidget;
class ProjectManagerWidget;
class FileSelectWidget;

#ifndef MIN
    #define MIN(x,y)		( ( x >= y ) ? y : x )
#endif

#ifndef MAX
    #define MAX(x,y)		( ( x >= y ) ? x : y )
#endif

/*
class DipSmoothingWidgetOpenDialog : public QDialog
{
	Q_OBJECT
public:
	DipSmoothingWidgetOpenDialog(QString path, char *out, QWidget *parent=0);
	DipSmoothingWidgetOpenDialog(std::vector<QString> list, char *out, QWidget *parent);
	DipSmoothingWidgetOpenDialog(QString title, QString msg, char *out, QWidget *parent);

private:
	QListWidget *textInfo;
	QPushButton *qpb_cancel, *qpb_ok;
	char *pout;

private slots:
	void accept();
	void cancel();
	void listviewdoubleclick(QListWidgetItem *item);
};
*/


class MyThreadSpectrumCompute;

class SpectrumComputeWidget : public QWidget{
	Q_OBJECT
public:
	SpectrumComputeWidget(QWidget* parent = 0);
	virtual ~SpectrumComputeWidget();
	// void trt_start();
	int GLOBAL_RUN_TYPE, GLOBAL_RUNRGB2TORGB1_TYPE, GLOBAL_RUNRGB1TOXT_TYPE;
	int m_functionType;

	static std::size_t getFileSize(const char* filePath);

	static bool checkSeismicsSizeMatch(const QString& cube1, const QString& cube2);

	static QString getFFMPEGTime(double);
	static QString formatTimeWithMinCharacters(double time, int minCharNumber=2);
	static QString formatColorForFFMPEG(const QColor& color);

private:
	// GeotimeProjectManagerWidget *m_selectorWidget;
	ProjectManagerWidget *m_projectManager = nullptr;
	MeanSeismicSpectrumWidget *m_meanSeismicSpectrumWidget;
	RgtToGCC16Widget *m_rgtToGCC16Widget = nullptr;
	Rgb2ToXtWidget *m_rgb2ToXtWidget;
	QTabWidget *tabwidget_table1;

	QLabel *label_rgb2Name, *label_rgb1Name, *label_isostep, *label_layerNumber ;
	QLineEdit *lineedit_seismicFilename, *lineedit_rgtFilename, *lineedit_f1, *lineedit_f2, *lineedit_f3, *lineedit_wsize, *lineedit_isostep,
	*lineedit_rgb2Filename, *lineedit_alpha, *lineedit_ratio, *lineedit_rgb1Filename, *lineedit_rgb2prefix, *lineedit_rgb1prefix,
	*lineedit_fmin, *lineedit_fmax, *linedit_stepf, *lineedit_rawtoaviFilename, *lineedit_rgb2toaviFilename, *lineedit_aviPrefix,
	*lineedit_layer_number,
	*lineedit_rgb1toxtFilename, *lineedit_xtPrefix, *lineedit_rgb1toxtRGB2Filename, *lineedit_aviviewAviFilename,
	*le_SesismicMeanRgtFilename, *le_outMeanPrefix, *le_outMeanWindowSize, *le_outMeanIsoStep;
	QProgressBar *qpb_progress, *qpb_progress_rgb2torgb1, *qpb_progress_rgb1toxt,
	*qpb_seismicMean;
	QPushButton *pushbutton_startstop, *pushbutton_rgtb2torgb1StartStop, *qpb_rgb1toxtStart, *qbp_seismicMeanStart;
	QTableWidget *tableWidget_freq;
	QListWidget *listwidget_horizons;
	QComboBox *combobox_horizonchoice;
	QWidget* horizonWidget;
	QComboBox* directionComboBox;
	QSlider* sectionPositionSlider;
	QSpinBox* sectionPositionSpinBox;
	QPushButton* colorHolder;
	ProcessWatcherWidget* processwatcher_rawtoaviprocess;

	QString // seismicTinyName, seismicFullName,
	// rgtTinyName, rgtFullName,
	rgb2TinyName, rgb2FullName,
	isoTinyName, isoFullName,
	rgb1TinyName, rgb1FullName,
	// rgb2_2_TinyName, rgb2_2_FullName, // for rgb2torgb1 process
	// rgb1_2_TinyName, rgb1_2_FullName, // for rgb1toavi
	// rgb1_3_TinyName, rgb1_3_Fullname, // for rgb1toxt
	// rgb2_3_TinyName, rgb2_3_Fullname, // for rgb1toxt
	aviTinyName, aviFullName,
	// avi_2_TinyName, avi_2_FullName, // for aviview rgb1
	rgb2_Avi_TinyName, rgb2_Avi_FullName; // for aviview rgb2
	std::vector<QString> horizonTinyName, horizonFullname;
	bool startStop;
	MyThreadSpectrumCompute *thread;
	QTimer *timer;
	QDoubleSpinBox* spinboxvideoScale;
	QSpinBox* spinboxRgb1TOrigin, *spinboxRgb1TStep, *spinboxFps, *spinboxFirstIso, *spinboxLastIso;
	QCheckBox* checkboxRgb1IsReversed;

	double videoScale = 1.0;

	// depend of rgb2 generation process, current defaults are :
	int rgb1TOrigin = 0;
	int rgb1TStep = 25;
	int framePerSecond = 25;
	int firstIso = 0;
	int lastIso = 31975;
	int textSize = 24;
	QColor textColor = Qt::white;

	bool rgb1IsReversed = true;

	std::unique_ptr<QMetaObject::Connection> m_processwatcher_connection;

	/*
	GeotimeSystemInfo *systemInfo;
	QLineEdit *qlineedit_datatin, *qlineedit_dipxz, *qlineedit_dipxy, *qlineedit_outfilename,
	*qlineedit_dx, *qlineedit_dy, *qlineedit_dz;
	QComboBox *qcombobox_type;
	QProgressBar *qpb_progress;
	QString seismic_name, seismic_filename, dipx_name, dipx_filename, dipz_name, dipz_filename;
	int getIndexFromVectorString(std::vector<QString> list, QString txt);
	GeotimeProjectManagerWidget *m_selectorWidget;
	void trt_open_file(std::vector<QString> name_list, char *filename_out);
	void get_size_from_filename(QString filename, int *size);
	*/
	// int parameters_check();

	QGroupBox *initMeanSeismicSpectrumGroupBox();

	std::tuple<double, bool, double> getAngleFromFilename(QString seismicFullName);
	bool getPropertiesFromDatasetPath(QString filename, int* size, double* steps, double* origins=nullptr);
	QSizeF newSizeFromSizeAndAngle(QSizeF oriSize, double angle);
	int getIndexFromVectorString(std::vector<QString> list, QString txt);
	std::pair<QString, QString> getPadCropFFMPEG(double wRatio, double hRatio);
	void trt_open_file(std::vector<QString> name_list, char *filename_out, bool multiselection);
	QString filenameToPath(QString fullName);
	void rgb2FilenameUpdate(int depth);
	void rgb2LabelUpdate(QString name);
	void rgb1FilenameUpdate();
	void rgb1LabelUpdate(QString name);
	void frequencyTableWidgetRead(int *arrayFreq, int *arrayIso, int *count);
	void aviFilenameUpdate();
	void DisplayHorizonType();
	void rawToAviRunFfmeg(bool onlyFirstImage);
	void setAviTextColor(const QColor&);

	void closeEvent (QCloseEvent *event);

	FileSelectWidget *m_seismicFileSelectWidget = nullptr;
	FileSelectWidget *m_rgtFileSelectWidget = nullptr;
	FileSelectWidget *m_rb2FileSelectWidget = nullptr;

	FileSelectWidget *m_rawToAviRgb1FileSelectWidget = nullptr;
	FileSelectWidget *m_rawToAviRgb2FileSelectWidget = nullptr;

	FileSelectWidget *m_rgb8BitsXtRgb1FileSelectWidget = nullptr;
	FileSelectWidget *m_rgb8BitsXtRgb2FileSelectWidget = nullptr;

	FileSelectWidget *m_aviViewFileSelectWidget = nullptr;

	FileSelectWidget *m_seismicMeanRgtFileSelectWidget = nullptr;



public:
	GeotimeSystemInfo *systemInfo;
	QString getSeismicTinyName();
	QString getSeismicFullName();
	QString getRgtTinyName();
	QString getRgtFullName();
	void getTraceParameter(char *filename, float *pasech, float *tdeb);
	int getSizeFromFilename(QString filename, int *size);
	void trt_rgt2rgbStartStop();
	void trt_rgb2rgb1StartStop();
	void trt_rgb1toxtStartStop();
	void trt_seismicMeanStartStop();
	void horizonRead(std::vector<QString> horizonTinyName, std::vector<QString> horizonFullname,
			int dimx, int dimy, int dimz,
			float pasech, float tdeb,
			float **horizon1, float **horizon2);


private slots:
	// void trt_dipx_open();
	// void trt_dipz_open();
	void trt_horizonchoiceclick(int idx);
	void trt_seismic_open();
	void trt_rgt_open();
	void trt_launch_thread();
	void showTime();
	void trt_rgb2_open();
	void trt_launch_rgb2torgb1_thread();
	void trt_loadSession();
	void trt_saveSession();
	void trt_rawToAviRun();
	void trt_rawToAviDisplay();
	void trt_aviviewRun();
	void trt_raw1Open();
	void trt_rgb2AviOpen();
	void trt_horizon_add();
	void trt_horizon_sub();
	void trt_rgb1toxt_open();
	void trt_rgb1toxtRGB2_open();
	void trt_launch_rgb1toxt_thread();
	void trt_aviview_open();
	void trt_videoScaleChanged(double);
	void trt_rgb1TOriginChanged(int);
	void trt_rgb1TStepChanged(int);
	void trt_rgb1IsReversedChanged(int);
	void trt_fpsChanged(int);
	void trt_firstIsoChanged(int);
	void trt_lastIsoChanged(int);
	void trt_textSizeChanged(int val);
	void trt_changeAviTextColor();
	void updateRGBD();
	void computeoptimalscale_rawToAvi();
	void trt_rgtMeanSeismicOpen();
	void trt_lauchRgtMeanSeismicStart();
	void trt_directionChanged(int newComboBoxIndex);
	void trt_sectionIndexChanged(int sectionIndex);

	// void trt_launch_thread();
	// void trt_stop();
	// void showTime();
	// void trt_session_load();
	// void trt_session_save();

};

class MyThreadSpectrumCompute : public QThread
{
     // Q_OBJECT
	 public:
	MyThreadSpectrumCompute(SpectrumComputeWidget *p);
	 private:
	SpectrumComputeWidget *pp;

protected:
     void run();
};

/*
class MyThreadDipSmoothing : public QThread
{
     // Q_OBJECT
	 public:
	MyThreadDipSmoothing(DipSmoothingWidget *p);
	 private:
	DipSmoothingWidget *pp;

protected:
     void run();
};
*/


class DialogSp : public QDialog
{
	Q_OBJECT
public:
    DialogSp(QString path, char *out, QWidget *parent=0);
	DialogSp(std::vector<QString> list, char *out, QWidget *parent);
	DialogSp(QString title, QString msg, char *out, QWidget *parent);
	DialogSp(std::vector<QString> list, QWidget *parent);
	void setMultiselection();


private:
	QListWidget *textInfo;
	QPushButton *qpb_cancel, *qpb_ok;
	char *pout;
	SpectrumComputeWidget *pparent;
	QFileInfoList get_dirlist(QString path);

private slots:
	void accept();
	void cancel();
	void listviewdoubleclick(QListWidgetItem *item);
};




#endif /* MURATAPP_SRC_TOOLS_XCOM_MARFACOMPUTATIONWIDGET_H_ */
