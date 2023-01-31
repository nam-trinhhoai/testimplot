/*
 *
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */

#ifndef MURATAPP_SRC_TOOLS_XCOM_GEOTIMECONFIGURATIONWIDGET_H_
#define MURATAPP_SRC_TOOLS_XCOM_GEOTIMECONFIGURATIONWIDGET_H_

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

#include "GeotimeSystemInfo.h"
#include "GeotimeProjectManagerWidget.h"
#include "ProjectManagerWidget.h"
#include <ihm2.h>

#include <vector>
#include <math.h>

class QTableView;
class QStandardItemModel;
class FaultDetectionWidget;
class FileSelectWidget;
class OrientationWidget;
class HorizonSelectWidget;
class DipFilterWidget;

/*
#ifndef MIN
    #define MIN(x,y)		( ( x >= y ) ? y : x )
#endif

#ifndef MAX
    #define MAX(x,y)		( ( x >= y ) ? x : y )
#endif
*/

class GeotimeConfigurationWidget;
class RgtVolumicGraphicOut;
class ProcessRelay;

class PatchParametersWidget;
class RgtPatchParametersWidget;
class RgtStackingParametersWidget;
class RgtStackingWidget;
class RgtAndPatchWidget;
class WorkingSetManager;
class HorizonAttributComputeDialog;
template <typename T> class RgtVolumicCPU;

// class GeotimeConfigurationWidget;


class RgtVolumicScaleParam
{
public:
	int nbIter = 50;
	double epsilon = 0.05;
	int decimY = 1;
	int decimZ = 1;
	RgtVolumicScaleParam(int _nbIter, double _epsilon, int _decim)
	{
		this->nbIter = _nbIter;
		this->epsilon = _epsilon;
		this->decimY = _decim;
		this->decimZ = _decim;
	}
};

class RgtVolumicParam
{
public:
	// int nbIter = 50;
	// double epsilon = 0.05;
	// int decimY = 1;
	// int decimZ = 1;

	RgtVolumicScaleParam initScaleParam = RgtVolumicScaleParam(200, 0.5, 10);
	RgtVolumicScaleParam scaleParam = RgtVolumicScaleParam(10, 0.5, 2);

	bool initScaleEnable = true;
	short idleDipMax = 20000;
	// QString rgt0Name = "";
	// QString rgt0Filename = "";
	// QString patchFilename = "";
	// QString patchName = "";
};

class PatchParam
{
public:
	int patchSize = 16;
	int patchFitThreshold = 90;
	int patchFaultMaskThreshold = 10000;
	std::string patchPolarity = "positive"; // positive; negative; both
	double deltaVoverV = .3;
	int patchGradMax = 3.0;
};




class GeotimeConfigurationWidget : public QWidget{
	Q_OBJECT
public:
	GeotimeConfigurationWidget(ProjectManagerWidget *_projectmanager, QWidget* parent = 0);
	virtual ~GeotimeConfigurationWidget();
	void setProcessRelay(ProcessRelay *relay);
	int decimation_factor, stack_format, nbiter, partial_rgt_save, rgt_saverate, output_format, nb_threads;
	int seed_threshold, seed_threshold_valid;
	int traceLimitX1 = -1, traceLimitX2 = -1;
	double dip_threshold, sigma_stack, sigmagradient, sigmatensor, rgt_compresserror;
	bool bool_snapping;
	void trt_compute();
	void trt_start_rgtPatch();
	std::vector<QString> listQStringDialog;
	static QString paramColorStyle;
	void setSystemInfo(GeotimeSystemInfo *m_systemInfo);

public:
	QComboBox *qcb_rgtcpugpu;
	GeotimeSystemInfo *systemInfo;
	GeotimeSystemInfo *m_systemInfo = nullptr;
	// QString seismic_filename;
	RgtVolumicParam rgtVolumicParam;
	PatchParam patchParam;
	QString getSeismicName();
	QString getSeismicPath();

private:
	WorkingSetManager *m_workingSetManager = nullptr;
	ProcessRelay *m_processRelay = nullptr;
	QGroupBox* qgb_seismic, *qgb_orientation, *qgb_stackrgt;
	QLineEdit *lineedit_nbthreads, *lineedit_sigmagradient, *lineedit_sigmatensor, *lineedit_rgtsuffix,
		*lineedit_horizont0, *lineedit_horizont1,
		*lineedit_patchRgtRgtName,
		*lineedit_rgtVolumicRgt0;

	QLabel *label_dimensions, *label_cpumem, *label_gpumem;
	QCheckBox *qcb_rgtcwt, *qcb_snapping,
	*cb_rgtformat, *qCheckBoxSeedsInsideHorizons,
	*qcb_rgtPatchMaskEnable, *qcb_patchCompute, *qcb_patchRgtCompute, *qcb_patchRgtSeismicWeight, *qcb_stackingRgtEnable,
	*qcb_rgtVolumicRgt0;
	QComboBox *qcb_rgt_format;
	QPlainTextEdit *textInfo, *rgtPatchTextInfo;
	ProjectManagerWidget *projectmanager;
	QProgressBar *qpb_progress, *qpbRgtPatchProgress;
	QPushButton *pushbutton_compute, *pushbutton_abort, *pushbutton_expert, *pushbutton_rgtpartialsave,
	*pushbutton_rgtPatchMaskFilename, *pushbutton_rgtPatchParamExpert,
	*qpbRgtPatchStart, *qpbRgtPatchStop, *qpbRgtPatchSave, *qpbRgtPatchKill;

	// QListWidget *listwidget_horizons;
	// double gpu_total_mem, gpu_free_mem, cpu_total_mem, cpu_free_mem;
	int bool_abort, cuda_nb_devices;
	QString patchConstraintsFilename; //, fault_name = "", fault_filename = "";
	char dipxyRecomposeFilename[10000], dipxzRecomposeFilename[10000];
	char rgtRecomposeFilename[10000], rgtRecomposeFilename1[10000];

	QString current_path, config_filename, QFileDialogTempPath;
	FaultDetectionWidget *faultDetectionWidget = nullptr;;
	DipFilterWidget *m_dipFilterWidget = nullptr;

	RgtVolumicGraphicOut *rgtVolumicGraphicOut = nullptr;
	PatchParametersWidget *m_patchParameters = nullptr;
	RgtPatchParametersWidget *m_rgtPatchParameters = nullptr;
	RgtStackingParametersWidget *m_rgtStackParameters = nullptr;
	RgtStackingWidget *m_rgtStackingWidget = nullptr;
	RgtAndPatchWidget *m_rgtAndPatch = nullptr;
	HorizonAttributComputeDialog *m_horizonAttribut = nullptr;
	QString data_path_read();
	// deleted
	// QString horizon_path_read();
	void data_path_write(QString filename);
	QString filename_to_path_create(QString filename);
	QString filename_format_create(QString base, QString separator, QString suffix);
	void get_size_from_filename(QString filename, int *size);
	void xt_header_copy(char *src_filename, char *dst_filename);
	int getIndexFromVectorString(std::vector<QString> list, QString txt);
	void sizeRectifyWithTraceLimits(int *size, int *sizeX);


	// double qt_ram_free_memory();
	// double qt_ram_total_memory();
	// double qt_cuda_total_memory();
	// double qt_cuda_free_memory();
	double qt_ram_needed_memory(int nbthreads, int *size, int decim, int sizeof_stack, int nbsurfaces, bool polarity);
	double qt_cuda_needed_memory(int *size, int decim, int rgt_format, int nbsurfaces, bool polarity);

	// void qt_endian_short_swap(short *data, long size);
	// void dip_read_decim(char *filename, int *size_in, int *size_out, int decim, short *data);
	// size_t c_btesti_NK(char *a, size_t k);
	// void c_bset_NK(char *a,size_t k);
	// char *polarity_data_create(char *seismic_filename, int decim, int *polarity_size);

	void window_enable(bool val);
	void field_fill();
	void update_label_size(int size[3]);

	int check_fields_for_compute();
	int check_memory_for_compute();

	void msgDisplay(char *txt);
	// void trt_open_file(std::vector<QString> name_list, char *filename_out,  bool multiselection);
	// void trt_rgtMethodeEnable(int idx);
	int checkGpuTextureSize();

	// patch
	QString patchMainDirectoryGet();
	QString patchDirectoryGet();
	QString graphFilenameGet();
	QString graphlabel0FilenameGet();
	QString graphlabel1FilenameGet();
	QString graphLabelRawFilenameGet();
	QString patchMainDirectoryFromPatchName(QString patchName);


	bool dipFilenameUpdate();
	void dipCompute();
	void rgtStackingCompute();
	void rgtVolumicRun();
	bool checkRgtGraphLaunch();
	int rgtVolumicDecimationFactorEstimation();

	// new
	FileSelectWidget *m_seismicFileSelectWidget = nullptr;
	FileSelectWidget *m_patchFileSelectWidget = nullptr;
	OrientationWidget *m_orientationWidget = nullptr;
	FileSelectWidget *m_faultSelectWidget = nullptr;
	FileSelectWidget *m_rgtInitSelectWidget = nullptr;
	HorizonSelectWidget *m_horizonSelectWidget = nullptr;


	Ihm2 *pIhmPatch = nullptr;
	QString checkForRGTPatch();
	double memoryEstimationForPatchProcess();
	long getVertexnbreEstimation();
	// void rgtFileInfoWrite(RgtVolumicCPU<float> *p);
	void setWindowTitle0();


private slots:
		void trt_expert();
		void trt_rgtPatchExpert();
		void trt_seed_info();
		void trt_launch_thread();
		void trt_abort();
		void trt_save_rgt_partial();
		void showTime();
		void trt_file_conversion();
		void trt_session_load();
		void trt_launch_rgtGraphThread();
		void trt_rgtGraph_stop();
		void trt_rgtGraph_Save();
		void trt_rgtGraph_Kill();
		void trt_graphicOut();
		void projectChanged();
		void surveyChanged();

	// void computeZvsRho();
};


class MyThread0 : public QThread
{
     // Q_OBJECT
	 public:
	 MyThread0(GeotimeConfigurationWidget *p);
	 private:
	 GeotimeConfigurationWidget *pp;

protected:
     void run();
};

class rgtGraphThread : public QThread
{
     // Q_OBJECT
	 public:
	rgtGraphThread(GeotimeConfigurationWidget *p);
	 private:
	 GeotimeConfigurationWidget *pp;

protected:
     void run();
};


#endif /* MURATAPP_SRC_TOOLS_XCOM_MARFACOMPUTATIONWIDGET_H_ */
