/*
 *
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */




#include <QTableView>
#include <QHeaderView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QRadioButton>
#include <QGroupBox>
#include <QLabel>
#include <QPainter>
#include <QChart>
#include <QLineEdit>
#include <QToolButton>
#include <QLineSeries>
#include <QScatterSeries>
#include <QtCharts>
#include <QRandomGenerator>
#include <QTimer>
#include <QFileInfo>

#include <QVBoxLayout>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/sysinfo.h>


#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
// #include "FileConvertionXTCWT.h"
#include <RgtPatchManagerWidget.h>
#include "GeotimeConfiguratorExpertWidget.h"
#include "rgtPatchExpertWidget.h"
#include "GeotimeConfiguratorWidget.h"
#include "rgtGraphLabelRead.h"
#include "globalconfig.h"
#include "Xt.h"
#include "processrelay.h"
#include <rgtVolumicVolumeComputationOperator.h>
#include <horizonAttributComputeDialog.h>


// #define __LINUX__
#define EXPORT_LIB __attribute__((visibility("default")))
#include <config.h>
#include <normal.h>
#include <util.h>
#include <cuda_utils.h>
#include <fileio.h>
#include <fileio2.h>
#include <surface_stack.h>
#include <gradient_multiscale.cuh>
#include <ihm.h>
#include <ihm2.h>
// #include <rgtProcessing.h>
#include <sampleLimitsChooser.h>
#include <rgtGraph.h>
#include <rgtGraphLabelRead.h>
// #include <rgtVolumicCPU.hpp>

#include <workingsetmanager.h>
#include "ProjectManager.h"
#include "SurveyManager.h"
#include <rgtVolumic.h>
#include <file_convert.h>
#include <faultDetectionWidget.h>
#include <rgtVolumicGraphicOut.h>
#include <rgtDisplayData.hpp>
#include <orientationWidget.h>
#include <fileSelectWidget.h>
#include <dataFileCreationInfo.h>
#include <dipFilterWidget.h>
#include "collapsablescrollarea.h"
#include <patchParametersWidget.h>
#include <rgtPatchParametersWidget.h>
#include <orientationParametersWidget.h>
#include "rgtStackingParametersWidget.h"
#include <rgtStackingWidget.h>
#include <horizonSelectWidget.h>
#include <rgtAndPatchWidget.h>


using namespace std;


#define DEBUG0() fprintf(stderr, "debug : %s - %d\n", __FILE__, __LINE__);
unsigned long mem_avail();

int GLOBAL_RUN_TYPE;

void msgDisplay0(char *txt)
{
	fprintf(stderr, "%s", txt);
}

GeotimeConfigurationWidget::GeotimeConfigurationWidget(ProjectManagerWidget *_projectmanager, QWidget* parent) :
				QWidget(parent) {

	GLOBAL_RUN_TYPE = 0;
	this->bool_abort = 0;
	this->cuda_nb_devices = cuda_get_nbre_devices();

	QSize size0 = this->size();

	QHBoxLayout * mainLayout00 = new QHBoxLayout(this);

	QGroupBox *qgbProgramManager = new QGroupBox;
	QVBoxLayout *qvb_programmanager = new QVBoxLayout(qgbProgramManager);
	if ( _projectmanager )
		projectmanager = _projectmanager;
	else
	{
		projectmanager = new ProjectManagerWidget;
		QPushButton *qpb_loadsession = new QPushButton("load session");
		qvb_programmanager->addWidget(projectmanager);
		m_workingSetManager = new WorkingSetManager(this);
		m_workingSetManager->setManagerWidgetV2(projectmanager);
	}


	// DATASETS MANAGERS
	/*
	QGroupBox *qgbProgramManager = new QGroupBox;
	QVBoxLayout *qvb_programmanager = new QVBoxLayout(qgbProgramManager);
	projectmanager = new ProjectManagerWidget;
	QPushButton *qpb_loadsession = new QPushButton("load session");
	qvb_programmanager->addWidget(projectmanager);
	*/

	// m_workingSetManager = manager;

//	QGroupBox *qgbSystem = new QGroupBox;
//	this->systemInfo = new GeotimeSystemInfo(this);
//	this->systemInfo->setVisible(true);
//	systemInfo->setMinimumWidth(350);
//	QVBoxLayout *layout2 = new QVBoxLayout(qgbSystem);
//	layout2->addWidget(systemInfo);

	// computation
	QGroupBox *qgbMainCompute = new QGroupBox;
	QVBoxLayout *mainlayoutCompute = new QVBoxLayout(qgbMainCompute);

	// label_dimensions = new QLabel("...");

	m_seismicFileSelectWidget = new FileSelectWidget();
	m_seismicFileSelectWidget->setProjectManager(projectmanager);
	m_seismicFileSelectWidget->setLabelText("seismic filename");
	m_seismicFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Seismic);
	m_seismicFileSelectWidget->setLabelDimensionVisible(true);


	// ==== orientation
	m_orientationWidget = new OrientationWidget(projectmanager, true);

	qgb_orientation = new QGroupBox();
	qgb_orientation->setTitle("Orientation");
	// qgb_orientation->setMaximumHeight(300);

	QVBoxLayout *qvb_orientation = new QVBoxLayout(qgb_orientation);
	qvb_orientation->addWidget(m_orientationWidget);

	// === stacking
	QGroupBox *qgbRgtStacking0 = new QGroupBox(this);
	QVBoxLayout *qvbRgtStacking0 = new QVBoxLayout(qgbRgtStacking0);
	m_rgtStackingWidget = new RgtStackingWidget(m_workingSetManager);
	m_rgtStackingWidget->setSystemInfo(m_systemInfo);
	// m_rgtStackingWidget->setWorkingSetManager(m_workingSetManager);
	qvbRgtStacking0->addWidget(m_rgtStackingWidget);

	// ==== patch
	QGroupBox *qgbRgtAndPatch = new QGroupBox(this);
	QVBoxLayout *qvbRgtAndPatch = new QVBoxLayout(qgbRgtAndPatch);
	m_rgtAndPatch = new RgtAndPatchWidget(m_workingSetManager);
	m_rgtAndPatch->setSystemInfo(m_systemInfo);

	qvbRgtAndPatch->addWidget(m_rgtAndPatch);

	faultDetectionWidget = new FaultDetectionWidget(m_workingSetManager);
	m_dipFilterWidget = new DipFilterWidget(m_workingSetManager);

	// horizon attribut
	m_horizonAttribut = new HorizonAttributComputeDialog(nullptr);
	m_horizonAttribut->setProjectManager(projectmanager);
	m_horizonAttribut->setWorkingSetManager(m_workingSetManager);
	m_horizonAttribut->setTreeUpdate(false);

	qgb_orientation->setAlignment(Qt::AlignTop);
	QTabWidget *computeTabwidget = new QTabWidget();
	// computeTabwidget->insertTab(0, qgb_orientation, QIcon(QString("")), "Orientation");
	computeTabwidget->insertTab(0, qgbRgtStacking0, QIcon(":/slicer/icons/iconsRGT.svg"), "Rgt\nStacking");
	computeTabwidget->insertTab(1, qgbRgtAndPatch, QIcon(":/slicer/icons/RGTPatch.svg"), "Rgt\npatches");
//	// computeTabwidget->insertTab(3, qgbRgtStacking, QIcon(QString("")), "RGT");
	computeTabwidget->insertTab(2, faultDetectionWidget, QIcon(":/slicer/icons/Fault.svg"), "Fault\ndetection");
	computeTabwidget->insertTab(3, m_dipFilterWidget, QIcon(":/slicer/icons/dipFilter.svg"), "Dip\nfilter");
	computeTabwidget->setStyleSheet("QTabBar::tab { height: 40px; width: 100px; }");
	computeTabwidget->setIconSize(QSize(40, 40));
	// computeTabwidget->setMaximumHeight(500);

	/*
	QGroupBox* qgb_textinfo = new QGroupBox(this);
	qgb_textinfo->setTitle("info");
	// qgb_textinfo->setMaximumHeight(200);
	QVBoxLayout* qhb_textinfo = new QVBoxLayout(qgb_textinfo);//(qgb_orientation);
	*/


	m_seismicFileSelectWidget->setFixedHeight(20);
	// mainlayoutCompute->addWidget(m_seismicFileSelectWidget);
	// mainlayoutCompute->addLayout(qvb_orientation);
	mainlayoutCompute->addWidget(computeTabwidget); //addLayout(qvbRgtMethodes);
	// mainlayoutCompute->addWidget(qgb_textinfo);
	mainlayoutCompute->setAlignment(Qt::AlignTop);

	QTabWidget *tabWidgetMain = new QTabWidget();
	// tabWidgetMain->insertTab(1, qgbMainCompute, QIcon(QString("")), "Compute");
	// tabWidgetMain->insertTab(2, qgbSystem, QIcon(QString("")), "System");
	tabWidgetMain->insertTab(0, qgbRgtStacking0, QIcon(":/slicer/icons/iconsRGT.svg"), "Rgt\nStacking");
	tabWidgetMain->insertTab(1, qgbRgtAndPatch, QIcon(":/slicer/icons/RGTPatch.svg"), "Rgt\npatches");
	//	// computeTabwidget->insertTab(3, qgbRgtStacking, QIcon(QString("")), "RGT");
	tabWidgetMain->insertTab(2, faultDetectionWidget, QIcon(":/slicer/icons/Fault.svg"), "Fault\ndetection");
	tabWidgetMain->insertTab(3, m_dipFilterWidget, QIcon(":/slicer/icons/dipFilter.svg"), "Dip\nfilter");
	tabWidgetMain->insertTab(4, m_horizonAttribut, QIcon(""), "Horizon attributs");
	tabWidgetMain->setIconSize(QSize(40, 40));


	QGroupBox *g1 = new QGroupBox;
	QGroupBox *g2 = new QGroupBox;
	tabWidgetMain->insertTab(5, g1, QIcon(QString("")), "");
	// tabWidgetMain->insertTab(5, g2, QIcon(QString("")), "");
	tabWidgetMain->setTabEnabled(5, false);
	// tabWidgetMain->setTabEnabled(5, false);

	tabWidgetMain->insertTab(6, qgbProgramManager, QIcon(":/slicer/icons/earth.png"), "Manager");
	tabWidgetMain->setStyleSheet("QTabBar::tab { height: 40px; width: 100px; }");
	tabWidgetMain->setIconSize(QSize(40, 40));
	// computeTabwidget->setMaximumHeight(500);

	QScrollArea *scrollArea = new QScrollArea;
	scrollArea->setWidget(tabWidgetMain);
	scrollArea->setWidgetResizable(true);
	mainLayout00->addWidget(scrollArea);

	field_fill();
	current_path = QDir::currentPath();
	config_filename = current_path + "/config.txt";
	window_enable(true);
	if ( pIhmPatch == nullptr ) pIhmPatch = new Ihm2();

	connect(projectmanager->m_projectManager, &ProjectManager::projectChanged, this, &GeotimeConfigurationWidget::projectChanged);
	connect(projectmanager->m_surveyManager, &SurveyManager::surveyChanged, this, &GeotimeConfigurationWidget::surveyChanged);
	setWindowTitle0();
}

GeotimeConfigurationWidget::~GeotimeConfigurationWidget() {
	if ( pIhmPatch )
	{
		delete pIhmPatch;
		pIhmPatch = nullptr;
	}
}

void GeotimeConfigurationWidget::setWindowTitle0() {
	QString title = "NextVision Processing - ";
	if ( projectmanager != nullptr )
	{
		title += projectmanager->getProjectName() + " - " + projectmanager->getSurveyName();
	}
	this->setWindowTitle(title);
}

void GeotimeConfigurationWidget::setProcessRelay(ProcessRelay* relay) {
	m_processRelay = relay;
	if ( m_rgtAndPatch ) m_rgtAndPatch->setProcessRelay(m_processRelay);
	if ( m_rgtStackingWidget ) m_rgtStackingWidget->setProcessRelay(m_processRelay);
}

void GeotimeConfigurationWidget::setSystemInfo(GeotimeSystemInfo *systemInfo)
{
	m_systemInfo = systemInfo;
	if ( m_rgtStackingWidget ) m_rgtStackingWidget->setSystemInfo(m_systemInfo);
	if ( m_rgtAndPatch ) m_rgtAndPatch->setSystemInfo(m_systemInfo);
}

QString GeotimeConfigurationWidget::getSeismicName()
{
	return this->m_seismicFileSelectWidget->getFilename();
}

QString GeotimeConfigurationWidget::getSeismicPath()
{
	return this->m_seismicFileSelectWidget->getPath();
}


// *****************************************************************
//
// *****************************************************************
QString GeotimeConfigurationWidget::data_path_read()
{
	return NULL; //this->projectmanager->get_seismic_directory();
}

void GeotimeConfigurationWidget::data_path_write(QString filename)
{

}

QString GeotimeConfigurationWidget::filename_to_path_create(QString filename)
{
	int lastPoint = filename.lastIndexOf("/");
	QString path = filename.left(lastPoint);
	return path;
}

QString GeotimeConfigurationWidget::filename_format_create(QString base, QString separator, QString suffix)
{
	int lastPoint = base.lastIndexOf(separator);
	QString prefix = base.left(lastPoint);
	QString out = prefix + "_" + suffix;
	return out;
}

void GeotimeConfigurationWidget::get_size_from_filename(QString filename, int *size)
{
	char c_filename[10000];
	strcpy(c_filename, (char*)(filename.toStdString().c_str()));
	char *p = c_filename;
	FILEIO2 *pf = new FILEIO2();
	pf->openForRead(p);
	size[0] = pf->get_dimy();
	size[1] = pf->get_dimx();
	size[2] = pf->get_dimz();
	delete pf;
}

void GeotimeConfigurationWidget::sizeRectifyWithTraceLimits(int *size, int *sizeX)
{
	int x1 = traceLimitX1;
	int x2 = traceLimitX2;
	if ( x1 < 0 ) x1 = 0;
	if ( x2 < 0 ) x2 = size[1]-1;
	sizeX[0] = size[0];
	sizeX[1] = x2-x1+1;
	sizeX[2] = size[2];
}

double GeotimeConfigurationWidget::qt_ram_needed_memory(int nbthreads, int *size, int decim, int sizeof_stack, int nbsurfaces, bool polarity)
{
	double ret = 0.0;
	double surface = (double)size[0]*size[2];

	ret = (double)nbthreads * 64.0;

	if ( sizeof_stack == 0 )
		ret += 6 * size[1];
	else
		ret += 8 * size[1];

	if ( polarity )
		ret += size[1]/8.0;
	ret += (float)nbsurfaces*sizeof(float);
	ret *= (double)surface;
	ret /= (double)((double)decim*decim);
	ret /= 1e9;
	return ret;
}

double GeotimeConfigurationWidget::qt_cuda_needed_memory(int *size, int decim, int rgt_format, int nbsurfaces, bool polarity)
{
	double ret = 0.0;
	float cpumem = 0.0f;
	float gpumem = 0.0f;
	surface_stack_get_memory(1, 1, size[1], size[0], size[2], 1, nbsurfaces, rgt_format, &cpumem, &gpumem);
	ret = gpumem/1e9;
	return ret;
}


// *****************************************************************
//
// *****************************************************************
void GeotimeConfigurationWidget::window_enable(bool val)
{
	/*
	this->lineedit_rgtsuffix->setReadOnly(!val);
	this->lineedit_horizont0->setReadOnly(!val);
	this->lineedit_horizont1->setReadOnly(!val);
	pushbutton_expert->setEnabled(val);
	pushbutton_abort->setEnabled(!val);
	pushbutton_rgtpartialsave->setEnabled(!val);
	pushbutton_compute->setEnabled(val);
	*/
}



void GeotimeConfigurationWidget::field_fill()
{
	this->nb_threads = 30;
	// this->lineedit_rgtsuffix->setText("rgt");
	// this->lineedit_dipxfilename->setText("dipxy");
	// this->lineedit_dipzfilename->setText("dipxz");
	// this->decimation_factor = 1;
	this->stack_format = SURFACE_STACK_FORMAT_SHORT;
	this->dip_threshold = 5.0;
	this->stack_format == SURFACE_STACK_FORMAT_SHORT;	
	this->nbiter = 20;
	this->sigma_stack = 0.0;
	this->partial_rgt_save = 1;
	this->rgt_saverate = 4;
	// this->qcb_rgtcpugpu->setCurrentIndex(1);
	this->sigmagradient = 1.0;
	this->sigmatensor = 1.5;
	// qcb_orientationcpugpu->setCurrentIndex(1);
	this->rgt_compresserror = 0.001;
	this->bool_snapping = false;
	this->output_format = 0; // xt
	this->seed_threshold = 10000;
	this->seed_threshold_valid = 1;

	// parameters
	// orientation
	m_orientationWidget->setGradient(this->sigmagradient);
	m_orientationWidget->setTensor(this->sigmatensor);

	// rgt stack
	m_rgtStackingWidget->setCompute(false);
	m_rgtStackingWidget->setCpuGpu(1);
	m_rgtStackingWidget->setRgtSuffix("stacking_rgt");
	m_rgtStackingWidget->setPropagationSeedOnHorizon(false);
	m_rgtStackingWidget->setIter(this->nbiter);
	m_rgtStackingWidget->setDecimation(1);
	m_rgtStackingWidget->setDipThreshold(this->dip_threshold);
	m_rgtStackingWidget->setSnapping(this->bool_snapping);
	m_rgtStackingWidget->setEnableSeedMax(this->seed_threshold_valid==1);
	m_rgtStackingWidget->setSeedMax(seed_threshold);

	// patch

	/*
	m_rgtStackParameters->setNbIter(this->nbiter);
	m_rgtStackParameters->setDipThreshold(this->dip_threshold);
	m_rgtStackParameters->setDecimation(this->decimation_factor);
	m_rgtStackParameters->setSnapping(this->bool_snapping);
	m_rgtStackParameters->setSeedMaxvalid(this->seed_threshold_valid==1);
	m_rgtStackParameters->setSeedMax(seed_threshold);

	m_patchParameters->setPatchSize(patchParam.patchSize);
	m_patchParameters->setPatchPolarity(0);
	m_patchParameters->setGradientMax(patchParam.patchGradMax);
	m_patchParameters->setPatchRatio(patchParam.patchFitThreshold);
	m_patchParameters->setFaultThreshold(patchParam.patchFaultMaskThreshold);

	m_rgtPatchParameters->setScaleInitIter(rgtVolumicParam.initScaleParam.nbIter);
	m_rgtPatchParameters->setScaleInitEpsilon(rgtVolumicParam.initScaleParam.epsilon);
	m_rgtPatchParameters->setScaleInitDecim(rgtVolumicParam.initScaleParam.decimY);
	m_rgtPatchParameters->setIter(rgtVolumicParam.scaleParam.nbIter);
	m_rgtPatchParameters->setEpsilon(rgtVolumicParam.scaleParam.epsilon);
	m_rgtPatchParameters->setDecim(rgtVolumicParam.scaleParam.decimY);
	*/
}

void GeotimeConfigurationWidget::update_label_size(int size[3])
{
	/*
	QString str = "size: " + QString::number(size[1]) + " - " + QString::number(size[0]) + " - " + QString::number(size[2]);
	this->label_dimensions->setText(str);
	*/
}

int GeotimeConfigurationWidget::check_fields_for_compute()
{
	QString filename1, filename2;
	QMessageBox *msgBox = new QMessageBox(parentWidget());

	filename1 = m_seismicFileSelectWidget->getFilename();
	if ( filename1 == "" )
	{
		msgBox->setText("warning");
		msgBox->setInformativeText("You have to specify a seismic filename");
		msgBox->setStandardButtons(QMessageBox::Ok);
		msgBox->exec();
		return 0;
	}
	filename1 = m_seismicFileSelectWidget->getFilename();
	if ( /*qcb_rgtstackpolarity->isChecked() &&*/ filename1 == "" )
	{
	}
	filename1 = m_orientationWidget->getDipxyFilename();
	filename2 = m_orientationWidget->getDipxzFilename();
	qDebug() << filename1 << " -- " << filename2;
	if ( filename1 == "" || filename2 == "" )
	{
		msgBox->setText("warning");
		msgBox->setInformativeText("You have to specify dipx/dipy names or prefix");
		msgBox->setStandardButtons(QMessageBox::Ok);
		msgBox->exec();
		return 0;
	}
	return 1; 	
}

int GeotimeConfigurationWidget::check_memory_for_compute()
{
	int ret = 1;
	int nbthreads = this->nb_threads; // xxx jd this->lineedit_nbthreads->text().toInt();
	// int decim = m_rgtStackParameters->getDecimation(); // this->decimation_factor;
	// int nbsurfaces = listwidget_horizons->count()+2;
	int decim = m_rgtStackingWidget->getDecimation();
	int nbsurfaces = m_horizonSelectWidget->getNames().size()+2;
	QString filename;
	if ( m_orientationWidget->getComputationChecked() )
	{
		filename = getSeismicPath();
	}
	else
	{
		filename = m_orientationWidget->getDipxyPath();
	}
	if ( filename == "" ) return -1;
	int size[3];
	get_size_from_filename(filename, size);
	int stack_cpu_gpu = this->qcb_rgtcpugpu->currentIndex();
	if ( stack_cpu_gpu == 1 )
	{
		double cuda_mem = this->systemInfo->cuda_min_free_memory();
		double gpu_need = qt_cuda_needed_memory(size, 1, stack_format, nbsurfaces, 1);
		int decim_factor0 = (int)ceil(sqrt(gpu_need/(cuda_mem*.90)));
		if ( decim_factor0 >  m_rgtStackingWidget->getDecimation() )
		{

			QMessageBox *msgBox = new QMessageBox(parentWidget());
			QString txt;
			txt = "The computation needs " + QString::number(gpu_need, 'f', 2) + " GB for the GPU process.\nAnd you have only " + QString::number(cuda_mem, 'f', 2) + " GB free\n";
			txt += "Do you want to modify the decimation factor and put it to the value of " + QString::number(decim_factor0, 'd', 0) + " ?";
			msgBox->setText("Memory problem                              -");
			msgBox->setInformativeText(txt);
			msgBox->setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel );DEBUG0()
			// msgBox.setDefaultButton(QMessageBox::Save);
			int ret0 = msgBox->exec(); DEBUG0()
			if ( ret0 == QMessageBox::Ok ){ m_rgtStackingWidget->setDecimation(decim_factor0); /*this->decimation_factor = decim_factor0;*/ ret = 1; }DEBUG0()
			//if ( ret0 == QMessageBox::Discard ){ret = 1; }
			if ( ret0 == QMessageBox::Cancel ){ret = 0; }DEBUG0()
		}
		DEBUG0()
	}
	else
	{
		double ram_mem = this->systemInfo->qt_cpu_free_memory();
		double ram_need = qt_ram_needed_memory(nbthreads, size, 1, stack_format, nbsurfaces, 1);

		fprintf(stderr, "---> %f %f\n", ram_mem, ram_need);
		int decim_factor0 = (int)ceil(sqrt(ram_need/(ram_mem*.90)));
		if ( decim_factor0 !=  this->m_rgtStackingWidget->getDecimation() )
		{
			QMessageBox *msgBox = new QMessageBox(parentWidget());
			QString txt;
			txt = "The computation needs " + QString::number(ram_need, 'f', 2) + " GB ram memory.\nAnd you have only " + QString::number(ram_mem, 'f', 3) + "GB free\n";
			txt += "Do you want to modify the decimation factor and put it at the value of " + QString::number(decim_factor0, 'd', 0) + " ?";
			msgBox->setText("Memory problem.");
			msgBox->setInformativeText(txt);
			msgBox->setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
			// msgBox.setDefaultButton(QMessageBox::Save);
			int ret0 = msgBox->exec();
			if ( ret0 == QMessageBox::Ok ){ m_rgtStackingWidget->setDecimation(decim_factor0); ret = 1; }
			if ( ret0 == QMessageBox::Cancel ){ret = 0; }
		}
	}
	return ret;
}

// *****************************************************************

int GeotimeConfigurationWidget::getIndexFromVectorString(std::vector<QString> list, QString txt)
{
	for (int i=0; i<list.size(); i++)
	{
		if ( list[i].compare(txt) == 0 )
			return i;
	}
	return -1;
}


void file_convertion_xt_cwt(QString src, float cwt_error);
void GeotimeConfigurationWidget::trt_expert()
{
	GeotimeConfigurationExpertWidget* GeotimeConfigurationEx = new GeotimeConfigurationExpertWidget(this);
	GeotimeConfigurationEx->setAttribute(Qt::WA_DeleteOnClose);
	GeotimeConfigurationEx->setVisible(true);
	// GeotimeConfigurationEx->resize(500, 400);
}

void GeotimeConfigurationWidget::trt_rgtPatchExpert()
{
	RgtPatchExpertWidget* GeotimeConfigurationEx = new RgtPatchExpertWidget(this);
	GeotimeConfigurationEx->setAttribute(Qt::WA_DeleteOnClose);
	GeotimeConfigurationEx->setVisible(true);
	// GeotimeConfigurationEx->resize(500, 400);
}


void GeotimeConfigurationWidget::trt_save_rgt_partial()
{
	ihm_set_trt(IHM_TRT_SAVE_RGT_PARTIAL);
	QMessageBox *msgBox = new QMessageBox(parentWidget());

	msgBox->setText("Info");
	msgBox->setInformativeText("The save procedure will start after finishing the seeds of the current trace\nIt can take few seconds");
	msgBox->setStandardButtons(QMessageBox::Ok);
	msgBox->exec();
}

void GeotimeConfigurationWidget::showTime()
{
	// GLOBAL_textInfo->appendPlainText(QString("timer"));
	char txt[1000], txt2[1000];
	int type = 0;
	long idx, vmax;
	int msg_new = ihm_get_global_msg(&type, &idx, &vmax, txt);
	if ( GLOBAL_RUN_TYPE == 1 || type == IHM_TYPE_RGT_END )// && GLOBAL_PSTACK != NULL )
	{
		// if ( msg_new == 0 && !pIhmPatch ) return;
		// fprintf(stderr, "msg: %d\n", type);
		if ( msg_new != 0 )
		{
			switch ( type )
			{
			case IHM_TYPE_SURFACE_STACK:
			{
				if ( idx >= 0 && vmax > 0 )
				{
					float val_f = 100.0*idx/vmax;
					int val = (int)(val_f);
					qpb_progress->setValue(val);
					sprintf(txt2, "run %.1f%%", val_f);
					qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
					qpb_progress->setFormat(txt2);
				}
				this->textInfo->appendPlainText(QString(txt));
				break;
			}
			case IHM_TYPE_DIPXY:
			{
				if ( idx >= 0 && vmax > 0 )
				{
					float val_f = 100.0*idx/vmax;
					int val = (int)(val_f);
					qpb_progress->setValue(val);
					sprintf(txt2, "dip read: %.1f%%", val_f);
					qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,100,0)}");
					qpb_progress->setFormat(txt2);
				}
				this->textInfo->appendPlainText(QString(txt));
				break;
			}
			case IHM_TYPE_POLARITY:
			{
				if ( idx >= 0 && vmax > 0 )
				{
					float val_f = 100.0*idx/vmax;
					int val = (int)(val_f);
					qpb_progress->setValue(val);
					sprintf(txt2, "polarity read: %.1f%%", val_f);
					qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,100,0)}");
					qpb_progress->setFormat(txt2);
				}
				this->textInfo->appendPlainText(QString(txt));
				break;
			}
			case IHM_TYPE_MASK2D:
			{
				if ( idx >= 0 && vmax > 0 )
				{
					float val_f = 100.0*idx/vmax;
					int val = (int)(val_f);
					qpb_progress->setValue(val);
					sprintf(txt2, "mask2d: %.1f%%", val_f);
					qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,100,0)}");
					qpb_progress->setFormat(txt2);
				}
				this->textInfo->appendPlainText(QString(txt));
				break;
			}
			case IHM_TYPE_ORIENTATION:
			{
				if ( idx >= 0 && vmax > 0 )
				{
					float val_f = 100.0*idx/vmax;
					int val = (int)(val_f);
					qpb_progress->setValue(val);
					sprintf(txt2, "orientation: %.1f%%", val_f);
					qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,100,0)}");
					qpb_progress->setFormat(txt2);
					this->textInfo->appendPlainText(QString(txt2));
				}
				break;
			}
			case IHM_TYPE_RGT_SAVE:
			{
				if ( idx >= 0 && vmax > 0 )
				{
					float val_f = 100.0*idx/vmax;
					int val = (int)(val_f);
					qpb_progress->setValue(val);
					sprintf(txt2, "rgt save: %.1f%%", val_f);
					qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,0,200)}");
					qpb_progress->setFormat(txt2);
				}
				this->textInfo->appendPlainText(QString(txt));
				break;
			}
			case IHM_TYPE_RGT_END:
			{
				qpb_progress->setValue(0);
				qpb_progress->setFormat("");
				QMessageBox::information(this, "Geotime", "Process Geotime complete");
				break;
			}

			/*
		// RGT Volumic
		case IHM_TYPE_RGT_GRAPH:
		case IHM_TYPE_DATAINTERPOLATIONTOFILE:
		{
			float val_f = 100.0*idx/vmax;
			int val = (int)(val_f);
			qpb_progress->setValue(val);
			sprintf(txt2, "%s - %.1f%%", txt, val_f);
			qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
			qpb_progress->setFormat(txt2);
			break;
		}

		case IHM_TYPE_RGT_GRAPH_END:
		case IHM_TYPE_DATAINTERPOLATIONTOFILE_END:
		{
			// GLOBAL_RUN_TYPE = 0;
			qpb_progress->setValue(0);
			qpb_progress->setFormat("");
			// QMessageBox::information(this, "Geotime", "Process complete");
			break;
		}

		case IHM_TYPE_CONJUGATEGRADIENT:
		{
			float val_f = 100.0*idx/vmax;
			int val = (int)(val_f);
			qpb_progress->setValue(val);
			sprintf(txt2, "%s - %.1f%%", txt, val_f);
			qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
			qpb_progress->setFormat(txt2);
			break;
		}

		case IHM_TYPE_CONJUGATEGRADIENT_END:
		{
			GLOBAL_RUN_TYPE = 0;
			qpb_progress->setValue(0);
			qpb_progress->setFormat("");
			QMessageBox::information(this, "RGT Volumic", "Process complete");
			break;
		}
			 */
			}
		}

		// new ihmMessage
		if ( pIhmPatch )
		{
			if ( pIhmPatch->isSlaveMessage() )
			{
				Ihm2Message mess = pIhmPatch->getSlaveMessage();
				std::string message = mess.message;
				long count = mess.count;
				long countMax = mess.countMax;
				int trtId = mess.trtId;
				bool valid = mess.valid;
				float val = 100.0*count/countMax;
				QString barMessage = QString(message.c_str()) + " [ " + QString::number(val, 'f', 1) + " % ]";
				qpb_progress->setValue((int)val);
				qpb_progress->setFormat(barMessage);
				// qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
			}
			std::vector<std::string> mess = pIhmPatch->getSlaveInfoMessage();
			for (int n=0; n<mess.size(); n++)
			{
				this->textInfo->appendPlainText(QString(mess[n].c_str()));
			}
		}
	}
	else
	{
		qpb_progress->setValue(0);
		qpb_progress->setFormat("");
		qpbRgtPatchProgress->setValue(0);
		qpbRgtPatchProgress->setFormat("");
	}
}


static long seed_nbre_compute(int *size, int stepx, int stepy, int stepz)
{
	long nbseed = ((long)size[0]/stepx) * ((long)size[1]/stepy) * ((long)size[2]/stepz);
	return nbseed;
}
void GeotimeConfigurationWidget::trt_seed_info()
{

}


void GeotimeConfigurationWidget::msgDisplay(char *txt)
{

}


int GeotimeConfigurationWidget::checkGpuTextureSize()
{
	int cpuGpu = qcb_rgtcpugpu->currentIndex();
	if ( cpuGpu == 0 ) return SUCCESS;
	int *tab_gpu = nullptr, tab_gpu_size;
	tab_gpu = (int*)calloc(this->systemInfo->get_gpu_nbre(), sizeof(int));
	this->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	int maxSize;
	int textureMaxSize[3];
	cudaGetMaxTexture3D(textureMaxSize, tab_gpu[0]);
	maxSize = textureMaxSize[0];
	for (int i=1; i<tab_gpu_size; i++)
	{
		cudaGetMaxTexture3D(textureMaxSize, tab_gpu[0]);
		maxSize = MIN(textureMaxSize[0], maxSize);
	}

	char seismicFilename[10000];
	strcpy(seismicFilename, (char*)(getSeismicPath().toStdString().c_str()));
	if ( !FILEIO2::exist(seismicFilename) ) return FAIL;
	FILEIO2 *pf = new FILEIO2();
	pf->openForRead(seismicFilename);
	int dimx = pf->get_dimx();
	delete pf;

	int xx1 = this->traceLimitX1; if ( xx1 < 0 ) xx1 = 0;
	int xx2 = this->traceLimitX2; if ( xx2 < 0 ) xx2 = dimx-1;
	int dimxx = xx2-xx1+1;
	if ( dimxx <= maxSize ) return SUCCESS;

	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	QString txt = "Your memory GPU is lower than the time direction size [ " + QString::number(dimx) + " / " + QString::number(maxSize) + " ]";
	txt += "You have to reduce the processing size of your data (options Expert -> Sample limits";
	msgBox->setInformativeText(txt);
	msgBox->setStandardButtons(QMessageBox::Close);
	msgBox->exec();
	return FAIL;
}




// RUN
bool GeotimeConfigurationWidget::dipFilenameUpdate()
{
	bool dip_compute = m_orientationWidget->getComputationChecked();
	if ( dip_compute )
	{
		QString qext = QString(".xt");
		QString qs_dipx_filename = filename_format_create(getSeismicPath(), ".", m_orientationWidget->getDipxyFilename() + qext);
		QString qs_dipz_filename = filename_format_create(getSeismicPath(), ".", m_orientationWidget->getDipxzFilename() + qext);
		strcpy(this->dipxyRecomposeFilename, (char*)(qs_dipx_filename.toStdString().c_str()));
		strcpy(this->dipxzRecomposeFilename, (char*)(qs_dipz_filename.toStdString().c_str()));
		fprintf(stderr, "run: %d %s\n", __LINE__, getSeismicPath().toStdString().c_str());
	}
	else
	{
		QString qs_dipx_filename = m_orientationWidget->getDipxyPath();
		QString qs_dipz_filename = m_orientationWidget->getDipxzPath();
		strcpy(this->dipxyRecomposeFilename, (char*)(qs_dipx_filename.toStdString().c_str()));
		strcpy(this->dipxzRecomposeFilename, (char*)(qs_dipz_filename.toStdString().c_str()));
		if ( !FILEIO2::exist(this->dipxyRecomposeFilename) || !FILEIO2::exist(this->dipxzRecomposeFilename) )
		{
			QMessageBox *msgBox = new QMessageBox(this);
			msgBox->setText("warning");
			msgBox->setInformativeText("The files dipxy or dipxz seem not exist.\nPlease check them.");
			msgBox->setStandardButtons(QMessageBox::Ok);
			msgBox->exec();
			window_enable(true);
			return false;
		}
	}
	return true;
}


void cuda_props_print(void *_pFile, int id);

void GeotimeConfigurationWidget::dipCompute()
{
	cuda_props_print(stderr, 0);
	void *p = nullptr;
	char seismic_filename0[10000];
	double sigmag = m_orientationWidget->getGradient(); // this->sigmagradient;
	double sigmat = m_orientationWidget->getTensor(); // this->sigmatensor;
	int size[3];
	char dipx_filename0[10000];
	char dipz_filename0[10000];
	int *tab_gpu = NULL, tab_gpu_size;
	tab_gpu = (int*)calloc(this->systemInfo->get_gpu_nbre(), sizeof(int));
	this->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	int dip_cpu_gpu = m_orientationWidget->getProcessingTypeIndex();
	int nbthreads = this->nb_threads;

	QString qs_seismic_filename = m_seismicFileSelectWidget->getPath();
	strcpy(seismic_filename0, (char*)(qs_seismic_filename.toStdString().c_str()));
	get_size_from_filename(seismic_filename0, size);

//	FILEIO2 *pf = new FILEIO2();
//	pf->createNew(seismic_filename0, dipxyRecomposeFilename, size[1], size[0], size[2], 2);
//	delete pf;
//	pf = new FILEIO2();
//	pf->createNew(seismic_filename0, dipxzRecomposeFilename, size[1], size[0], size[2], 2);
//	delete pf;

	inri::Xt xtSeismic(seismic_filename0);
	inri::Xt xtDipxy(dipxyRecomposeFilename, xtSeismic);
	inri::Xt xtDipxz(dipxzRecomposeFilename, xtSeismic);

	char *p_dipx_filename = this->dipxyRecomposeFilename;
	char *p_dipz_filename = this->dipxzRecomposeFilename;


	p = normal_init();
	normal_set_gpu_list(p, tab_gpu, tab_gpu_size);

	normal_set_source_size(p, size[0], size[1], size[2]);
	normal_set_tile_size(p, 16, 16, 16);
	normal_set_block_size(p, 256, 256, 256);
	// normal_set_block_size(p, 64, 64, 64);
	normal_set_datain_filename(p, (char*)seismic_filename0);
	normal_set_sigma_grad(p, sigmag);
	normal_set_sigma_tens(p, sigmat);
	normal_set_nb_scales(p, 1);
	normal_set_out_filename(p, &p_dipx_filename, &p_dipz_filename, NULL, NULL);
	normal_set_output_type(p, NORMAL_OUTPUT_TYPE_DIP);
	normal_set_output_format(p, NORMAL_OUTPUT_FORMAT_16BITS);
	normal_set_type(p, NORMAL_TYPE_INLINE);
	normal_set_cpu_gpu(p, dip_cpu_gpu);
	normal_enable_chronos(p, 1);
	normal_set_nb_threads(p, nbthreads);
	normal_set_dip_type(p, 1);
	normal_set_data_cache_enable(p, 1);
	// normal_framwork_thread_run(p);
	GLOBAL_RUN_TYPE = 1;

	normal_framwork_run(p);
	p = normal_release(p);
}

void GeotimeConfigurationWidget::rgtStackingCompute()
{
	int processing_size[3], polarity_size[3], size[3];
	get_size_from_filename(this->dipxyRecomposeFilename, size);
	processing_size[0] = size[0] / m_rgtStackingWidget->getDecimation(); // this->decimation_factor;
	processing_size[1] = size[1];
	processing_size[2] = size[2] / m_rgtStackingWidget->getDecimation(); // this->decimation_factor;
	char rgt_filename[10000], rgt_filename2[10000];
	QString qs_seismic_filename = getSeismicPath();
	int *tab_gpu = NULL, tab_gpu_size;
	tab_gpu = (int*)calloc(this->systemInfo->get_gpu_nbre(), sizeof(int));
	this->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	int nbthreads = this->nb_threads;
	char seismic_filename0[10000];
	strcpy(seismic_filename0, (char*)(qs_seismic_filename.toStdString().c_str()));
	float horizon_t0 = (float)this->lineedit_horizont0->text().toInt();
	float horizon_t1 = (float)this->lineedit_horizont1->text().toInt();
	int onlySeedInsideHorizons = qCheckBoxSeedsInsideHorizons->isChecked() ? 1 : 0;

	QString rgt_sufix = this->lineedit_rgtsuffix->text();DEBUG0()
	if ( rgt_sufix != "" )
	{
		if ( this->output_format == 0 )
		{
			QString qs_rgt_filename = filename_format_create(qs_seismic_filename, ".", rgt_sufix + QString(".xt"));
			strcpy(rgt_filename, (char*)(qs_rgt_filename.toStdString().c_str()));DEBUG0()
			rgt_filename2[0] = 0;
		}
		else if ( this->output_format == 1 )
		{
			QString qs_rgt_filename = filename_format_create(qs_seismic_filename, ".", rgt_sufix + QString(".cwt"));
			strcpy(rgt_filename, (char*)(qs_rgt_filename.toStdString().c_str()));
			rgt_filename2[0] = 0;
		}
		else
		{
			QString qs_rgt_filename = filename_format_create(qs_seismic_filename, ".", rgt_sufix + QString(".xt"));
			strcpy(rgt_filename, (char*)(qs_rgt_filename.toStdString().c_str()));
			qs_rgt_filename = filename_format_create(qs_seismic_filename, ".", rgt_sufix + QString(".cwt"));
			strcpy(rgt_filename2, (char*)(qs_rgt_filename.toStdString().c_str()));
		}
	}

	if ( strlen(rgt_filename) != 0 )
	{
		if ( this->stack_format == SURFACE_STACK_FORMAT_SHORT )
		{
			FILEIO2 *pf = new FILEIO2();
			pf->createNew(this->dipxyRecomposeFilename, rgt_filename, size[1], size[0], size[2], 2);
			delete pf;
		}
		else if ( this->stack_format == SURFACE_STACK_FORMAT_FLOAT )
		{
			FILEIO2 *pf = new FILEIO2();
			pf->createNew(this->dipxyRecomposeFilename, rgt_filename, size[1], size[0], size[2], 8);
			delete pf;
		}
	}

	if ( strlen(rgt_filename2) != 0 )
	{
		if ( this->stack_format == SURFACE_STACK_FORMAT_SHORT )
		{
			FILEIO2 *pf = new FILEIO2();
			pf->createNew(this->dipxyRecomposeFilename, rgt_filename2, size[1], size[0], size[2], 2);
			delete pf;
		}
		else if ( this->stack_format == SURFACE_STACK_FORMAT_FLOAT )
		{
			FILEIO2 *pf = new FILEIO2();
			pf->createNew(this->dipxyRecomposeFilename, rgt_filename2, size[1], size[0], size[2], 8);
			delete pf;
		}
	}

	int bool_polarity = 1; // this->qcb_rgtstackpolarity->isChecked();
	int bool_mask2d = 1;

	int dimx_step = 20; //this->lineedit_stepdimy->text().toInt();
	int dimy_step = 100; //this->lineedit_stepdimx->text().toInt();
	int dimz_step = 100; //this->lineedit_stepdimz->text().toInt();
	int stack_cpu_gpu = qcb_rgtcpugpu->currentIndex(); //this->qcb_rgtstackgpu->isChecked();
	char **horizon_filename0 = NULL;
	int nbhorizons = m_horizonSelectWidget->getPaths().size();
	fprintf(stderr, "nbre: %d\n", nbhorizons);

	if ( traceLimitX1 >= 0 || traceLimitX2 >= 0 )
		fprintf(stderr, "trace limits: %d %d\n", traceLimitX1, traceLimitX2);

	int nativeSize[3], nativeSizeLimit[3];
	int processingSize[3], processingSizeLimit[3];

	get_size_from_filename(this->dipxyRecomposeFilename, nativeSize);
	sizeRectifyWithTraceLimits(nativeSize, nativeSizeLimit);

	processingSize[0] = nativeSize[0] / m_rgtStackingWidget->getDecimation(); // this->decimation_factor;
	processingSize[1] = nativeSize[1];
	processingSize[2] = nativeSize[2] / m_rgtStackingWidget->getDecimation(); // this->decimation_factor;
	sizeRectifyWithTraceLimits(processingSize, processingSizeLimit);

	void *p = surface_stack_init();
	surface_stack_set_gpu_list(p, tab_gpu, this->systemInfo->get_gpu_nbre());
	surface_stack_set_dip_threshold(p, dip_threshold);

	// deprecated ?
	surface_stack_set_xgrid(p, 0, processing_size[0] - 1, MAX(1, dimx_step/m_rgtStackingWidget->getDecimation())); // decimation_factor));
	// surface_stack_set_ygrid(p, 0, processing_size[1] - 1, MAX(1, dimy_step/decimation_factor));
	surface_stack_set_ygrid(p, 0, processing_size[1] - 1, MAX(1, dimy_step));DEBUG0()
	surface_stack_set_zgrid(p, 0, processing_size[2] - 1, MAX(1, dimz_step/m_rgtStackingWidget->getDecimation())); // decimation_factor));DEBUG0()

	surface_stack_set_nbthreads(p, nbthreads);
	surface_stack_domain_set(p, 0, processingSizeLimit[0] - 1, processingSizeLimit[0], 0, processingSizeLimit[2] - 1, processingSizeLimit[2]);
	surface_stack_iteration_set(p, this->nbiter);
	surface_stack_dip_set_dims(p, processingSizeLimit[0], processingSizeLimit[1], processingSizeLimit[2]);

	// surface_starck_dip_set_dips(p, (void*)v_dipx, (void*)v_dipz, dip_precision);
	surface_stack_set_dipx_filename(p, this->dipxyRecomposeFilename);
	surface_stack_set_dipz_filename(p, this->dipxzRecomposeFilename);
	surface_stack_set_seismic_filename(p, seismic_filename0);


	surface_stack_set_native_size(p, nativeSize[0], nativeSize[1], nativeSize[2]);
	surface_stack_set_data_in_filename(p, nullptr);
	surface_stack_set_decimation_factor(p, m_rgtStackingWidget->getDecimation()); // this->decimation_factor);
	surface_stack_set_stack_format(p, this->stack_format);
	surface_stack_set_cuda(p, stack_cpu_gpu);

	// surface_stack_set_polarity_data(p, pol, polarity_size);
	// surface_stack_set_mask(p, v_mask);
	surface_stack_set_bool_polarity(p, bool_polarity);
	surface_stack_set_bool_mask2d(p, bool_mask2d);

	// surface_stack_set_stack_crit_filename(p, stack_crit_filename);
	// if ( strlen(stack_filename) != 0 ) surface_stack_set_stack_filename(p, stack_filename); else surface_stack_set_stack_filename(p, nullptr);

	if ( strlen(rgt_filename) != 0 ) surface_stack_set_rgt_filename(p, rgt_filename); else surface_stack_set_rgt_filename(p, nullptr);
	if ( strlen(rgt_filename2) != 0 ) surface_stack_set_rgt_filename2(p, rgt_filename2); else surface_stack_set_rgt_filename2(p, nullptr);
	surface_stack_set_sigma_stack(p, this->sigma_stack);
	surface_stack_set_enable_partial_rgt_save(p, this->partial_rgt_save);
	// deprecated ?
	surface_stack_set_rgt_saverate(p, this->rgt_saverate);

	GLOBAL_RUN_TYPE = 1;
	// GLOBAL_PSTACK = p;

	// horizons
	std::vector<QString> v_horizon_filename = m_horizonSelectWidget->getPaths();
	if ( nbhorizons > 0 )
	{
		horizon_filename0 = (char**)calloc(nbhorizons, sizeof(char*));
		for (int i=0; i<nbhorizons; i++)
		{
			horizon_filename0[i] = (char*)calloc(10000, sizeof(char));
			// QListWidgetItem *phorizon = listwidget_horizons->item(i);
			// QString horizonname = phorizon->text();
			QString horizonname = v_horizon_filename[i];
			sprintf(horizon_filename0[i], (char*)horizonname.toStdString().c_str());
			surface_stack_add_horizon_filename(p, horizon_filename0[i]);
			fprintf(stderr, "horizon: n°: %d - %s\n", i, horizon_filename0[i]);
		}
	}
	surface_stack_set_horizon_t0(p, horizon_t0);
	surface_stack_set_horizon_t1(p, horizon_t1);
	surface_stack_set_bool_snapping(p, bool_snapping);
	surface_stack_set_rgt_cwtcompressionerror(p, rgt_compresserror);

	if ( seed_threshold_valid == 1 )
		surface_stack_set_seedMax(p, seed_threshold);
	else
		surface_stack_set_seedMax(p, -1);

	float seismic_step_sample = 1.0f;
	float seismic_start_sample = 0.0f;
	void *p1 = fileio_open(seismic_filename0);
	seismic_step_sample = file_get_step_sample(p1);
	seismic_start_sample = file_get_start_sample(p1);
	p1 = fileio_close(p1);

	surface_stack_set_seismic_start_sample(p, seismic_start_sample);
	surface_stack_set_seismic_step_sample(p, seismic_step_sample);
	surface_stack_set_only_seeds_inside_horizons(p, onlySeedInsideHorizons);

	surface_stack_set_trace_limits(p, traceLimitX1, traceLimitX2);


	// surface_stack_set_msg_display(p, &msgDisplay0);
	surface_stack_run(p);
	// surface_stack_debug_test_minmax(p);
	// surface_stack_debug_test_minmax(pol, polarity_size);

	p = surface_stack_release(p);
	// FREE(v_dipx)
	// FREE(v_dipz)
	// FREE(pol)
	// void *p = normal_init();
	// normal_framwork_run(p);


	FREE(tab_gpu);
	this->pushbutton_compute->setEnabled(true);
	for (int i=0; i<nbhorizons; i++) if ( horizon_filename0[i] != NULL ) free(horizon_filename0[i]);
	if ( horizon_filename0 != NULL ) free(horizon_filename0);
	this->bool_abort = 0;
	window_enable(true);
	GLOBAL_RUN_TYPE = 0;
}





void GeotimeConfigurationWidget::trt_compute()
{
	bool dip_compute = m_orientationWidget->getComputationChecked();
	bool rgtStackingEnable = qcb_stackingRgtEnable->isChecked();
	this->bool_abort = 0;
	window_enable(false);
	this->pushbutton_compute->setEnabled(false);

	int size[3];

	if ( !dipFilenameUpdate() ) return;
	if ( dip_compute )
	{
		dipCompute();
	}

	if ( checkGpuTextureSize() == FAIL ) {
		window_enable(true);
		return;
	}

	if ( rgtStackingEnable )
	{
		rgtStackingCompute();
	}

	if ( this->bool_abort == 1 )
	{
		GLOBAL_RUN_TYPE = 0;
		this->bool_abort = 0;
		window_enable(true);
		GLOBAL_RUN_TYPE = 0;
		return;
	}


	this->pushbutton_compute->setEnabled(true);
	this->bool_abort = 0;
	window_enable(true);
	GLOBAL_RUN_TYPE = 0;
}


// void MyThread::run()
// {

// }
MyThread0::MyThread0(GeotimeConfigurationWidget *p)
{
	this->pp = p;
}

void MyThread0::run()
{
	pp->trt_compute();
}

// void MyThread0::stop() {}
void GeotimeConfigurationWidget::trt_launch_thread()
{
	if ( check_fields_for_compute() == 0 ) return;
	if ( check_memory_for_compute() == 0 ) return;
	MyThread0 *thread = new MyThread0(this);
	thread->start();
}

void GeotimeConfigurationWidget::trt_abort()
{
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort the processing ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		ihm_set_trt(IHM_TRT_SURFACE_STACK_STOP);
		this->bool_abort = 1;
	}
}

void GeotimeConfigurationWidget::trt_file_conversion()
{
	// FileConversionXTCWT *p = new FileConversionXTCWT(projectmanager);
	// p->setModal(true);
	// p->exec();
}



void GeotimeConfigurationWidget::trt_session_load()
{
	projectmanager->load_session_gui();
}


// ==========================================================================

rgtGraphThread::rgtGraphThread(GeotimeConfigurationWidget *p)
{
	this->pp = p;
}

void rgtGraphThread::run()
{
	pp->trt_start_rgtPatch();
}


int GeotimeConfigurationWidget::rgtVolumicDecimationFactorEstimation()
{
	int size[3];
	int blocSize[3] = {0,0,0};
	long nVertex = -1;
	long memSize = 0;

	get_size_from_filename(getSeismicPath(), size);
	long size0 = (long)size[0]*size[1]*size[2];
	long nbVertex = -1;

	if ( qcb_patchCompute->isChecked() )
	{
		nbVertex = getVertexnbreEstimation();
		blocSize[1] = m_patchParameters->getPatchSize(); // patchParam.patchSize;
		blocSize[2] = m_patchParameters->getPatchSize(); // patchParam.patchSize;
	}
	else
	{
		RgtGraphLabelRead::getBlocSizeFromFile(m_patchFileSelectWidget->getPath().toStdString(), blocSize);
		nVertex = RgtGraphLabelRead::getVertexNbreFromFile(m_patchFileSelectWidget->getPath().toStdString());
	}
	fprintf(stderr, "patch: size [%d %d] %ld vertices\n", blocSize[1], blocSize[2], nVertex);
	memSize = (long)10 * size0 * sizeof(float) + nVertex * 3 * sizeof(int) * (long)blocSize[1] * (long)blocSize[2];
	double decimD = memSize / (0.9*1e9*systemInfo->qt_cpu_free_memory());
	int decim = (int)ceil(sqrt(decimD));
	decim = MAX(1, decim);
	return decim;
}

long GeotimeConfigurationWidget::getVertexnbreEstimation()
{
	long nbre = -1.0;
	QString seismicFilename = getSeismicPath();
	QFileInfo file(seismicFilename);
	if ( !file.exists() ) return nbre;
	inri::Xt xt((char*)seismicFilename.toStdString().c_str());
	if ( !xt.is_valid() )  return nbre;
	long dimx = xt.nSamples();
	long dimy = xt.nRecords();
	long dimz = xt.nSlices();
	long nbCylindres = ((dimy-1)/patchParam.patchSize+1) * ((dimy-1)/patchParam.patchSize+1);
	long nbSurfacesPerCylindre = dimx / 4;
	return nbCylindres * nbSurfacesPerCylindre;
}

double GeotimeConfigurationWidget::memoryEstimationForPatchProcess()
{
	double mem = -1.0;
	long nbVertex = getVertexnbreEstimation();
	if ( nbVertex < 0.0 ) return mem;
	long nbEdges = nbVertex * 2;
	return mem;
}



QString GeotimeConfigurationWidget::checkForRGTPatch()
{
	QString msg = "";

	if ( !dipFilenameUpdate() ) return "error";
	if ( qcb_patchCompute->isChecked() )
	{


	}
	else
	{


	}

	return msg;
}

void GeotimeConfigurationWidget::trt_launch_rgtGraphThread()
{
	// RgtGraph::debug_testFusion();
	// return;
	// ==========================================================

//	if ( GLOBAL_RUN_TYPE == 1 ) return;
//	if ( !dipFilenameUpdate() ) return;
//	if ( qcb_patchRgtCompute->isChecked() && !checkRgtGraphLaunch() )
//	{
//		QMessageBox msgBox;
//		QString txt = "Size incompatibility between data ";
//		msgBox.setText(txt);
//		msgBox.setStandardButtons(QMessageBox::Ok);
//		msgBox.exec();
//		return;
//	}
//
//	if ( qcb_patchRgtCompute->isChecked() )
//	{
//		int decim = rgtVolumicDecimationFactorEstimation();
//		if ( decim > m_rgtPatchParameters->getDecim() )
//		{
//			QMessageBox msgBox;
//			QString txt = "The process needs to decimate the data with a factor of " + QString::number(decim);
//			msgBox.setText(txt);
//			msgBox.setInformativeText(tr("Do you want to continue ?"));
//			msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
//			msgBox.setDefaultButton(QMessageBox::Ok);
//			int ret = msgBox.exec();
//			if ( ret == QMessageBox::Cancel )return;
//			m_rgtPatchParameters->setDecim(decim);
//		}
//	}
//	rgtGraphThread *thread = new rgtGraphThread(this);
//	thread->start();
}


QString GeotimeConfigurationWidget::patchMainDirectoryGet()
{
	// QString seismicFilename = lineedit_seismicfilename->text();
	QString seismicFilename = m_seismicFileSelectWidget->getFilename();
	if ( seismicFilename.isEmpty() ) return "";
	QString patchPath = projectmanager->getPatchPath();
	if ( patchPath.isEmpty() ) return "";
	return patchPath + seismicFilename + "_" + m_patchFileSelectWidget->getLineEditText() + "_" +  QString::number(patchParam.patchSize) + "/";
}

QString GeotimeConfigurationWidget::patchDirectoryGet()
{
	QString path = patchMainDirectoryGet();
	if ( path.isEmpty() ) return "";
	return path + "patch/";
}

QString GeotimeConfigurationWidget::graphFilenameGet()
{
	QString path = patchMainDirectoryGet();
	if ( path.isEmpty() ) return "";
	return path + "graph.bin";
}

QString GeotimeConfigurationWidget::graphLabelRawFilenameGet()
{
	QString path = patchMainDirectoryGet();
	if ( path.isEmpty() ) return "";
	return path + "label.raw";
}


QString GeotimeConfigurationWidget::graphlabel0FilenameGet()
{
	// QString seismicName = lineedit_seismicfilename->text();
	QString seismicName = m_seismicFileSelectWidget->getFilename();
	QString filename = getSeismicPath();
	int lastPoint = filename.lastIndexOf("/");
	filename = filename.left(lastPoint) + "/" + seismicName;
	filename = filename + "_" + m_patchFileSelectWidget->getLineEditText() + "_" + QString::number(patchParam.patchSize) + "__nextvisionpatch.xt";
	return filename;
}

QString GeotimeConfigurationWidget::graphlabel1FilenameGet()
{
	QString filename = getSeismicPath();
	int lastPoint = filename.lastIndexOf(".");
	filename = filename.left(lastPoint) + "_" + m_patchFileSelectWidget->getLineEditText() + "_" + QString::number(patchParam.patchSize) + "_label1.xt";
	return filename;
}


QString GeotimeConfigurationWidget::patchMainDirectoryFromPatchName(QString patchName)
{
	int lastPoint = patchName.lastIndexOf("__nextvisionpatch");
	QString path = projectmanager->getPatchPath() + "/" + patchName.left(lastPoint);
	qDebug() << "patch name" + patchName;
	qDebug() << patchName.left(lastPoint);
	return path;
}



bool GeotimeConfigurationWidget::checkRgtGraphLaunch()
{
	int sizeRef[3], size[3];
	std::string dyFilename = dipxyRecomposeFilename;
	std::string dzFilename = dipxzRecomposeFilename;

	if ( FILEIO2::exist((char*)dyFilename.c_str()) && FILEIO2::exist((char*)dzFilename.c_str()) )
	{
		get_size_from_filename((char*)dyFilename.c_str(), sizeRef);
		get_size_from_filename((char*)dzFilename.c_str(), size);
		for (int i=0; i<3; i++)
			if ( sizeRef[i ] != size[i] )
				return false;
	}

	if ( qcb_rgtVolumicRgt0->isChecked() && FILEIO2::exist((char*)dyFilename.c_str()) && FILEIO2::exist((char*)dzFilename.c_str()) )
	{
		get_size_from_filename((char*)m_rgtInitSelectWidget->getPath().toStdString().c_str(), size);
		for (int i=0; i<3; i++)
			if ( sizeRef[i ] != size[i] )
				return false;
	}
	return true;
}


//void GeotimeConfigurationWidget::rgtFileInfoWrite(RgtVolumicCPU<float> *p)
//{
//	if ( p == nullptr ) return;
//	DataFileCreationInfo *info = new DataFileCreationInfo();
//	info->addComment(QString::fromStdString("Type: "), QString::fromStdString("RGT Volumic"));
//	info->setDataPath(QString::fromStdString(p->getRgtPath()));
//	info->addComment(QString::fromStdString("Seismic: "), QString::fromStdString(p->getSeismicPath()));
//	info->write();
//	delete info;
//}

void GeotimeConfigurationWidget::rgtVolumicRun()
{
//	if ( qcb_patchRgtCompute->isChecked() )
//	{
//		int size[3];
//		double alpha = 81.0;
//		get_size_from_filename(getSeismicPath(), size);
//
//		QString qsRgtFilename = filename_format_create(getSeismicPath(), ".", lineedit_patchRgtRgtName->text() + QString(".xt"));
//		std::string dyFilename = dipxyRecomposeFilename;
//		std::string dzFilename = dipxzRecomposeFilename;
//		std::string rgtFilenameT = qsRgtFilename.toStdString();
//		// fprintf(stderr, "%s\n%s\n", dyFilename.c_str(), dzFilename.c_str());
//		QString patchMainDirectory = patchMainDirectoryFromPatchName(m_patchFileSelectWidget->getFilename());
//		QString label1Filename = patchMainDirectory + "/" + "label.raw";
//		fprintf(stderr, "graph filename:     %s\n", m_patchFileSelectWidget->getPath().toStdString().c_str());
//		fprintf(stderr, "patchMainDirectory: %s\n", patchMainDirectory.toStdString().c_str());
//
//
//#ifdef __DEF_rgtVolumicGraphicOut
//
//		if ( rgtVolumicGraphicOut == nullptr )
//		{
//			rgtVolumicGraphicOut = new RgtVolumicGraphicOut();
//		}
//
//		rgtVolumicGraphicOut->setVisible(true);
//		rgtVolumicGraphicOut->setWindowFlags(Qt::Window | Qt::WindowTitleHint | Qt::CustomizeWindowHint);
//		return;
//		xxx
//#endif
//
//		RgtDisplayData<float> *display = new RgtDisplayData<float>();
//
//		// if ( pIhmPatch != nullptr ) { delete pIhmPatch; pIhmPatch = nullptr; };
//		// if ( pIhmPatch == nullptr ) pIhmPatch = new Ihm2();
//		QString rgtName = lineedit_patchRgtRgtName->text();
//
//		RgtVolumicComputationOperator *rgtVolumicOperator =	new RgtVolumicComputationOperator(getSeismicPath().toStdString(), rgtFilenameT, rgtName.toStdString());
//		rgtVolumicOperator->setSurveyPath(projectmanager->getSurveyPath());
//		m_processRelay->addProcess(rgtVolumicOperator);
//
//
//		RgtVolumicCPU<float> *p = new RgtVolumicCPU<float>();
//		// RgtVolumic *p = new RgtVolumic();
//		p->setRgtVolumicComputationOperator(rgtVolumicOperator);
//		p->setSeismicFilename(getSeismicPath().toStdString());
//		p->setDipxyFilename(dyFilename);
//		p->setDipxzFilename(dzFilename);
//		if ( qcb_rgtVolumicRgt0->isChecked() )
//		{
//			p->setRgt0Filename(m_rgtInitSelectWidget->getPath().toStdString());
//			// fprintf(stderr, "rgt0 filename: %s\n", rgtVolumicParam.rgt0Filename.toStdString().c_str());
//		}
//		else
//			p->setRgt0Filename("");
//		p->setRgtFilename(rgtFilenameT);
//		p->setlabel1Filename(label1Filename.toStdString());
//
//		QString patchFilename = m_patchFileSelectWidget->getPath();
//		if ( qcb_patchCompute->isChecked() )
//		{
//			patchFilename = graphlabel0FilenameGet();
//		}
//		p->setPatchFilename(patchFilename.toStdString());
//		p->setEpsilon(m_rgtPatchParameters->getEpsilon());
//		p->setDecim(m_rgtPatchParameters->getDecim(), m_rgtPatchParameters->getDecim());
//		p->setNbIter(m_rgtPatchParameters->getIter());
//
//		int arrayDecim[2];
//		int arrayNbIter[2];
//		double arrayEpsilon[2];
//		int nbScales = 0;
//		if ( !rgtVolumicParam.initScaleEnable )
//		{
//			nbScales = 1;
//			arrayDecim[0] = m_rgtPatchParameters->getDecim();
//			arrayNbIter[0] = m_rgtPatchParameters->getIter();
//			arrayEpsilon[0] = m_rgtPatchParameters->getEpsilon();
//		}
//		else
//		{
//			nbScales = 2;
//			arrayDecim[0] = m_rgtPatchParameters->getScaleInitDecim();
//			arrayNbIter[0] = m_rgtPatchParameters->getScaleInitIter();
//			arrayEpsilon[0] = m_rgtPatchParameters->getScaleInitEpsilon();
//			arrayDecim[1] = m_rgtPatchParameters->getDecim();
//			arrayNbIter[1] = m_rgtPatchParameters->getIter();
//			arrayEpsilon[1] = m_rgtPatchParameters->getEpsilon();
//		}
//		p->setNbScales(nbScales);
//		p->setArrayDecim(arrayDecim);
//		p->setArrayEpsilon(arrayEpsilon);
//		p->setArrayNbIter(arrayNbIter);
//		p->setIhm(pIhmPatch);
//		// p->setIdleDipMax(rgtVolumicParam.idleDipMax);
//
//		p->setDepthMax(-1);
//		GLOBAL_RUN_TYPE = 1;
//		// p->run();
//		// p->setRgtDisplayData(display);
//		p->runScale();
//		delete p;
//		m_processRelay->removeProcess(rgtVolumicOperator);
//		delete rgtVolumicOperator;
//	}
}

/*
void GeotimeConfigurationWidget::rgtVolumicRun()
{
	if ( qcb_patchRgtCompute->isChecked() )
	{
		int size[3];
		double alpha = 81.0;
		get_size_from_filename(this->seismic_filename, size);

		QString qsRgtFilename = filename_format_create(this->seismic_filename, ".", lineedit_patchRgtRgtName->text() + QString(".xt"));
		std::string dyFilename = dipxyRecomposeFilename;
		std::string dzFilename = dipxzRecomposeFilename;
		std::string rgtFilenameT = qsRgtFilename.toStdString();
		// fprintf(stderr, "%s\n%s\n", dyFilename.c_str(), dzFilename.c_str());
		QString patchMainDirectory = patchMainDirectoryFromPatchName(rgtVolumicParam.patchName);
		QString label1Filename = patchMainDirectory + "/" + "label.raw";
		fprintf(stderr, "graph filename:     %s\n", rgtVolumicParam.patchFilename.toStdString().c_str());
		fprintf(stderr, "patchMainDirectory: %s\n", patchMainDirectory.toStdString().c_str());

		RgtVolumicCPU<float> *p = new RgtVolumicCPU<float>();
		p->setDipxyFilename(dyFilename);
		p->setDipxzFilename(dzFilename);
		if ( qcb_rgtVolumicRgt0->isChecked() )
			p->setRgt0Filename(rgtVolumicParam.rgt0Filename.toStdString());
		else
			p->setRgt0Filename("");
		p->setRgtFilename(rgtFilenameT);
		p->setlabel1Filename(label1Filename.toStdString());
		p->setPatchFilename(rgtVolumicParam.patchFilename.toStdString());

		p->setEpsilon(rgtVolumicParam.scaleParam.epsilon);
		p->setDecim(rgtVolumicParam.scaleParam.decimY, rgtVolumicParam.scaleParam.decimY); // no decimZ
		p->setNbIter(rgtVolumicParam.scaleParam.nbIter);

		int arrayDecim[2];
		int arrayNbIter[2];
		double arrayEpsilon[2];
		int nbScales = 0;
		if ( !rgtVolumicParam.initScaleEnable )
		{
			nbScales = 1;
			arrayDecim[0] = rgtVolumicParam.scaleParam.decimY;
			arrayNbIter[0] = rgtVolumicParam.scaleParam.nbIter;
			arrayEpsilon[0] = rgtVolumicParam.scaleParam.epsilon;
		}
		else
		{
			nbScales = 2;
			arrayDecim[0] = rgtVolumicParam.initScaleParam.decimY;
			arrayNbIter[0] = rgtVolumicParam.initScaleParam.nbIter;
			arrayEpsilon[0] = rgtVolumicParam.initScaleParam.epsilon;
			arrayDecim[1] = rgtVolumicParam.scaleParam.decimY;
			arrayNbIter[1] = rgtVolumicParam.scaleParam.nbIter;
			arrayEpsilon[1] = rgtVolumicParam.scaleParam.epsilon;
		}
		p->setNbScales(nbScales);
		p->setArrayDecim(arrayDecim);
		p->setArrayEpsilon(arrayEpsilon);
		p->setArrayNbIter(arrayNbIter);

		p->setDepthMax(-1);
		GLOBAL_RUN_TYPE = 1;
		// p->run();
		p->runScale();
		delete p;
	}
}
*/

void GeotimeConfigurationWidget::trt_start_rgtPatch()
{
	bool dip_compute = m_orientationWidget->getComputationChecked();
	if ( !dipFilenameUpdate() ) return;
	if ( dip_compute )
	{
		dipCompute();
	}
	int ret = SUCCESS;
	// projectmanager->createPatchDir();
	if ( qcb_patchCompute->isChecked() )
	{
		std::string label0Filename = graphlabel0FilenameGet().toStdString();
		std::string label1Filename = graphlabel1FilenameGet().toStdString();
		std::string labelRawFilename = graphLabelRawFilenameGet().toStdString();
		std::string graphFilename = graphFilenameGet().toStdString();
		std::string isoPath = patchDirectoryGet().toStdString();

		// rgtVolumicParam.patchFilename = QString::fromStdString(label0Filename);

		QDir dir(patchDirectoryGet());
		if ( !dir.exists() ) dir.mkpath(".");

		fprintf(stderr, "seismic filename:   %s\n", getSeismicPath().toStdString().c_str());
		fprintf(stderr, "label0 filename:    %s\n", label0Filename.c_str());
		fprintf(stderr, "label1 filename:    %s\n", label1Filename.c_str());
		fprintf(stderr, "label raw filename: %s\n", labelRawFilename.c_str());
		fprintf(stderr, "graph filename:     %s\n", graphFilename.c_str());
		fprintf(stderr, "isoPath filename:   %s\n", isoPath.c_str());
		fprintf(stderr, "graphFilename: %s\n", graphFilename.c_str());
		GLOBAL_RUN_TYPE = 1;

		// if ( pIhmPatch != nullptr ) { delete pIhmPatch; pIhmPatch = nullptr; };
		// if ( pIhmPatch == nullptr ) pIhmPatch = new Ihm2();
		RgtGraph *p = new RgtGraph();
		p->setSeismicFilename(getSeismicPath().toStdString());
		p->setLabel0Filename(label0Filename);
		p->setLabel1Filename(label1Filename);
		p->setLabel1RawFilename(labelRawFilename);
		p->setLabelChaineFilename(graphFilename);
		p->setIsoPatchFilenamePrefix(isoPath);
		p->setBlocSize(patchParam.patchSize, patchParam.patchSize);
		if ( qcb_rgtPatchMaskEnable->isChecked() )
			p->setFaultFilename(m_faultSelectWidget->getPath().toStdString());
		p->setFaultMaskThreshold(patchParam.patchFaultMaskThreshold);
		p->setPatchPolarity(patchParam.patchPolarity);
		p->setIhm(pIhmPatch);
		p->setDeltaVoverV(patchParam.deltaVoverV);
		p->setGradMax(patchParam.patchGradMax);

		std::vector<QString> Qnames = m_horizonSelectWidget->getPaths();
		std::vector<std::string> horizonPath;
		if ( Qnames.size() > 0 )
		{
			horizonPath.resize(Qnames.size());
			for (int n=0; n<Qnames.size(); n++)
			{
				horizonPath[n] = Qnames[n].toStdString();
			}
			p->setHorizons(horizonPath);
		}
		// todo ret
		ret = p->run();
	}
	if ( ret == SUCCESS )
	{
		rgtVolumicRun();
	}
	GLOBAL_RUN_TYPE = 0;
	if ( projectmanager )
		projectmanager->seimsicDatabaseUpdate();
}

void GeotimeConfigurationWidget::trt_rgtGraph_stop()
{
	if ( GLOBAL_RUN_TYPE == 0 ) return;

	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort the processing ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		ihm_set_trt(IHM_TRT_RGT_GRAPH_STOP);
		this->bool_abort = 1;
		if ( pIhmPatch )
		{
			pIhmPatch->setMasterMessage("stop", 0, 0, 0);
		}
	}
}

void GeotimeConfigurationWidget::trt_rgtGraph_Save()
{
	if ( GLOBAL_RUN_TYPE == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to save the current result ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		// ihm_set_trt(IHM_TRT_RGT_GRAPH_STOP);
		// this->bool_abort = 1;
		if ( pIhmPatch )
		{
			pIhmPatch->setMasterMessage("save", 0, 0, 0);
		}
	}
}

void GeotimeConfigurationWidget::trt_rgtGraph_Kill()
{
	if ( GLOBAL_RUN_TYPE == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		// ihm_set_trt(IHM_TRT_RGT_GRAPH_STOP);
		// this->bool_abort = 1;
		if ( pIhmPatch )
		{
			pIhmPatch->setMasterMessage("kill", 0, 0, 0);
		}
	}
}


void GeotimeConfigurationWidget::trt_graphicOut()
{

}


// ==========================================================================
QFileInfoList get_dirlist(QString path)
{
	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.xt" << "*.cwt";
	dir.setNameFilters(filters);
	QFileInfoList list = dir.entryInfoList();
	return list;
}


void GeotimeConfigurationWidget::projectChanged()
{
	setWindowTitle0();
}

void GeotimeConfigurationWidget::surveyChanged()
{
	setWindowTitle0();
}

 QString GeotimeConfigurationWidget::paramColorStyle = QString("background-color: rgb(40,56,72);");
