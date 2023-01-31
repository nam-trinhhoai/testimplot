/*
 *
 *
 *  Created on:
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

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <algorithm>


#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
// #include "FileConvertionXTCWT.h"
#include "collapsablescrollarea.h"
#include <rgtStackingParametersWidget.h>
#include <rgtStackingWidget.h>
#include <ProjectManager.h>
#include <fileSelectWidget.h>
#include <orientationWidget.h>
#include <GeotimeSystemInfo.h>
#include <surface_stack.h>
#include <ihm2.h>
#include "processrelay.h"
#include <rgtStackingComputationOperator.h>
#include <normal.h>
#include <geotimeFlags.h>
#include <workingsetmanager.h>
#include <GeotimeConfiguratorWidget.h>
#include <cuda_utils.h>
#include <util.h>

#include "Xt.h"


RgtStackingWidget::RgtStackingWidget(WorkingSetManager *workingSetManager, QWidget* parent) :
				QWidget(parent) {

	m_workingSetManager = workingSetManager;
	m_processing = new QLabel(".");
	m_projectManagerWidget = m_workingSetManager->getManagerWidgetV2();
	QVBoxLayout *layoutMain = new QVBoxLayout(this);

	QGroupBox *qgb_orientation = new QGroupBox(this);
	qgb_orientation->setTitle("Orientation");
	// qgb_orientation->setMaximumHeight(300);

	QVBoxLayout *qvb_orientation = new QVBoxLayout(qgb_orientation);
	m_orientationWidget = new OrientationWidget(m_projectManagerWidget, true, m_workingSetManager);
	qvb_orientation->addWidget(m_orientationWidget);
	qgb_orientation->setAlignment(Qt::AlignTop);

	m_compute = new QCheckBox("compute");

	QHBoxLayout* qhb_rgtsuffix = new QHBoxLayout;//(qgb_orientation);
	QLabel *label_rgtsuffix = new QLabel("rgt suffix");
	m_rgtSuffix = new QLineEdit("rgt");
	m_cpuGpu = new QComboBox();
	m_cpuGpu->addItem("CPU");
	m_cpuGpu->addItem("GPU");
	m_cpuGpu->setStyleSheet("QComboBox::item{height: 20px}");

	qhb_rgtsuffix->addWidget(label_rgtsuffix);
	qhb_rgtsuffix->addWidget(m_rgtSuffix);
	qhb_rgtsuffix->addWidget(m_cpuGpu);

	m_propagateSeed = new QCheckBox("propagate only seeds inside horizons");
	m_propagateSeed->setCheckState(Qt::Unchecked);


	CollapsableScrollArea* param = new CollapsableScrollArea("Parameters");
	m_rgtStackParameters = new RgtStackingParametersWidget(m_workingSetManager);
	QVBoxLayout *qvbRgtStackingParam = new QVBoxLayout;
	qvbRgtStackingParam->addWidget(m_rgtStackParameters);
	param->setContentLayout(*qvbRgtStackingParam);

	// QString style = QString("background-color: rgb(%1,%2,%3);").arg(90).arg(125).arg(160);
	param->setStyleSheet(GeotimeConfigurationWidget::paramColorStyle);

	m_seismicFileSelectWidget = new FileSelectWidget();
	m_seismicFileSelectWidget->setProjectManager(m_projectManagerWidget);
	m_seismicFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_seismicFileSelectWidget->setLabelText("seismic filename");
	m_seismicFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Seismic);
	m_seismicFileSelectWidget->setLabelDimensionVisible(false);
	m_seismicFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);


	QGroupBox *qgb_rgt = new QGroupBox();
	QVBoxLayout *qvb_rgt = new QVBoxLayout(qgb_rgt);
	qgb_rgt->setTitle("Rgt");
	qvb_rgt->addWidget(m_compute);
	qvb_rgt->addLayout(qhb_rgtsuffix);
	qvb_rgt->addWidget(m_propagateSeed);
	qvb_rgt->addWidget(param);
	qgb_rgt->setAlignment(Qt::AlignTop);
	qvb_rgt->setAlignment(Qt::AlignTop);

	QGroupBox *qgb_control = new QGroupBox("Run");
	QVBoxLayout *qvb_control = new QVBoxLayout(qgb_control);
	m_textInfo = new QPlainTextEdit("ready");
	m_progress = new QProgressBar();
	// qpbRgtPatchProgress->setGeometry(5, 45, 240, 20);
	m_progress->setMinimum(0);
	m_progress->setMaximum(100);
	m_progress->setValue(0);
	m_progress->setTextVisible(true);
	m_progress->setFormat("");

	QHBoxLayout *qhb_buttons = new QHBoxLayout(qgb_control);
	m_startStop = new QPushButton("start");
	m_save = new QPushButton("save");
	m_kill = new QPushButton("kill process");
	m_launchBatch = new QPushButton("run batch");
	m_preview = new QPushButton("preview");
	qhb_buttons->addWidget(m_startStop);
	qhb_buttons->addWidget(m_save);
	qhb_buttons->addWidget(m_kill);
	// qhb_buttons->addWidget(m_launchBatch);
	// qhb_buttons->addWidget(m_preview);

	qvb_control->addWidget(m_textInfo);
	qvb_control->addWidget(m_progress);
	qvb_control->addLayout(qhb_buttons);

	layoutMain->addWidget(m_processing);
	layoutMain->addWidget(m_seismicFileSelectWidget);
	layoutMain->addWidget(qgb_orientation);
	layoutMain->addWidget(qgb_rgt);
	layoutMain->addWidget(qgb_control);

//	layoutMain->addWidget(m_compute);
//	layoutMain->addLayout(qhb_rgtsuffix);
//	layoutMain->addWidget(m_propagateSeed);
//	layoutMain->addWidget(param);
	layoutMain->setAlignment(Qt::AlignTop);

	if ( pIhm == nullptr ) pIhm = new Ihm2();

	QTimer *timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));

	connect(m_startStop, SIGNAL(clicked()), this, SLOT(trt_launch()));
	connect(m_kill, SIGNAL(clicked()), this, SLOT(trt_rgtGraph_Kill()));
	connect(m_save, SIGNAL(clicked()), this, SLOT(trt_rgtSave()));
	connect(m_seismicFileSelectWidget, &FileSelectWidget::filenameChanged, this, &RgtStackingWidget::seismicFilenameChanged);
	startStopConfigDisplay();
}

RgtStackingWidget::~RgtStackingWidget() {

}

void RgtStackingWidget::setCompute(bool val)
{
	m_compute->setChecked(val);
}

void RgtStackingWidget::setCpuGpu(int val)
{
	m_cpuGpu->setCurrentIndex(val);
}

void RgtStackingWidget::setRgtSuffix(QString val)
{
	m_rgtSuffix->setText(val);
}

void RgtStackingWidget::setPropagationSeedOnHorizon(bool val)
{
	m_propagateSeed->setChecked(val);
}

void RgtStackingWidget::setIter(int val) { m_rgtStackParameters->setNbIter(val); }

void RgtStackingWidget::setDipThreshold(double val) { m_rgtStackParameters->setDipThreshold(val); }

void RgtStackingWidget::setDecimation(int val) { m_rgtStackParameters->setDecimation(val); }

void RgtStackingWidget::setSnapping(bool val) { m_rgtStackParameters->setSnapping(val); }

void RgtStackingWidget::setEnableSeedMax(bool val) { m_rgtStackParameters->setSeedMaxvalid(val); }

void RgtStackingWidget::setSeedMax(long val) { m_rgtStackParameters->setSeedMax(val); }

bool RgtStackingWidget::getCompute() { return m_compute->isChecked(); }

int RgtStackingWidget::getCpuGpu() { return m_cpuGpu->currentIndex(); }

QString RgtStackingWidget::getRgtSuffix() { return m_rgtSuffix->text(); }

bool RgtStackingWidget::getPropagationSeedOnHorizon() {return m_propagateSeed->isChecked(); }

int RgtStackingWidget::getIter() { return m_rgtStackParameters->getNbIter(); }

double RgtStackingWidget::getDipThreshold() { return m_rgtStackParameters->getDipThreshold(); }

int RgtStackingWidget::getDecimation() { return m_rgtStackParameters->getDecimation(); }

bool RgtStackingWidget::getSnapping() { return m_rgtStackParameters->getSnapping(); }

bool RgtStackingWidget::getEnableSeedMax() { return m_rgtStackParameters->getSeedMaxvalid(); }

long RgtStackingWidget::getSeedMax() { return m_rgtStackParameters->getSeedMax(); }


void RgtStackingWidget::setSystemInfo(GeotimeSystemInfo *val)
{
	m_systemInfo = val;
}

void RgtStackingWidget::startStopConfigDisplay()
{
	if ( pStatus == 0 )
	{
		m_startStop->setText("start");
		m_save->setEnabled(false);
		m_kill->setEnabled(false);
	}
	else if ( pStatus == 1 )
	{
		m_startStop->setText("stop");
		m_save->setEnabled(true);
		m_kill->setEnabled(true);
	}
}

// ===================================
bool RgtStackingWidget::createDataSet(char *src, char *dst)
{
	if ( src == nullptr || dst == nullptr ) return false;

	inri::Xt xtSrc(src);
	inri::Xt xtDst(dst, xtSrc);
	return true;
}

QString RgtStackingWidget::getDipxyPath()
{
	if ( !m_orientationWidget->getComputationChecked() )
	{
		return m_orientationWidget->getDipxyPath();
	}
	else
	{
		QString seismicPath = m_seismicFileSelectWidget->getPath();
		int lastPoint = seismicPath.lastIndexOf(".");
		QString prefix = seismicPath.left(lastPoint);
		QString out = prefix + "_" + m_orientationWidget->getDipxyFilename() + ".xt";
		return out;
	}
}

QString RgtStackingWidget::getDipxzPath()
{
	if ( !m_orientationWidget->getComputationChecked() )
	{
		return m_orientationWidget->getDipxzPath();
	}
	else
	{
		QString seismicPath = m_seismicFileSelectWidget->getPath();
		int lastPoint = seismicPath.lastIndexOf(".");
		QString prefix = seismicPath.left(lastPoint);
		QString out = prefix + "_" + m_orientationWidget->getDipxzFilename() + ".xt";
		return out;
	}
}

QString RgtStackingWidget::getRgtPath()
{
	QString seismicPath = m_seismicFileSelectWidget->getPath();
	int lastPoint = seismicPath.lastIndexOf(".");
	QString prefix = seismicPath.left(lastPoint);
	QString out = prefix + "_" + getRgtSuffix() + ".xt";
	return out;
}

int RgtStackingWidget::checkFieldsForCompute()
{
	QString filename1, filename2;
	QMessageBox *msgBox = new QMessageBox(parentWidget());

	QString seismicPath = m_seismicFileSelectWidget->getPath();
	if ( seismicPath == "" )
	{
		msgBox->setText("warning");
		msgBox->setInformativeText("You have to specify a seismic filename");
		msgBox->setStandardButtons(QMessageBox::Ok);
		msgBox->exec();
		return 0;
	}

	if ( m_orientationWidget->getComputationChecked() )
	{
		QString dipxySuffix = m_orientationWidget->getDipxyFilename();
		QString dipxzSuffix = m_orientationWidget->getDipxzFilename();
		if ( dipxySuffix.isEmpty() || dipxySuffix.isEmpty() )
		{
			msgBox->setText("warning");
			msgBox->setInformativeText("You have to specify dip suffix");
			msgBox->setStandardButtons(QMessageBox::Ok);
			msgBox->exec();
			return 0;
		}
	}
	else
	{
		QString dipxyPath = m_orientationWidget->getDipxyPath();
		QString dipxzPath = m_orientationWidget->getDipxzPath();

		if ( dipxyPath.isEmpty() || dipxzPath.isEmpty() )
		{
			msgBox->setText("warning");
			msgBox->setInformativeText("You have to specify dip filename");
			msgBox->setStandardButtons(QMessageBox::Ok);
			msgBox->exec();
			return 0;
		}
		inri::Xt xt((char*)seismicPath.toStdString().c_str());
		if ( !xt.is_valid() ) return 0;
		inri::Xt xtDipx((char*)dipxyPath.toStdString().c_str());
		if ( !xtDipx.is_valid() ) return 0;
		inri::Xt xtDipz((char*)dipxzPath.toStdString().c_str());
		if ( !xtDipz.is_valid() ) return 0;
		if ( xt.nSamples() != xtDipx.nSamples() ||
			 xt.nSamples() != xtDipz.nSamples() ||
			 xt.nRecords() != xtDipx.nRecords() ||
			 xt.nRecords() != xtDipz.nRecords() ||
			 xt.nSlices() != xtDipx.nSlices() ||
			 xt.nSlices() != xtDipz.nSlices() )
			return 0;
	}
	return 1;
}


double RgtStackingWidget::qt_cuda_needed_memory(int *size, int decim, int rgt_format, int nbsurfaces, bool polarity)
{
	double ret = 0.0;
	float cpumem = 0.0f;
	float gpumem = 0.0f;
	surface_stack_get_memory(1, 1, size[1], size[0], size[2], 1, nbsurfaces, rgt_format, &cpumem, &gpumem);
	ret = gpumem/1e9;
	return ret;
}

double RgtStackingWidget::qt_ram_needed_memory(int nbthreads, int *size, int decim, int sizeof_stack, int nbsurfaces, bool polarity)
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

void RgtStackingWidget::sizeRectifyWithTraceLimits(int *size, int *sizeX)
{
	int traceLimitX1 = m_rgtStackParameters->getXlimit1();
	int traceLimitX2 = m_rgtStackParameters->getXlimit2();
	int x1 = traceLimitX1;
	int x2 = traceLimitX2;
	if ( x1 < 0 ) x1 = 0;
	if ( x2 < 0 ) x2 = size[1]-1;
	sizeX[0] = size[0];
	sizeX[1] = x2-x1+1;
	sizeX[2] = size[2];
}


int RgtStackingWidget::checkMemoryForCompute()
{
	int ret = 1;
	int nbthreads = 30; //todo
	int stack_format = SURFACE_STACK_FORMAT_SHORT;
	// int decim = m_rgtStackParameters->getDecimation(); // this->decimation_factor;
	// int nbsurfaces = listwidget_horizons->count()+2;
	int decim = getDecimation();
	int nbsurfaces = m_rgtStackParameters->getHorizonPath().size() + 2;
//	QString filename;
//	if ( m_orientationWidget->getComputationChecked() )
//	{
//		filename = getSeismicPath();
//	}
//	else
//	{
//		filename = m_orientationWidget->getDipxyPath();
//	}
//	if ( filename == "" ) return -1;
	int size[3];
	inri::Xt xt(m_seismicFileSelectWidget->getPath().toStdString().c_str());
	if ( !xt.is_valid() ) return 0;
	size[0] = xt.nRecords();
	size[1] = xt.nSamples();
	size[2] = xt.nSlices();
	if ( getCpuGpu() == 1 )
	{
		double cuda_mem = m_systemInfo->cuda_min_free_memory();
		double gpu_need = qt_cuda_needed_memory(size, 1, stack_format, nbsurfaces, true);
		int decim_factor0 = (int)ceil(sqrt(gpu_need/(cuda_mem*.90)));
		if ( decim_factor0 >  getDecimation() )
		{
			QMessageBox *msgBox = new QMessageBox(parentWidget());
			QString txt;
			txt = "The computation needs " + QString::number(gpu_need, 'f', 2) + " GB for the GPU process.\nAnd you have only " + QString::number(cuda_mem, 'f', 2) + " GB free\n";
			txt += "Do you want to modify the decimation factor and put it to the value of " + QString::number(decim_factor0, 'd', 0) + " ?";
			msgBox->setText("Memory problem                              -");
			msgBox->setInformativeText(txt);
			msgBox->setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel );
			int ret0 = msgBox->exec();
			if ( ret0 == QMessageBox::Ok ){ setDecimation(decim_factor0); ret = 1; }
			if ( ret0 == QMessageBox::Cancel ){ ret = 0; }
		}
	}
	else
	{
		double ram_mem = m_systemInfo->qt_cpu_free_memory();
		double ram_need = qt_ram_needed_memory(nbthreads, size, 1, stack_format, nbsurfaces, 1);
		int decim_factor0 = (int)ceil(sqrt(ram_need/(ram_mem*.90)));
		if ( decim_factor0 !=  getDecimation() )
		{
			QMessageBox *msgBox = new QMessageBox(parentWidget());
			QString txt;
			txt = "The computation needs " + QString::number(ram_need, 'f', 2) + " GB ram memory.\nAnd you have only " + QString::number(ram_mem, 'f', 3) + "GB free\n";
			txt += "Do you want to modify the decimation factor and put it at the value of " + QString::number(decim_factor0, 'd', 0) + " ?";
			msgBox->setText("Memory problem.");
			msgBox->setInformativeText(txt);
			msgBox->setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
			int ret0 = msgBox->exec();
			if ( ret0 == QMessageBox::Ok ){ setDecimation(decim_factor0); ret = 1; }
			if ( ret0 == QMessageBox::Cancel ){ret = 0; }
		}
	}
	return ret;
}


void RgtStackingWidget::trt_launch()
{
	if ( pStatus == 0 )
	{
		if ( checkFieldsForCompute() == 0 ) return;
		if ( checkMemoryForCompute() == 0 ) return;
		RgtStackingWidget::MyThread0 *thread = new RgtStackingWidget::MyThread0(this);
		thread->start();
	}
	else if ( pStatus == 1 )
	{
		QMessageBox *msgBox = new QMessageBox(parentWidget());
		msgBox->setText("warning");
		msgBox->setInformativeText("Do you really want to stop the processing ?");
		msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
		int ret = msgBox->exec();
		if ( ret == QMessageBox::Yes )
		{
			if ( pIhm ) pIhm->setMasterMessage("stop", 0, 0, GeotimeFlags::RGT_STACKING_STOP);
			QMessageBox *msgBox2 = new QMessageBox(parentWidget());
			msgBox2->setText("Info");
			msgBox2->setInformativeText("The save procedure will start after finishing the seeds of the current trace\nIt can take few seconds");
			msgBox2->setStandardButtons(QMessageBox::Ok);
			msgBox2->exec();
		}
	}
}

void RgtStackingWidget::trt_compute()
{
	// bool dip_compute = m_orientationWidget->getComputationChecked();
	bool rgtStackingEnable = getCompute();
	// this->bool_abort = 0;
	// window_enable(false);
	// this->pushbutton_compute->setEnabled(false);

	int size[3];
	bool processOk = true;

	if ( getCompute() )
	{
		if (  checkGpuTextureSize() == false )
			return;
	}

	// if ( !dipFilenameUpdate() ) return;
	if ( m_orientationWidget->getComputationChecked() )
	{
		processOk = dipCompute();
		m_projectManagerWidget->seimsicDatabaseUpdate();
	}

	/*
	if ( checkGpuTextureSize() == FAIL ) {
		window_enable(true);
		return;
	}
	*/

	if ( processOk && getCompute() )
	{
		rgtStackingCompute();
		m_projectManagerWidget->seimsicDatabaseUpdate();
	}

	/*
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
	*/
}


bool RgtStackingWidget::dipCompute()
{
	// cuda_props_print(stderr, 0);
	void *p = nullptr;
	double sigmag = m_orientationWidget->getGradient(); // this->sigmagradient;
	double sigmat = m_orientationWidget->getTensor(); // this->sigmatensor;
	int size[3];
	int *tab_gpu = NULL, tab_gpu_size;
	tab_gpu = (int*)calloc(m_systemInfo->get_gpu_nbre(), sizeof(int));
	m_systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	int dip_cpu_gpu = m_orientationWidget->getProcessingTypeIndex();
	int nbthreads = 30;

	QString seismicPath = m_seismicFileSelectWidget->getPath();
	inri::Xt xtSeismic((char*)seismicPath.toStdString().c_str());
	size[0] = xtSeismic.nRecords();
	size[1] = xtSeismic.nSamples();
	size[2] = xtSeismic.nSlices();

	QString dipxyPath = getDipxyPath();
	QString dipxzPath = getDipxzPath();

	char dipxyFilename[10000];
	char dipxzFilename[10000];
	char seismicFilename[10000];
	char *p_dipxyFilename = &dipxyFilename[0];
	char *p_dipxzFilename = &dipxzFilename[0];
	char *p_seismicFilename = &seismicFilename[0];
	strcpy(p_dipxyFilename, (char*)dipxyPath.toStdString().c_str());
	strcpy(p_dipxzFilename, (char*)dipxzPath.toStdString().c_str());
	strcpy(p_seismicFilename, (char*)seismicPath.toStdString().c_str());

	createDataSet((char*)seismicPath.toStdString().c_str(), (char*)dipxyPath.toStdString().c_str());
	createDataSet((char*)seismicPath.toStdString().c_str(), (char*)dipxzPath.toStdString().c_str());

	p = normal_init();
	normal_set_gpu_list(p, tab_gpu, tab_gpu_size);

	normal_set_source_size(p, size[0], size[1], size[2]);
	normal_set_tile_size(p, 16, 16, 16);
	normal_set_block_size(p, 256, 256, 256);
	// normal_set_block_size(p, 64, 64, 64);
	normal_set_datain_filename(p, p_seismicFilename);
	normal_set_sigma_grad(p, sigmag);
	normal_set_sigma_tens(p, sigmat);
	normal_set_nb_scales(p, 1);
	normal_set_out_filename(p, &p_dipxyFilename, &p_dipxzFilename, NULL, NULL);
	normal_set_output_type(p, NORMAL_OUTPUT_TYPE_DIP);
	normal_set_output_format(p, NORMAL_OUTPUT_FORMAT_16BITS);
	normal_set_type(p, NORMAL_TYPE_INLINE);
	normal_set_cpu_gpu(p, dip_cpu_gpu);
	normal_enable_chronos(p, 1);
	normal_set_nb_threads(p, nbthreads);
	normal_set_dip_type(p, 1);
	normal_set_data_cache_enable(p, 1);
	normal_set_ihm(p, (void*)pIhm);
	// normal_framwork_thread_run(p);
//	GLOBAL_RUN_TYPE = 1;
	pStatus = 1;
	bool ret = normal_framwork_run(p);
	p = normal_release(p);
	pStatus = 0;
	m_projectManagerWidget->seimsicDatabaseUpdate();
	free(tab_gpu);
	return ret;
}

int RgtStackingWidget::rgtStackingCompute()
{
	int processing_size[3], polarity_size[3], size[3];

	QString seismicPath = m_seismicFileSelectWidget->getPath();
	inri::Xt xtSeismic((char*)seismicPath.toStdString().c_str());
	size[0] = xtSeismic.nRecords();
	size[1] = xtSeismic.nSamples();
	size[2] = xtSeismic.nSlices();

	processing_size[0] = size[0] / getDecimation(); // this->decimation_factor;
	processing_size[1] = size[1];
	processing_size[2] = size[2] / getDecimation(); // this->decimation_factor;

	char rgtFilename[10000], rgtFilename2[10000];
	char seismicFilename[10000];
	// QString qs_seismic_filename = getSeismicPath();
	int *tab_gpu = NULL, tab_gpu_size;
	tab_gpu = (int*)calloc(m_systemInfo->get_gpu_nbre(), sizeof(int));
	m_systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	int nbthreads = 32;

	strcpy(seismicFilename, (char*)(seismicPath.toStdString().c_str()));
	// float horizon_t0 = (float)this->lineedit_horizont0->text().toInt();
	// float horizon_t1 = (float)this->lineedit_horizont1->text().toInt();

	int onlySeedInsideHorizons = getPropagationSeedOnHorizon() ? 1 : 0;
	QString rgtPath = getRgtPath();
	strcpy(rgtFilename, (char*)rgtPath.toStdString().c_str());
	strcpy(seismicFilename, (char*)seismicPath.toStdString().c_str());

	QString dipxyPath = getDipxyPath();
	QString dipxzPath = getDipxzPath();
	char dipxyFilename[10000];
	char dipxzFilename[10000];
	char *p_dipxyFilename = &dipxyFilename[0];
	char *p_dipxzFilename = &dipxzFilename[0];
	char *p_seismicFilename = &seismicFilename[0];
	char *p_rgtFilename = &rgtFilename[0];
	strcpy(p_dipxyFilename, (char*)dipxyPath.toStdString().c_str());
	strcpy(p_dipxzFilename, (char*)dipxzPath.toStdString().c_str());
	strcpy(p_seismicFilename, (char*)seismicPath.toStdString().c_str());
	// inri::Xt xtRgt(rgtFilename, xtSeismic);
	createDataSet(p_seismicFilename, p_rgtFilename);
	m_projectManagerWidget->seimsicDatabaseUpdate();

	int bool_polarity = 1; // this->qcb_rgtstackpolarity->isChecked();
	int bool_mask2d = 1;

	int dimx_step = 20; //this->lineedit_stepdimy->text().toInt();
	int dimy_step = 100; //this->lineedit_stepdimx->text().toInt();
	int dimz_step = 100; //this->lineedit_stepdimz->text().toInt();
	int stack_cpu_gpu = getCpuGpu(); //this->qcb_rgtstackgpu->isChecked();
	char **horizon_filename0 = nullptr;

	// todo
	int nbhorizons = 0;
	fprintf(stderr, "nbre: %d\n", nbhorizons);

	// if ( traceLimitX1 >= 0 || traceLimitX2 >= 0 )
	//	fprintf(stderr, "trace limits: %d %d\n", traceLimitX1, traceLimitX2);

	int nativeSize[3], nativeSizeLimit[3];
	int processingSize[3], processingSizeLimit[3];

	for (int i=0; i<3; i++) nativeSize[i] = size[i];

	// todo
	for (int i=0; i<3; i++) processingSizeLimit[i] = size[i];

	// todo
	sizeRectifyWithTraceLimits(nativeSize, nativeSizeLimit);

	processingSize[0] = nativeSize[0] / getDecimation(); // this->decimation_factor;
	processingSize[1] = nativeSize[1];
	processingSize[2] = nativeSize[2] / getDecimation(); // this->decimation_factor;
	sizeRectifyWithTraceLimits(processingSize, processingSizeLimit);

	void *p = surface_stack_init();

	RgtStackingComputationOperator *rgtStackingOperator = new RgtStackingComputationOperator(m_seismicFileSelectWidget->getPath().toStdString(), rgtPath.toStdString(), m_rgtSuffix->text().toStdString());
	rgtStackingOperator->setSurveyPath(m_projectManagerWidget->getSurveyPath());
	rgtStackingOperator->setRgtStacking(p);
	if ( m_processRelay) m_processRelay->addProcess(rgtStackingOperator);

	int stack_format = SURFACE_STACK_FORMAT_SHORT;
	surface_stack_set_gpu_list(p, tab_gpu, m_systemInfo->get_gpu_nbre());
	surface_stack_set_dip_threshold(p, getDipThreshold());

		// deprecated ?
	surface_stack_set_xgrid(p, 0, processing_size[0] - 1, std::max(1, dimx_step/getDecimation())); // decimation_factor));
		// surface_stack_set_ygrid(p, 0, processing_size[1] - 1, MAX(1, dimy_step/decimation_factor));
	surface_stack_set_ygrid(p, 0, processing_size[1] - 1, std::max(1, dimy_step));
	surface_stack_set_zgrid(p, 0, processing_size[2] - 1, std::max(1, dimz_step/getDecimation())); // decimation_factor));DEBUG0()

	surface_stack_set_nbthreads(p, nbthreads);
	surface_stack_domain_set(p, 0, processingSizeLimit[0] - 1, processingSizeLimit[0], 0, processingSizeLimit[2] - 1, processingSizeLimit[2]);
	surface_stack_iteration_set(p, getIter());
	surface_stack_dip_set_dims(p, processingSizeLimit[0], processingSizeLimit[1], processingSizeLimit[2]);

		// surface_starck_dip_set_dips(p, (void*)v_dipx, (void*)v_dipz, dip_precision);
	surface_stack_set_dipx_filename(p, p_dipxyFilename);
	surface_stack_set_dipz_filename(p, p_dipxzFilename);
	surface_stack_set_seismic_filename(p, p_seismicFilename);

	surface_stack_set_native_size(p, nativeSize[0], nativeSize[1], nativeSize[2]);
	surface_stack_set_data_in_filename(p, nullptr);
	surface_stack_set_decimation_factor(p, getDecimation()); // this->decimation_factor);
	surface_stack_set_stack_format(p, stack_format);
	surface_stack_set_cuda(p, stack_cpu_gpu);

	// surface_stack_set_polarity_data(p, pol, polarity_size);
	// surface_stack_set_mask(p, v_mask);
	surface_stack_set_bool_polarity(p, bool_polarity);
	surface_stack_set_bool_mask2d(p, bool_mask2d);

	// surface_stack_set_stack_crit_filename(p, stack_crit_filename);
	// if ( strlen(stack_filename) != 0 ) surface_stack_set_stack_filename(p, stack_filename); else surface_stack_set_stack_filename(p, nullptr);

	surface_stack_set_rgt_filename(p, p_rgtFilename);
	surface_stack_set_rgt_filename2(p, nullptr);
	surface_stack_set_sigma_stack(p, 1.0);
	surface_stack_set_enable_partial_rgt_save(p, 1);
			// deprecated ?
	surface_stack_set_rgt_saverate(p, 4);

		// horizons
	// todo
	std::vector<QString> v_horizon_filename;// = m_horizonSelectWidget->getPaths();
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

	// surface_stack_set_horizon_t0(p, horizon_t0);
	// surface_stack_set_horizon_t1(p, horizon_t1);
	surface_stack_set_bool_snapping(p, getSnapping());
	surface_stack_set_rgt_cwtcompressionerror(p, 0.001);

	if ( getEnableSeedMax() )
		surface_stack_set_seedMax(p, getSeedMax());
	else
		surface_stack_set_seedMax(p, -1);

	float seismic_step_sample = xtSeismic.stepSamples();
	float seismic_start_sample = xtSeismic.startSamples();
	surface_stack_set_seismic_start_sample(p, seismic_start_sample);
	surface_stack_set_seismic_step_sample(p, seismic_step_sample);
	surface_stack_set_only_seeds_inside_horizons(p, onlySeedInsideHorizons);

	// todo
	// surface_stack_set_trace_limits(p, -1, -1);
	// surface_stack_set_msg_display(p, &msgDisplay0);

	int traceLimitX1 = m_rgtStackParameters->getXlimit1();
	int traceLimitX2 = m_rgtStackParameters->getXlimit2();
	surface_stack_set_trace_limits(p, traceLimitX1, traceLimitX2);
	surface_stack_set_ihm(p, pIhm);
	pStatus = 1;
	surface_stack_run(p);
	pStatus = 0;
	m_projectManagerWidget->seimsicDatabaseUpdate();
	// surface_stack_debug_test_minmax(p);
	// surface_stack_debug_test_minmax(pol, polarity_size);
	p = surface_stack_release(p);
	// FREE(v_dipx)
	// FREE(v_dipz)
	// FREE(pol)
	// void *p = normal_init();
	// normal_framwork_run(p);


	free(tab_gpu);

	m_processRelay->removeProcess(rgtStackingOperator);
	delete rgtStackingOperator;

	// this->pushbutton_compute->setEnabled(true);
	// for (int i=0; i<nbhorizons; i++) if ( horizon_filename0[i] != NULL ) free(horizon_filename0[i]);
	//	if ( horizon_filename0 != NULL ) free(horizon_filename0);
	//	this->bool_abort = 0;
	//	window_enable(true);
	//	GLOBAL_RUN_TYPE = 0;
	return 1;
}





RgtStackingWidget::MyThread0::MyThread0(RgtStackingWidget *p)
{
	this->pp = p;
}

void RgtStackingWidget::MyThread0::run()
{
	pp->trt_compute();
}


void RgtStackingWidget::trt_rgtSave()
{
	if ( pIhm ) pIhm->setMasterMessage("write", 0, 0, GeotimeFlags::RGT_STACKING_SAVE);
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("Info");
	msgBox->setInformativeText("The save procedure will start after finishing the seeds of the current trace\nIt can take few seconds");
	msgBox->setStandardButtons(QMessageBox::Ok);
	msgBox->exec();

}

void RgtStackingWidget::trt_rgtGraph_Kill()
{
	if ( pStatus == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		QMessageBox *msgBox2 = new QMessageBox(parentWidget());
		msgBox2->setText("Information");
		msgBox2->setInformativeText("The killing process could take few seconds due to data unloading.");
		msgBox2->setStandardButtons(QMessageBox::Ok);
		if ( pIhm )
		{
			pIhm->setMasterMessage("kill", 0, 0, GeotimeFlags::MASTER_KILL);
		}
		msgBox2->exec();
	}
}

void RgtStackingWidget::processingDisplay()
{
	if ( pStatus == 0 )
	{
		m_progress->setValue(0); m_progress->setFormat(""); m_processing->setText("waiting ...");
		m_processing->setStyleSheet("QLabel { color : white; }");
		return;
	}
	m_processing->setText("PROCESSING");
	m_cptProcessing++;
	if ( m_cptProcessing%2 == 0 )
	{
		m_processing->setStyleSheet("QLabel { color : red; }");
	}
	else
	{
		m_processing->setStyleSheet("QLabel { color : white; }");
	}
}



void RgtStackingWidget::showTime()
{
	if ( pIhm == nullptr ) return;
	processingDisplay();
	startStopConfigDisplay();
	if ( pIhm->isSlaveMessage() )
	{
		Ihm2Message mess = pIhm->getSlaveMessage();
		std::string message = mess.message;
		long count = mess.count;
		long countMax = mess.countMax;
		int trtId = mess.trtId;
		bool valid = mess.valid;
		float val = 100.0*count/countMax;
		QString barMessage = QString(message.c_str()) + " [ " + QString::number(val, 'f', 1) + " % ]";
		m_progress->setValue((int)val);
		m_progress->setFormat(barMessage);
		// qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
	}
	std::vector<std::string> mess = pIhm->getSlaveInfoMessage();
	for (int n=0; n<mess.size(); n++)
	{
		m_textInfo->appendPlainText(QString(mess[n].c_str()));
	}
}


void RgtStackingWidget::seismicFilenameChanged()
{
	QString seismicFilename = m_seismicFileSelectWidget->getPath();
	inri::Xt xt(seismicFilename.toStdString().c_str());
	int dimx = xt.nSamples();
	int dimy = xt.nRecords();
	int dimz = xt.nSlices();
	m_orientationWidget->setConstraintsDims(dimx, dimy, dimz);
}



bool RgtStackingWidget::checkGpuTextureSize()
{
	int cpuGpu = getCpuGpu();
	if ( cpuGpu == 0 ) return true;
	// todo
	int tab_gpu[] = {0,0,0,0,0}, tab_gpu_size = 1;
	// tab_gpu = (int*)calloc(this->systemInfo->get_gpu_nbre(), sizeof(int));
	// this->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	int maxSize;
	int textureMaxSize[3];
	cudaGetMaxTexture3D(textureMaxSize, tab_gpu[0]);
	maxSize = textureMaxSize[0];
	for (int i=1; i<tab_gpu_size; i++)
	{
		cudaGetMaxTexture3D(textureMaxSize, tab_gpu[0]);
		maxSize = MIN((int)textureMaxSize[0], (int)maxSize);
	}
	QString seismicFilename = m_seismicFileSelectWidget->getPath();
	inri::Xt xt(seismicFilename.toStdString().c_str());
	int dimx = xt.nSamples();

	int traceLimitX1 = m_rgtStackParameters->getXlimit1();
	int traceLimitX2 = m_rgtStackParameters->getXlimit2();
	int xx1 = traceLimitX1; if ( xx1 < 0 ) xx1 = 0;
	int xx2 = traceLimitX2; if ( xx2 < 0 ) xx2 = dimx-1;
	int dimxx = xx2-xx1+1;
	if ( dimxx <= maxSize ) return true;

	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	QString txt = "Your memory GPU is lower than the time direction size [ " + QString::number(dimx) + " / " + QString::number(maxSize) + " ]\n";
	txt += "You have to reduce the processing size of your data (options limit x1 and limit x2)";
	msgBox->setInformativeText(txt);
	msgBox->setStandardButtons(QMessageBox::Close);
	msgBox->exec();
	return false;
}




