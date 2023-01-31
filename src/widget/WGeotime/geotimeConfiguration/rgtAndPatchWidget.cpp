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
#include <QDir>
#include <QDateTime>
#include <QTime>
#include <QStringLiteral>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDebug>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <QMessageBox>
#include <sys/sysinfo.h>


#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
#include <ProjectManagerWidget.h>
#include <workingsetmanager.h>
// #include "FileConvertionXTCWT.h"
#include "collapsablescrollarea.h"
#include <patchWidget.h>
#include <rgtPatchWidget.h>
#include <orientationWidget.h>
#include <rgtAndPatchWidget.h>
#include <normal.h>
#include <ihm2.h>
#include <rgtGraph.h>
#include <GeotimeSystemInfo.h>
#include <fileSelectWidget.h>
#include "processrelay.h"
#include <rgtVolumicVolumeComputationOperator.h>
#include <rgtVolumicCPU.hpp>

#include "Xt.h"



RgtAndPatchWidget::RgtAndPatchWidget(WorkingSetManager *workingSetManager, QWidget* parent):QWidget(parent)
{
	m_workingSetManager = workingSetManager;
	m_projectManager = m_workingSetManager->getManagerWidgetV2();
	QVBoxLayout *layoutMain = new QVBoxLayout(this);

	m_processing = new QLabel(".");

	m_seismicFileSelectWidget = new FileSelectWidget();
	m_seismicFileSelectWidget->setProjectManager(m_projectManager);
	m_seismicFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_seismicFileSelectWidget->setLabelText("seismic filename");
	m_seismicFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Seismic);
	m_seismicFileSelectWidget->setLabelDimensionVisible(false);
	m_seismicFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);

	QGroupBox *qgb_orientation = new QGroupBox(this);
	qgb_orientation->setTitle("Orientation");
	// qgb_orientation->setMaximumHeight(300);

	QVBoxLayout *qvb_orientation = new QVBoxLayout(qgb_orientation);
	m_orientationWidget = new OrientationWidget(m_projectManager, true, m_workingSetManager);
	qvb_orientation->addWidget(m_orientationWidget);
	qgb_orientation->setAlignment(Qt::AlignTop);

	m_patchWidget = new PatchWidget(m_workingSetManager);
	m_rgtPatchWidget = new RgtPatchWidget(m_workingSetManager);

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
	m_scaleStop = new QPushButton("scale stop");
	m_kill = new QPushButton("kill process");
	m_launchBatch = new QPushButton("run batch");
	m_preview = new QPushButton("preview");

	m_launchBatch->setEnabled(false);

	qhb_buttons->addWidget(m_startStop);
	qhb_buttons->addWidget(m_save);
	qhb_buttons->addWidget(m_scaleStop);
	qhb_buttons->addWidget(m_kill);
	qhb_buttons->addWidget(m_launchBatch);
	// qhb_buttons->addWidget(m_preview);

	qvb_control->addWidget(m_textInfo);
	qvb_control->addWidget(m_progress);
	qvb_control->addLayout(qhb_buttons);


	layoutMain->addWidget(m_processing);
	layoutMain->addWidget(m_seismicFileSelectWidget);
	layoutMain->addWidget(qgb_orientation);
	layoutMain->addWidget(m_patchWidget);
	layoutMain->addWidget(m_rgtPatchWidget);
	layoutMain->addWidget(qgb_control);
	layoutMain->setAlignment(Qt::AlignTop);

	if ( pIhm == nullptr ) pIhm = new Ihm2();
	QTimer *timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	connect(m_startStop, SIGNAL(clicked()), this, SLOT(trt_launch()));

	connect(m_scaleStop, SIGNAL(clicked()), this, SLOT(trt_scaleStop()));
	connect(m_save, SIGNAL(clicked()), this, SLOT(trt_save()));
	connect(m_kill, SIGNAL(clicked()), this, SLOT(trt_kill()));
	connect(m_launchBatch, SIGNAL(clicked()), this, SLOT(trt_launchBatch()));
	connect(m_seismicFileSelectWidget, &FileSelectWidget::filenameChanged, this, &RgtAndPatchWidget::seismicFilenameChanged);


}

RgtAndPatchWidget::~RgtAndPatchWidget()
{

}

void RgtAndPatchWidget::setSystemInfo(GeotimeSystemInfo *val)
{
	m_systemInfo = val;
}

void RgtAndPatchWidget::setPatchCompute(bool val)
{}

void RgtAndPatchWidget::setPatchSuffix(QString val)
{}

void RgtAndPatchWidget::setPatchEnableFaultInput(bool val)
{}

void RgtAndPatchWidget::setPatchSize(int val)
{}

void RgtAndPatchWidget::setPatchPolarity(int val)
{}

void RgtAndPatchWidget::setPatchGradMax(int val)
{}

void RgtAndPatchWidget::setPatchDvOverV(double val)
{}

void RgtAndPatchWidget::setPatchRatio(int val)
{}

void RgtAndPatchWidget::setPatchMaskThreshold(int val)
{

}

bool RgtAndPatchWidget::getPatchCompute()
{
}

QString RgtAndPatchWidget::getPatchSuffix()
{
	return m_rgtPatchWidget->getRgtSuffix();

}

bool RgtAndPatchWidget::getPatchEnableFaultInput()
{}

int RgtAndPatchWidget::getPatchSize()
{
	return m_patchWidget->getPatchSize();
}

QString RgtAndPatchWidget::getPatchPolarity()
{
	return  m_patchWidget->getPatchPolarity();
}

int RgtAndPatchWidget::getPatchGradMax()
{
	return m_patchWidget->getGradMax();
}

double RgtAndPatchWidget::getPatchDvOverV()
{
	return m_patchWidget->getDeltaVOverV();
}

int RgtAndPatchWidget::getPatchRatio()
{}

int RgtAndPatchWidget::getPatchMaskThreshold()
{
	return m_patchWidget->getFaultThreshold();
}


bool RgtAndPatchWidget::getFaultMaskInput()
{
	return m_patchWidget->getFaultMaskInput();
}


QString RgtAndPatchWidget::getFaultMaskPath()
{
	return m_patchWidget->getFaultMaskPath();
}

std::vector<QString> RgtAndPatchWidget::getHorizonPaths()
{
	return m_patchWidget->getHorizonPaths();
}


int RgtAndPatchWidget::getScaleInitIter()
{
	return m_rgtPatchWidget->getScaleInitIter();
}

double RgtAndPatchWidget::getScaleInitEpsilon()
{
	return m_rgtPatchWidget->getScaleInitEpsilon();
}

int RgtAndPatchWidget::getScaleInitDecim()
{
	return m_rgtPatchWidget->getScaleInitDecim();
}

int RgtAndPatchWidget::getIter()
{
	return m_rgtPatchWidget->getIter();
}

double RgtAndPatchWidget::getEpsilon()
{
	return m_rgtPatchWidget->getEpsilon();
}

int RgtAndPatchWidget::getDecim()
{
	return m_rgtPatchWidget->getDecim();
}

bool RgtAndPatchWidget::getScaleInitValid()
{
	return m_rgtPatchWidget->getScaleInitValid();
}


void RgtAndPatchWidget::setProcessRelay(ProcessRelay* relay) {
	m_processRelay = relay;
	// if ( m_rgtPatchWidget ) m_rgtPatchWidget->
}


QString RgtAndPatchWidget::getDipxyPath()
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

QString RgtAndPatchWidget::getDipxzPath()
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


QString RgtAndPatchWidget::getDatasetFormat(QString filename)
{
	inri::Xt xt((char*)filename.toStdString().c_str());
	if ( !xt.is_valid() ) return "invalid";
	inri::Xt::Type type = xt.type();
	return QString::fromStdString(xt.type2str(type));
}


int RgtAndPatchWidget::checkFieldsForCompute()
{
	QString seismicPath = m_seismicFileSelectWidget->getPath();
	if ( m_seismicFileSelectWidget->getPath().isEmpty() ) { displayWarning("you have to specify a dataset"); return 0; }
	if ( m_orientationWidget->getComputationChecked() ) { }
	if ( getDatasetFormat(m_seismicFileSelectWidget->getPath()) != "Signed_16" ) { displayWarning("the dataset must be in short format"); return 0; }
	if ( m_patchWidget->getCompute() )
	{
		if ( m_patchWidget->getPatchName().isEmpty() ) { displayWarning("you have to specify a patch name"); return 0; }
	}
	if ( m_rgtPatchWidget->getCompute() )
	{
		if ( !m_orientationWidget->getComputationChecked() )
		{
			if ( !datasetValid(getDipxyPath()) ) { displayWarning("dataset dipxy is not valid"); return 0; }
			if ( !datasetValid(getDipxzPath()) ) { displayWarning("dataset dipxz is not valid"); return 0; }
			if ( !fitDatasetSize(seismicPath, getDipxyPath()) ) { displayWarning("size error between seismic and dipxy"); return 0; }
			if ( !fitDatasetSize(seismicPath, getDipxzPath()) ) { displayWarning("size error between seismic and dipxz"); return 0; }
			if ( getDatasetFormat(getDipxyPath()) != "Signed_16" ) { displayWarning("dipxy be in short format"); return 0; }
			if ( getDatasetFormat(getDipxzPath()) != "Signed_16" ) { displayWarning("dipxz be in short format"); return 0; }
		}
		if ( !m_patchWidget->getCompute() )
		{
			if ( m_patchWidget->getPatchEnable() )
			{
				QString patchFilename = getPatchPath();
				if ( patchFilename.isEmpty()  ) { displayWarning("patch is not valid"); return 0; }
				if ( !fitDatasetSize(seismicPath, patchFilename) ) { displayWarning("size error between seismic and patch"); return 0; }
			}
		}
	}
	return 1;
}

bool RgtAndPatchWidget::checkMemoryForCompute()
{
	return true;
}


int RgtAndPatchWidget::rgtVolumicDecimationFactorEstimation()
{
	int size[3];
	int blocSize[3] = {0,0,0};
	long nVertex = -1;
	long memSize = 0;

	QString seismicPath = m_seismicFileSelectWidget->getPath();
	inri::Xt xt(seismicPath.toStdString().c_str());
	size[0] = xt.nSamples();
	size[1] = xt.nRecords();
	size[2] = xt.nSlices();
	long size0 = (long)size[0]*size[1]*size[2];
	long nbVertex = -1;

	if ( m_rgtPatchWidget->getCompute() )
	{
		nbVertex = getVertexnbreEstimation();
		blocSize[1] = m_patchWidget->getPatchSize(); // patchParam.patchSize;
		blocSize[2] = m_patchWidget->getPatchSize(); // patchParam.patchSize;
	}
	else
	{
		RgtGraphLabelRead::getBlocSizeFromFile(m_patchWidget->getPatchPath().toStdString(), blocSize);
		nVertex = RgtGraphLabelRead::getVertexNbreFromFile(m_patchWidget->getPatchPath().toStdString());
	}
	fprintf(stderr, "patch: size [%d %d] %ld vertices\n", blocSize[1], blocSize[2], nVertex);
	memSize = (long)10 * size0 * sizeof(float) + nVertex * 3 * sizeof(int) * (long)blocSize[1] * (long)blocSize[2];
	double decimD = memSize / (0.9*1e9*m_systemInfo->qt_cpu_free_memory());
	int decim = (int)ceil(sqrt(decimD));
	decim = std::max(1, decim);
	return decim;
}

long RgtAndPatchWidget::getVertexnbreEstimation()
{
	long nbre = -1.0;
	QString seismicFilename =m_seismicFileSelectWidget->getPath();
	QFileInfo file(seismicFilename);
	if ( !file.exists() ) return nbre;
	inri::Xt xt((char*)seismicFilename.toStdString().c_str());
	if ( !xt.is_valid() )  return nbre;
	long dimx = xt.nSamples();
	long dimy = xt.nRecords();
	long dimz = xt.nSlices();
	long nbCylindres = ((dimy-1)/m_patchWidget->getPatchSize()+1) * ((dimy-1)/m_patchWidget->getPatchSize()+1);
	long nbSurfacesPerCylindre = dimx / 4;
	return nbCylindres * nbSurfacesPerCylindre;
}

void RgtAndPatchWidget::trt_launch()
{
	//	if ( checkFieldsForCompute() == 0 ) return;
	//	if ( checkMemoryForCompute() == 0 ) return;

	if ( checkFieldsForCompute() == 0 ) return;
	if ( checkMemoryForCompute() == 0 ) return;
	if ( m_rgtPatchWidget->getCompute() )
	{
		int decim = rgtVolumicDecimationFactorEstimation();
		if ( decim > m_rgtPatchWidget->getDecim() )
		{
			QMessageBox msgBox;
			QString txt = "The process needs to decimate the data with a factor of " + QString::number(decim);
			msgBox.setText(txt);
			msgBox.setInformativeText(tr("Do you want to continue ?"));
			msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
			msgBox.setDefaultButton(QMessageBox::Ok);
			int ret = msgBox.exec();
			if ( ret == QMessageBox::Cancel )return;
			m_rgtPatchWidget->setDecim(decim);
		}
	}

	RgtAndPatchWidget::MyThread0 *thread = new RgtAndPatchWidget::MyThread0(this);
	thread->start();
}

void RgtAndPatchWidget::trt_scaleStop()
{
	if ( pStatus == 0 ) return;

	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to stop this scale process and continue ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		// ihm_set_trt(IHM_TRT_RGT_GRAPH_STOP);
		if ( pIhm )
		{
			pIhm->setMasterMessage("stop", 0, 0, 0);
		}
	}

}

void RgtAndPatchWidget::trt_save()
{
	if ( pStatus == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to save the current result ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		// ihm_set_trt(IHM_TRT_RGT_GRAPH_STOP);
		// this->bool_abort = 1;
		if ( pIhm )
		{
			pIhm->setMasterMessage("save", 0, 0, 0);
		}
	}
}

void RgtAndPatchWidget::trt_kill()
{
	if ( pStatus == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		if ( pIhm )
		{
			pIhm->setMasterMessage("kill", 0, 0, 0);
		}
	}
}

void RgtAndPatchWidget::showTime()
{
	if ( pIhm == nullptr ) return;
	processingDisplay();
	if ( pStatus == 0 ) return;

	if ( pStatus == 99 )
	{
		displayProcessFinish();
		pStatus = 0;
	}

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



QString RgtAndPatchWidget::getPatchPath()
{
	if ( m_patchWidget->getCompute() )
	{
		QString seismicName = m_seismicFileSelectWidget->getFilename();
		QString path = m_seismicFileSelectWidget->getPath();
		int lastPoint = path.lastIndexOf("/");
		path = path.left(lastPoint) + "/" + seismicName;
		path = path + "_" + m_patchWidget->getPatchName() + "_" + QString::number(m_patchWidget->getPatchSize()) + "__nextvisionpatch.xt";
		return path;
		/*
		QString seismicName = m_seismicFileSelectWidget->getFilename();
		QString path = m_projectManager->getNextVisionSeismicPath() + "seismic3d." + seismicName + "_" + m_patchWidget->getPatchName() +"_" + QString::number(m_patchWidget->getPatchSize()) + "__nextvisionpatch.xt";
		return path;
		*/
	}
	else
		return m_patchWidget->getPatchPath();
}

QString RgtAndPatchWidget::getGraphDir()
{
	/*
	// QString seismicFilename = lineedit_seismicfilename->text();
	QString seismicFilename = m_seismicFileSelectWidget->getFilename();
	if ( seismicFilename.isEmpty() ) return "";
	QString patchPath = m_projectManager->getPatchPath();
	if ( patchPath.isEmpty() ) return "";
	return patchPath + seismicFilename + "_" + m_patchWidget->getPatchName() + "_" +  QString::number(m_patchWidget->getPatchSize()) + "/";
	*/
	QString seismicFilename = m_seismicFileSelectWidget->getFilename();
	QString patchPath = m_projectManager->getPatchPath();
	if ( patchPath.isEmpty() ) return "";
	return patchPath + seismicFilename + "_" + m_patchWidget->getPatchName() + "_" +  QString::number(m_patchWidget->getPatchSize()) + "/";
}


QString RgtAndPatchWidget::getGraphPath()
{
	QString path = getGraphDir();
	if ( path.isEmpty() ) return "";
	return path + "graph.bin";
}


QString RgtAndPatchWidget::getPatchDir()
{
	QString path = getGraphDir();
	if ( path.isEmpty() ) return "";
	return path + "patch/";
}

QString RgtAndPatchWidget::getRgtPath()
{
	QString base = m_seismicFileSelectWidget->getPath();
	int lastPoint = base.lastIndexOf(".");
	QString prefix = base.left(lastPoint);
	QString out = prefix + "_" + getPatchSuffix() + ".xt";
	return out;
}




void RgtAndPatchWidget::trt_compute()
{
	m_returnParam.timeOrientationAll = 0.0;
	m_returnParam.timeOrientationRead = 0.0;
	m_returnParam.timeOrientationNormal = 0.0;
	m_returnParam.timeOrientationWrite = 0.0;
	m_returnParam.timePatchPatch = 0.0;
	m_returnParam.timePatchFusion = 0.0;
	m_returnParam.timeRgtAll = 0.0;

	int ret = 1;
	// todo
	// test ok
	if ( m_orientationWidget->getComputationChecked() )
	{
		ret = dipCompute();
		m_projectManager->seimsicDatabaseUpdate();
	}
	if ( m_patchWidget->getCompute() )
	{
		ret = patchCompute();
		m_projectManager->seimsicDatabaseUpdate();
	}
	if ( m_rgtPatchWidget->getCompute() )
	{
		ret = rgtPatchCompute();
		m_projectManager->seimsicDatabaseUpdate();
	}
	pStatus = 99;
}

int RgtAndPatchWidget::dipCompute()
{
	// cuda_props_print(stderr, 0);
	void *p = nullptr;
	double sigmag = m_orientationWidget->getGradient(); // this->sigmagradient;
	double sigmat = m_orientationWidget->getTensor(); // this->sigmatensor;
	int size[3];
	int *tab_gpu = nullptr, tab_gpu_size;
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
	inri::Xt xtDipxy(p_dipxyFilename, xtSeismic);
	inri::Xt xtDipxz(p_dipxzFilename, xtSeismic);

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
	normal_framwork_run(p);

	m_returnParam.timeOrientationAll = normal_chronos_get(p, NORMAL_CHRONOS_ALL);
	m_returnParam.timeOrientationRead = normal_chronos_get(p, NORMAL_CHRONOS_READ);
	m_returnParam.timeOrientationNormal = normal_chronos_get(p, NORMAL_CHRONOS_NORMAL);
	m_returnParam.timeOrientationWrite = normal_chronos_get(p, NORMAL_CHRONOS_WRITE);
	p = normal_release(p);
	pStatus = 0;

	free(tab_gpu);
	return 1;

}

int RgtAndPatchWidget::patchCompute()
{
	int ret = 1;
	bool dip_compute = m_orientationWidget->getComputationChecked();

	std::string label0Filename = getPatchPath().toStdString();
	// std::string label1Filename = graphlabel1FilenameGet().toStdString();
	// std::string labelRawFilename = graphLabelRawFilenameGet().toStdString();
	std::string graphFilename = getGraphPath().toStdString();
	std::string isoPath = getPatchDir().toStdString();

	// rgtVolumicParam.patchFilename = QString::fromStdString(label0Filename);

	m_projectManager->mkdirPatchPath();
	m_projectManager->mkdirNextVisionSeismicPath();
	QDir d;
	if ( !d.mkpath(getPatchDir()) ) { qDebug() << "unable to create directory: " + getPatchDir(); return 0; }

//	QDir dir(getPatchDir());
//	if ( !dir.exists() ) dir.mkpath(".");
	//	fprintf(stderr, "seismic filename:   %s\n", getSeismicPath().toStdString().c_str());
	//	fprintf(stderr, "label0 filename:    %s\n", label0Filename.c_str());
	//	fprintf(stderr, "label1 filename:    %s\n", label1Filename.c_str());
	//	fprintf(stderr, "label raw filename: %s\n", labelRawFilename.c_str());
	//	fprintf(stderr, "graph filename:     %s\n", graphFilename.c_str());
	//	fprintf(stderr, "isoPath filename:   %s\n", isoPath.c_str());
	//	fprintf(stderr, "graphFilename: %s\n", graphFilename.c_str());

	RgtGraph *p = new RgtGraph();
	p->setSeismicFilename(m_seismicFileSelectWidget->getPath().toStdString());
	p->setLabel0Filename(label0Filename);
	p->setLabel1Filename("");
	p->setLabel1RawFilename("");
	p->setLabelChaineFilename(graphFilename);
	p->setIsoPatchFilenamePrefix(isoPath);
	p->setBlocSize(getPatchSize(), getPatchSize());
	if ( getFaultMaskInput() )
		p->setFaultFilename(getFaultMaskPath().toStdString());
	p->setFaultMaskThreshold(getPatchMaskThreshold());
	p->setPatchPolarity(getPatchPolarity().toStdString());
	p->setIhm(pIhm);
	p->setDeltaVoverV(getPatchDvOverV());
	p->setGradMax(getPatchGradMax());
	std::vector<QString> Qnames = getHorizonPaths();
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
	pStatus = 1;
	ret = p->run();
	m_returnParam.timePatchPatch = p->getChronos(0);
	m_returnParam.timePatchFusion = p->getChronos(1);
	pStatus = 0;
	return ret;
}


int RgtAndPatchWidget::rgtPatchCompute()
{
	int ret = 1;
	int size[3];
	inri::Xt xtSeismic((char*)m_seismicFileSelectWidget->getPath().toStdString().c_str());
	size[0] = xtSeismic.nSamples();
	size[1] = xtSeismic.nRecords();
	size[2] = xtSeismic.nSlices();

	QString qsRgtFilename = getRgtPath();
	std::string dyFilename = getDipxyPath().toStdString();
	std::string dzFilename = getDipxzPath().toStdString();
	std::string rgtFilenameT = qsRgtFilename.toStdString();
	// fprintf(stderr, "%s\n%s\n", dyFilename.c_str(), dzFilename.c_str());
	// QString patchMainDirectory = patchMainDirectoryFromPatchName(m_patchFileSelectWidget->getFilename());
	// QString label1Filename = patchMainDirectory + "/" + "label.raw";
	// fprintf(stderr, "graph filename:     %s\n", m_patchFileSelectWidget->getPath().toStdString().c_str());
	// fprintf(stderr, "patchMainDirectory: %s\n", patchMainDirectory.toStdString().c_str());


	// if ( pIhmPatch != nullptr ) { delete pIhmPatch; pIhmPatch = nullptr; };
	// if ( pIhmPatch == nullptr ) pIhmPatch = new Ihm2();
	// QString rgtName = lineedit_patchRgtRgtName->text();


	RgtVolumicComputationOperator *rgtVolumicOperator =	new RgtVolumicComputationOperator(m_seismicFileSelectWidget->getPath().toStdString(), rgtFilenameT, m_rgtPatchWidget->getRgtSuffix().toStdString());
	rgtVolumicOperator->setSurveyPath(m_projectManager->getSurveyPath());
	if ( m_processRelay) m_processRelay->addProcess(rgtVolumicOperator);

	RgtVolumicCPU<float> *p = new RgtVolumicCPU<float>();
	p->setRgtVolumicComputationOperator(rgtVolumicOperator);
	p->setSeismicFilename(m_seismicFileSelectWidget->getPath().toStdString());
	p->setDipxyFilename(dyFilename);
	p->setDipxzFilename(dzFilename);

//	if ( qcb_rgtVolumicRgt0->isChecked() )
//	{
//		p->setRgt0Filename(m_rgtInitSelectWidget->getPath().toStdString());
//		// fprintf(stderr, "rgt0 filename: %s\n", rgtVolumicParam.rgt0Filename.toStdString().c_str());
//	}
//	else
//		p->setRgt0Filename("");

	p->setRgtFilename(rgtFilenameT);
	// p->setlabel1Filename(label1Filename.toStdString());
	QString patchFilename = "";
	if ( m_patchWidget->getPatchEnable() ) patchFilename = getPatchPath();
	p->setPatchFilename(patchFilename.toStdString());
	p->setEpsilon(getEpsilon());
	p->setDecim(getDecim(), getDecim());
	p->setNbIter(getIter());

	int arrayDecim[2];
	int arrayNbIter[2];
	double arrayEpsilon[2];
	int nbScales = 0;
	if ( !getScaleInitValid() )
	{
		nbScales = 1;
		arrayDecim[0] = getDecim();
		arrayNbIter[0] = getIter();
		arrayEpsilon[0] = getEpsilon();
	}
	else
	{
		nbScales = 2;
		arrayDecim[0] = getScaleInitDecim();
		arrayNbIter[0] = getScaleInitIter();
		arrayEpsilon[0] = getScaleInitEpsilon();
		arrayDecim[1] = getDecim();
		arrayNbIter[1] = getIter();
		arrayEpsilon[1] = getEpsilon();
	}
	p->setNbScales(nbScales);
	p->setArrayDecim(arrayDecim);
	p->setArrayEpsilon(arrayEpsilon);
	p->setArrayNbIter(arrayNbIter);
	p->setIhm(pIhm);
	// p->setIdleDipMax(rgtVolumicParam.idleDipMax);
	p->setDepthMax(-1);
	if ( m_rgtPatchWidget->getIsRgtInit() )
	{
		p->setRgt0Filename(m_rgtPatchWidget->getRgtInit().toStdString());
	}
	pStatus = 1;
	p->runScale();

	m_returnParam.timeRgtAll = p->getChronos(0);

	pStatus = 0;
	delete p;
	m_processRelay->removeProcess(rgtVolumicOperator);
	delete rgtVolumicOperator;
	return 1;
}



bool RgtAndPatchWidget::fitDatasetSize(QString path1, QString path2)
{
	inri::Xt xt1((char*)path1.toStdString().c_str());
	if ( !xt1.is_valid() ) return false;
	inri::Xt xt2((char*)path1.toStdString().c_str());
	if ( !xt2.is_valid() ) return false;
	if ( xt1.nSamples() != xt2.nSamples() ||
			xt1.nRecords() != xt2.nRecords() ||
			xt1.nSlices() != xt2.nSlices() ) return false;
	return true;
}

bool RgtAndPatchWidget::datasetValid(QString path1)
{
	inri::Xt xt1((char*)path1.toStdString().c_str());
	return xt1.is_valid();
}

bool RgtAndPatchWidget::displayWarning(QString msg)
{
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText(msg);
	msgBox->setStandardButtons(QMessageBox::Ok);
	msgBox->exec();
	return true;
}


void RgtAndPatchWidget::processingDisplay()
{
	if ( pStatus == 0 )
	{
		m_progress->setValue(0); m_progress->setFormat("");
		m_processing->setText("waiting ...");
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



RgtAndPatchWidget::MyThread0::MyThread0(RgtAndPatchWidget *p)
{
	this-> pp = p;
}

void RgtAndPatchWidget::MyThread0::run()
{
	pp->trt_compute();
}

// ==================================================
void RgtAndPatchWidget::trt_launchBatch()
{

}

QString RgtAndPatchWidget::timeMSToString(double val)
{
	// QTime t(0,0,(int)(val/1000.0),0);
	// return t.toString("hh:mm:ss");
	long val0 = (long)(val / 1000.0);

	int hh = (int) (val0/3600.0);

	long ss0 = val0 - hh * 3600;
	int mm = (int) (ss0/60.0);

	ss0 = ss0 - mm * 60;

	// QString number = QStringLiteral("%1").arg(yourNumber, 5, 10, QLatin1Char('0'));
	return QString::number(hh) + ":" + QString::number(mm) + ":" + QString::number((int)ss0);
}



void RgtAndPatchWidget::displayProcessFinish()
{
	QMessageBox msgBox;
	QString txt = "Process ended\n\n";

	double totalTime = m_returnParam.timeOrientationAll + m_returnParam.timePatchPatch + m_returnParam.timePatchFusion + m_returnParam.timeRgtAll;
	txt += "total time: " + timeMSToString(totalTime);

	if ( m_returnParam.timeOrientationAll != 0.0 )
	{
		QString t = timeMSToString(m_returnParam.timeOrientationAll);
		txt += "\norientation :" + t;
	}
	if ( m_returnParam.timePatchPatch != 0.0 && m_returnParam.timePatchFusion != 0.0 )
	{
		QString total = timeMSToString(m_returnParam.timePatchPatch+m_returnParam.timePatchFusion);
		txt += "\npatch:\nsurfaces:" + timeMSToString(m_returnParam.timePatchPatch);
		txt += "\nfusion: " + timeMSToString(m_returnParam.timePatchFusion);
		txt += "\ntotal: " + total;
	}
	if ( m_returnParam.timeRgtAll != 0.0 )
	{
		txt += "\nrgt: " + timeMSToString(m_returnParam.timeRgtAll);
	}

	msgBox.setText(txt);
	// msgBox.setInformativeText(tr("Information"));
	msgBox.setStandardButtons(QMessageBox::Ok);
	msgBox.setDefaultButton(QMessageBox::Ok);
	int ret = msgBox.exec();

}

void RgtAndPatchWidget::seismicFilenameChanged()
{
	QString seismicFilename = m_seismicFileSelectWidget->getPath();
	inri::Xt xt(seismicFilename.toStdString().c_str());
	int dimx = xt.nSamples();
	int dimy = xt.nRecords();
	int dimz = xt.nSlices();
	m_orientationWidget->setConstraintsDims(dimx, dimy, dimz);
	m_patchWidget->setConstraintsDims(dimx, dimy, dimz);
	m_rgtPatchWidget->setConstraintsDims(dimx, dimy, dimz);
}

