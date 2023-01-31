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
#include <QComboBox>

#include <QVBoxLayout>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
#include <sampleLimitsChooser.h>
#include <cuda_utils.h>
#include <fileio2.h>
#include <util.h>
#include "GeotimeConfiguratorExpertWidget.h"


// #define __LINUX__

using namespace std;



GeotimeConfigurationExpertWidget::GeotimeConfigurationExpertWidget(GeotimeConfigurationWidget *pconfig, QWidget* parent) :
		QWidget(parent) {

    this->pconf = pconfig;

    // new
    setWindowTitle("Geotime Computation Advanced Parameters");

    QVBoxLayout * mainLayout=new QVBoxLayout(this);

    QHBoxLayout *qhb_nbthreads = new QHBoxLayout;
    QLabel *label_nbthreads = new QLabel("threads:");
    lineedit_nbthreads = new QLineEdit;
    qhb_nbthreads->addWidget(label_nbthreads);
    qhb_nbthreads->addWidget(lineedit_nbthreads);


    QHBoxLayout *qhbox_iteration = new  QHBoxLayout;
    QLabel *label_iteration = new QLabel("iteration: ");
    lineedit_iteration = new QLineEdit;
    qhbox_iteration->addWidget(label_iteration);
    qhbox_iteration->addWidget(lineedit_iteration);

    QHBoxLayout *qhbox_dipthreshold = new  QHBoxLayout;
    QLabel *label_dipthreshold = new QLabel("dip threshold:");
    lineedit_dipthreshold = new QLineEdit;
    qhbox_dipthreshold->addWidget(label_dipthreshold);
    qhbox_dipthreshold->addWidget(lineedit_dipthreshold);

    QHBoxLayout *qhbox_decimationfactor = new  QHBoxLayout;
    QLabel *label_decimationfactor = new QLabel("decimation factor:");
    lineedit_decimationfactor = new QLineEdit;
    qhbox_decimationfactor->addWidget(label_decimationfactor);
    qhbox_decimationfactor->addWidget(lineedit_decimationfactor);

    QHBoxLayout *qhbox_stackformat = new  QHBoxLayout;
    QLabel *label_stackformat = new QLabel("stack/rgt format:");
    cb_stackformat = new QComboBox;
    this->cb_stackformat->addItem("short");
    this->cb_stackformat->addItem("float");
    this->cb_stackformat->setStyleSheet("QComboBox::item{height: 20px}");
    qhbox_stackformat->addWidget(label_stackformat);
    qhbox_stackformat->addWidget(cb_stackformat);

    QHBoxLayout *qhbox_tensor = new  QHBoxLayout;
    QLabel *label_sigmagradient = new QLabel("gradient:");
   	lineedit_sigmagradient = new QLineEdit;
   	QLabel *label_sigmatensor = new QLabel("tensor:");
   	lineedit_sigmatensor = new QLineEdit;
   	qhbox_tensor->addWidget(label_sigmagradient);
   	qhbox_tensor->addWidget(lineedit_sigmagradient);
   	qhbox_tensor->addWidget(label_sigmatensor);
   	qhbox_tensor->addWidget(lineedit_sigmatensor);

   	QHBoxLayout *qhbox_rgtcompress = new  QHBoxLayout;
   	QLabel *label_rgtcompresserror = new QLabel("rgt compress error:");
    lineedit_rgtcompresserror = new QLineEdit;
    qhbox_rgtcompress->addWidget(label_rgtcompresserror);
    qhbox_rgtcompress->addWidget(lineedit_rgtcompresserror);

    QHBoxLayout *qhbox_seedthreshold = new  QHBoxLayout;
    // QLabel *label_seedthreshold = new QLabel("seed threshold:");
    qcb_seedthreshold_valid = new QCheckBox("seed max:");
    lineedit_seedthreshold = new QLineEdit;
    // qhbox_seedthreshold->addWidget(label_seedthreshold);
    qhbox_seedthreshold->addWidget(qcb_seedthreshold_valid);
    qhbox_seedthreshold->addWidget(lineedit_seedthreshold);

    QHBoxLayout *qhbox_buttons = new  QHBoxLayout;
    QPushButton *pushbutton_ok = new QPushButton("OK");
    QPushButton *pushbutton_cancel = new QPushButton("cancel");
    qhbox_buttons->addWidget(pushbutton_ok);
    qhbox_buttons->addWidget(pushbutton_cancel);

   	QHBoxLayout *qhbox_snapping = new  QHBoxLayout;
   	qcb_snapping = new QCheckBox("snapping");
   	qhbox_snapping->addWidget(qcb_snapping);

   	QHBoxLayout *qhbox_fileformat = new  QHBoxLayout;
   	QLabel *label_fileformat = new QLabel("file format:");
   	qcb_fileformat = new QComboBox;
   	this->qcb_fileformat->addItem("xt (sismage)");
   	this->qcb_fileformat->addItem("cwt (compressed)");
   	this->qcb_fileformat->addItem("xt & cwt (sismage & compressed)");
    this->qcb_fileformat->setStyleSheet("QComboBox::item{height: 20px}");
   	qhbox_fileformat->addWidget(label_fileformat);
   	qhbox_fileformat->addWidget(qcb_fileformat);

   	pbSampleLimits = new QPushButton("sample limits");


   	mainLayout->addLayout(qhb_nbthreads);
    mainLayout->addLayout(qhbox_iteration);
    mainLayout->addLayout(qhbox_dipthreshold);
    mainLayout->addLayout(qhbox_decimationfactor);
    mainLayout->addLayout(qhbox_snapping);
    mainLayout->addLayout(qhbox_stackformat);
    mainLayout->addLayout(qhbox_tensor);
    mainLayout->addLayout(qhbox_seedthreshold);
    mainLayout->addLayout(qhbox_fileformat);
    mainLayout->addWidget(pbSampleLimits);
    mainLayout->addLayout(qhbox_buttons);

    connect(pushbutton_ok, SIGNAL(clicked()), this, SLOT(trt_ok()));
    connect(pushbutton_cancel, SIGNAL(clicked()), this, SLOT(trt_cancel()));
    connect(qcb_seedthreshold_valid, SIGNAL(clicked(bool)), this, SLOT(trt_seedThresholdValid(bool)));
    connect(pbSampleLimits, SIGNAL(clicked(bool)), this, SLOT(trt_sampleLimits()));
    fill_fields();
}

GeotimeConfigurationExpertWidget::~GeotimeConfigurationExpertWidget() {
}

// ******************************************************************
// 
// ******************************************************************
void GeotimeConfigurationExpertWidget::fill_fields()
{
	this->lineedit_nbthreads->setText(QString::number(this->pconf->nb_threads));
    this->lineedit_iteration->setText(QString::number(this->pconf->nbiter));
    this->lineedit_dipthreshold->setText(QString::number((double)this->pconf->dip_threshold, 'f', 1 ));
    this->lineedit_decimationfactor->setText(QString::number(this->pconf->decimation_factor));
    this->cb_stackformat->setCurrentIndex(this->pconf->stack_format);
    this->lineedit_sigmagradient->setText(QString::number(this->pconf->sigmagradient, 'f', 1));
	this->lineedit_sigmatensor->setText(QString::number(this->pconf->sigmatensor, 'f', 1));
    this->lineedit_rgtcompresserror->setText(QString::number(this->pconf->rgt_compresserror, 'f', 6));
    this->lineedit_seedthreshold->setText(QString::number(this->pconf->seed_threshold));
    if ( this->pconf->bool_snapping )
    	qcb_snapping->setCheckState(Qt::Checked);
    else
    	qcb_snapping->setCheckState(Qt::Unchecked);
    qcb_fileformat->setCurrentIndex(this->pconf->output_format);

    qcb_seedthreshold_valid->setCheckState(this->pconf->seed_threshold_valid == 1 ? Qt::Checked : Qt::Unchecked);
    this->lineedit_seedthreshold->setEnabled(qcb_seedthreshold_valid->isChecked());

    pbSampleLimits->setEnabled(checkEnableButtonSampleLimits());
}


bool GeotimeConfigurationExpertWidget::checkEnableButtonSampleLimits()
{
	if ( pconf->getSeismicPath().isEmpty() ) return false;
	char seismicFilename[1000];
	strcpy(seismicFilename, (char*)(pconf->getSeismicPath().toStdString().c_str()));
	if ( !FILEIO2::exist(seismicFilename) ) return false;

	int cpuGpu = pconf->qcb_rgtcpugpu->currentIndex();
    if ( cpuGpu == 0 ) return false;
    int *tab_gpu = nullptr, tab_gpu_size;
    tab_gpu = (int*)calloc(pconf->systemInfo->get_gpu_nbre(), sizeof(int));
    pconf->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
 	int maxSize;
   	int textureMaxSize[3];
   	cudaGetMaxTexture3D(textureMaxSize, tab_gpu[0]);
   	maxSize = textureMaxSize[0];
   	for (int i=1; i<tab_gpu_size; i++)
   	{
   		cudaGetMaxTexture3D(textureMaxSize, tab_gpu[0]);
   		maxSize = MIN(textureMaxSize[0], maxSize);
   	}
   	free(tab_gpu);

   	FILEIO2 *pf = new FILEIO2();
   	pf->openForRead(seismicFilename);
   	int dimx = pf->get_dimx();
   	delete pf;
   	if ( dimx <= maxSize ) return false;
   	return true;
}

// *****************************************************************
// 
// *****************************************************************
void GeotimeConfigurationExpertWidget::trt_seedThresholdValid(bool val)
{
	this->lineedit_seedthreshold->setEnabled(qcb_seedthreshold_valid->isChecked());
}


void GeotimeConfigurationExpertWidget::trt_sampleLimits()
{
	int cpuGpu = pconf->qcb_rgtcpugpu->currentIndex();
	if ( cpuGpu == 0 ) return;
    int *tab_gpu = nullptr, tab_gpu_size;
    tab_gpu = (int*)calloc(pconf->systemInfo->get_gpu_nbre(), sizeof(int));
    pconf->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	int maxSize;
	int textureMaxSize[3];
	cudaGetMaxTexture3D(textureMaxSize, tab_gpu[0]);
	maxSize = textureMaxSize[0];
	for (int i=1; i<tab_gpu_size; i++)
	{
		cudaGetMaxTexture3D(textureMaxSize, tab_gpu[0]);
		maxSize = MIN(textureMaxSize[0], maxSize);
	}

	char seismicFilename[1000];
	strcpy(seismicFilename, (char*)(pconf->getSeismicPath().toStdString().c_str()));
	if ( !FILEIO2::exist(seismicFilename) ) return;
	FILEIO2 *pf = new FILEIO2();
	pf->openForRead(seismicFilename);
	int dimx = pf->get_dimx();
	delete pf;
	if ( dimx <= maxSize ) return;
	float startSample, stepSample;
    FILEIO2::getSeismicStartAndStepSample(seismicFilename, &startSample, &stepSample);

    // float t1 = startSample, t2 = t1 + (dimx-1)*stepSample;
    int *x1 = &pconf->traceLimitX1;
    int *x2 = &pconf->traceLimitX2;
    SampleLimitsChooser *p = new SampleLimitsChooser(this, dimx, maxSize, startSample, stepSample, x1, x2);
    p->exec();
}


void GeotimeConfigurationExpertWidget::trt_ok()
{
	this->pconf->nb_threads = this->lineedit_nbthreads->text().toInt();
    this->pconf->nbiter = this->lineedit_iteration->text().toInt();
    this->pconf->dip_threshold = this->lineedit_dipthreshold->text().toFloat();
    this->pconf->decimation_factor = this->lineedit_decimationfactor->text().toInt();
    this->pconf->stack_format = this->cb_stackformat->currentIndex();
    this->pconf->sigmagradient = this->lineedit_sigmagradient->text().toDouble();
	this->pconf->sigmatensor = this->lineedit_sigmatensor->text().toDouble();
    this->pconf->rgt_compresserror = this->lineedit_rgtcompresserror->text().toDouble();
    this->pconf->seed_threshold = this->lineedit_seedthreshold->text().toInt();
    this->pconf->seed_threshold_valid = qcb_seedthreshold_valid->isChecked() ? 1 : 0;
    this->pconf->bool_snapping = qcb_snapping->isChecked();
    this->pconf->output_format = qcb_fileformat->currentIndex();
    QWidget::close();
}

void GeotimeConfigurationExpertWidget::trt_cancel()
{
    QWidget::close();
}



