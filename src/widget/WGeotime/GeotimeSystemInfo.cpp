/*
 *
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */


#include <QSizeGrip>
#include <QTableView>
#include <QVBoxLayout>
#include <QHeaderView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QRadioButton>
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
#include <QThread>

#include <QVBoxLayout>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <stdio.h>
#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
#include <sys/sysinfo.h>
#include "sys/types.h"


#include <malloc.h>
#define EXPORT_LIB __attribute__((visibility("default")))
#include <cuda_utils.h>
#include "GeotimeSystemInfo.h"


// #define __LINUX__

using namespace std;


// namespace XCom = process::XCom;

GeotimeSystemInfo::GeotimeSystemInfo(QWidget* parent) :
		QWidget(parent) 
{
	setWindowTitle("System Info");

    int qtmargin = 5;
	this->nb_threads = QThread::idealThreadCount();
	this->nb_gpu = cuda_get_nbre_devices();	

	QVBoxLayout* mainlayout = new QVBoxLayout();
	mainlayout->setSpacing(0);
	mainlayout->setContentsMargins(0, 0, 0, 0);
	setLayout(mainlayout);
	setContentsMargins(0, 0, 0, 0);

	setMinimumWidth(350);
	setMinimumHeight(160+25*nb_gpu);

	this->qgb_systeminfo = new QGroupBox;
	mainlayout->addWidget(qgb_systeminfo);
	qgb_systeminfo->setTitle("System informations");
	qgb_systeminfo->setGeometry(QRect(0, 0, 350, 150+25*nb_gpu));

	this->qpb_gpumemory = (QProgressBar**)calloc(this->nb_gpu, sizeof(QProgressBar*));
	this->qcb_validgpu = (QCheckBox**)calloc(this->nb_gpu, sizeof(QCheckBox*));
	this->qlabel_gpumem = (QLabel**)calloc(this->nb_gpu, sizeof(QLabel*));

	qlabel_nbthreads = new QLabel(qgb_systeminfo);
	qlabel_nbthreads->setGeometry(QRect(qtmargin+5, 25, 150, 50));
	qlabel_nbthreads->setText(QString("cpu cores: ") + QString::number(nb_threads));

	qpb_cpumemory = new QProgressBar(qgb_systeminfo);
	qpb_cpumemory->setMinimum(0);
	qpb_cpumemory->setMaximum(100);    
	qpb_cpumemory->setGeometry(QRect(qtmargin+5, 70, 150, 20));

	qlabel_cpumem = new QLabel(qgb_systeminfo);
	qlabel_cpumem->setGeometry(QRect(qtmargin+5+150+5, 70, 150, 20));
	qlabel_cpumem->setText(QString("cpu: "));

	QLabel *qlabel_gpunbre = new QLabel(qgb_systeminfo);
	qlabel_gpunbre->setGeometry(QRect(qtmargin+5, 100, 150, 50));
	qlabel_gpunbre->setText(QString("gpu: ") + QString::number(nb_gpu));

	for (int i=0; i<this->nb_gpu; i++)
	{
		this->qpb_gpumemory[i] = new QProgressBar(qgb_systeminfo);
		this->qpb_gpumemory[i]->setMinimum(0);
		this->qpb_gpumemory[i]->setMaximum(100);    
		this->qpb_gpumemory[i]->setGeometry(QRect(qtmargin+40+5, 145+25*i, 150, 20));

		this->qcb_validgpu[i] = new QCheckBox(qgb_systeminfo);
		this->qcb_validgpu[i]->setGeometry(QRect(qtmargin+5, 145+25*i, 20, 20));
		// this->qcb_validgpu[i]->setText("[ " + QString::number(i) + " ]");		
		this->qcb_validgpu[i]->setChecked(qgb_systeminfo);

		this->qlabel_gpumem[i] = new QLabel(qgb_systeminfo);
		this->qlabel_gpumem[i]->setGeometry(QRect(qtmargin+200, 145+25*i, 140, 20));			
	}

	QSizeGrip* sizegrip = new QSizeGrip(this);
	sizegrip->setContentsMargins(0, 0, 0, 0);
	mainlayout->addWidget(sizegrip, 0, Qt::AlignRight);

	timer = new QTimer(this);
    timer->start(1000);
    timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));



	/*
	this->nb_threads = QThread::idealThreadCount();
	this->nb_gpu = cuda_get_nbre_devices();
	this->qpb_gpumemory = (QProgressBar**)calloc(this->nb_gpu, sizeof(QProgressBar*));
	this->qcb_validgpu = (QCheckBox**)calloc(this->nb_gpu, sizeof(QCheckBox*));
	this->qlabel_gpumem = (QLabel**)calloc(this->nb_gpu, sizeof(QLabel*));

	QVBoxLayout * mainLayout00 = new QVBoxLayout(this);

	QVBoxLayout *layout2 = new QVBoxLayout;
	QVBoxLayout *layout2p = new QVBoxLayout;

	QGroupBox *qgb = new QGroupBox("System informations");
	qgb->setMinimumHeight(350);
	qgb->setMaximumHeight(350);

	qlabel_nbthreads = new QLabel(QString("cpu cores: ") + QString::number(nb_threads));

	layout2p->addWidget(qlabel_nbthreads);

	QHBoxLayout *layout2_1 = new QHBoxLayout;
	qpb_cpumemory = new QProgressBar;
	qpb_cpumemory->setMinimum(0);
	qpb_cpumemory->setMaximum(100);
	qpb_cpumemory->setMaximumWidth(200);
	qlabel_cpumem = new QLabel;
	qlabel_cpumem->setText(QString("cpu: "));
	layout2_1->addWidget(qpb_cpumemory);
	layout2_1->addWidget(qlabel_cpumem);

	QLabel *qlabel_gpunbre = new QLabel(this);
	qlabel_gpunbre->setText(QString("gpu number: ") + QString::number(nb_gpu));

	layout2p->addLayout(layout2_1);
	layout2p->addWidget(qlabel_gpunbre);

	for (int i=0; i<this->nb_gpu; i++)
	{
		QHBoxLayout *layout2_2 = new QHBoxLayout;

		this->qpb_gpumemory[i] = new QProgressBar;
		this->qpb_gpumemory[i]->setMinimum(0);
		this->qpb_gpumemory[i]->setMaximum(100);
		qpb_gpumemory[i]->setMaximumWidth(200);

		this->qcb_validgpu[i] = new QCheckBox;
		this->qcb_validgpu[i]->setChecked(true);

		this->qlabel_gpumem[i] = new QLabel("mem");

		layout2_2->addWidget(qcb_validgpu[i]);
		layout2_2->addWidget(qpb_gpumemory[i]);
		layout2_2->addWidget(qlabel_gpumem[i]);
		// layout2_2->setAlignment(Qt::AlignLeft);


		layout2p->addLayout(layout2_2);
	}

	qgb->setLayout(layout2p);

	layout2->addWidget(qgb);
	mainLayout00->addLayout(layout2);

	timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	*/
}


GeotimeSystemInfo::~GeotimeSystemInfo()
{
	free(this->qpb_gpumemory);
	free(this->qcb_validgpu);
	free(this->qlabel_gpumem);
}

int GeotimeSystemInfo::get_gpu_nbre()
{
	return this->nb_gpu;
}

void GeotimeSystemInfo::get_valid_gpu(int *tab, int *size)
{
	if ( tab == NULL || size == NULL ) return;
	int idx = 0;
	for (int i=0; i<this->nb_gpu; i++)
	{
		if ( this->qcb_validgpu[i]->isChecked() )
		{
			tab[idx++] = i;
		}
	}
	*size = idx;
}

int GeotimeSystemInfo::get_valid_gpu_nbre()
{	int nb = 0;

	for (int i=0; i<this->nb_gpu; i++)
	{
		if ( this->qcb_validgpu[i]->isChecked() ) nb++;
	}
	return nb;
}

double GeotimeSystemInfo::cuda_min_free_memory()
{
	double mem = -1.0;
	for (int i=0; i<this->nb_gpu; i++)
	{
		if ( this->qcb_validgpu[i]->isChecked() )
		{
			double mem0 = cuda_free_memory2(i)/1e9;
			if ( mem <= 0.0 || mem0 < mem )
			{
				mem = mem0;
			}
		}
	}
    return mem;
}

// Use /proc/meminfo file
long long getCacheRam() {
	long long cacheRam = 0;
	bool ok;
	// https://stackoverflow.com/questions/8122277/getting-memory-information-with-qt
	QProcess p;
	p.start("awk", QStringList() << "/^Cached/ { print $2 }" << "/proc/meminfo");
	p.waitForFinished();
	QString memory = p.readAllStandardOutput();
	cacheRam = memory.toLong(&ok) * 1024; // /proc/meminfo is in kB
	p.close();

	if (!ok) {
		cacheRam = 0;
	}

	return cacheRam;
}


double GeotimeSystemInfo::qt_cpu_free_memory()
{    
	/*
    struct sysinfo info;
    if (sysinfo(&info) < 0)
    {
      printf("error\n");
      return 0;
    }    
    return (double)info.freeram/1e9;
    */
	struct sysinfo memInfo;
	sysinfo (&memInfo);

	// sysinfo does not provide cacheram using another way to retrieve it
	long long cacheram = getCacheRam();

	long long totalVirtualMem = memInfo.totalram;
	//Add other values in next statement to avoid int overflow on right hand side...
	totalVirtualMem += memInfo.totalswap;
	totalVirtualMem *= memInfo.mem_unit;

	long long virtualMemUsed = memInfo.totalram - memInfo.freeram - memInfo.bufferram - cacheram;
	//Add other values in next statement to avoid int overflow on right hand side...
	virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
	virtualMemUsed *= memInfo.mem_unit;

	double free_mem = (double)totalVirtualMem-virtualMemUsed;
	return free_mem / 1e9;
}

double GeotimeSystemInfo::qt_cpu_total_memory()
{
    struct sysinfo info;
    if (sysinfo(&info) < 0)
    {
      printf("error\n");
      return 0;
    }   
    long long totalVirtualMem = info.totalram;
    totalVirtualMem += info.totalswap;
    totalVirtualMem *= info.mem_unit;
    return (double)(totalVirtualMem)/1e9;
}


void GeotimeSystemInfo::showTime()
{
	char txt[1000];

	float fmem = (float)qt_cpu_free_memory();
    float tmem = (float)qt_cpu_total_memory();
    float val_f = 100.0*fmem/tmem;
    this->qpb_cpumemory->setValue(100-(int)val_f);
    sprintf(txt, "%.1f / %.1f GB", tmem-fmem, tmem);
    qlabel_cpumem->setText(QString(txt));

	for (int i=0; i<this->nb_gpu; i++)
	{
		float fmem = (float)cuda_free_memory2(i);
    	float tmem = (float)cuda_total_memory2(i);
		float val_f = 100.0*fmem/tmem;
		this->qpb_gpumemory[i]->setValue(100-(int)val_f);

		sprintf(txt, "%.1f / %.1f GB", (tmem-fmem)/1e9, tmem/1e9);
		this->qlabel_gpumem[i]->setText(QString(txt));
	}

}





