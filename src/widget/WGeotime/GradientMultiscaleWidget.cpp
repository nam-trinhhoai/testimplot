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

#include <QVBoxLayout>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>
#include "GradientMultiscaleWidget.h"
#include <gradient_multiscale.cuh>


// #define __LINUX__
#define EXPORT_LIB __attribute__((visibility("default")))
#include <config.h>
#include <normal.h>
#include <util.h>
#include <cuda_utils.h>
#include <fileio.h>
#include <gradient_multiscale.cuh>

using namespace std;



GradientMultiscaleWidget::GradientMultiscaleWidget(QWidget* parent) :
		QWidget(parent) {

    // gradient_multiscale_gpu_run(NULL, NULL, 0, 0, 0, 0, 0, 0, 0, NULL);

	int qtmargin = 10;
	QSize size0 = this->size();
	int width0 = size0.rwidth();
	int height0 = size0.rheight();
	int width1 = 480;

	QLabel *label_version = new QLabel(this);
	label_version->setText("version: " + QString(__DATE__));
	label_version->setGeometry(QRect(qtmargin, 5, width1, 15));


	QGroupBox* qgb_seismic = new QGroupBox(this);
	qgb_seismic->setTitle("Seismic");
	qgb_seismic->setGeometry(QRect(qtmargin, 50, width1, 90));
/*
	QHBoxLayout* qhb_seismic = new QHBoxLayout;    
	QLabel *label_seismicfilename = new QLabel("seismic filename");
	lineedit_seismicfilename = new QLineEdit("");
	QPushButton *pushbutton_seismicfilename = new QPushButton("...");	
	qhb_seismic->addWidget(label_seismicfilename);
	qhb_seismic->addWidget(lineedit_seismicfilename);
	qhb_seismic->addWidget(pushbutton_seismicfilename);
	*/

/*
    QHBoxLayout* qhb_rgt = new QHBoxLayout;    
	QLabel *label_rgtfilename = new QLabel("rgt filename");
	lineedit_rgtfilename = new QLineEdit("");
	QPushButton *pushbutton_rgtfilename = new QPushButton("...");	
	qhb_seismic->addWidget(label_seismicfilename);
	qhb_seismic->addWidget(lineedit_seismicfilename);
	qhb_seismic->addWidget(pushbutton_seismicfilename);

	QVBoxLayout* mainLayout0 = new QVBoxLayout(qgb_seismic);	
	mainLayout0->addLayout(qhb_seismic);
*/

	// QGroupBox* qgb_orientation = new QGroupBox(this);
	// qgb_orientation->setTitle("Orientation");
	// qgb_orientation->setGeometry(QRect(qtmargin, 150, width1, 200));

	// QHBoxLayout* qhb_orientationcompute = new QHBoxLayout;//(qgb_orientation);
	// qcb_orientationcompute = new QCheckBox("compute");
	// qcb_orientationgpu = new QCheckBox("gpu");
	// qhb_orientationcompute->setContentsMargins(0,0,0,0);
	// qhb_orientationcompute->addWidget(qcb_orientationcompute, 0, Qt::AlignLeft);
	// // qhb_orientationcompute->addStretch();
	// qhb_orientationcompute->addWidget(qcb_orientationgpu, 0, Qt::AlignLeft);
	
	// QHBoxLayout* qhb_dipx = new QHBoxLayout;//(qgb_orientation);
	// QLabel *label_dipxfilename = new QLabel("dipx filename");
	// lineedit_dipxfilename = new QLineEdit("");
	// QPushButton *pushbutton_dipxfilename = new QPushButton("...");
	// qhb_dipx->addWidget(label_dipxfilename);
	// qhb_dipx->addWidget(lineedit_dipxfilename);
	// qhb_dipx->addWidget(pushbutton_dipxfilename);

	// QHBoxLayout* qhb_dipz = new QHBoxLayout;//(qgb_orientation);
	// QLabel *label_dipzfilename = new QLabel("dipz filename");
	// lineedit_dipzfilename = new QLineEdit("");
	// QPushButton *pushbutton_dipzfilename = new QPushButton("...");
	// qhb_dipz->addWidget(label_dipzfilename);
	// qhb_dipz->addWidget(lineedit_dipzfilename);
	// qhb_dipz->addWidget(pushbutton_dipzfilename);	

	// // QHBoxLayout* qhb_seismic = new QHBoxLayout;//(qgb_orientation);
	// // QLabel *label_seismicfilename = new QLabel("seismic filename");
	// // QLineEdit *lineedit_seismicfilename = new QLineEdit("");
	// // QPushButton *pushbutton_seismicfilename = new QPushButton("...");	
	// // qhb_seismic->addWidget(label_seismicfilename);
	// // qhb_seismic->addWidget(lineedit_seismicfilename);
	// // qhb_seismic->addWidget(pushbutton_seismicfilename);	

	// QHBoxLayout* qhb_orientationsigma = new QHBoxLayout;//(qgb_orientation);
	// QLabel *label_sigmagradient = new QLabel("gradient");
	// lineedit_sigmagradient = new QLineEdit("1.0");
	// QLabel *label_sigmatensor = new QLabel("tensor");
	// lineedit_sigmatensor = new QLineEdit("1.5");
	// qhb_orientationsigma->addWidget(label_sigmagradient);
	// qhb_orientationsigma->addWidget(lineedit_sigmagradient);
	// qhb_orientationsigma->addWidget(label_sigmatensor);
	// qhb_orientationsigma->addWidget(lineedit_sigmatensor);

	// QVBoxLayout* mainLayout = new QVBoxLayout(qgb_orientation);	
	// mainLayout->addLayout(qhb_orientationcompute);
	// mainLayout->addLayout(qhb_dipx);
	// mainLayout->addLayout(qhb_dipz);
	// // mainLayout->addLayout(qhb_seismic);
	// mainLayout->addLayout(qhb_orientationsigma);


	// QGroupBox* qgb_stackrgt = new QGroupBox(this);
	// qgb_stackrgt->setTitle("Stack - RGT");
	// qgb_stackrgt->setGeometry(QRect(qtmargin, 360, width1, 180));

	// QHBoxLayout* qhb_stacksuffix = new QHBoxLayout;//(qgb_orientation);
	// QLabel *label_stacksuffix = new QLabel("stack suffix");
	// lineedit_stacksuffix = new QLineEdit("stack");
	// qhb_stacksuffix->addWidget(label_stacksuffix);
	// qhb_stacksuffix->addWidget(lineedit_stacksuffix);

	// QHBoxLayout* qhb_rgtsuffix = new QHBoxLayout;//(qgb_orientation);
	// QLabel *label_rgtsuffix = new QLabel("rgt suffix");
	// lineedit_rgtsuffix = new QLineEdit("rgt");	
	// qhb_rgtsuffix->addWidget(label_rgtsuffix);
	// qhb_rgtsuffix->addWidget(lineedit_rgtsuffix);	

	// QHBoxLayout* qhb_rgtstackpolaritygpu = new QHBoxLayout;//(qgb_orientation);
	// qcb_rgtstackpolarity = new QCheckBox("polarity");
	// qcb_rgtstackgpu = new QCheckBox("gpu");
	// qhb_rgtstackpolaritygpu->setContentsMargins(0,0,0,0);
	// qhb_rgtstackpolaritygpu->addWidget(qcb_rgtstackpolarity, 0, Qt::AlignLeft);
	// // qhb_orientationcompute->addStretch();
	// qhb_rgtstackpolaritygpu->addWidget(qcb_rgtstackgpu, 0, Qt::AlignLeft);	

	// QHBoxLayout* qhb_dimstep = new QHBoxLayout;//(qgb_orientation);
	// QLabel *label_stepdimx = new QLabel("step dimx");
	// lineedit_stepdimx = new QLineEdit("100");	
	// QLabel *label_stepdimy = new QLabel("step dimy");
	// lineedit_stepdimy = new QLineEdit("100");	
	// QLabel *label_stepdimz = new QLabel("step dimz");
	// lineedit_stepdimz = new QLineEdit("100");	
	// qhb_dimstep->addWidget(label_stepdimx);
	// qhb_dimstep->addWidget(lineedit_stepdimx);
	// qhb_dimstep->addStretch(1);
	// qhb_dimstep->addWidget(label_stepdimy);	
	// qhb_dimstep->addWidget(lineedit_stepdimy);
	// qhb_dimstep->addStretch(1);
	// qhb_dimstep->addWidget(label_stepdimz);
	// qhb_dimstep->addWidget(lineedit_stepdimz);		

	// QVBoxLayout* mainLayout2 = new QVBoxLayout(qgb_stackrgt);	
	// mainLayout2->addLayout(qhb_stacksuffix);
	// mainLayout2->addLayout(qhb_rgtsuffix);
	// mainLayout2->addLayout(qhb_rgtstackpolaritygpu);
	// mainLayout2->addLayout(qhb_dimstep);



	// QGroupBox* qgb_action = new QGroupBox(this);
	// qgb_action->setTitle("Actions");
	// qgb_action->setGeometry(QRect(qtmargin, 550, width1, 80));	

	// QHBoxLayout* qhb_actions = new QHBoxLayout(qgb_action);//(qgb_orientation);
	// QPushButton *pushbutton_expert = new QPushButton("expert");
	// QPushButton *pushbutton_seedinfo = new QPushButton("seed info");
	// QPushButton *pushbutton_compute = new QPushButton("compute");
	// qhb_actions->addWidget(pushbutton_expert);
	// qhb_actions->addWidget(pushbutton_seedinfo);
	// qhb_actions->addWidget(pushbutton_compute);
	// QVBoxLayout* mainLayout3 = new QVBoxLayout(this);	
	// mainLayout3->addLayout(qhb_actions);

	// QGroupBox* qgb_textinfo = new QGroupBox(this);
	// qgb_textinfo->setTitle("info");
	// qgb_textinfo->setGeometry(QRect(qtmargin, 635, width1, 150));	
	// QHBoxLayout* qhb_textinfo = new QHBoxLayout(qgb_textinfo);//(qgb_orientation);
	// textInfo = new QPlainTextEdit(".");
	// textInfo->setReadOnly(true);
	// qhb_textinfo->addWidget(textInfo);
	// QVBoxLayout* mainLayout4 = new QVBoxLayout(this);	
	// mainLayout4->addLayout(qhb_textinfo);

	// // textInfo->appendPlainText("string to append. ");
	// // textInfo->appendPlainText("string to append. 2");	




	// lineedit_dipxfilename->setFocus();

	// // slots
	// connect(pushbutton_expert, SIGNAL(clicked()), this, SLOT(trt_expert()));
	// connect(pushbutton_compute, SIGNAL(clicked()), this, SLOT(trt_compute()));
    // connect(pushbutton_dipxfilename, SIGNAL(clicked()), this, SLOT(trt_dipx_open()));
	// connect(pushbutton_dipzfilename, SIGNAL(clicked()), this, SLOT(trt_dipz_open()));
	// connect(pushbutton_seismicfilename, SIGNAL(clicked()), this, SLOT(trt_seismic_open()));

	// field_fill();
	// current_path = QDir::currentPath();
    // config_filename = current_path + "/config.txt";



}

GradientMultiscaleWidget::~GradientMultiscaleWidget() {
}

// *****************************************************************
//
// *****************************************************************



