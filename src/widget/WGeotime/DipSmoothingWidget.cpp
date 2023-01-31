/*
 *
 *
 *  Created on: 11 September 2020
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

#include <ihm.h>
#include <fileio2.h>
#include <filter_dip.h>
// #include "FileConvertionXTCWT.h"
#include "DipSmoothingWidget.h"
#include <fileSelectWidget.h>
#include <orientationWidget.h>
#include <fileSelectorDialog.h>


#include "globalconfig.h"


// #define __LINUX__
#define EXPORT_LIB __attribute__((visibility("default")))

using namespace std;




DipSmoothingWidget::DipSmoothingWidget(QWidget* parent) :
		QWidget(parent) {

	setWindowTitle("Dip Smoothing");

	GLOBAL_RUN_TYPE = 0;

	QHBoxLayout * mainLayout00 = new QHBoxLayout(this);

	QVBoxLayout *layout0 = new QVBoxLayout;
	QPushButton *qpb_session_load = new QPushButton("Load session");
	QPushButton *qpb_session_save = new QPushButton("Save session");

	// DATASETS MANAGERS
	QGroupBox *qgbProgramManager = new QGroupBox;
	QVBoxLayout *qvb_programmanager = new QVBoxLayout(qgbProgramManager);
	m_projectManager = new ProjectManagerWidget;
	QPushButton *qpb_loadsession = new QPushButton("load session");
	qvb_programmanager->addWidget(m_projectManager);

	/*
	m_selectorWidget = new GeotimeProjectManagerWidget(this);
	m_selectorWidget->removeTabHorizons();
	m_selectorWidget->removeTabCulturals();
	m_selectorWidget->removeTabWells();
	m_selectorWidget->removeTabNeurons();
	m_selectorWidget->removeTabPicks();
	*/

	layout0->addWidget(qpb_session_load);
	// layout0->addWidget(m_selectorWidget);
	layout0->addWidget(qpb_session_save);

	QVBoxLayout *layout2 = new QVBoxLayout;
	QVBoxLayout *layout2p = new QVBoxLayout;

	QGroupBox *qgb = new QGroupBox("Smooth Filtering");

	QHBoxLayout *layout2_1 = new QHBoxLayout;
	QLabel *ql_label2_1 = new QLabel("data in");

	lw_dataIn = new QListWidget();
	lw_dataIn->setSelectionMode(QAbstractItemView::MultiSelection);
	lw_dataIn->setMaximumHeight(300);

	QVBoxLayout *vbdataInOpenButton = new QVBoxLayout();
	QPushButton *qpushbutton_datainopen = new QPushButton("Add basket");
	QPushButton *qpushbutton_datainClear = new QPushButton("clear");
	vbdataInOpenButton->addWidget(qpushbutton_datainopen);
	vbdataInOpenButton->addWidget(qpushbutton_datainClear);
	layout2_1->addWidget(ql_label2_1);
	layout2_1->addWidget(lw_dataIn);
	layout2_1->addLayout(vbdataInOpenButton);

	m_orientationWidget = new OrientationWidget(m_projectManager);

	QHBoxLayout *layout2_4 = new QHBoxLayout;
	QLabel *ql_label2_4 = new QLabel("out suffix filename");
	qlineedit_outfilename = new QLineEdit("dipfiltered");
	layout2_4->addWidget(ql_label2_4);
	layout2_4->addWidget(qlineedit_outfilename);

	QHBoxLayout *layout2_6 = new QHBoxLayout;
	QLabel *ql_label2_6 = new QLabel("type");
	qcombobox_type = new QComboBox;
	qcombobox_type->addItem("MIN");
	qcombobox_type->addItem("MAX");
	qcombobox_type->addItem("MEAN");
	qcombobox_type->addItem("MEDIAN");
	qcombobox_type->setCurrentIndex(2);
	qcombobox_type->setStyleSheet("QComboBox::item{height: 20px}");

	QHBoxLayout *layout2_8 = new QHBoxLayout;
	QFrame *linev1 = new QFrame;
	linev1->setFrameShape(QFrame::VLine);
	QFrame *linev2 = new QFrame;
	linev2->setFrameShape(QFrame::VLine);
	QLabel *ql_label2_8_1 = new QLabel("dx");
	qlineedit_dx = new QLineEdit("3");
	QLabel *ql_label2_8_2 = new QLabel("dy");
	qlineedit_dy = new QLineEdit("7");
	QLabel *ql_label2_8_3 = new QLabel("dz");
	qlineedit_dz = new QLineEdit("7");

	layout2_8->addWidget(ql_label2_8_1);
	layout2_8->addWidget(qlineedit_dx);
	layout2_8->addWidget(linev1);
	layout2_8->addWidget(ql_label2_8_2);
	layout2_8->addWidget(qlineedit_dy);
	layout2_8->addWidget(linev2);
	layout2_8->addWidget(ql_label2_8_3);
	layout2_8->addWidget(qlineedit_dz);

	QHBoxLayout *layout2_7 = new QHBoxLayout;
	qpb_progress = new QProgressBar();
	// qpb_progress->setGeometry(5, 45, 240, 20);
	qpb_progress->setMinimum(0);
	qpb_progress->setMaximum(100);
	qpb_progress->setValue(0);
	// qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,0,0)}");
	qpb_progress->setTextVisible(true);
	qpb_progress->setValue(0);
	qpb_progress->setFormat("");
	layout2_7->addWidget(qpb_progress);

	QHBoxLayout *layout2_5 = new QHBoxLayout;
	QPushButton *qpb_start = new QPushButton("Start");
	QPushButton *qpb_stop = new QPushButton("Stop");
	layout2_5->addWidget(qpb_start);
	layout2_5->addWidget(qpb_stop);
/*
	QHBoxLayout *layout2_9 = new QHBoxLayout;
	QPushButton *qpb_session_load = new QPushButton("Load session");
	QPushButton *qpb_session_save = new QPushButton("Save session");
	layout2_9->addWidget(qpb_session_load);
	layout2_9->addWidget(qpb_session_save);
*/

	// this->systemInfo = new GeotimeSystemInfo();
	// this->systemInfo->setVisible(true);
	QGroupBox *qgbSystem = new QGroupBox;
	this->systemInfo = new GeotimeSystemInfo(this);
	this->systemInfo->setVisible(true);
	systemInfo->setMinimumWidth(350);
	QVBoxLayout *layout2s = new QVBoxLayout(qgbSystem);
	layout2s->addWidget(systemInfo);

	// this->systemInfo->setGeometry(QRect(qtmargin+500, 5, 350, 500));

	QFrame *line;
	line = new QFrame;
	line->setFrameShape(QFrame::HLine);

	// layout2p->addWidget(systemInfo);
	// layout2p->addLayout(layout2_9);
	// layout2p->addWidget(line);
	layout2p->addLayout(layout2_1);
	// layout2p->addLayout(layout2_2);
	// layout2p->addLayout(layout2_3);
	layout2p->addWidget(m_orientationWidget);
	layout2p->addLayout(layout2_4);
	layout2p->addLayout(layout2_8);
	layout2p->addLayout(layout2_7);
	layout2p->addLayout(layout2_5);
	// qgb->setMinimumHeight(350*1.1);
	// qgb->setMaximumHeight(350*1.1);
	layout2p->setAlignment(Qt::AlignTop);

	qgb->setLayout(layout2p);

	// layout2->addWidget(systemInfo);
	// layout2->addWidget(qgb);
	// layout2->setAlignment(Qt::AlignBottom);

	// ==============================================================================
	QTabWidget *tabWidgetMain = new QTabWidget();
	tabWidgetMain->insertTab(0, qgbProgramManager, QIcon(QString("")), "Project Manager");
	tabWidgetMain->insertTab(1, qgb, QIcon(QString("")), "Filter");
	tabWidgetMain->insertTab(2, qgbSystem, QIcon(QString("")), "System");

	QScrollArea *scrollArea = new QScrollArea;
	scrollArea->setWidget(tabWidgetMain);
	scrollArea->setWidgetResizable(true);
	mainLayout00->addWidget(scrollArea);

	// mainLayout00->addLayout(layout0);
	// mainLayout00->addLayout(layout2);

    QTimer *timer = new QTimer(this);
    timer->start(1000);
//    connect(timer, &QTimer::timeout, this, &Endoveinous::showTime);
    timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));

	// connect(qpushbutton_dipxyopen, SIGNAL(clicked()), this, SLOT(trt_dipx_open()));
	// connect(qpushbutton_dipxzopen, SIGNAL(clicked()), this, SLOT(trt_dipz_open()));
	connect(qpushbutton_datainopen, SIGNAL(clicked()), this, SLOT(trt_seismic_open()));
	connect(qpushbutton_datainClear, SIGNAL(clicked()), this, SLOT(trt_seismic_clear()));
	connect(qpb_start, SIGNAL(clicked()), this, SLOT(trt_launch_thread()));
	connect(qpb_stop, SIGNAL(clicked()), this, SLOT(trt_stop()));
	connect(qpb_session_load, SIGNAL(clicked()), this, SLOT(trt_session_load()));
	connect(qpb_session_save, SIGNAL(clicked()), this, SLOT(trt_session_save()));

	// setMinimumHeight(900);
	// setMinimumWidth(700);
	resize(900, 900);
}

DipSmoothingWidget::~DipSmoothingWidget() {

}

int DipSmoothingWidget::getIndexFromVectorString(std::vector<QString> list, QString txt)
{
    for (int i=0; i<list.size(); i++)
    {
        if ( list[i].compare(txt) == 0 )
            return i;
    }
    return -1;
}

void DipSmoothingWidget::trt_seismic_open()
{
	if ( m_projectManager == nullptr ) return;
	std::vector<QString> seismicNames;
	std::vector<QString> seismicPath;
	seismicNames = m_projectManager->getSeismicAllNames();
	seismicPath = m_projectManager->getSeismicAllPath();

	FileSelectorDialog dialog(&seismicNames, "Select file name");
	dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::seismic);
	dialog.setMultipleSelection(true);
	int code = dialog.exec();
	if (code==QDialog::Accepted)
	{
		std::vector<int> selectedIdx = dialog.getMultipleSelectedIndex();
		for (int n=0; n<selectedIdx.size(); n++)
		{
			lw_dataIn->addItem(seismicNames[selectedIdx[n]]);
			v_seismicFilename.push_back(seismicPath[selectedIdx[n]]);
		}
	}
}

void DipSmoothingWidget::trt_seismic_clear()
{
	lw_dataIn->clear();
	v_seismicFilename.clear();
}

void DipSmoothingWidget::get_size_from_filename(QString filename, int *size)
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


// TODO
int DipSmoothingWidget::parameters_check()
{
	bool ok = false;
	int a = 0;
	if ( m_orientationWidget->getDipxyPath().isEmpty() ) return 0;
	if ( m_orientationWidget->getDipxzPath().isEmpty() ) return 0;
	int size0[3];
	int size[3];

	get_size_from_filename(m_orientationWidget->getDipxyPath().toStdString().c_str(), size0);
	get_size_from_filename(m_orientationWidget->getDipxzPath().toStdString().c_str(), size);
	for (int i=0; i<3; i++)
		if ( size0[i] != size[i] )
		{
			return 0;
		}
	if ( v_seismicFilename.size() == 0 ) return 0;
	for (int n=0; n<v_seismicFilename.size(); n++ )
	{
		get_size_from_filename(v_seismicFilename[n].toStdString().c_str(), size);
		for (int i=0; i<3; i++)
			if ( size0[i] != size[i] )
			{
				return 0;
			}
	}
	// if ( seismic_filename.isEmpty() ) return 0;
	if ( qlineedit_outfilename->text().isEmpty() ) return 0;
	a = qlineedit_dx->text().toInt(&ok); if ( ok == false ) return 0;
	a = qlineedit_dy->text().toInt(&ok); if ( ok == false ) return 0;
	a = qlineedit_dz->text().toInt(&ok); if ( ok == false ) return 0;
	return 1;
}


void DipSmoothingWidget::trt_launch_thread()
{
	if ( GLOBAL_RUN_TYPE == 1 ) return;
	if ( parameters_check() == 0 )
	{
		QMessageBox *msgBox = new QMessageBox(parentWidget());
		msgBox->setText("warning");
		msgBox->setInformativeText("Wrong parameters\nPlease fill in all the fields and check the data size");
		msgBox->setStandardButtons(QMessageBox::Ok );
		int ret = msgBox->exec();
		return;
	}
	GLOBAL_RUN_TYPE = 1;
	MyThreadDipSmoothing *thread = new MyThreadDipSmoothing(this);
    thread->start();
}

void DipSmoothingWidget::trt_stop()
{
	if ( GLOBAL_RUN_TYPE == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort the process ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
	    ihm_set_trt(IHM_TRT_DIP_SMOOTHING_STOP);
	    this->GLOBAL_RUN_TYPE = 0;
	}
}


QString DipSmoothingWidget::seismicFilenameToOutFilename(QString seismicFilename, QString suffix)
{
	int lastPoint = seismicFilename.lastIndexOf(".");
	QString prefix = seismicFilename.left(lastPoint);
	return prefix + "_" + suffix + ".xt";;
}

// todo
void DipSmoothingWidget::trt_start()
{
	int size[3];
	QString suffix = qlineedit_outfilename->text();
	int N = v_seismicFilename.size();
	if ( N == 0 ) return;
	vector <QString> outFilename;
	outFilename.resize(N);
	for (int n=0; n<N; n++)
	{
		outFilename[n] = seismicFilenameToOutFilename(v_seismicFilename[n], suffix);
	}
	get_size_from_filename(v_seismicFilename[0], size);

	int *tab_gpu = NULL, tab_gpu_size;
	tab_gpu = (int*)calloc(this->systemInfo->get_gpu_nbre(), sizeof(int));
	this->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);

	int filter_type = 2; // qcombobox_type->currentIndex();

	int dx0 = qlineedit_dx->text().toInt(); // 3
	int dy0 = qlineedit_dy->text().toInt(); // 7
	int dz0 = qlineedit_dz->text().toInt(); // 7

	GLOBAL_RUN_TYPE = 1;
	for (int n=0; n<N; n++)
	{
		QString txt = "current data: " + QString::number(n+1) + " / " + QString::number(N);
		int ret = filter_dip(filter_type, m_orientationWidget->getDipxyPath().toStdString().c_str(), m_orientationWidget->getDipxzPath().toStdString().c_str(), v_seismicFilename[n].toStdString().c_str(),
				outFilename[n].toStdString().c_str(), size[1], size[0], size[2], dx0, dy0, dz0, tab_gpu, tab_gpu_size,
				txt.toStdString().c_str());
		if ( GLOBAL_RUN_TYPE == 0 ) break;
	}
	GLOBAL_RUN_TYPE = 0;
	if ( tab_gpu ) free(tab_gpu);
}



void DipSmoothingWidget::showTime()
{
    // GLOBAL_textInfo->appendPlainText(QString("timer"));
    char txt[1000], txt2[1000];

    if ( GLOBAL_RUN_TYPE == 0 )
    {
    	qpb_progress->setValue(0);
    	qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
    	qpb_progress->setFormat("");
    	return;
    }

    int type;
    long idx, vmax;
    int msg_new = ihm_get_global_msg(&type, &idx, &vmax, txt);
    if ( msg_new == 0 ) return;
    if ( type == IHM_TYPE_DIP_FILTER )
    {
    	float val_f = 100.0*idx/vmax;
    	int val = (int)(val_f);
    	qpb_progress->setValue(val);
    	sprintf(txt2, "run %s - %.1f%%", txt, val_f);
    	qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
    	qpb_progress->setFormat(txt2);
    }
}

void DipSmoothingWidget::trt_session_load()
{
	m_projectManager->load_session_gui();
}

void DipSmoothingWidget::trt_session_save()
{
	m_projectManager->save_session_gui();
}


MyThreadDipSmoothing::MyThreadDipSmoothing(DipSmoothingWidget *p)
{
    this->pp = p;
}

void MyThreadDipSmoothing::run()
{
    pp->trt_start();
}




