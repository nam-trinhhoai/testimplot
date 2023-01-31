

#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>

#include <OpenFileWidget.h>
#include <ihm.h>
#include <fileio2.h>
#include <SpectrumComputeWidget.h>
#include "gradient_multiscale/rgtToGCCProcess.h"
#include <rgtToGCC16Widget.h>




RgtToGCC16Widget::RgtToGCC16Widget(ProjectManagerWidget *projectManager, QWidget* parent)
{
	setProjectManagerWidget(projectManager);
	mainGroupBox = new QGroupBox();

	QVBoxLayout *layout = new QVBoxLayout(mainGroupBox);

	/*
	QHBoxLayout *hlayout1 = new QHBoxLayout;
	QLabel *rgtLabel = new QLabel("rgt filename");
	qleRgtFilename = new QLineEdit();
	QPushButton *pb_rgtFilename = new QPushButton("...");
	hlayout1->addWidget(rgtLabel);
	hlayout1->addWidget(qleRgtFilename);
	hlayout1->addWidget(pb_rgtFilename);
	*/

	m_rgtFileSelectWidget = new FileSelectWidget();
	m_rgtFileSelectWidget->setProjectManager(m_projectManager);
	m_rgtFileSelectWidget->setLabelText("rgt filename");
	m_rgtFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Rgt);
	m_rgtFileSelectWidget->setLabelDimensionVisible(false);

	QHBoxLayout *hlayout2 = new QHBoxLayout;
	QLabel *qlGccFilenamePrefix = new QLabel("prefix");
	qleGccFilenamePrefix = new QLineEdit("gcc16");
	hlayout2->addWidget(qlGccFilenamePrefix);
	hlayout2->addWidget(qleGccFilenamePrefix);

	QHBoxLayout *hlayout3 = new QHBoxLayout;
	QLabel *qlWindowSize = new QLabel("window size");
	qleWindowSize = new QLineEdit("6");
	hlayout3->addWidget(qlWindowSize);
	hlayout3->addWidget(qleWindowSize);

	QHBoxLayout *hlayout4 = new QHBoxLayout;
	qcbRgtHorizonChoice = new QComboBox;
	qcbRgtHorizonChoice->addItem("Isovalue");
	qcbRgtHorizonChoice->addItem("Horizons");
	qcbRgtHorizonChoice->setMaximumWidth(200);
	qcbRgtHorizonChoice->setEnabled(false);
	QLabel *qlIsoMin = new QLabel("iso min");
	qleIsoMin = new QLineEdit("0");
	QLabel *qlIsoMax = new QLabel("iso max");
	qleIsoMax = new QLineEdit("32000");
	QLabel *qlIsoStep = new QLabel("iso step");
	qleIsoStep = new QLineEdit("25");
	hlayout4->addWidget(qcbRgtHorizonChoice);

	/*
	QFrame* line = new QFrame();
	line->setFrameShape(QFrame::VLine);
	line->setFrameShadow(QFrame::Sunken);
	hlayout4->addWidget(line);
	*/

	hlayout4->addWidget(qlIsoMin);
	hlayout4->addWidget(qleIsoMin);
	hlayout4->addWidget(qlIsoMax);
	hlayout4->addWidget(qleIsoMax);
	hlayout4->addWidget(qlIsoStep);
	hlayout4->addWidget(qleIsoStep);

	QHBoxLayout *hlayout5 = new QHBoxLayout;
	QLabel *qlW = new QLabel("W");
	qleW = new QLineEdit("7");
	hlayout5->addWidget(qlW);
	hlayout5->addWidget(qleW);

	QHBoxLayout *hlayout6 = new QHBoxLayout;
	QLabel *qlShift = new QLabel("shift");
	qleShift = new QLineEdit("5");
	hlayout5->addWidget(qlShift);
	hlayout5->addWidget(qleShift);


	qpb_progress = new QProgressBar;
	qpbStart = new QPushButton("Start");

	layout->addWidget(m_rgtFileSelectWidget);
	layout->addLayout(hlayout2);
	// layout->addLayout(hlayout3);
	layout->addLayout(hlayout4);
	layout->addLayout(hlayout5);
	layout->addWidget(qpb_progress);
	layout->addWidget(qpb_progress);
	layout->addWidget(qpbStart);
	mainGroupBox->setMaximumHeight(300);

	connect(qpbStart, SIGNAL(clicked()), this, SLOT(trt_launchThread()));
	// connect(pb_rgtFilename, SIGNAL(clicked()), this, SLOT(trt_rgtFilenameOpen()));

	timer = new QTimer(this);
    timer->start(1000);
    timer->disconnect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
}


RgtToGCC16Widget::~RgtToGCC16Widget()
{
	if ( thread != nullptr ) delete thread;
	if ( timer != nullptr ) delete timer;
}



void RgtToGCC16Widget::setProjectManagerWidget(ProjectManagerWidget *projectManagerWidget)
{
	m_projectManager = projectManagerWidget;
}

void RgtToGCC16Widget::setSpectrumComputeWidget(SpectrumComputeWidget *spectrumComputeWidget)
{
	m_spectrumComputeWidget = spectrumComputeWidget;
}

QGroupBox *RgtToGCC16Widget::getMainGroupBox()
{
	return mainGroupBox;
}

int RgtToGCC16Widget::getSizeFromFilename(QString filename, int *size)
{
	if ( filename.compare("") == 0 ) { for (int i=0; i<3;i++) size[i] = 0; return 0; }
    FILEIO2 *pf = new FILEIO2();
    pf->openForRead((char*)filename.toStdString().c_str());
    size[0] = pf->get_dimy();
    size[1] = pf->get_dimx();
    size[2] = pf->get_dimz();
    delete pf;
    if ( size[0] == 0 && size[1] == 0 && size[2] == 0 ) return 0;
    return 1;
}

void RgtToGCC16Widget::trt_rgtFilenameOpen()
{
	/*
	std::vector<QString> v_seismic_names = m_projectManagerWidget->get_seismic_names();
	std::vector<QString> v_seismic_filenames = m_projectManagerWidget->get_seismic_fullpath_names();
	if ( v_seismic_names.empty() ) return;

	OpenFileWidget *p = new OpenFileWidget(this, v_seismic_names, v_seismic_filenames);
	if ( !p->exec() ) return;
	rgtTinyName = p->getSelectedTinyName();
	rgtFullName = p->getSelectedFullName();
	inri::Xt xt(rgtFullName.toStdString().c_str());
	if (!xt.is_valid()) return;
	if (xt.type()!=inri::Xt::Signed_16) {
		QMessageBox::warning(this, "Fail to load cube", "Selected cube is not of type : \"signed short\", abort selection");
		rgtTinyName = "";
		rgtFullName = "";
	}
	this->qleRgtFilename->setText(rgtTinyName);
	*/
}



int RgtToGCC16Widget::filenamesUpdate()
{
	QString seismicTinyName = m_spectrumComputeWidget->getSeismicTinyName();
	QString seismicFullName = m_spectrumComputeWidget->getSeismicFullName();

	if ( seismicTinyName.compare("") == 0 )
	{
		gccFilename = "";
		return 0;
	}

	QString ImportExportPath = m_projectManager->getImportExportPath();
	QString IJKPath = m_projectManager->getIJKPath();
	QString seimsicNamePath = m_projectManager->getIJKPath() + QString(seismicTinyName) + "/";
	QString cubeRgt2RgbPath = seimsicNamePath + "cubeRgt2RGB/";
	QString prefix = qleGccFilenamePrefix->text();

	QDir ImportExportDir(ImportExportPath);
	if ( !ImportExportDir.exists() )
	{
		QDir dir;
		dir.mkdir(ImportExportPath);
	}

	QDir IJKDir(IJKPath);
	if ( !IJKDir.exists() )
	{
		QDir dir;
		dir.mkdir(IJKPath);
	}

	QDir seismicNameDir(seimsicNamePath);
	if ( !seismicNameDir.exists() )
	{
		QDir dir;
		dir.mkdir(seimsicNamePath);
	}

	QDir cubeRgt2RgbDir(cubeRgt2RgbPath);
	if ( !cubeRgt2RgbDir.exists() )
	{
		QDir dir;
		dir.mkdir(cubeRgt2RgbPath);
	}


	int size[3];
	getSizeFromFilename(seismicFullName, size);
	int width = size[0];
	int height = size[2];
	int depth = 32000 / qleIsoStep->text().toInt();

	QString ret = "";
	QString path = cubeRgt2RgbPath;
	gccFilename = path + "/" + "rgb2_GCC_" + prefix + "_from_" + seismicTinyName + "_size_" + QString::number(width) + "x" + QString::number(height) + "x" + QString::number(depth) + ".raw";

	return 1;
}


void RgtToGCC16Widget::showTime()
{
    char txt[1000], txt2[1000];
    if ( globalRun == 0  )
    {
    	qpb_progress->setValue(0);
    	qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
    	qpb_progress->setFormat("");
    	return;
    }

    int type = -1;
    long idx, vmax;
    int msg_new = ihm_get_global_msg(&type, &idx, &vmax, txt);
    if ( msg_new == 0 ) return;
    switch ( type )
    {
    	case IHM_TYPE_RGTTOGCC:
    	{
        	float val_f = 100.0*idx/vmax;
        	int val = (int)(val_f);
        	qpb_progress->setValue(val);
        	sprintf(txt2, "run %.1f%%", val_f);
        	qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
        	qpb_progress->setFormat(txt2);
        	break;
    	}
    }
}

void RgtToGCC16Widget::trt_run()
{
	int isoMin = qleIsoMin->text().toInt();
	int isoMax = qleIsoMax->text().toInt();
	int isoStep = qleIsoStep->text().toInt();
	int windowSize = qleWindowSize->text().toInt();
	int w = qleW->text().toInt();
	int shift = qleShift->text().toInt();
	std::string seismicFilename = m_spectrumComputeWidget->getSeismicFullName().toStdString();
	// std::string seismicFilename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/DATA/SEISMIC/seismic3d.HR_NEAR.xt";
	// std::string rgtFilename = m_spectrumComputeWidget->getRgtFullName().toStdString();
	// std::string rgtFilename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/DATA/SEISMIC/seismic3d.HR_NEAR_rgt.xt";
	// std::string gccFilename = "/data/PLI/NKDEEP/jacques/gcc.raw";

	// todo
	// message box
	if ( filenamesUpdate() == 0 ) return;
	RgtToGCCProcess *p = new RgtToGCCProcess();
	p->setSeismicFilename(seismicFilename);
	p->setRgtFilename(m_rgtFileSelectWidget->getPath().toStdString());
	p->setGCCFilename(gccFilename.toStdString());
	// p->setIsoStep(isoStep);
	p->setIsoValues(isoMin, isoMax, isoStep);
	p->setWindowSize(windowSize);
	p->setW(w);
	p->setShift(shift);
	globalRun = 1;
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	qpbStart->setText("stop");
	p->run();
	delete p;
	globalRun = 0;
	timer->disconnect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	qpb_progress->setValue(0);
	qpbStart->setText("start");
}

void RgtToGCC16Widget::trt_launchThread()
{
	if ( globalRun == 0 )
	{
		if ( thread == nullptr )
			thread = new MyThreadRgtToGCC16Widget(this);
		// m_functionType = 1;
		thread->start();
	}
	else
	{
		QMessageBox *msgBox = new QMessageBox(parentWidget());
		msgBox->setText("warning");
		msgBox->setInformativeText("Do you really want to abort the process ?");
		msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
		int ret = msgBox->exec();
		if ( ret == QMessageBox::Yes )
		{
		    ihm_set_trt(IHM_TYPE_RGTTOGCC_END);
			qpbStart->setText("start");
		}
	}
}

// ================== THREAD

MyThreadRgtToGCC16Widget::MyThreadRgtToGCC16Widget(RgtToGCC16Widget *p)
 {
     this->pp = p;
 }

 void MyThreadRgtToGCC16Widget::run()
 {
	 pp->trt_run();
 }
