
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>

#include <OpenFileWidget.h>
#include <ihm.h>
#include <fileio2.h>
#include <SpectrumComputeWidget.h>
#include <rgb2ToXt.h>
#include <rgb2ToXtWidget.h>




Rgb2ToXtWidget::Rgb2ToXtWidget(ProjectManagerWidget *projectManager, QWidget* parent)
{
	m_projectManager = projectManager;
	mainGroupBox = new QGroupBox();

	QVBoxLayout *layout = new QVBoxLayout(mainGroupBox);

	/*
	QHBoxLayout *hlayout1 = new QHBoxLayout;
	QLabel *rgb2Label = new QLabel("rgb2 filename");
	qleRgb2Filename = new QLineEdit();
	QPushButton *pb_rgb2Filename = new QPushButton("...");
	hlayout1->addWidget(rgb2Label);
	hlayout1->addWidget(qleRgb2Filename);
	hlayout1->addWidget(pb_rgb2Filename);
	*/
	m_rgb2FileSelectWidget = new FileSelectWidget();
	m_rgb2FileSelectWidget->setProjectManager(m_projectManager);
	m_rgb2FileSelectWidget->setLabelText("rgb2 filename");
	m_rgb2FileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::rgtCubeToAttribut);
	m_rgb2FileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Raw);
	m_rgb2FileSelectWidget->setLabelDimensionVisible(false);


	QHBoxLayout *hlayout2 = new QHBoxLayout;
	QLabel *qlXtFilenamePrefix = new QLabel("prefix");
	qleXtFilenamePrefix = new QLineEdit("rgb2");
	hlayout2->addWidget(qlXtFilenamePrefix);
	hlayout2->addWidget(qleXtFilenamePrefix);

	qpb_progress = new QProgressBar;
	qpbStart = new QPushButton("Start");

	layout->addWidget(m_rgb2FileSelectWidget);
	layout->addLayout(hlayout2);
	layout->addWidget(qpb_progress);
	layout->addWidget(qpbStart);
	mainGroupBox->setMaximumHeight(300);

	connect(qpbStart, SIGNAL(clicked()), this, SLOT(trt_launchThread()));
	// connect(pb_rgb2Filename, SIGNAL(clicked()), this, SLOT(trt_rgb2FilenameOpen()));


	timer = new QTimer(this);
    timer->start(1000);
    // timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
    timer->disconnect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
}


Rgb2ToXtWidget::~Rgb2ToXtWidget()
{

}



void Rgb2ToXtWidget::setProjectManagerWidget(ProjectManagerWidget *projectManagerWidget)
{
	m_projectManager = projectManagerWidget;
}

void Rgb2ToXtWidget::setSpectrumComputeWidget(SpectrumComputeWidget *spectrumComputeWidget)
{
	m_spectrumComputeWidget = spectrumComputeWidget;
}

QGroupBox *Rgb2ToXtWidget::getMainGroupBox()
{
	return mainGroupBox;
}



void Rgb2ToXtWidget::showTime()
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
    	case IHM_TYPE_RGB2TOXT:
    	{
        	float val_f = 100.0*idx/vmax;
        	int val = (int)(val_f);
        	qpb_progress->setValue(val);
        	sprintf(txt2, "%s %.1f%%", txt, val_f);
        	qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
        	qpb_progress->setFormat(txt2);
        	break;
    	}
    }
}

void Rgb2ToXtWidget::trt_rgb2FilenameOpen()
{
	/*
	std::vector<QString> v_rgb_names = this->m_projectManagerWidget->get_rgb_names();
	std::vector<QString> v_rgb_filenames = this->m_projectManagerWidget->get_rgb_fullnames();
	char buff[10000];
	trt_open_file(v_rgb_names, buff, false);
	if ( buff[0] == 0 ) return;
	this->rgb2Name = QString(buff);
	this->qleRgb2Filename->setText(this->rgb2Name);
	int idx = getIndexFromVectorString(v_rgb_names, this->rgb2Name);
	if ( idx >= 0 )
		this->rgb2Filename = v_rgb_filenames[idx];
	fprintf(stderr, "%s\n", v_rgb_filenames[0].toStdString().c_str());
	*/

	/*
	std::vector<QString> v_seismic_names = m_projectManagerWidget->get_rgb_names();
	std::vector<QString> v_seismic_filenames = m_projectManagerWidget->get_rgb_fullnames();
	if ( v_seismic_names.empty() ) return;

	OpenFileWidget *p = new OpenFileWidget(this, v_seismic_names, v_seismic_filenames);
	if ( !p->exec() ) return;
	rgb2Name = p->getSelectedTinyName();
	rgb2Filename = p->getSelectedFullName();
	this->qleRgb2Filename->setText(rgb2Name);
	*/
}



int Rgb2ToXtWidget::filenamesUpdate()
{
	QString seismicTinyName = m_spectrumComputeWidget->getSeismicTinyName();
	QString seismicFullName = m_spectrumComputeWidget->getSeismicFullName();

	if ( seismicTinyName.compare("") == 0 )
	{
		xtFilename = "";
		return 0;
	}

	QString ImportExportPath = m_projectManager->getImportExportPath();
	QString IJKPath = m_projectManager->getIJKPath();
	QString seimsicNamePath = m_projectManager->getIJKPath() + QString(seismicTinyName) + "/";
	QString cubeRgt2RgbPath = seimsicNamePath + "cubeRgt2RGB/";
	QString prefix = qleXtFilenamePrefix->text();

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


	QString ret = "";
	QString path = cubeRgt2RgbPath;
	xtFilename = path + "/" + "xt_from_" + prefix + "_from_" + m_rgb2FileSelectWidget->getFilename() + ".xt";

	return 1;
}


void Rgb2ToXtWidget::trt_run()
{
	if ( filenamesUpdate() == 0 ) return;
	// std::string seismicFilename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/DATA/SEISMIC/seismic3d.HR_NEAR.xt";
	// std::string rgb2Filename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/HR_NEAR/cubeRgt2RGB/rgb2_GCC_JDgcc16_from_HR_NEAR_size_1500x700x1280.raw";
	// std::string xtFilename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/HR_NEAR/cubeRgt2RGB/xt_from_rgb1__from_rgb2_spectrum_from_HR_NEAR2_size_1500x700x1280__alpha_1x0__ratio_x0001_size_1500x700x1280_xt-lk_dims_400x1500x700_V2.xt";
	std::string seismicFilename = m_spectrumComputeWidget->getSeismicFullName().toStdString();
	Rgb2ToXt *p = new Rgb2ToXt();
	p->setSeismicFilename(seismicFilename);
	p->setRgb2Filename(m_rgb2FileSelectWidget->getPath().toStdString());
	p->setXtFilename(xtFilename.toStdString());
	globalRun = 1;
	qpbStart->setText("stop");
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	p->run();
	delete p;
	globalRun = 0;
	timer->disconnect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	qpb_progress->setValue(0);
	qpbStart->setText("start");
}

void Rgb2ToXtWidget::trt_launchThread()
{
	if ( globalRun == 0 )
	{
		if ( thread == nullptr )
			thread = new MyThreadRGB2TOXTWidget(this);
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
		    ihm_set_trt(IHM_TYPE_RGB2TOXT_END);
			qpbStart->setText("start");
		}
	}
}

// ================== THREAD

MyThreadRGB2TOXTWidget::MyThreadRGB2TOXTWidget(Rgb2ToXtWidget *p)
 {
     this->pp = p;
 }

 void MyThreadRGB2TOXTWidget::run()
 {
	 pp->trt_run();
 }
