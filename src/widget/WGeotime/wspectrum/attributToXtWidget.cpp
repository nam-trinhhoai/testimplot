
#include <QMessageBox>

#include <stdio.h>
#include <malloc.h>

#include <rgtSpectrumHeader.h>
#include <QFileUtils.h>
#include <fileSelectorDialog.h>
#include <Xt.h>
#include <QTemporaryFile>
#include <QDebug>
#include <cuda_rgb2torgb1.h>
#include <attributToXtWidget.h>

AttributToXtWidget::AttributToXtWidget(ProjectManagerWidget *selectorWidget, QWidget* parent)
{
	m_selectorWidget = selectorWidget;

	QVBoxLayout * mainLayout = new QVBoxLayout(this);

	QHBoxLayout *qhbAttributType = new QHBoxLayout;
	QLabel *label_attributType = new QLabel("data type");
	m_attributType = new QComboBox;
	m_attributType->addItem("spectrum");
	m_attributType->addItem("gcc");
	// m_attributType->addItem("mean");
	m_attributType->setMaximumWidth(200);
	qhbAttributType->addWidget(label_attributType);
	qhbAttributType->addWidget(m_attributType);
	qhbAttributType->setAlignment(Qt::AlignLeft);

	QLabel *labelAttributDirectory = new QLabel("data");
	m_lineEditAttributDirectory = new QLineEdit;
	QPushButton *m_pushButtonAttributDirectory = new QPushButton("...");

	QHBoxLayout *hbAttributDirectory = new QHBoxLayout;
	hbAttributDirectory->addWidget(labelAttributDirectory);
	hbAttributDirectory->addWidget(m_lineEditAttributDirectory);
	hbAttributDirectory->addWidget(m_pushButtonAttributDirectory);

	/*
	m_attributDirectoty = new FileSelectWidget();
	m_attributDirectoty->setLabelText("data filename");
	m_attributDirectoty->setProjectManager(m_selectorWidget);
	m_attributDirectoty->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Rgt);
	m_attributDirectoty->setFileType(FileSelectWidget::FILE_TYPE::seismic);
	*/

	m_prefixFilename = new LabelLineEditWidget();
	m_prefixFilename->setLabelText("prefix");
	m_prefixFilename->setLineEditText("spectrum");

	QHBoxLayout *qhbButtons = new QHBoxLayout;
	m_start = new QPushButton("Start");
	m_stop = new QPushButton("Stop");
	// m_kill = new QPushButton("Kill");
	qhbButtons->addWidget(m_start);
	qhbButtons->addWidget(m_stop);
	// qhbButtons->addWidget(m_kill);



	m_progressBar = new QProgressBar();
	m_progressBar->setMinimum(0);
	m_progressBar->setMaximum(100);
	m_progressBar->setValue(0);
	// m_progressBar->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,0,0)}");
	m_progressBar->setTextVisible(true);
	m_progressBar->setFormat("");

	mainLayout->addLayout(qhbAttributType);
	// mainLayout->addWidget(m_attributDirectoty);
	mainLayout->addLayout(hbAttributDirectory);
	mainLayout->addWidget(m_prefixFilename);
	mainLayout->addWidget(m_progressBar);
	mainLayout->addLayout(qhbButtons);

	mainLayout->setAlignment(Qt::AlignTop);

	setMaximumHeight(450);

	timer = new QTimer();
	timer->start(1000);
    timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	// connect(m_isoValOrHorizon, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_horizonTypeDisplay(int)));
	connect(m_start, SIGNAL(clicked()), this, SLOT(trt_start()));
	connect(m_stop, SIGNAL(clicked()), this, SLOT(trt_stop()));
	connect(m_pushButtonAttributDirectory, SIGNAL(clicked()), this, SLOT(trt_attributDirectoryOpen()));
	// horizonTypeDisplay();
	setStartStopStatus(STATUS_STOP);

}

AttributToXtWidget::~AttributToXtWidget()
{

}

void AttributToXtWidget::trt_attributDirectoryOpen()
{
	std::vector<QString> names = m_selectorWidget->getHorizonIsoValueListName();
	std::vector<QString> path = m_selectorWidget->getHorizonIsoValueListPath();

	FileSelectorDialog dialog(&names, "Select file name");
	int code = dialog.exec();
	if (code==QDialog::Accepted)
	{
		int selectedIdx = dialog.getSelectedIndex();
		if (selectedIdx>=0 && selectedIdx<names.size())
		{
			m_lineEditAttributDirectory->setText(names[selectedIdx]);
			m_attributDirectory = path[selectedIdx];
		}
	}
}


void AttributToXtWidget::setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget)
{
	m_spectrumProcessWidget = spectrumProcessWidget;
}

std::vector<std::string> AttributToXtWidget::getIsoPath(QString path)
{
	// todo
	// if ( m_attributDirectoty->getPath().isEmpty() ) return std::vector<std::string>();
	std::vector<std::string> out;

	QDir *dir = new QDir(path);
	dir->setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	// dir->setSorting(QDir::Name);
	QFileInfoList list = dir->entryInfoList();
	for (int n=0; n<list.size(); n++)
	{
		QString path0 = list.at(n).fileName();
		if ( path0.contains("iso_") )
		{
			QString tmp = path + "/" + path0 + "/";
			out.push_back(tmp.toStdString());
		}
	}
	for (std::string str:out)
	{
		qDebug() << QString::fromStdString(str);
	}
	return out;
}


int AttributToXtWidget::paramCreate()
{
	// param.directory = m_attributDirectoty->getPath();
	// param.outFilename = m_selectorWidget->getSeismicDirectory() + "/" + m_prefixFilename->getLineEditText() + ".xt";
	// param.outFilename = m_attributDirectoty->getPath() + "/../xt_" + m_attributType->currentText() + "_from_" + m_spectrumProcessWidget->getSeismicName() + "_" + m_prefixFilename->getLineEditText() + ".xt";
	param.directory = m_attributDirectory + "/";
	QString seismicName = m_spectrumProcessWidget->getSeismicName();
	param.outFilename = m_selectorWidget->getIJKPath() + "/" + seismicName + "/cubeRgt2RGB/" + "xt_" + m_attributType->currentText() + "_from_" + m_spectrumProcessWidget->getSeismicName() + "_" + m_prefixFilename->getLineEditText() + ".xt";

	mkdirPathIfNotExist(m_selectorWidget->getIJKPath());
	mkdirPathIfNotExist(m_selectorWidget->getIJKPath() + "/" + seismicName);
	mkdirPathIfNotExist(m_selectorWidget->getIJKPath() + "/" + seismicName + "/cubeRgt2RGB/");

	qDebug() << param.outFilename;

	param.seismicFilename = m_spectrumProcessWidget->getSeismicPath();
	if ( m_attributType->currentText().compare("spectrum") == 0 )
	{
		param.dataType = SPECTRUM_NAME;
		param.type = AttributToXt::TYPE0::rgb2;
	}
	else if ( m_attributType->currentText().compare("gcc") == 0 )
	{
		param.dataType = GCC_NAME;
		param.type = AttributToXt::TYPE0::gcc;
	}
	// else if ( m_attributType->currentText().compare("mean") == 0 )
	//	param.type = AttributToXt::TYPE0::mean;
	return 1;
}



void AttributToXtWidget::trt_startStop()
{
}


void AttributToXtWidget::trt_start()
{
	if ( valStartStop == 1 ) return;
	if ( !paramCreate() ) return;
	AttributToXtWidgetTHREAD *thread = new AttributToXtWidgetTHREAD(this);
	thread->start();
}

void AttributToXtWidget::trt_stop()
{
	if ( valStartStop == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort the process ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		if ( pIhm2 ) pIhm2->setMasterMessage("stop", 0, 1, 1);
		setStartStopStatus(STATUS_STOP);
	}
}

void AttributToXtWidget::setStartStopStatus(START_STOP_STATUS status)
{
	if ( status == STATUS_STOP )
	{
		valStartStop = 0;
		m_start->setEnabled(true);
		m_stop->setEnabled(false);
		// m_progressBar->setValue(0);
		// m_progressBar->setFormat("");
	}
	else
	{
		valStartStop = 1;
		m_start->setEnabled(false);
		m_stop->setEnabled(true);
		// timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
		if ( pIhm2 ) pIhm2->clearSlaveMessage();
	}
}


void AttributToXtWidget::showTime()
{
	if ( !pIhm2 ) return;
	if ( valStartStop == 1 && pIhm2->isSlaveMessage() )
	{
		Ihm2Message mess = pIhm2->getSlaveMessage();
		float val_f = 100.0*mess.count/mess.countMax;
		int val = (int)(val_f);
		m_progressBar->setValue(val);
		m_progressBar->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
		QString text = QString::fromStdString(mess.message) + " " + QString::number(val_f, 'f', 1) + "%";
		m_progressBar->setFormat(text);
	}
	else if ( valStartStop == 0 )
	{
		m_progressBar->setValue(0);
		m_progressBar->setFormat("");
	}
}

void AttributToXtWidget::trt_threadRun()
{
	QString ImportExportPath = m_selectorWidget->getImportExportPath();
	QString IJKPath = m_selectorWidget->getIJKPath();
	// QString rgtPath = IJKPath + "/" + m_attributDirectoty->getFilename() + "/";
	QString rgtPath = m_attributDirectory;

	int *tab_gpu = NULL, tab_gpu_size;
	tab_gpu = (int*)calloc(m_spectrumProcessWidget->m_systemInfo->get_gpu_nbre(), sizeof(int));
	m_spectrumProcessWidget->m_systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	std::vector<std::string> isoPath = getIsoPath(rgtPath);

	inri::Xt xt((char*)param.seismicFilename.toStdString().c_str());
	int dimx = xt.nSamples();
	int dimy = xt.nRecords();
	int dimz = xt.nSlices();

	// QDir tempDir = QDir(m_attributDirectoty->getPath());
	// QString tmp = tempDir.absoluteFilePath("NextVision_rgb1_XXXXXX.rgb");

	/*
	QDir tempDir = QDir(m_attributDirectoty->getPath());
	QTemporaryFile rgb1RawFile;
	rgb1RawFile.setFileTemplate(tempDir.absoluteFilePath("NextVision_rgb1_XXXXXX.rgb"));
	// outSectionVideoRawFile.setFileTemplate(QDir("/data/PLI/NKDEEP/jacques/").absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	rgb1RawFile.setAutoRemove(false);
	rgb1RawFile.open();
	rgb1RawFile.close();
	QString tmp = rgb1RawFile.fileName();
	qDebug() << tmp;
	*/

	QString tmp = getTmpFilename(rgtPath, "rgb1_", ".rgb");
	qDebug() << tmp;

	if ( !pIhm2 ) pIhm2 = new Ihm2();

	valStartStop = 1;
	setStartStopStatus(STATUS_START);

	int ret = 1;
	// todo
	/*
	int ret = cuda_rgb2torgb1ByDirectories(isoPath, dimy, dimz, isoPath.size(), rgb2Torgb1Ratio, rgb2Torgb1Alpha, (char*)tmp.toStdString().c_str(),
			param.dataType.toStdString(),
			m_spectrumProcessWidget->getSeismicName().toStdString(),
			pIhm2);
			*/

	if ( ret == 1 )
	{
		AttributToXt *p = new AttributToXt();
		p->setDataPath(param.directory.toStdString());
		p->setSeismicFilename(param.seismicFilename.toStdString());
		p->setOutFilename(param.outFilename.toStdString());
		p->setDataType(param.type);
		p->setIsoPath(isoPath);
		p->setRgb1Filename(tmp.toStdString());
		p->setSeismicName(m_spectrumProcessWidget->getSeismicName().toStdString());
		p->setIhm(pIhm2);
		p->run();
		if ( m_selectorWidget ) m_selectorWidget->RgbRawUpdateNames();
	}
	valStartStop = 0;
	setStartStopStatus(STATUS_STOP);


	// qDebug() << param.outFilename;
	// qDebug() << param.directory;

/*
	RgtToAttribut *p = new RgtToAttribut();
	p->setSeismicFilename(paramInit.seismicFilename0);
	p->setRgtFilename((char*)paramInit.rgtFilename0);
	p->setOutRawFilename(paramInit.rgb2TmpFullName0);
	p->setAttributType(RgtToAttribut::TYPE::mean);
	p->setAttributData(RgtToAttribut::DATA::iso);
	p->setIsoStep(paramInit.isoStep);
	p->setWSize(paramInit.windowSize);
	p->setMeanDirectory(paramInit.strOutMeanDirectory);
	p->setIsoDirectory(paramInit.strOutHorizonsDirectory);
	p->setMeanDirectory(paramInit.strOutMeanDirectory);
	p->run();

	FREE(tab_gpu)
	setStartStopStatus(STATUS_STOP);
	*/

}



// =====================================================
AttributToXtWidgetTHREAD::AttributToXtWidgetTHREAD(AttributToXtWidget *p)
 {
     this->pp = p;
 }

 void AttributToXtWidgetTHREAD::run()
 {
	 fprintf(stderr, "thread start\n");
	 pp->trt_threadRun();
 }


