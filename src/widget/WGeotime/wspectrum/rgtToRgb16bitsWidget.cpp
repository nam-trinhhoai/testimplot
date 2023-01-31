
#include <QVBoxLayout>
#include <QMessageBox>
#include <QDebug>
#include <QTimer>
#include <QMessageBox>

#include <QFileUtils.h>

#include <Xt.h>
#include <util.h>
#include <cuda_rgt2rgb.h>
#include <spectrumProcessWidget.h>
#include <checkDataSizeMatch.h>
#include <rgtToAttribut.h>
#include <horizonUtils.h>
#include <rgtToRgb16bitsWidget.h>

RgtToRgb16bitsWidget::RgtToRgb16bitsWidget(ProjectManagerWidget *selectorWidget, QWidget* parent)
{
	m_selectorWidget = selectorWidget;

	QVBoxLayout * mainLayout = new QVBoxLayout(this);

	m_rgtFileSelectWidget = new FileSelectWidget();
	m_rgtFileSelectWidget->setLabelText("rgt filename");
	m_rgtFileSelectWidget->setProjectManager(m_selectorWidget);
	m_rgtFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Rgt);


	m_prefixFilename = new LabelLineEditWidget();
	m_prefixFilename->setLabelText("prefix");
	m_prefixFilename->setLineEditText(param.prefix);
	m_windowSize = new LabelLineEditWidget();
	m_windowSize->setLabelText("window size");
	m_windowSize->setLineEditText(QString::number(param.windowSize));

	QHBoxLayout *qhbIsoValOrHorizon = new QHBoxLayout;

	m_isoValOrHorizon = new QComboBox;
	m_isoValOrHorizon->addItem("Isovalue");
	m_isoValOrHorizon->addItem("Horizons");
	m_isoValOrHorizon->setMaximumWidth(200);
	m_isoStep = new LabelLineEditWidget();
	m_isoStep->setLabelText("iso step");
	m_isoStep->setLineEditText(QString::number(param.isoStep));

	m_layerNumber = new LabelLineEditWidget();
	m_layerNumber->setLabelText("layer number");
	m_layerNumber->setLineEditText(QString::number(param.layerNumber));

	qhbIsoValOrHorizon->addWidget(m_isoValOrHorizon);
	qhbIsoValOrHorizon->addWidget(m_isoStep);
	qhbIsoValOrHorizon->addWidget(m_layerNumber);

	m_horizonSelectWidget = new HorizonSelectWidget(this);
	m_horizonSelectWidget->setProjectManager(m_selectorWidget);

	m_progressBar = new QProgressBar();
	m_progressBar->setMinimum(0);
	m_progressBar->setMaximum(100);
	m_progressBar->setValue(0);
	// m_progressBar->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,0,0)}");
	m_progressBar->setTextVisible(true);
	m_progressBar->setFormat("");

	QHBoxLayout *qhbButtons = new QHBoxLayout;
	m_start = new QPushButton("Start");
	m_stop = new QPushButton("Stop");
	m_kill = new QPushButton("Kill");
	qhbButtons->addWidget(m_start);
	qhbButtons->addWidget(m_stop);
	// qhbButtons->addWidget(m_kill);

	mainLayout->addWidget(m_rgtFileSelectWidget);
	// mainLayout->addWidget(m_prefixFilename);
	mainLayout->addWidget(m_windowSize);
	mainLayout->addLayout(qhbIsoValOrHorizon);
	mainLayout->addWidget(m_horizonSelectWidget);
	mainLayout->addWidget(m_progressBar);
	mainLayout->addLayout(qhbButtons);

	mainLayout->setAlignment(Qt::AlignTop);
	setMaximumHeight(450);

	timer = new QTimer();
	timer->start(1000);

    connect(m_isoValOrHorizon, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_horizonTypeDisplay(int)));
    connect(m_start, SIGNAL(clicked()), this, SLOT(trt_start()));
    connect(m_stop, SIGNAL(clicked()), this, SLOT(trt_stop()));
    timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
    // connect(m_start, SIGNAL(clicked()), this, SLOT(trt_start()));
    horizonTypeDisplay();
    // setStartStopStatus(STATUS_STOP);
    m_start->setEnabled(true);
    m_stop->setEnabled(false);
}



RgtToRgb16bitsWidget::~RgtToRgb16bitsWidget()
{
	FREE(paramInit.horizon1)
	FREE(paramInit.horizon2)
	if ( pIhm2 ) delete pIhm2;
}

void RgtToRgb16bitsWidget::setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget)
{
	m_spectrumProcessWidget = spectrumProcessWidget;
}


void RgtToRgb16bitsWidget::horizonTypeDisplay()
{
	int idx = m_isoValOrHorizon->currentIndex();
	switch (idx)
	{
		case 0:
			m_isoStep->setVisible(true);
			m_layerNumber->setVisible(false);
			m_horizonSelectWidget->setVisible(false);
			break;
		case 1:
			m_isoStep->setVisible(false);
			m_layerNumber->setVisible(true);
			m_horizonSelectWidget->setVisible(true);
			break;
	}
}


bool RgtToRgb16bitsWidget::paramInitCreate()
{
	paramInit.rgtFilename = m_rgtFileSelectWidget->getPath();
	paramInit.seismicFilename = m_spectrumProcessWidget->getSeismicPath();
	paramInit.isoStep = m_isoStep->getLineEditText().toInt();
	std::vector<QString> horizonFilename = m_horizonSelectWidget->getPaths();

	FREE(paramInit.horizon1)
	FREE(paramInit.horizon2)

	if ( paramInit.seismicFilename.isEmpty() || paramInit.rgtFilename.isEmpty() )
	{
		QMessageBox::warning(this, "Filenames empty", "Please, fill the fields seismic and rgt filename");
		return false;
	}
	if ( !checkSeismicsSizeMatch(paramInit.seismicFilename, paramInit.rgtFilename) )
	{
		QMessageBox::warning(this, "Volumes mismatch", "Seismic and RGT volumes do not match, try again with matching volumes");
		return false;
	}

	QString seismicName = m_spectrumProcessWidget->getSeismicName();
	QString ImportExportPath = m_selectorWidget->getImportExportPath();
	QString IJKPath = m_selectorWidget->getIJKPath();
	QString horizonPath = m_selectorWidget->getHorizonsPath();
	QString isoValPath = m_selectorWidget->getHorizonsIsoValPath();
	QString rgtPath = isoValPath + m_rgtFileSelectWidget->getFilename() + "/";

	qDebug() << "rgt filename  " << m_rgtFileSelectWidget->getFilename();
	qDebug() << "rgt path      " << rgtPath;

	mkdirPathIfNotExist(ImportExportPath);
	mkdirPathIfNotExist(IJKPath);
	mkdirPathIfNotExist(horizonPath);
	mkdirPathIfNotExist(isoValPath);
	mkdirPathIfNotExist(rgtPath);
	paramInit.outMainDirectory = rgtPath;

	inri::Xt xt(paramInit.seismicFilename.toStdString().c_str());
	paramInit.dimx = xt.nSamples();
	paramInit.dimy = xt.nRecords();
	paramInit.dimz = xt.nSlices();
	int width0 = xt.nRecords();
	int height0 = xt.nSamples();
	int depth0 = xt.nSlices();
	paramInit.tdeb = xt.startSamples();
	paramInit.pasech = xt.stepSamples();
	paramInit.windowSize = m_windowSize->getLineEditText().toInt();
	paramInit.nbLayers = m_layerNumber->getLineEditText().toInt();
	int depth = 0;

	if ( m_isoValOrHorizon->currentIndex() == 0 )
	{
		depth = 32000 / paramInit.isoStep;
	}
	else
	{
		bool ret;
		if ( horizonFilename.size() < 2 )
		{
			QMessageBox::warning(this, "horizon number mismatch", "Please, choose 2 horizons");
			return false;
		}
		ret = horizonRead(horizonFilename[0].toStdString(), paramInit.dimx, paramInit.dimy, paramInit.dimz, paramInit.pasech, paramInit.tdeb, &paramInit.horizon1);
		ret = horizonRead(horizonFilename[1].toStdString(), paramInit.dimx, paramInit.dimy, paramInit.dimz, paramInit.pasech, paramInit.tdeb, &paramInit.horizon2);
		depth = paramInit.nbLayers;
	}

	QString prefix = m_prefixFilename->getLineEditText();
	// paramInit.rgb2tmpfilename = cubeRgt2RgbPath + "rgb2_" + prefix + "_from_" + seismicName + "_size_" + QString::number(width0) + "x" + QString::number(depth0) + "x" + QString::number(depth) + ".raw";
	paramInit.rgb2tmpfilename = getTmpFilename(rgtPath, "rgb2_", ".raw");
	qDebug() << "main dir:       " << paramInit.outMainDirectory;
	qDebug() << "rgb2 tmp:       " << paramInit.rgb2tmpfilename;
	qDebug() << "seismic prefix: " << m_spectrumProcessWidget->getSeismicName();
	return true;
}


void RgtToRgb16bitsWidget::trt_horizonTypeDisplay(int idx)
{
	horizonTypeDisplay();
}

void RgtToRgb16bitsWidget::setStartStopStatus(START_STOP_STATUS status)
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

void RgtToRgb16bitsWidget::trt_start()
{
	if ( valStartStop == 1 ) return;
	if ( !paramInitCreate() ) return;
	RgtToRgb16bitsWidgetTHREAD *thread = new RgtToRgb16bitsWidgetTHREAD(this);
	thread->start();
}

void RgtToRgb16bitsWidget::trt_stop()
{
	if ( valStartStop == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort the process ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		if ( pIhm2 )
			pIhm2->setMasterMessage("stop", 0, 1, RgtToAttribut::TRT_ABORT);
		setStartStopStatus(STATUS_STOP);
	}
}

void RgtToRgb16bitsWidget::trt_threadRun()
{
	int *tab_gpu = NULL, tab_gpu_size;

	if ( !pIhm2 ) pIhm2 = new Ihm2(); else { pIhm2->clearSlaveMessage(); pIhm2->clearMasterMessage(); }
	tab_gpu = (int*)calloc(m_spectrumProcessWidget->m_systemInfo->get_gpu_nbre(), sizeof(int));
	m_spectrumProcessWidget->m_systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	// GLOBAL_RUN_TYPE = 1;
	// pushbutton_startstop->setText("stop");
	RgtToAttribut *p = new RgtToAttribut();
	p->setSeismicFilename(paramInit.seismicFilename.toStdString());
	p->setRgtFilename(paramInit.rgtFilename.toStdString());
	p->setOutRawFilename(paramInit.rgb2tmpfilename.toStdString());
	p->setOutMainDirectory(paramInit.outMainDirectory.toStdString());
	p->setAttributType(RgtToAttribut::TYPE::spectrum);
	p->setAttributData(RgtToAttribut::DATA::iso);
	p->setIsoStep(paramInit.isoStep);
	p->setWSize(paramInit.windowSize);
	p->setRgbSubDirectory("rgb2");
	p->setDataOutFormat(m_spectrumProcessWidget->getDataOutFormat());
	p->setDataOutFormat(1);
	p->setMainPrefix(m_spectrumProcessWidget->getSeismicName().toStdString());
	// p->setRgbDirectory(paramInit.strOutRGB2Directory);
	// p->setIsoDirectory(paramInit.strOutHorizonsDirectory);
	p->setIhmMessage(pIhm2);
	setStartStopStatus(STATUS_START);
	p->run();
	fileRemove(paramInit.rgb2tmpfilename);
	FREE(tab_gpu)
	setStartStopStatus(STATUS_STOP);
	if ( m_selectorWidget ) m_selectorWidget->RgbRawUpdateNames();
}

void RgtToRgb16bitsWidget::showTime()
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


// =====================================================
RgtToRgb16bitsWidgetTHREAD::RgtToRgb16bitsWidgetTHREAD(RgtToRgb16bitsWidget *p)
 {
     this->pp = p;
 }

 void RgtToRgb16bitsWidgetTHREAD::run()
 {
	 fprintf(stderr, "thread start\n");
	 pp->trt_threadRun();
 }
