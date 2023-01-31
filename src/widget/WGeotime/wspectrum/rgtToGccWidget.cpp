
#include <QVBoxLayout>
#include <QMessageBox>
#include <QDebug>

#include <QFileUtils.h>

#include <Xt.h>
#include <util.h>
#include <cuda_rgt2rgb.h>
#include <spectrumProcessWidget.h>
#include <checkDataSizeMatch.h>
#include <rgtToAttribut.h>
#include <horizonUtils.h>
#include <rgtToGccWidget.h>

RgtToGccWidget::RgtToGccWidget(ProjectManagerWidget *selectorWidget, QWidget* parent)
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
	m_isoMin = new LabelLineEditWidget();
	m_isoMin->setLabelText("iso min");
	m_isoMin->setLineEditText(QString::number(param.isoMin));
	m_isoMax = new LabelLineEditWidget();
	m_isoMax->setLabelText("iso max");
	m_isoMax->setLineEditText(QString::number(param.isoMax));
	m_isoStep = new LabelLineEditWidget();
	m_isoStep->setLabelText("iso step");
	m_isoStep->setLineEditText(QString::number(param.isoStep));

	m_layerNumber = new LabelLineEditWidget();
	m_layerNumber->setLabelText("layer number");
	m_layerNumber->setLineEditText(QString::number(param.layerNumber));

	qhbIsoValOrHorizon->addWidget(m_isoValOrHorizon);
	qhbIsoValOrHorizon->addWidget(m_isoMin);
	qhbIsoValOrHorizon->addWidget(m_isoMax);
	qhbIsoValOrHorizon->addWidget(m_isoStep);
	qhbIsoValOrHorizon->addWidget(m_layerNumber);

	m_horizonSelectWidget = new HorizonSelectWidget(this);
	m_horizonSelectWidget->setProjectManager(m_selectorWidget);


	QHBoxLayout *qhbWShift = new QHBoxLayout;
	m_w = new LabelLineEditWidget();
	m_w->setLabelText("w");
	m_w->setLineEditText(QString::number(param.w));
	m_shift = new LabelLineEditWidget();
	m_shift->setLabelText("shift");
	m_shift->setLineEditText(QString::number(param.shift));


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
    horizonTypeDisplay();
    m_start->setEnabled(true);
    m_stop->setEnabled(false);
}



RgtToGccWidget::~RgtToGccWidget()
{
	FREE(paramInit.horizon1)
	FREE(paramInit.horizon2)
}

void RgtToGccWidget::setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget)
{
	m_spectrumProcessWidget = spectrumProcessWidget;
}


void RgtToGccWidget::horizonTypeDisplay()
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


bool RgtToGccWidget::paramInitCreate()
{
	paramInit.rgtFilename = m_rgtFileSelectWidget->getPath();
	// paramInit.strRgtFilename = paramInit.rgtFilename.toStdString();
	// paramInit.rgtFilename0 = (char*)paramInit.strRgtFilename.c_str();

	paramInit.seismicFilename = m_spectrumProcessWidget->getSeismicPath();
	// paramInit.strSeismicFilename = paramInit.seismicFilename.toStdString();
	// paramInit.seismicFilename0 = (char*)paramInit.strSeismicFilename.c_str();

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
	// QString rgtPath = IJKPath + "/" + m_rgtFileSelectWidget->getFilename() + "/";

	// QString seimsicNamePath = IJKPath + seismicName + "/";
	// QString cubeRgt2RgbPath = seimsicNamePath + "cubeRgt2RGB/";
	// QString outDirectory = rgtPath;
	// QString outHorizonsDirectory = outDirectory+ "horizons/";
	// QString outRGB2Directory = outDirectory+ "rgb2/";
	// QString outMeanDirectory = outDirectory+ "mean/";
	// QString outGccDirectory = outDirectory+ "gcc/";

	mkdirPathIfNotExist(ImportExportPath);
	mkdirPathIfNotExist(IJKPath);
	mkdirPathIfNotExist(horizonPath);
	mkdirPathIfNotExist(isoValPath);
	mkdirPathIfNotExist(rgtPath);

	// mkdirPathIfNotExist(seimsicNamePath);
	// mkdirPathIfNotExist(cubeRgt2RgbPath);
	// mkdirPathIfNotExist(outDirectory);
	// mkdirPathIfNotExist(outHorizonsDirectory);
	// mkdirPathIfNotExist(outRGB2Directory);
	// mkdirPathIfNotExist(outMeanDirectory);
	// mkdirPathIfNotExist(outGccDirectory);

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

	// paramInit.strOutHorizonsDirectory = outHorizonsDirectory.toStdString();
	// paramInit.strOutRGB2Directory = outRGB2Directory.toStdString();
	// paramInit.strOutMeanDirectory = outMeanDirectory.toStdString();
	// paramInit.strOutGccDirectory = outGccDirectory.toStdString();
	// paramInit.rgb2TmpFullName = cubeRgt2RgbPath + "rgb2_" + m_prefixFilename->getLineEditText() + "_from_" + seismicName + "_size_" + QString::number(width0) + "x" + QString::number(depth0) + "x" + QString::number(depth) + ".raw";
	// paramInit.rgb2TmpFullName = cubeRgt2RgbPath + seismicName + "_mean_ws_" + QString::number(paramInit.windowSize) + "_size_" + QString::number(width0) + "x" + QString::number(depth0) + "x" + QString::number(depth) + ".raw";;
	paramInit.rgb2TmpFullName = getTmpFilename(rgtPath, "gcc_", ".raw");

	/*
	paramInit.strRgb2TmpFullName = paramInit.rgb2TmpFullName.toStdString();
	paramInit.rgb2TmpFullName0 = (char*)paramInit.strRgb2TmpFullName.c_str();
	paramInit.outHorizonsDirectory0 = (char*)paramInit.strOutHorizonsDirectory.c_str();
	paramInit.outRGB2Directory0 = (char*)paramInit.strOutRGB2Directory.c_str();
	paramInit.outMeanDirectory0 = (char*)paramInit.strOutMeanDirectory.c_str();
	paramInit.outGccDirectory0 = (char*)paramInit.strOutGccDirectory.c_str();
	*/

	paramInit.outMainDirectory = rgtPath;
	qDebug() << "main dir:       " << paramInit.outMainDirectory;
	qDebug() << "rgb2 tmp:       " << paramInit.rgb2TmpFullName;
	qDebug() << "seismic prefix: " << m_spectrumProcessWidget->getSeismicName();
	return true;
}


void RgtToGccWidget::trt_horizonTypeDisplay(int idx)
{
	horizonTypeDisplay();
}

void RgtToGccWidget::setStartStopStatus(START_STOP_STATUS status)
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

void RgtToGccWidget::trt_start()
{
	if ( !paramInitCreate() ) return;
	RgtToGccWidgetTHREAD *thread = new RgtToGccWidgetTHREAD(this);
	thread->start();
}

void RgtToGccWidget::trt_stop()
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


void RgtToGccWidget::showTime()
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

void RgtToGccWidget::trt_threadRun()
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
	p->setOutRawFilename(paramInit.rgb2TmpFullName.toStdString());
	p->setAttributType(RgtToAttribut::TYPE::gcc);
	p->setAttributData(RgtToAttribut::DATA::iso);
	p->setIsoStep(paramInit.isoStep);
	p->setWSize(paramInit.windowSize);
	p->setDataOutFormat(m_spectrumProcessWidget->getDataOutFormat());
	p->setOutMainDirectory(paramInit.outMainDirectory.toStdString());
	// p->setMeanDirectory(paramInit.strOutMeanDirectory);
	// p->setIsoDirectory(paramInit.strOutHorizonsDirectory);
	// p->setGccDirectory(paramInit.strOutGccDirectory);
	p->setMainPrefix(m_spectrumProcessWidget->getSeismicName().toStdString());
	p->setIhmMessage(pIhm2);
	setStartStopStatus(STATUS_START);
	p->run();
	fileRemove(paramInit.rgb2TmpFullName);
	FREE(tab_gpu)
	setStartStopStatus(STATUS_STOP);
	if ( m_selectorWidget ) m_selectorWidget->RgbRawUpdateNames();
}

/*
void RgtToSeismicMeanWidget::trt_threadRun()
{
	int *tab_gpu = NULL, tab_gpu_size;

	tab_gpu = (int*)calloc(m_spectrumProcessWidget->m_systemInfo->get_gpu_nbre(), sizeof(int));
	m_spectrumProcessWidget->m_systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
	// GLOBAL_RUN_TYPE = 1;
	// pushbutton_startstop->setText("stop");

	Rgt2Rgb *p = new Rgt2Rgb();

	qDebug() << paramInit.seismicFilename;

	p->setSeismicFilename(paramInit.seismicFilename0);
	p->setRgtFilename((char*)paramInit.rgtFilename0);
	p->setTDeb(paramInit.tdeb);
	p->setPasEch(paramInit.pasech);
	p->setIsoVal(0, 32000, paramInit.isoStep);

	if ( paramInit.horizon1 && paramInit.horizon2 )
	{
		if ( horizonMean(paramInit.horizon1, paramInit.dimy, paramInit.dimz) <= horizonMean(paramInit.horizon2, paramInit.dimy, paramInit.dimz) )
			p->setHorizon(paramInit.horizon1, paramInit.horizon2, paramInit.nbLayers);
		else
			p->setHorizon(paramInit.horizon2, paramInit.horizon1, paramInit.nbLayers);
	}
	p->setSize(paramInit.windowSize);
	// p->setArrayFreq(arrayFreq, arrayIso, Freqcount);
	p->setRgbFilename(paramInit.rgb2TmpFullName0);
	// p->setIsoFilename(isoFilename);
	p->setGPUList(tab_gpu, tab_gpu_size);
	p->setRgbDirectory(paramInit.outRGB2Directory0);
	p->setIsoDirectory(paramInit.outHorizonsDirectory0);
	p->setOutputType(0);
	setStartStopStatus(STATUS_START);
	p->run();
	delete p;
	FREE(tab_gpu)
	setStartStopStatus(STATUS_STOP);
}
*/









// =====================================================
RgtToGccWidgetTHREAD::RgtToGccWidgetTHREAD(RgtToGccWidget *p)
{
	this->pp = p;
}

void RgtToGccWidgetTHREAD::run()
{
	fprintf(stderr, "thread start\n");
	pp->trt_threadRun();
}
