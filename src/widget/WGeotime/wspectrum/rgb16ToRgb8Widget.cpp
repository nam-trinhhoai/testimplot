
#include <QMessageBox>

#include <Xt.h>
#include <spectrumProcessWidget.h>
#include <rgb16ToRgb8Widget.h>

#include <rgtToAttribut.h>
#include <cuda_rgb2torgb1.h>


Rgb16ToRgb8Widget::Rgb16ToRgb8Widget(ProjectManagerWidget *selectorWidget, QWidget* parent)
{
	m_selectorWidget = selectorWidget;

	QVBoxLayout * mainLayout = new QVBoxLayout(this);

	m_rgb2FileSelectWidget = new FileSelectWidget();
	m_rgb2FileSelectWidget->setLabelText("rgb2 filename");
	m_rgb2FileSelectWidget->setProjectManager(m_selectorWidget);
	m_rgb2FileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::rgtCubeToAttribut);
	m_rgb2FileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Raw);

	m_prefixFilename = new LabelLineEditWidget();
	m_prefixFilename->setLabelText("prefix");
	m_prefixFilename->setLineEditText("rgb1");

	QHBoxLayout *paramLayout = new QHBoxLayout(this);
	m_ratio = new LabelLineEditWidget();
	m_ratio->setLabelText("ratio: ");
	m_ratio->setLineEditText("0.0001");

	m_alpha = new LabelLineEditWidget();
	m_alpha->setLabelText("alpha: ");
	m_alpha->setLineEditText("1.0");
	paramLayout->addWidget(m_ratio);
	paramLayout->addWidget(m_alpha);

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
	qhbButtons->addWidget(m_kill);

	mainLayout->addWidget(m_rgb2FileSelectWidget);
	mainLayout->addWidget(m_prefixFilename);
	mainLayout->addLayout(paramLayout);
	mainLayout->addWidget(m_progressBar);
	mainLayout->addLayout(qhbButtons);

	mainLayout->setAlignment(Qt::AlignTop);

	setMaximumHeight(450);
	timer = new QTimer();
	timer->start(1000);

    connect(m_start, SIGNAL(clicked()), this, SLOT(trt_start()));
    connect(m_stop, SIGNAL(clicked()), this, SLOT(trt_stop()));
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
    m_start->setEnabled(true);
    m_stop->setEnabled(false);

}

Rgb16ToRgb8Widget::~Rgb16ToRgb8Widget()
{

}

void Rgb16ToRgb8Widget::setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget)
{
	m_spectrumProcessWidget = spectrumProcessWidget;
}

void Rgb16ToRgb8Widget::setStartStopStatus(START_STOP_STATUS status)
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

void Rgb16ToRgb8Widget::trt_start()
{
	if ( valStartStop == 1 ) return;
	Rgb16ToRgb8WidgetTHREAD *thread = new Rgb16ToRgb8WidgetTHREAD(this);
	thread->start();
}

void Rgb16ToRgb8Widget::trt_stop()
{
	if ( valStartStop == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort the process ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		if ( pIhm2 ) pIhm2->setMasterMessage("stop", 0, 1, RgtToAttribut::TRT_ABORT);
		setStartStopStatus(STATUS_STOP);
	}
}


void Rgb16ToRgb8Widget::showTime()
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

QString Rgb16ToRgb8Widget::filenameToPath(QString fullName)
{
	int lastPoint = fullName.lastIndexOf("/");
	QString path = fullName.left(lastPoint);
	return path;
}

void Rgb16ToRgb8Widget::trt_threadRun()
{
	int *tab_gpu = NULL, tab_gpu_size;
	if ( !pIhm2 ) pIhm2 = new Ihm2(); else { pIhm2->clearSlaveMessage(); pIhm2->clearMasterMessage(); }

	if ( m_spectrumProcessWidget->getSeismicPath().compare("") == 0 || m_rgb2FileSelectWidget->getPath().compare("") == 0 || m_alpha->getLineEditText().compare("") == 0 || m_ratio->getLineEditText().compare("") == 0 ) return;
	int size[3];
	inri::Xt xt((const char*)m_spectrumProcessWidget->getSeismicPath().toStdString().c_str());
	size[0] = xt.nSamples();
	size[1] = xt.nRecords();
	size[2] = xt.nSlices();

	QFile file(m_rgb2FileSelectWidget->getPath());
	long size0 = file.size();
	int dimx = size[2];
	int dimy = size[0];
	int dimz = size0/dimx/dimy/4/2;

	QString path = filenameToPath(m_rgb2FileSelectWidget->getPath());
	float alpha = m_alpha->getLineEditText().toFloat();
	float ratio = m_ratio->getLineEditText().toFloat();
	QString prefix = m_prefixFilename->getLineEditText();

	QString	rgb1FullName = path + "/" + "rgb1_" + prefix + "_from_" +
				m_rgb2FileSelectWidget->getFilename() + "__alpha_" + m_alpha->getLineEditText().replace(".", "x") + "__ratio_" + m_ratio->getLineEditText().replace(".", "x")
				+ QString("_size_") + QString::number(dimy) + "x" + QString::number(dimx) + "x" + QString::number(dimz) + ".rgb";

	QFileInfo info(rgb1FullName);
	QString rgb1TinyName = info.fileName();
	char rgb2Filename[10000], rgb1Filename[10000];
	strcpy(rgb2Filename, m_rgb2FileSelectWidget->getPath().toStdString().c_str());
	strcpy(rgb1Filename, rgb1FullName.toStdString().c_str());
	fprintf(stderr, "size: %ld %d %d %d\n", size0, dimx, dimy, dimz);

	setStartStopStatus(STATUS_START);
	cuda_rgb2torgb1(rgb2Filename, dimx, dimy, dimz, ratio, alpha, rgb1Filename, pIhm2);
	setStartStopStatus(STATUS_STOP);
}



// =====================================================
Rgb16ToRgb8WidgetTHREAD::Rgb16ToRgb8WidgetTHREAD(Rgb16ToRgb8Widget *p)
{
	this->pp = p;
}

void Rgb16ToRgb8WidgetTHREAD::run()
{
	fprintf(stderr, "thread start\n");
	pp->trt_threadRun();
}
