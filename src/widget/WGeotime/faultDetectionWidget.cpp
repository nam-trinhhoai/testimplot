
#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QGroupBox>
#include <QTimer>
#include <QProgressBar>
#include <QMessageBox>


#include <ctime>

#include <stdio.h>
#include <workingsetmanager.h>
#include <QFileUtils.h>
#include <ihm.h>
#include "GeotimeConfiguratorWidget.h"
#include <ProjectManagerWidget.h>
#include <labelLineEditWidget.h>
#include <fileSelectWidget.h>
#include <faultDetection.h>
#include <faultDetectionWidget.h>


FaultDetectionWidget::FaultDetectionWidget(WorkingSetManager *workingSetManager, QWidget* parent)
{
	m_workingSetManager = workingSetManager;
	int height = 150;
	m_projectManager = m_workingSetManager->getManagerWidgetV2();

	QVBoxLayout * mainLayout00 = new QVBoxLayout(this);

	m_processing = new QLabel(".");

	m_seismicFileSelectWidget = new FileSelectWidget();
	m_seismicFileSelectWidget->setProjectManager(m_projectManager);
	m_seismicFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_seismicFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);

	QHBoxLayout *qhbNormalizationDetection = new QHBoxLayout;

	QGroupBox *qgbNormalization = new QGroupBox("Normalization");
	qgbNormalization->setMaximumHeight(height);

	// qgbNormalization->setTitle("Normalization");
	QVBoxLayout *qvbNormalization = new QVBoxLayout(qgbNormalization);
	QHBoxLayout *qvbNormalizationType = new QHBoxLayout;
	QLabel *qlabelNormalizationType = new QLabel("Type");
	qleNormalizationType = new QLineEdit("1");
	qvbNormalizationType->addWidget(qlabelNormalizationType);
	qvbNormalizationType->addWidget(qleNormalizationType);

	QHBoxLayout *qvbNormalizationAlpha1 = new QHBoxLayout;
	QLabel *qlabelNormalizationAlpha1 = new QLabel("Alpha 1");
	qleNormalizationAlpha1 = new QLineEdit("0.1");
	qvbNormalizationAlpha1->addWidget(qlabelNormalizationAlpha1);
	qvbNormalizationAlpha1->addWidget(qleNormalizationAlpha1);

	QHBoxLayout *qvbNormalizationAlpha2 = new QHBoxLayout;
	QLabel *qlabelNormalizationAlpha2 = new QLabel("Alpha 2");
	qleNormalizationAlpha2 = new QLineEdit("0.05");
	qvbNormalizationAlpha2->addWidget(qlabelNormalizationAlpha2);
	qvbNormalizationAlpha2->addWidget(qleNormalizationAlpha2);

	qvbNormalization->addLayout(qvbNormalizationType);
	qvbNormalization->addLayout(qvbNormalizationAlpha1);
	qvbNormalization->addLayout(qvbNormalizationAlpha2);

	QGroupBox *qgbDetection = new QGroupBox("Detection");
	qgbDetection->setMaximumHeight(height);
	// qgbNormalization->setTitle("Normalization");
	QVBoxLayout *qvbDetection = new QVBoxLayout(qgbDetection);
	QHBoxLayout *qvbDetectionLength = new QHBoxLayout;
	QLabel *qlabelDetectionLength = new QLabel("Length");
	qleDetectionLength = new QLineEdit("71");
	qvbDetectionLength->addWidget(qlabelDetectionLength);
	qvbDetectionLength->addWidget(qleDetectionLength);

	QHBoxLayout *qvbDetectionWidth = new QHBoxLayout;
	QLabel *qlabelDetectionWidth = new QLabel("Width");
	qleDetectionWidth = new QLineEdit("1");
	qvbDetectionWidth->addWidget(qlabelDetectionWidth);
	qvbDetectionWidth->addWidget(qleDetectionWidth);

	QHBoxLayout *qvbDetectionPartition = new QHBoxLayout;
	QLabel *qlabelDetectionPartition = new QLabel("Partition into");
	qleDetectionPartition = new QLineEdit("9");
	qvbDetectionPartition->addWidget(qlabelDetectionPartition);
	qvbDetectionPartition->addWidget(qleDetectionPartition);

	qvbDetection->addLayout(qvbDetectionLength);
	qvbDetection->addLayout(qvbDetectionWidth);
	qvbDetection->addLayout(qvbDetectionPartition);

	qhbNormalizationDetection->addWidget(qgbNormalization);
	qhbNormalizationDetection->addWidget(qgbDetection);

	QHBoxLayout *qhbDipFilter = new QHBoxLayout;

	QGroupBox *qgbDip = new QGroupBox("Dip");
	qgbDip->setMaximumHeight(height);
	// qgbNormalization->setTitle("Normalization");
	QVBoxLayout *qvbDip = new QVBoxLayout(qgbDip);
	QHBoxLayout *qvbDipMaximum = new QHBoxLayout;
	QLabel *qlabelDipMaximum = new QLabel("Maximum");
	qleDipMaximum = new QLineEdit("40");
	qvbDipMaximum->addWidget(qlabelDipMaximum);
	qvbDipMaximum->addWidget(qleDipMaximum);

	QHBoxLayout *qvbDipStep = new QHBoxLayout;
	QLabel *qlabelDipStep = new QLabel("Step");
	qleDipStep = new QLineEdit("2");
	qvbDipStep->addWidget(qlabelDipStep);
	qvbDipStep->addWidget(qleDipStep);

	qvbDip->addLayout(qvbDipMaximum);
	qvbDip->addLayout(qvbDipStep);


	QGroupBox *qgbFilter = new QGroupBox("Filter");
	qgbFilter->setMaximumHeight(height);
	// qgbNormalization->setTitle("Normalization");
	QVBoxLayout *qvbFilter = new QVBoxLayout(qgbFilter);
	QHBoxLayout *qvbFilterSize = new QHBoxLayout;
	QLabel *qlabelFilterSize = new QLabel("Size");
	qleFilterSize = new QLineEdit("51");
	qvbFilterSize->addWidget(qlabelFilterSize);
	qvbFilterSize->addWidget(qleFilterSize);

	QHBoxLayout *qvbFilterBright = new QHBoxLayout;
	QLabel *qlabelFilterBright = new QLabel("Bright point");
	qleFilterBright = new QLineEdit("0.05");
	qvbFilterBright->addWidget(qlabelFilterBright);
	qvbFilterBright->addWidget(qleFilterBright);

	QHBoxLayout *qvbFilterVariance = new QHBoxLayout;
	QLabel *qlabelFilterVariance = new QLabel("Variance");
	qleFilterVariance = new QLineEdit("0");
	qvbFilterVariance->addWidget(qlabelFilterVariance);
	qvbFilterVariance->addWidget(qleFilterVariance);

	qvbFilter->addLayout(qvbFilterSize);
	qvbFilter->addLayout(qvbFilterBright);
	qvbFilter->addLayout(qvbFilterVariance);

	qhbDipFilter->addWidget(qgbDip);
	qhbDipFilter->addWidget(qgbFilter);

	m_btnStartStop = new QPushButton("Start");

	m_progressBar = new QProgressBar();
	m_progressBar->setMinimum(0);
	m_progressBar->setMaximum(100);
	m_progressBar->setValue(0);
	// m_progressBar->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,0,0)}");
	m_progressBar->setTextVisible(true);
	m_progressBar->setFormat("");

	m_faulFileName = new LabelLineEditWidget();
	m_faulFileName->setLabelText("suffix filename");
	m_faulFileName->setLineEditText("fault");

	m_validCrestDetection = new QCheckBox("crest detection");
	m_validCrestDetection->setCheckState(Qt::Checked);
	m_crestDetection = new LabelLineEditWidget();
	m_crestDetection->setLabelText("");
	m_crestDetection->setLineEditText("1");
	QHBoxLayout *qhbCrestDetection = new QHBoxLayout;
	qhbCrestDetection->addWidget(m_validCrestDetection);
	qhbCrestDetection->addWidget(m_crestDetection);
	qhbCrestDetection->setAlignment(Qt::AlignLeft);

	QLabel *qlLabelMinSize = new QLabel("Min size threshold:");
	qleLabelMinSize = new QLineEdit(QString::number(this->parameters.labelMinSize));
	QHBoxLayout *qhbLabelMinSize = new QHBoxLayout;
	qhbLabelMinSize->addWidget(qlLabelMinSize);
	qhbLabelMinSize->addWidget(qleLabelMinSize);

	QHBoxLayout *qhbDirection = new QHBoxLayout;
	QLabel *lDirection = new QLabel("direction: ");
	m_direction = new QComboBox();
	m_direction->addItem("inline");
	m_direction->addItem("xline");
	m_direction->addItem("both");
	m_direction->setCurrentIndex(2);
	qhbDirection->addWidget(lDirection);
	qhbDirection->addWidget(m_direction);
	qhbDirection->setAlignment(Qt::AlignLeft);

	mainLayout00->addWidget(m_processing);
	mainLayout00->addWidget(m_seismicFileSelectWidget);
	mainLayout00->addWidget(m_faulFileName);
	mainLayout00->addLayout(qhbNormalizationDetection);
	mainLayout00->addLayout(qhbDipFilter);
	mainLayout00->addLayout(qhbCrestDetection);
	mainLayout00->addLayout(qhbLabelMinSize);
	mainLayout00->addLayout(qhbDirection);
	mainLayout00->addWidget(m_btnStartStop);
	mainLayout00->addWidget(m_progressBar);
	mainLayout00->setAlignment(Qt::AlignTop);


	timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	connect(m_btnStartStop, SIGNAL(clicked()), this, SLOT(trt_startStop()));
	setStartStopStatus(STATUS_STOP);
}

FaultDetectionWidget::~FaultDetectionWidget()
{


}

void FaultDetectionWidget::setGeotimeConfigurationWidget(GeotimeConfigurationWidget *val)
{
	geotimeConfigurationWidget = val;
}



void FaultDetectionWidget::showTime()
{
	char txt[1000], txt2[1000];
	int type;
	long idx, vmax;
	int msg_new = ihm_get_global_msg(&type, &idx, &vmax, txt);
	processingDisplay();
	if ( valStartStop == 1 && type == imhId )
	{
		if ( msg_new == 0 ) return;
		if ( idx >= 0 && vmax > 0 )
		{
			float val_f = 100.0*idx/vmax;
			int val = (int)(val_f);
			m_progressBar->setValue(val);
			sprintf(txt2, "%s %.1f%%", txt, val_f);
			m_progressBar->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
			m_progressBar->setFormat(txt2);
		}
		// this->textInfo->appendPlainText(QString(txt));
	}
	if ( flagEnd )
	{
		m_progressBar->setValue(0);
		m_progressBar->setFormat("");
		QMessageBox::information(this, "Fault", "Process end");
		flagEnd = false;
		timer->disconnect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	}
}


void FaultDetectionWidget::setStartStopStatus(START_STOP_STATUS status)
{
	if ( status == STATUS_STOP )
	{
		valStartStop = 0;
		m_btnStartStop->setText("start");
	}
	else
	{
		valStartStop = 1;
		m_btnStartStop->setText("stop");
		timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	}
}

void FaultDetectionWidget::trt_start()
{
	FaultDetectionWidgetTHREAD *thread = new FaultDetectionWidgetTHREAD(this);
	thread->start();
}

void FaultDetectionWidget::trt_stop()
{
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort the processing ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		ihm_set_trt(imhAbortId);
		setStartStopStatus(STATUS_STOP);
	}
}

void FaultDetectionWidget::trt_threadRun()
{
	QString prefix = m_faulFileName->getLineEditText();
	int normalizationType = qleNormalizationType->text().toInt();
	double normalizationAlpha1 = qleNormalizationAlpha1->text().toDouble();
	double normalizationAlpha2 = qleNormalizationAlpha2->text().toDouble();
	double normalizationCoef = 100.0f;
	int detectionLength = qleDetectionLength->text().toInt();
	int detectionWidth = qleDetectionWidth->text().toInt();
	int detectionPartitionInto = qleDetectionPartition->text().toInt();
	int dipMaximum = qleDipMaximum->text().toInt();
	int dipStep = qleDipStep->text().toInt();
	int filterSize = qleFilterSize->text().toInt();
	double filterBrightPoint = qleFilterBright->text().toDouble();
	float filterVariance = qleFilterVariance->text().toFloat();
	float crestDetection = m_crestDetection->getLineEditText().toFloat();
	if (  !m_validCrestDetection->isChecked() ) crestDetection = -1.0;
	// if (geotimeConfigurationWidget == nullptr ) return;
	// QString seismicPath = geotimeConfigurationWidget->getSeismicPath();
	QString seismicPath = m_seismicFileSelectWidget->getPath();
	QString faultPath = makeFilePathWithSuffix(seismicPath, m_faulFileName->getLineEditText());
	fprintf(stderr, "%s\n%s\n", (char*)seismicPath.toStdString().c_str(), (char*)faultPath.toStdString().c_str());

	std::srand(std::time(nullptr));
	imhId = std::rand();
	imhAbortId = std::rand();
	imhEndId = std::rand();

	FaultDetection *p = new FaultDetection();
	p->setIhmId(imhId);
	p->setIhmAbortId(imhAbortId);
	p->setIhmEndId(imhEndId);
	p->setNormalizationType(normalizationType);
	p->setNormalizationAlpha1(normalizationAlpha1);
	p->setNormalizationAlpha2(normalizationAlpha2);
	p->setNormalizationCoef(normalizationCoef);
	p->setDetectionLength(detectionLength);
	p->setDetectionWidth(detectionWidth);
	p->setDetectionPartitionInto(detectionPartitionInto);
	p->setDipMaximum(dipMaximum);
	p->setDipStep(dipStep);
	p->setFilterSize(filterSize);
	p->setFilterBrightPoint(filterBrightPoint);
	p->setFilterVariance(filterVariance);
	p->setCrestDetection(crestDetection);
	p->setSeismicFilename(seismicPath.toStdString());
	p->setfaultFilename(faultPath.toStdString());
	p->setLabelMinSize(qleLabelMinSize->text().toInt());
	if ( m_direction->currentIndex() == 0 ) p->setDirection(FaultDetection::DIRECTION::_inline);
	else if ( m_direction->currentIndex() == 1 ) p->setDirection(FaultDetection::DIRECTION::_xline);
	else if ( m_direction->currentIndex() == 2 ) p->setDirection(FaultDetection::DIRECTION::_both);
	setStartStopStatus(STATUS_START);
	p->run();
	setStartStopStatus(STATUS_STOP);
	delete p;
	flagEnd = true;
	m_projectManager->seimsicDatabaseUpdate();
}


void FaultDetectionWidget::trt_startStop()
{
	if ( valStartStop == 0 )
	{
		trt_start();
	}
	else
	{
		trt_stop();
	}
}

void FaultDetectionWidget::processingDisplay()
{
	if ( valStartStop == 0 )
	{
		// m_progress->setValue(0); m_progress->setFormat("");
		m_processing->setText("waiting ...");
		m_processing->setStyleSheet("QLabel { color : white; }");
		return;
	}
	m_processing->setText("PROCESSING");
	m_cptProcessing++;
	if ( m_cptProcessing%2 == 0 )
	{
		m_processing->setStyleSheet("QLabel { color : red; }");
	}
	else
	{
		m_processing->setStyleSheet("QLabel { color : white; }");
	}
}

// ====

FaultDetectionWidgetTHREAD::FaultDetectionWidgetTHREAD(FaultDetectionWidget *p)
 {
     this->pp = p;
 }

 void FaultDetectionWidgetTHREAD::run()
 {
	 fprintf(stderr, "thread start\n");
	 pp->trt_threadRun();
 }




