
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QFrame>
#include <QProgressBar>
#include <QMessageBox>
#include <QFormLayout>
#include "sismagedbmanager.h"
#include <seismicsurvey.h>
#include <DataSelectorDialog.h>
#include <workingsetmanager.h>
#include <GeotimeProjectManagerWidget.h>

#include <Xt.h>
#include <geotimeFlags.h>
#include <freeHorizonManager.h>
#include <gccOnSpectrumProcess.h>
#include <gccOnSpectrumAttributWidget.h>


GccOnSpectrumAttributWidget::GccOnSpectrumAttributWidget(QString surveyPath, QString dirPath, QString spectrumName, WorkingSetManager *manager, QWidget *parent)
{
	m_spectrumName = spectrumName;
	m_dirPath = dirPath;
	m_path = m_dirPath + "/" + m_spectrumName + QString::fromStdString(FreeHorizonManager::attributExt);
	m_surveyPath = surveyPath;
	m_nbFreq = FreeHorizonManager::getNbreSpectrumFreq(m_path.toStdString());
	m_manager = manager;

	QFileInfo fi(m_dirPath);
	m_horizonName = fi.baseName();

	QVBoxLayout *mainLayout = new QVBoxLayout(this);

	QLabel *mainLabel = new QLabel("Ggc on " + m_spectrumName);

	m_processing = new QLabel(".");

	QHBoxLayout *qhbNbFreq = new QHBoxLayout;
	QLabel *qlNbfreq = new QLabel("Frequency nbre:");
	QLineEdit *qleNbFreq = new QLineEdit(QString::number(m_nbFreq));
	qleNbFreq->setReadOnly(true);
	qleNbFreq->setStyleSheet("QLineEdit { border: none }");
	qhbNbFreq->addWidget(qlNbfreq);
	qhbNbFreq->addWidget(qleNbFreq);

	// QFormLayout *formFreqCentral = new QFormLayout;
	QHBoxLayout *qhbCentralFrequency = new QHBoxLayout;
	m_centralFrequency = new QSpinBox;
	m_centralFrequency->setFixedHeight(25);
	m_centralFrequency->setMaximumWidth(spinBoxWidth);
	m_centralFrequency->setMinimumWidth(75);
	m_fcentral_hz = new QLineEdit();
	m_fcentral_hz->setStyleSheet("QLineEdit { border: none }");
	qhbCentralFrequency->setAlignment(Qt::AlignLeft);
	qhbCentralFrequency->addWidget(m_centralFrequency);
	qhbCentralFrequency->addWidget(m_fcentral_hz);
	// formFreqCentral->addRow("central frequency", qhbCentralFrequency);

	QFormLayout *formScrollFreq = new QFormLayout;

	QHBoxLayout *qhbAmplitudeFrequency = new QHBoxLayout;
	qhbAmplitudeFrequency->setAlignment(Qt::AlignLeft);
	m_scrollFreq = new QSlider();
	m_scrollFreq->setOrientation(Qt::Horizontal);
	m_ampl = new QLineEdit;
	m_ampl->setStyleSheet("QLineEdit { border: none }");
	m_ampl->setMaximumWidth(200);
	qhbAmplitudeFrequency->addWidget(m_scrollFreq);
	qhbAmplitudeFrequency->addWidget(m_ampl);
	// m_scrollFreq->setMinimumWidth(200);
	// m_scrollFreq->setRange(0, dimx);




	QHBoxLayout *qhbF1 = new QHBoxLayout;
	qhbF1->setAlignment(Qt::AlignLeft);
	m_f1_idx = new QLineEdit("3");
	m_f1_idx->setStyleSheet("QLineEdit { border: none }");
	m_f1_idx->setMaximumWidth(100);
	m_f1_hz = new QLineEdit("3 hz");
	m_f1_hz->setStyleSheet("QLineEdit { border: none }");
	qhbF1->addWidget(m_f1_idx);
	qhbF1->addWidget(m_f1_hz);

	QHBoxLayout *qhbF2 = new QHBoxLayout;
	qhbF2->setAlignment(Qt::AlignLeft);
	m_f2_idx = new QLineEdit("3");
	m_f2_idx->setStyleSheet("QLineEdit { border: none }");
	m_f2_idx->setMaximumWidth(100);
	m_f2_hz = new QLineEdit("3 hz");
	m_f2_hz->setStyleSheet("QLineEdit { border: none }");
	qhbF2->addWidget(m_f2_idx);
	qhbF2->addWidget(m_f2_hz);

	m_spinW = new QSpinBox;
	m_spinW->setFixedHeight(25);
	m_spinW->setMinimumWidth(75);
	m_spinW->setMaximumWidth(100);
	m_spinW->setMinimum(1);

	formScrollFreq->addRow("Central frequency", qhbCentralFrequency);
	formScrollFreq->addRow("Amplitude", qhbAmplitudeFrequency);
	formScrollFreq->addRow("f1", qhbF1);
	formScrollFreq->addRow("f2", qhbF2);
	formScrollFreq->addRow("Scales", m_spinW);


	m_f1SpinBox = new QSpinBox;
	m_f1SpinBox->setMinimum(0);
	m_f1SpinBox->setMaximum(32);
	m_f1SpinBox->setValue(0);

	m_f2SpinBox = new QSpinBox;
	m_f2SpinBox->setMinimum(0);
	m_f2SpinBox->setMaximum(32);
	m_f2SpinBox->setValue(0);
	// formLayout->addRow("Frequency", m_freqSpinBox);


/*

	QHBoxLayout *qhbF1 = new QHBoxLayout;
	QLabel *ql_f1 = new QLabel("start frequency:");
	m_f1 = new QLineEdit("");
	m_f1->setReadOnly(true);
	qhbF1->addWidget(ql_f1);
	qhbF1->addWidget(m_f1SpinBox);
	qhbF1->addWidget(m_f1);

	QHBoxLayout *qhbF2 = new QHBoxLayout;
	QLabel *ql_f2 = new QLabel("end frequency:");
	m_f2 = new QLineEdit("");
	m_f2->setReadOnly(true);
	qhbF2->addWidget(ql_f2);
	qhbF2->addWidget(m_f2SpinBox);
	qhbF2->addWidget(m_f2);
	*/

	m_progress = new QProgressBar();
	// qpbRgtPatchProgress->setGeometry(5, 45, 240, 20);
	m_progress->setMinimum(0);
	m_progress->setMaximum(100);
	m_progress->setValue(0);
	m_progress->setTextVisible(true);
	m_progress->setFormat("");

	m_start = new QPushButton("start");

	QFrame *line1 = new QFrame;
	line1->setObjectName(QString::fromUtf8("line"));
	line1->setGeometry(QRect(320, 150, 118, 3));
	line1->setFrameShape(QFrame::HLine);

	mainLayout->addWidget(m_processing);
	mainLayout->addWidget(mainLabel);
	mainLayout->addLayout(qhbNbFreq);
	mainLayout->addWidget(line1);
	// mainLayout->addLayout(formFreqCentral);
	mainLayout->addLayout(formScrollFreq);

	// mainLayout->addLayout(qhbF1);
	// mainLayout->addLayout(qhbF2);
	mainLayout->addWidget(m_progress);
	mainLayout->addWidget(m_start);
	mainLayout->setAlignment(Qt::AlignTop);

	setWindowTitle("compute gcc on spectrum " + m_spectrumName);

	initParams();
	displayParams();

	if ( pIhm == nullptr ) pIhm = new Ihm2();
	QTimer *timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));

	connect(m_start, SIGNAL(clicked()), this, SLOT(trt_launch()));
	connect(m_centralFrequency, QOverload<int>::of(&QSpinBox::valueChanged), this, &GccOnSpectrumAttributWidget::fcChanged);
	// connect(m_scrollFreq, SIGNAL(valueChanged(int)), this, &GccOnSpectrumAttributWidget::famplChanged);
	connect(m_scrollFreq, SIGNAL(valueChanged(int)), this, SLOT(famplChanged(int)));
	// connect(m_f2SpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &GccOnSpectrumAttributWidget::f2Changed);
	setFixedWidth(600);
}

GccOnSpectrumAttributWidget::~GccOnSpectrumAttributWidget()
{

}

/*
QString GccOnSpectrumAttributWidget::getOutAttibutFilename()
{
	int f1 = m_f1SpinBox->value();
	int f2 = m_f2SpinBox->value();
	QString path = m_dirPath + "/gcc_on_spectrum_f1_" + QString::number(f1) + "_f2_" + QString::number(f2) + "_" + m_spectrumName + ".raw";
	return path;
}
 */


QString GccOnSpectrumAttributWidget::getOutAttibutPath()
{
	int f1 = m_f1SpinBox->value();
	int f2 = m_f2SpinBox->value();
	QString path = m_dirPath + "/" + getOutAttibutName() + QString::fromStdString(FreeHorizonManager::attributExt);
	return path;
}

QString GccOnSpectrumAttributWidget::getOutAttibutName()
{
	int ampl = m_scrollFreq->value();
	int fc = m_centralFrequency->value();
	int f1 = fc - ampl;
	int f2 = fc + ampl;
	int w = m_spinW->value();

	QString name = "/gcc_on_spectrum_f1_" + QString::number(f1) + "_f2_" + QString::number(f2) + "_w_" + QString::number(w) + "_" + m_spectrumName;
	return name;
}

void GccOnSpectrumAttributWidget::initParams()
{
	 std::string dataSetName = FreeHorizonManager::dataSetNameGet(m_dirPath.toStdString());
	m_dataSetPath = QString::fromStdString(SismageDBManager::datasetPathFromDatasetFileNameAndSurveyPath(dataSetName, m_surveyPath.toStdString()));
	inri::Xt xt(m_dataSetPath.toStdString().c_str());
	// if ( !xt.is_valid() ) return;
	// m_dimy = xt.nRecords();
	// m_dimz = xt.nSlices();
	FreeHorizonManager::getSize(m_path.toStdString(), &m_dimy, &m_dimz);
	m_nbFreq = FreeHorizonManager::getNbreSpectrumFreq(m_path.toStdString());
	m_dimx = m_nbFreq;
	m_fech = xt.stepSamples();

	m_spinW->setValue(m_w);
	m_centralFrequency->setValue(m_nbFreq / 2);
	m_centralFrequency->setMinimum(1);
	m_centralFrequency->setMaximum(m_nbFreq-2);

	int ampl = 3;
	m_scrollFreq->setValue(ampl);
	m_cacheAmpl = m_scrollFreq->value();

	m_scrollFreq->setMinimum(0);
	m_scrollFreq->setMaximum(m_nbFreq/2);
	m_cacheFc = m_scrollFreq->value();


	// m_f1SpinBox->setValue(0);
	// m_f2SpinBox->setValue(m_nbFreq/2);
	// m_f1SpinBox->setMaximum(m_nbFreq-2);
	// m_f1SpinBox->setValue(0);
	// m_f2SpinBox->setMinimum(1);
	// m_f2SpinBox->setMaximum(m_nbFreq-1);
}

double GccOnSpectrumAttributWidget::idxToFreq(int idx)
{
	return 1000.0/(m_fech*(m_nbFreq-1)) * idx;
}

void GccOnSpectrumAttributWidget::displayParams()
{
	int idxc = m_centralFrequency->value();
	int ampl = m_scrollFreq->value();
	int idx1 = idxc - ampl;
	int idx2 = idxc + ampl;

	double f1 = idxToFreq(idx1);
	double f2 = idxToFreq(idx2);
	double fc = idxToFreq(idxc);
	double amplHz = idxToFreq(ampl);

	m_f1_idx->setText(QString::number(idx1));
	m_f2_idx->setText(QString::number(idx2));
	m_f1_hz->setText(QString::number((double)f1) + " Hz");
	m_f2_hz->setText(QString::number((double)f2) + " Hz");
	m_fcentral_hz->setText(QString::number((double)fc) + " Hz");
	QString txt = QString::number(ampl) + "   ( " + QString::number(amplHz) + " Hz )";
	m_ampl->setText(txt);
}

void GccOnSpectrumAttributWidget::fcChanged(int value) {
	int ampl = m_scrollFreq->value();
	int fc = m_centralFrequency->value();
	int f1 = fc - ampl;
	int f2 = fc + ampl;
/*
	if ( f1 <= 1 )
	{
		m_cacheAmpl = 1;
		m_scrollFreq->setValue(m_cacheAmpl);
	}
	else if ( f2 >= m_nbFreq-1)
	{
		m_cacheAmpl = 1;
		m_scrollFreq->setValue(m_cacheAmpl);
	}
	*/
	if ( f1 < 0 || f2 > m_nbFreq-1 )
	{
		m_centralFrequency->setValue(m_cacheFc);
	}
	else
		m_cacheFc = fc;
	displayParams();
}

void GccOnSpectrumAttributWidget::famplChanged(int value) {
	int ampl = m_scrollFreq->value();
	int fc = m_centralFrequency->value();
	int f1 = fc - ampl;
	int f2 = fc + ampl;

	/*
	if ( f1 >=0 && f2 <= m_nbFreq-1 )
	{
		m_cacheAmpl = ampl;
	}
	else
	{
		if ( f1 < 0 ) fc = ampl;
		if ( f2 > m_nbFreq-1 ) fc = m_nbFreq-1-ampl;
		m_centralFrequency->setValue(fc);
		m_cacheFc = fc;
		m_cacheAmpl = ampl;
	}
	*/

	/*
	if ( f1 < 0 )
	{
		m_cacheFc = ampl+1;
		m_centralFrequency->setValue(m_cacheFc);
	}
	else if ( f2 > m_nbFreq-2)
	{
		m_cacheFc = m_nbFreq-3-ampl;
		m_centralFrequency->setValue(m_cacheFc);
	}
	m_cacheAmpl = ampl;
	*/

	if ( f1 < 0 || f2 > m_nbFreq-1 )
	{
		m_scrollFreq->setValue(m_cacheAmpl);
	}
	else
	{
		m_cacheAmpl = ampl;
	}
	displayParams();
}


void GccOnSpectrumAttributWidget::trt_compute()
{
	// int f1 = m_f1SpinBox->value();
	// int f2 = m_f2SpinBox->value();
	int ampl = m_scrollFreq->value();
	int fc = m_centralFrequency->value();
	int f1 = fc - ampl;
	int f2 = fc + ampl;
	int w = m_spinW->value();

	QString gccPath = getOutAttibutPath(); // m_dirPath + "/gcc_on_spectrum_" + m_spectrumName + ".raw";

	GccOnSpectrumProcess *p = new GccOnSpectrumProcess();
	p->setSpectrumPath(m_path.toStdString());
	p->setFreqInterval(f1, f2);
	p->setShift(0);
	p->setW(w);
	p->setSize(m_dimx, m_dimy, m_dimz);
	p->setGccPath(gccPath.toStdString());
	p->setIhm(pIhm);
	pStatus = 1;
	p->run();
	delete p;
	pStatus = 99;

	GeotimeProjectManagerWidget *selector = m_manager->getManagerWidget();
	QString surveyPath = selector->get_survey_fullpath_name();
	QString surveyName = selector->get_survey_name();
	bool bIsNewSurvey = false;
	SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(m_manager, surveyName, surveyPath, bIsNewSurvey);
	std::vector<QString> names; names.push_back(m_horizonName);
	std::vector<QString> pathes; pathes.push_back(m_dirPath);
	DataSelectorDialog::addNVHorizons(m_manager, survey, pathes, names);
}


void GccOnSpectrumAttributWidget::trt_launch()
{
	if ( pStatus == 0 )
	{
		GccOnSpectrumAttributWidget::MyThread0 *thread = new GccOnSpectrumAttributWidget::MyThread0(this);
		thread->start();
	}
	else
	{
		QMessageBox *msgBox = new QMessageBox(parentWidget());
		msgBox->setText("warning");
		msgBox->setInformativeText("Do you really want to stop the process ?");
		msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
		int ret = msgBox->exec();
		if ( ret == QMessageBox::Yes )
		{
			if ( pIhm )
			{
				pIhm->setMasterMessage("stop", 0, 0, GeotimeFlags::HORIZON_ATTRIBUT_STOP);
			}
		}
	}
}

void GccOnSpectrumAttributWidget::processingDisplay()
{
	if ( pStatus == 0 )
	{
		m_progress->setValue(0); m_progress->setFormat("");
		m_processing->setText("waiting ...");
		m_processing->setStyleSheet("QLabel { color : white; }");
		m_start->setText("start");
		return;
	}
	m_start->setText("stop");
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


void GccOnSpectrumAttributWidget::showTime()
{
	if ( pIhm == nullptr ) return;
	processingDisplay();
	if ( pStatus == 0 ) return;

	if ( pStatus == 99 )
	{
		displayProcessFinish();
		pStatus = 0;
	}

	if ( pIhm->isSlaveMessage() )
	{
		Ihm2Message mess = pIhm->getSlaveMessage();
		std::string message = mess.message;
		long count = mess.count;
		long countMax = mess.countMax;
		int trtId = mess.trtId;
		bool valid = mess.valid;
		float val = 100.0*count/countMax;
		QString barMessage = QString(message.c_str()) + " [ " + QString::number(val, 'f', 1) + " % ]";
		m_progress->setValue((int)val);
		m_progress->setFormat(barMessage);
		// qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
	}
	//	std::vector<std::string> mess = pIhm->getSlaveInfoMessage();
	//	for (int n=0; n<mess.size(); n++)
	//	{
	//		m_textInfo->appendPlainText(QString(mess[n].c_str()));
	//	}

}

void GccOnSpectrumAttributWidget::displayProcessFinish()
{
	QMessageBox msgBox;
	msgBox.setText("process finish");
	// msgBox.setInformativeText(tr("Information"));
	msgBox.setStandardButtons(QMessageBox::Ok);
	int ret = msgBox.exec();
}


GccOnSpectrumAttributWidget::MyThread0::MyThread0(GccOnSpectrumAttributWidget *p)
{
	this-> pp = p;
}

void GccOnSpectrumAttributWidget::MyThread0::run()
{
	pp->trt_compute();
}
