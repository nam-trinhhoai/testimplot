
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <QDebug>

#include <fileio2.h>
#include <ihm.h>
#include <OpenFileWidget.h>
#include <SpectrumComputeWidget.h>
#include <cuda_rgt2rgb.h>
#include <MeanSeismicSpectrumWidget.h>


MeanSeismicSpectrumWidget::MeanSeismicSpectrumWidget(ProjectManagerWidget *projectManagerWidget, QWidget* parent) : QWidget(parent)
{
	setProjectManagerWidget(projectManagerWidget);
	initIhm();
	thread = nullptr;
	GLOBAL_RUN_TYPE = 0;
}

MeanSeismicSpectrumWidget::~MeanSeismicSpectrumWidget()
{
	if ( thread ) delete thread;
	if ( timer ) {
		timer->disconnect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
		timer->deleteLater();
	}
}

void MeanSeismicSpectrumWidget::setSpectrumComputeWidget(SpectrumComputeWidget *spectrumComputeWidget)
{
	m_spectrumComputeWidget = spectrumComputeWidget;
}

void MeanSeismicSpectrumWidget::setProjectManagerWidget(ProjectManagerWidget *projectManagerWidget)
{
	m_projectManager = projectManagerWidget;
}


QGroupBox *MeanSeismicSpectrumWidget::getGroupBox()
{

	return mainGroupBox;
}


void MeanSeismicSpectrumWidget::initIhm()
{
	timer = nullptr;
	mainGroupBox = new QGroupBox();

	QVBoxLayout *layout = new QVBoxLayout(mainGroupBox);

	/*
	QHBoxLayout *hlayout1 = new QHBoxLayout;
	QLabel *rgtLabel = new QLabel("rgt filename");
	le_SeismicMeanRgtFilename = new QLineEdit();
	QPushButton *pb_rgtFilename = new QPushButton("...");
	hlayout1->addWidget(rgtLabel);
	hlayout1->addWidget(le_SeismicMeanRgtFilename);
	hlayout1->addWidget(pb_rgtFilename);
	*/

	m_seismicMeanRgtFileSelectWidget = new FileSelectWidget();
	m_seismicMeanRgtFileSelectWidget->setProjectManager(m_projectManager);
	m_seismicMeanRgtFileSelectWidget->setLabelText("rgt filename");
	m_seismicMeanRgtFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Rgt);
	m_seismicMeanRgtFileSelectWidget->setLabelDimensionVisible(false);
	m_seismicMeanRgtFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);

	QHBoxLayout *hlayout2 = new QHBoxLayout;
	QLabel *outPrefix = new QLabel("prefix");
	le_outMeanPrefix = new QLineEdit("mean");
	hlayout2->addWidget(outPrefix);
	hlayout2->addWidget(le_outMeanPrefix);

	QHBoxLayout *hlayout3 = new QHBoxLayout;
	QLabel *windowSizeLabel = new QLabel("window size");
	le_outMeanWindowSize = new QLineEdit("11");
	hlayout3->addWidget(windowSizeLabel);
	hlayout3->addWidget(le_outMeanWindowSize);


	QHBoxLayout *hlayout4 = new QHBoxLayout;
	cbHorizonChoice = new QComboBox;
	cbHorizonChoice->addItem("Isovalue");
	cbHorizonChoice->addItem("Horizons");
	cbHorizonChoice->setMaximumWidth(200);

	isoStepLabel = new QLabel("iso step");
	le_outMeanIsoStep = new QLineEdit("25");
	labelLayerNbre = new QLabel("Layer Number:");
	le_outMeanStepNbre = new QLineEdit("10");

	hlayout4->addWidget(cbHorizonChoice);
	hlayout4->addWidget(isoStepLabel);
	hlayout4->addWidget(le_outMeanIsoStep);
	hlayout4->addWidget(labelLayerNbre);
	hlayout4->addWidget(le_outMeanStepNbre);

	horizonWidget = new QWidget();
	QHBoxLayout *layout23_1 = new QHBoxLayout(horizonWidget);
	QLabel *label_horizons = new QLabel("horizons");
	lwHorizons = new QListWidget();
	lwHorizons->setMaximumHeight(50);
	QVBoxLayout *layout23_1_1 = new QVBoxLayout;
	QPushButton *pbHorizonAdd = new QPushButton("add");
	QPushButton *pbHorizonSub = new QPushButton("suppr");
	layout23_1_1->addWidget(pbHorizonAdd);
	layout23_1_1->addWidget(pbHorizonSub);
	layout23_1->addWidget(label_horizons);
	layout23_1->addWidget(lwHorizons);
	layout23_1->addLayout(layout23_1_1);

	qpb_seismicMean = new QProgressBar;
	qpb_seismicMean->setMinimum(0);
	qpb_seismicMean->setMaximum(100);
	qpb_seismicMean->setValue(0);
		// qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,0,0)}");
	qpb_seismicMean->setTextVisible(true);
	qpb_seismicMean->setValue(0);
	qpb_seismicMean->setFormat("");

	qbp_seismicMeanStart = new QPushButton("Start");

	layout->addWidget(m_seismicMeanRgtFileSelectWidget);
	layout->addLayout(hlayout2);
	layout->addLayout(hlayout3);
	layout->addLayout(hlayout4);
	layout->addWidget(horizonWidget); // layout2->addLayout(layout23_1);
	layout->addWidget(qpb_seismicMean);
	layout->addWidget(qbp_seismicMeanStart);

	mainGroupBox->setMaximumHeight(300);

	// connect(pb_rgtFilename, SIGNAL(clicked()), this, SLOT(trt_rgtMeanSeismicOpen()));
	connect(qbp_seismicMeanStart, SIGNAL(clicked()), this, SLOT(trt_lauchRgtMeanSeismicStart()));
	connect(cbHorizonChoice, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_horizonchoiceclick(int)));
	connect(pbHorizonAdd, SIGNAL(clicked()), this, SLOT(trt_horizonAdd()));
	connect(pbHorizonSub, SIGNAL(clicked()), this, SLOT(trt_horizonSub()));


	DisplayHorizonType();

}

void MeanSeismicSpectrumWidget::DisplayHorizonType()
{
	int idx = cbHorizonChoice->currentIndex();
	switch (idx)
	{
		case 0:
			isoStepLabel->setVisible(true);
			le_outMeanIsoStep->setVisible(true);
			labelLayerNbre->setVisible(false);
			le_outMeanStepNbre->setVisible(false);
			horizonWidget->setVisible(false);
			break;
		case 1:
			isoStepLabel->setVisible(false);
			le_outMeanIsoStep->setVisible(false);
			labelLayerNbre->setVisible(true);
			le_outMeanStepNbre->setVisible(true);
			horizonWidget->setVisible(true);
			break;
	}
}

void MeanSeismicSpectrumWidget::trt_horizonchoiceclick(int idx)
{
	DisplayHorizonType();
}

void MeanSeismicSpectrumWidget::trt_horizonAdd()
{
	std::vector<QString> vTinyNames = m_projectManager->get_horizon_names();
	std::vector<QString> vFullNames = m_projectManager->get_horizon_fullpath_names();
	if ( vTinyNames.empty() ) return;

	OpenFileWidget *p = new OpenFileWidget(this, vTinyNames, vFullNames);
	if ( !p->exec() ) return;
	// qDebug() << p->getSelectedTinyName();
	horizonTinyName.push_back(p->getSelectedTinyName());
	horizonFullname.push_back(p->getSelectedFullName());
	lwHorizons->clear();
	for (QString name:horizonTinyName)
	{
		QListWidgetItem *item = new QListWidgetItem;
		item->setText(name);
		item->setToolTip(name);
		lwHorizons->addItem(item);
	}
}

void MeanSeismicSpectrumWidget::trt_horizonSub()
{
	horizonTinyName.clear();
	horizonFullname.clear();
	lwHorizons->clear();
}


QString MeanSeismicSpectrumWidget::getOutDataFilename(QString mainPath, QString seismicTinyName, int windowSize, int width, int height, int depth)
{
	if ( !mainPath.compare("") || !seismicTinyName.compare("") ) return "";
	QString prefix = le_outMeanPrefix->text();
	QString fileName = mainPath + "/" + seismicTinyName +  "_" + prefix + "_ws_" + QString::number(windowSize) + "_size_" + QString::number(width) +"x" + QString::number(height) + "x" + QString::number(depth) + ".raw";
	return fileName;
}



QString MeanSeismicSpectrumWidget::getMainPath(QString seismicTinyName)
{
	QString ImportExportPath = m_projectManager->getImportExportPath();
	QString IJKPath = m_projectManager->getIJKPath();
	QString seimsicNamePath = m_projectManager->getIJKPath() + QString(seismicTinyName) + "/";
	QString cubeRgt2RgbPath = seimsicNamePath + "cubeRgt2RGB/";

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

	return cubeRgt2RgbPath;
}


void MeanSeismicSpectrumWidget::trt_rgtMeanSeismicOpen()
{
	/*
	std::vector<QString> v_seismic_names = m_projectManager->get_seismic_names();
	std::vector<QString> v_seismic_filenames = m_projectManagerWidget->get_seismic_fullpath_names();
	if ( v_seismic_names.empty() ) return;

	OpenFileWidget *p = new OpenFileWidget(this, v_seismic_names, v_seismic_filenames);
	if ( !p->exec() ) return;
	// qDebug() << p->getSelectedTinyName();
	rgtTinyName = p->getSelectedTinyName();
	rgtFullName = p->getSelectedFullName();
	inri::Xt xt(rgtFullName.toStdString().c_str());
	if (!xt.is_valid()) return;
	if (xt.type()!=inri::Xt::Signed_16) {
		QMessageBox::warning(this, "Fail to load cube", "Selected cube is not of type : \"signed short\", abort selection");
		rgtTinyName = "";
		rgtFullName = "";
	}
	this->le_SeismicMeanRgtFilename->setText(rgtTinyName);
	*/
}

void MeanSeismicSpectrumWidget::trt_lauchRgtMeanSeismicStart()
{
	/*
	if ( m_spectrumComputeWidget == nullptr ) return;
	QString seismicTinyName = m_spectrumComputeWidget->getSeismicTinyName();
	QString outFilename = getOutDataFilename(getMainPath(seismicTinyName), seismicTinyName);

	qDebug() << outFilename;
	*/
	if (!SpectrumComputeWidget::checkSeismicsSizeMatch(
			m_spectrumComputeWidget->getSeismicFullName(), m_seismicMeanRgtFileSelectWidget->getPath()) )
	{
		QMessageBox::warning(this, "Volumes mismatch",
				"Seismic and RGT volumes do not match, try again with matching volumes");
		return;
	}
	if ( timer == nullptr )
	{
		timer = new QTimer(this);
	    timer->start(1000);
	    timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	}

	if ( GLOBAL_RUN_TYPE == 0 )
	{
		// GLOBAL_RUN_TYPE = 1;
		if ( thread == nullptr )
			thread = new MyThreadMeanSeismicSpectrumCompute(this);
		// m_functionType = 4;
		// qDebug() << "start thread0";
		thread->start();
		// qDebug() << "start thread0 ok";
		// thread->wait();
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
		    ihm_set_trt(IHM_TRT_DIP_SMOOTHING_STOP);
		}
	}
}


void MeanSeismicSpectrumWidget::trt_StartStop()
{
	float pasech = 1.0f;
	float tdeb = 0.0f;
	float *horizon1 = nullptr, *horizon2 = nullptr;
	int arrayFreq[10], arrayIso[10], Freqcount = 0;
	int size[3];
	int isostep = le_outMeanIsoStep->text().toInt();
	int nbLayers = le_outMeanStepNbre->text().toInt();
	char seismicFilename[10000], rgtFilename[10000], rgbFilename[10000], isoFilename[10000];
	int idxHorizon = cbHorizonChoice->currentIndex(); // 0: iso 1: 2 horizons

	if (idxHorizon == 1 && horizonFullname.size() != 2)
	{
		QMessageBox::warning(this, "Fail", "You must choose 2 horizons");
		return;
	}


	int depth = 0;
	if ( idxHorizon ==  0 )
	{
		depth = 32000 / isostep;
	}
	else
	{
		depth = nbLayers;
	}

	int wsize = le_outMeanWindowSize->text().toInt();

	QString seismicTinyName = m_spectrumComputeWidget->getSeismicTinyName();
	QString seismicFullName = m_spectrumComputeWidget->getSeismicFullName();
	strcpy(seismicFilename, seismicFullName.toStdString().c_str());
	m_spectrumComputeWidget->getTraceParameter(seismicFilename, &pasech, &tdeb); // TODO in global functions
	int retsize = m_spectrumComputeWidget->getSizeFromFilename(seismicFilename, size);
	QString outFilename = getOutDataFilename(getMainPath(seismicTinyName), seismicTinyName, wsize, size[0], size[2], depth);
	strcpy(rgtFilename, m_seismicMeanRgtFileSelectWidget->getPath().toStdString().c_str());

	if (idxHorizon == 1)
		m_spectrumComputeWidget->horizonRead(horizonTinyName, horizonFullname, size[1], size[0], size[2], pasech, tdeb, &horizon1, &horizon2);

	strcpy(rgbFilename, outFilename.toStdString().c_str());
	strcpy(isoFilename, "");

	// qDebug() << outFilename;

	if ( FILEIO2::exist(seismicFilename) && FILEIO2::exist(rgtFilename) && retsize )
	{
		int *tab_gpu = NULL, tab_gpu_size;
		tab_gpu = (int*)calloc(m_spectrumComputeWidget->systemInfo->get_gpu_nbre(), sizeof(int));
		m_spectrumComputeWidget->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
		GLOBAL_RUN_TYPE = 1;
		qbp_seismicMeanStart->setText("stop");

		int tab_gpu2[] = {0,0,0,0,0,0,0};

		Rgt2Rgb *p = new Rgt2Rgb();
		p->setSeismicFilename(seismicFilename);
		p->setRgtFilename(rgtFilename);
		p->setTDeb(tdeb);
		p->setPasEch(pasech);
		p->setIsoVal(0, 32000, isostep);
		p->setHorizon(horizon1, horizon2, nbLayers);
		p->setSize(wsize);
		p->setArrayFreq(arrayFreq, arrayIso, Freqcount);
		p->setRgbFilename(rgbFilename);
		p->setIsoFilename(isoFilename);
		p->setGPUList(tab_gpu, tab_gpu_size);
		p->setOutputType(RGT2RGB_MEAN);
		p->run();
		delete p;

		GLOBAL_RUN_TYPE = 0;
		qbp_seismicMeanStart->setText("start");
		// m_projectManagerWidget->global_rgb_database_update();
	}
	if (horizon1) free(horizon1);
	if (horizon2) free(horizon2);
}


void MeanSeismicSpectrumWidget::showTime()
{
    // GLOBAL_textInfo->appendPlainText(QString("timer"));
    char txt[1000], txt2[1000];

    /* debug */
    // qDebug() << "Timer in";
    if ( GLOBAL_RUN_TYPE == 0  )
    {
    	qpb_seismicMean->setValue(0);
    	qpb_seismicMean->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
    	qpb_seismicMean->setFormat("");
    	return;
    }

    int type = -1;
    long idx, vmax;
    int msg_new = ihm_get_global_msg(&type, &idx, &vmax, txt);
    // qDebug() << "Timer message: " << QString::number(msg_new);
    if ( msg_new == 0 ) return;
    if ( type == IHM_TYPE_SEISMIC_MEAN )
    {
        float val_f = 100.0*idx/vmax;
        int val = (int)(val_f);
        qpb_seismicMean->setValue(val);
        sprintf(txt2, "run %.1f%%", val_f);
        qpb_seismicMean->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
        qpb_seismicMean->setFormat(txt2);
    }
}








MyThreadMeanSeismicSpectrumCompute::MyThreadMeanSeismicSpectrumCompute(MeanSeismicSpectrumWidget *p)
 {
     this->pp = p;
 }

 void MyThreadMeanSeismicSpectrumCompute::run()
 {
	 qDebug() << "start";
	 pp->trt_StartStop();
 }




