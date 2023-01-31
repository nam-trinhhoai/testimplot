
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QFileInfo>
#include <QGroupBox>
#include <QTabWidget>
#include <QMessageBox>

#include <GeotimeProjectManagerWidget.h>
#include <DataSelectorDialog.h>
#include <horizonAttributComputeDialog.h>
// #include <rgtToAttribut.h>
#include <multiHorizonSpectrum.h>
#include <horizonAttributSpectrumParam.h>
#include <horizonAttributMeanParam.h>
#include <horizonAttributGCCParam.h>
#include <multiHorizonsAttributProcessing.h>
#include <freeHorizonManager.h>
#include <freeHorizonQManager.h>
#include <DataSelectorDialog.h>
#include <workingsetmanager.h>
#include <seismicsurvey.h>
#include <geotimeFlags.h>
#include <fileSelectorDialog.h>
#include "collapsablescrollarea.h"
#include <ProjectManagerWidget.h>
#include <horizonSelectWidget.h>
#include <seismicinformationaggregator.h>
#include <managerFileSelectorWidget.h>
#include <GeotimeConfiguratorWidget.h>



HorizonAttributComputeDialog::HorizonAttributComputeDialog(DataSelectorDialog *dataSelectorDialog, QWidget *parent)
{
	m_dataSelectorDialog = dataSelectorDialog;
	if ( m_dataSelectorDialog ) m_selectorWidget = dataSelectorDialog->getSelectorWidget();

	// QString projectName = m_selectorWidget2->getProjectName();
	// QString surveyName = m_selectorWidget2->getSurveyName();
	setWindowTitle("Horizon attribut : ");
	QVBoxLayout *mainLayout = new QVBoxLayout(this);
	// QGroupBox *qgbMain = new QGroupBox(this);
	QVBoxLayout *layout0 = new QVBoxLayout;

	QGroupBox *qgbAttibutCheck = new QGroupBox("attribut");
	QVBoxLayout *attributLayout = new QVBoxLayout(qgbAttibutCheck);

	QHBoxLayout *qhbAttributSpectrum = new QHBoxLayout;
	m_cbSpectrum = new QCheckBox("spectrum");
	m_cbSpectrum->setChecked(true);
	m_spectrumParam = new HorizonAttributSpectrumParam();
	qhbAttributSpectrum->addWidget(m_cbSpectrum);
	qhbAttributSpectrum->addWidget(m_spectrumParam);
	qhbAttributSpectrum->setAlignment(Qt::AlignLeft);

	QHBoxLayout *qhbAttributMean = new QHBoxLayout;
	m_cbMean = new QCheckBox("mean");
	m_meanParam = new HorizonAttributMeanParam();
	m_meanParam->setWSize(m_meanWindowSize);
	qhbAttributMean->addWidget(m_cbMean);
	qhbAttributMean->addWidget(m_meanParam);
	qhbAttributMean->setAlignment(Qt::AlignLeft);

	QHBoxLayout *qhbAttributGcc = new QHBoxLayout;
	m_cbGcc = new QCheckBox("gcc");
	m_gccParam = new HorizonAttributGCCParam();
	m_gccParam->setOffset(m_gccOffset);
	m_gccParam->setW(m_w);
	m_gccParam->setShift(m_shift);
	qhbAttributGcc->addWidget(m_cbGcc);
	qhbAttributGcc->addWidget(m_gccParam);
	qhbAttributGcc->setAlignment(Qt::AlignLeft);

	attributLayout->addLayout(qhbAttributSpectrum);
	attributLayout->addLayout(qhbAttributMean);
	attributLayout->addLayout(qhbAttributGcc);
	qgbAttibutCheck->setAlignment(Qt::AlignTop);
	attributLayout->setAlignment(Qt::AlignTop);
	qgbAttibutCheck->setMaximumHeight(200);


	// ===========================================================================
	/*
	QWidget *topWidget = new QWidget;
	QHBoxLayout *horizonLayout = new QHBoxLayout(topWidget);
	QLabel *horizonLabel = new QLabel("horizons");
	m_horizonListWidget = new QListWidget;
	QVBoxLayout *horizonButton = new QVBoxLayout;
	QPushButton *horizonAdd = new QPushButton("+");
	QPushButton *horizonSub = new QPushButton("-");
	horizonButton->addWidget(horizonAdd);
	horizonButton->addWidget(horizonSub);
	horizonLayout->addWidget(horizonLabel);
	horizonLayout->addWidget(m_horizonListWidget);
	horizonLayout->addLayout(horizonButton);
	topWidget->setFixedHeight(200);
	*/

	QWidget *topWidget = new QWidget;
	QHBoxLayout *horizonLayout = new QHBoxLayout(topWidget);
	m_horizonSelectWidget = new HorizonSelectWidget();
	horizonLayout->addWidget(m_horizonSelectWidget);
	topWidget->setFixedHeight(200);

	QWidget *lineH = new QWidget;
	lineH->setFixedHeight(2);
	lineH->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	lineH->setStyleSheet(QString("background-color: #ffffff;"));

	QWidget *seismicWidget = new QWidget;
	QHBoxLayout *seismicLayout = new QHBoxLayout(seismicWidget);
	QLabel *seismicLabel = new QLabel("seismic");
	m_seismicListWidget = new QListWidget;
	// m_seismicListWidget->setIconSize(QSize(16, 16));

	QVBoxLayout *seismicButton = new QVBoxLayout;
	QPushButton *seismicAdd = new QPushButton("add");
	QPushButton *seismicSub = new QPushButton("suppress");
	seismicButton->addWidget(seismicAdd);
	seismicButton->addWidget(seismicSub);
	seismicLayout->addWidget(seismicLabel);
	seismicLayout->addWidget(m_seismicListWidget);
	seismicLayout->addLayout(seismicButton);
	seismicWidget->setFixedHeight(150);

	m_progressBar = new QProgressBar;
	m_progressBar->setOrientation(Qt::Horizontal);
	m_progressBar->setMinimum(0);
	m_progressBar->setMaximum(100);

	QHBoxLayout *buttonLayout = new QHBoxLayout;
	m_buttonStart = new QPushButton("start");
	m_buttonStop = new QPushButton("stop");
	buttonLayout->addWidget(m_buttonStart);
	buttonLayout->addWidget(m_buttonStop);

	layout0->addWidget(qgbAttibutCheck);
	layout0->addWidget(topWidget); //horizonLayout);
	// layout0->addWidget(m_horizonSelectWidget);
	layout0->addWidget(seismicWidget); //seismicLayout);
	layout0->addWidget(m_progressBar);
	layout0->addLayout(buttonLayout);

	mainLayout->addLayout(layout0);
	mainLayout->setAlignment(Qt::AlignTop);

	timer = new QTimer();
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));

	// connect (m_attributType, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &HorizonAttributComputeDialog::methodChanged);
	// connect(m_attributType, SIGNAL(currentIndexChanged(int)), SLOT(methodChanged(int)));
	// connect(horizonAdd, SIGNAL(clicked()), this, SLOT(trt_horizonAdd()));
	// connect(horizonSub, SIGNAL(clicked()), this, SLOT(trt_horizonSub()));
	connect(seismicAdd, SIGNAL(clicked()), this, SLOT(trt_seismicAdd()));
	connect(seismicSub, SIGNAL(clicked()), this, SLOT(trt_seismicSub()));
	connect(m_buttonStart, SIGNAL(clicked()), this, SLOT(trt_start()));
	connect(m_buttonStop, SIGNAL(clicked()), this, SLOT(trt_stop()));

	connect(m_cbSpectrum, SIGNAL(stateChanged(int)), this, SLOT(trt_cbSpectrumChange(int)));
	connect(m_cbMean, SIGNAL(stateChanged(int)), this, SLOT(trt_cbMeanChange(int)));
	connect(m_cbGcc, SIGNAL(stateChanged(int)), this, SLOT(trt_cbGccChange(int)));

	trt_cbSpectrumChange(1);
	trt_cbMeanChange(0);
	trt_cbGccChange(0);
	resize(800, this->height());
}


HorizonAttributComputeDialog::~HorizonAttributComputeDialog()
{

}

void HorizonAttributComputeDialog::setProjectManager(GeotimeProjectManagerWidget *p)
{
	m_selectorWidget = p;
	if ( m_selectorWidget == nullptr ) return;
	QString projectName = m_selectorWidget->get_projet_name();
	QString surveyName = m_selectorWidget->get_survey_name();
	setWindowTitle("Horizon attribut : " + projectName + " - " + surveyName);
}

void HorizonAttributComputeDialog::setProjectManager(ProjectManagerWidget *p)
{
	m_selectorWidget2 = p;
	if ( m_selectorWidget2 == nullptr ) return;
	QString projectName = m_selectorWidget2->getProjectName();
	QString surveyName = m_selectorWidget2->getSurveyName();
	setWindowTitle("Horizon attribut : " + projectName + " - " + surveyName);
}

void HorizonAttributComputeDialog::setWorkingSetManager(WorkingSetManager *p)
{
	m_workingSetManager = p;
	if ( m_horizonSelectWidget ) m_horizonSelectWidget->setWorkingSetManager(m_workingSetManager);
}

void HorizonAttributComputeDialog::setInputHorizons(QString path, QString name)
{
	m_horizonSelectWidget->clearData();
	m_horizonSelectWidget->addData(name, path);
}

void HorizonAttributComputeDialog::displaySeismicList()
{
	m_seismicListWidget->clear();
	for (int i=0; i<m_seismicNames.size(); i++)
	{
		QString str = m_seismicNames[i];
		QString path = m_seismicPath[i];
		QIcon icon = FreeHorizonQManager::getDataSetIcon(path);
		QListWidgetItem *item = new QListWidgetItem(icon, str);
		m_seismicListWidget->addItem(item);
		// m_horizonListWidget->addItem(str);
	}
}




void HorizonAttributComputeDialog::displayHorizonList()
{
	m_horizonListWidget->clear();
	for (int i=0; i<m_horizonNames.size(); i++)
	{
		QString str = m_horizonNames[i];
		QString path = m_horizonPath[i];
		QIcon icon = FreeHorizonQManager::getHorizonIcon(path, 16);
		QListWidgetItem *item = new QListWidgetItem(icon, str);
		m_horizonListWidget->addItem(item);
		// m_horizonListWidget->addItem(str);
	}
}

void HorizonAttributComputeDialog::trt_horizonAdd()
{
	std::vector<QString> names;
	std::vector<QString> path;

	if ( m_selectorWidget )
	{
		names = m_selectorWidget->get_freehorizon_names();
		path = m_selectorWidget->get_freehorizon_fullnames();
	}
	else if ( m_selectorWidget2 )
	{
		names = m_selectorWidget2->getFreeHorizonNames();
		path = m_selectorWidget2->getFreeHorizonFullName();
	}
	else
	{
		return;
	}

	FileSelectorDialog dialog(&names, "Select file name");
    dialog.setDataPath(&path);
	dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::horizon);
	dialog.setMultipleSelection(true);
	int code = dialog.exec();
	if (code==QDialog::Accepted)
	{
		std::vector<int> vIdx = dialog.getMultipleSelectedIndex();
		for (int idx:vIdx)
		{
			m_horizonNames.push_back(names[idx]);
			m_horizonPath.push_back(path[idx]);
		}
		displayHorizonList();
	}
	// for (QString str:names) qDebug() << str;
	// for (QString str:path) qDebug() << str;
}

void HorizonAttributComputeDialog::trt_horizonSub()
{
	m_horizonNames.clear();
	m_horizonPath.clear();
	displayHorizonList();
}

bool HorizonAttributComputeDialog::checkFormat(std::vector<QString> &path)
{
	for (int i=0; i<path.size(); i++)
	{
		QFileInfo fi(path[i]);
		if ( fi.suffix() != "xt" ) return false;
		inri::Xt xt(path[i].toStdString().c_str());
		if ( !xt.is_valid() ) return false;
		inri::Xt::Type xtType = xt.type();
		QString typeStr = QString::fromStdString(xt.type2str(xtType));
		if ( typeStr != "Signed_16" ) return false;
	}
	return true;
}

void HorizonAttributComputeDialog::trt_seismicAdd()
{
	std::vector<QString> names;
	std::vector<QString> path;

	SeismicInformationAggregator* aggregator = new SeismicInformationAggregator(m_workingSetManager, false);
    ManagerFileSelectorWidget *widget = new ManagerFileSelectorWidget(aggregator);

    int code = widget->exec();
    if (code==QDialog::Accepted)
    {
    	std::pair<std::vector<QString>, std::vector<QString>> names0 = widget->getSelectedNames();

    	if ( !checkFormat(names0.second) )
    	{
    		QMessageBox *msgBox = new QMessageBox(parentWidget());
    		msgBox->setText("warning");
    		msgBox->setInformativeText("Dataset has a wrong data format\nPlease choose a short int format or convert it.");
    		msgBox->setStandardButtons(QMessageBox::Ok );
    		int ret = msgBox->exec();
    		return;
    	}

    	for (int i=0; i<names0.first.size(); i++)
    	{
    		m_seismicNames.push_back(names0.first[i]);
    		m_seismicPath.push_back(names0.second[i]);
    	}
    	displaySeismicList();
    }
    delete widget;
}

void HorizonAttributComputeDialog::trt_seismicSub()
{
	m_seismicNames.clear();
	m_seismicPath.clear();
	displaySeismicList();
}


std::vector<float*> HorizonAttributComputeDialog::getHorizonData()
{
	int dimy = 1;
	int dimz = 1;
	std::vector<float*> horizons;
	int N = m_horizonPath.size();
	horizons.resize(N, nullptr);
	for (int n=0; n<N; n++)
	{
		QString filename = m_horizonPath[n];
		FILE *pf = fopen((char*)filename.toStdString().c_str(), "r");
		if ( pf == nullptr ) continue;
		float *data = (float*)calloc((long)dimy*dimz, sizeof(float));
		fread(data, sizeof(float), dimy*dimz, pf);
		fclose(pf);
		horizons[n] = data;
	}
	return horizons;
}

std::vector<std::string> HorizonAttributComputeDialog::QStringToStdString(std::vector<QString> &in)
{
	std::vector<std::string> out;
	out.resize(in.size());
	for (int i=0; i<in.size(); i++)
		out[i] = in[i].toStdString();
	return out;
}

std::vector<std::vector<std::string>> HorizonAttributComputeDialog::getAttributFilename(QString suffix)
{
	std::vector<std::vector<std::string>> filenames;
	m_horizonNames = m_horizonSelectWidget->getNames();
	m_horizonPath = m_horizonSelectWidget->getPaths();

	int nDataSet = m_seismicNames.size();
	int nHorizons = m_horizonNames.size();

	filenames.resize(nDataSet);
	for (int n=0; n<nDataSet; n++)
	{
		filenames[n].resize(nHorizons);
		for (int i=0; i<nHorizons; i++)
		{
			QFileInfo fileInfo(m_horizonPath[i]);
			QString filename = fileInfo.absoluteFilePath() + "/" + suffix + m_seismicNames[n] + ".raw";
			filenames[n][i] = filename.toStdString();
		}
	}
	return filenames;
}

std::vector<std::vector<std::string>> HorizonAttributComputeDialog::getAttributSpectrumFilename()
{
	QString suffix = QString::fromStdString(FreeHorizonManager::spectrumSuffix);
	int wsize = m_spectrumParam->getWSize();
	std::vector<std::vector<std::string>> filenames;

	m_horizonNames = m_horizonSelectWidget->getNames();
	m_horizonPath = m_horizonSelectWidget->getPaths();

	int nDataSet = m_seismicNames.size();
	int nHorizons = m_horizonNames.size();

	filenames.resize(nDataSet);
	for (int n=0; n<nDataSet; n++)
	{
		filenames[n].resize(nHorizons);
		for (int i=0; i<nHorizons; i++)
		{
			QFileInfo fileInfo(m_horizonPath[i]);
			QString filename = fileInfo.absoluteFilePath() + "/" + suffix + "_wsize_" + QString::number(wsize) + "_" + m_seismicNames[n] + QString::fromStdString(FreeHorizonManager::attributExt);
			filenames[n][i] = filename.toStdString();
		}
	}
	return filenames;
}

std::vector<std::vector<std::string>> HorizonAttributComputeDialog::getAttributGCCFilename()
{
	QString suffix = QString::fromStdString(FreeHorizonManager::gccSuffix);
	int offset = m_gccParam->getOffset();
	int shift= m_gccParam->getShift();
	int w= m_gccParam->getW();
	std::vector<std::vector<std::string>> filenames;

	m_horizonNames = m_horizonSelectWidget->getNames();
	m_horizonPath = m_horizonSelectWidget->getPaths();

	int nDataSet = m_seismicNames.size();
	int nHorizons = m_horizonNames.size();

	filenames.resize(nDataSet);
	for (int n=0; n<nDataSet; n++)
	{
		filenames[n].resize(nHorizons);
		for (int i=0; i<nHorizons; i++)
		{
			QFileInfo fileInfo(m_horizonPath[i]);
			QString filename = fileInfo.absoluteFilePath() + "/" + suffix +
					"_offset_" + QString::number(offset) +
					"_shift_" + QString::number(shift) +
					"_w_" + QString::number(w) +
					"_" + m_seismicNames[n] +
					QString::fromStdString(FreeHorizonManager::attributExt);
			filenames[n][i] = filename.toStdString();
		}
	}
	return filenames;
}

std::vector<std::vector<std::string>> HorizonAttributComputeDialog::getAttributMeanFilename()
{
	QString suffix = QString::fromStdString(FreeHorizonManager::meanSuffix);
	int size = m_meanParam->getWSize();
	std::vector<std::vector<std::string>> filenames;
	m_horizonNames = m_horizonSelectWidget->getNames();
	m_horizonPath = m_horizonSelectWidget->getPaths();

	int nDataSet = m_seismicNames.size();
	int nHorizons = m_horizonNames.size();

	filenames.resize(nDataSet);
	for (int n=0; n<nDataSet; n++)
	{
		filenames[n].resize(nHorizons);
		for (int i=0; i<nHorizons; i++)
		{
			QFileInfo fileInfo(m_horizonPath[i]);
			QString filename = fileInfo.absoluteFilePath() + "/" + suffix +
					"_size_" + QString::number(size) +
					"_" + m_seismicNames[n] +
					QString::fromStdString(FreeHorizonManager::attributExt);
			filenames[n][i] = filename.toStdString();
		}
	}
	return filenames;
}
void vectorPrint(std::vector<std::vector<std::string>>& tab)
{
	for (std::vector<std::string> vect:tab)
	{
		for (std::string str:vect)
			fprintf(stderr, "%s\n", str.c_str());
	}

}

bool HorizonAttributComputeDialog::checkFileExist(std::vector<std::vector<std::string>>& spectrumFilename,
		std::vector<std::vector<std::string>>& gccFilename,
		std::vector<std::vector<std::string>>& meanFilename)
{
	m_horizonNames = m_horizonSelectWidget->getNames();
	m_horizonPath = m_horizonSelectWidget->getPaths();

	int nDataSet = m_seismicNames.size();
	int nHorizons = m_horizonNames.size();

	// m_seismicNames

	QString msg = "";
	if ( spectrumFilename .size() > 0 )
	{
		for (int i=0; i<m_seismicNames.size(); i++)
		{
			for (int j=0; j<m_horizonNames.size(); j++)
			{
				QFile file(QString::fromStdString(spectrumFilename[i][j]));
				if ( file.exists() )
				{
					msg += "spectrum attribut on the horizon " + m_horizonNames[j] + " on the dataset " + m_seismicNames[i] + "\n\n";
				}
			}
		}
	}

	if ( gccFilename.size() > 0 )
	{
		for (int i=0; i<m_seismicNames.size(); i++)
		{
			for (int j=0; j<m_horizonNames.size(); j++)
			{
				QFile file(QString::fromStdString(gccFilename[i][j]));
				if ( file.exists() )
				{
					msg += "gcc attribut on the horizon " + m_horizonNames[j] + " on the dataset " + m_seismicNames[i] + "\n\n";
				}
			}
		}
	}

	if ( meanFilename.size() > 0 )
	{
		for (int i=0; i<m_seismicNames.size(); i++)
		{
			for (int j=0; j<m_horizonNames.size(); j++)
			{
				QFile file(QString::fromStdString(meanFilename[i][j]));
				if ( file.exists() )
				{
					msg += "mean attribut on the horizon " + m_horizonNames[j] + " on the dataset " + m_seismicNames[i] + "\n\n";
				}
			}
		}
	}

	if ( !msg.isEmpty() )
	{
		QString msg0 = "the following attributs already exist\n\n";
		msg0 += msg;
		msg0 += "Do you want to overwrite them ?";

		QMessageBox msgBox(this);
		msgBox.setText("warning");
		msgBox.setInformativeText("Some attributs are already computed, do you want to overwrite them ?");
		msgBox.setDetailedText(msg0);
		msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No );
		msgBox.setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
		// https://stackoverflow.com/questions/37668820/how-can-i-resize-qmessagebox
		QSpacerItem* horizontalSpacer = new QSpacerItem(500, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);
		QGridLayout* layout = (QGridLayout*)msgBox.layout();
		layout->addItem(horizontalSpacer, layout->rowCount(), 0, 1, layout->columnCount());

		int ret = msgBox.exec();
		if ( ret == QMessageBox::Yes )
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	return true;
}

void HorizonAttributComputeDialog::trt_threadRun()
{
	if ( !pIhm2 ) pIhm2 = new Ihm2();
	pIhm2->clearSlaveMessage();
	pIhm2->clearMasterMessage();

	std::vector<std::vector<std::string>> spectrumFilename;
	std::vector<std::vector<std::string>> gccFilename;
	std::vector<std::vector<std::string>> meanFilename;

	std::vector<std::string> dataSetPath = QStringToStdString(m_seismicPath);
	std::vector<std::string> horizonPath = QStringToStdString(m_horizonPath);
	for (int i=0; i<horizonPath.size(); i++)
		horizonPath[i] = horizonPath[i] + "/" + FreeHorizonManager::isoDataName; //isoData.raw";

	if ( m_cbSpectrum->checkState() == Qt::Checked ) spectrumFilename = getAttributSpectrumFilename();
	if ( m_cbGcc->checkState() == Qt::Checked ) gccFilename = getAttributGCCFilename();
	if ( m_cbMean->checkState() == Qt::Checked ) meanFilename = getAttributMeanFilename();

	MultiHorizonsAttributProcessing *p = new MultiHorizonsAttributProcessing();

	p->setDataSetPath(dataSetPath);
	p->setHorizonPath(horizonPath);

	p->setMethodSpectrum(m_cbSpectrum->checkState() == Qt::Checked);
	p->setMethodMean(m_cbMean->checkState() == Qt::Checked);
	p->setMethodGcc(m_cbGcc->checkState() == Qt::Checked);

	p->setSpectrumPath(spectrumFilename);
	p->setMeanPath(meanFilename);
	p->setGccPath(gccFilename);

	p->setSpectrumParamWindowSize(m_spectrumParam->getWSize());
	p->setSpectrumParamHatPower(m_spectrumParam->getHatPower());

	p->setMeanParamWindowSize(m_meanParam->getWSize());
	p->setGccParamWindowSize(m_gccParam->getOffset());
	p->setGccParamW(m_gccParam->getW());
	p->setGccParamShift(m_gccParam->getShift());
	p->setIhm(pIhm2);
	m_valStartStop = 1;
	p->run();
	m_valStartStop = 0;
	delete p;

	if ( m_workingSetManager && m_treeUpdate )
	{
		QString surveyPath = m_selectorWidget->get_survey_fullpath_name();
		QString surveyName = m_selectorWidget->get_survey_name();
		bool bIsNewSurvey = false;
		SeismicSurvey* survey = DataSelectorDialog::dataGetBaseSurvey(m_workingSetManager, surveyName, surveyPath, bIsNewSurvey);
		DataSelectorDialog::addNVHorizons(m_workingSetManager, survey, m_horizonPath, m_horizonNames);
	}
}


void HorizonAttributComputeDialog::trt_start()
{
	if ( m_valStartStop == 1 ) return;

	m_horizonNames = m_horizonSelectWidget->getNames();
	m_horizonPath = m_horizonSelectWidget->getPaths();

	std::vector<std::vector<std::string>> spectrumFilename;
	std::vector<std::vector<std::string>> gccFilename;
	std::vector<std::vector<std::string>> meanFilename;

	std::vector<std::string> dataSetPath = QStringToStdString(m_seismicPath);
	if (dataSetPath.size()==0)
	{
		QMessageBox::warning(this, tr("Compute attributes"), tr("There is no dataset to compute on. Please select a dataset."));
		return;
	}
	std::vector<std::string> horizonPath = QStringToStdString(m_horizonPath);
	if (horizonPath.size()==0)
	{
		QMessageBox::warning(this, tr("Compute attributes"), tr("There is no horizon to compute on. Please select an horizon."));
		return;
	}
	for (int i=0; i<horizonPath.size(); i++)
		horizonPath[i] = horizonPath[i] + "/" + FreeHorizonManager::isoDataName; //isoData.raw";

	bool doCompute = m_cbSpectrum->checkState() == Qt::Checked || m_cbGcc->checkState() == Qt::Checked ||
			m_cbMean->checkState() == Qt::Checked;
	if (!doCompute)
	{
		QMessageBox::warning(this, tr("Compute attributes"), tr("No process is selected, choose at least spectrum, gcc or mean."));
		return;
	}

	if ( m_cbSpectrum->checkState() == Qt::Checked ) spectrumFilename = getAttributSpectrumFilename();
	if ( m_cbGcc->checkState() == Qt::Checked ) gccFilename = getAttributGCCFilename();
	if ( m_cbMean->checkState() == Qt::Checked ) meanFilename = getAttributMeanFilename();

	bool warning = checkFileExist(spectrumFilename, gccFilename, meanFilename);
	if ( !warning ) return;
	// if ( !paramInitCreate() ) return;
	// trt_threadRun();
	HorizonAttributComputeDialogTHREAD *thread = new HorizonAttributComputeDialogTHREAD(this);
	thread->start();
}

void HorizonAttributComputeDialog::trt_stop()
{
	if ( m_valStartStop == 0 ) return;
	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort the process ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		if ( pIhm2 )
			if ( pIhm2 ) pIhm2->setMasterMessage("stop", 0, 0, GeotimeFlags::HORIZON_ATTRIBUT_STOP);
			// pIhm2->setMasterMessage("stop", 0, 1, 1);
		// setStartStopStatus(STATUS_STOP);
	}
}


void HorizonAttributComputeDialog::showTime()
{
	if ( !pIhm2 ) return;
	if ( m_valStartStop == 1 && pIhm2->isSlaveMessage() )
	{
		Ihm2Message mess = pIhm2->getSlaveMessage();
		float val_f = 100.0*mess.count/mess.countMax;
		int val = (int)(val_f);
		m_progressBar->setValue(val);
		m_progressBar->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
		QString text = QString::fromStdString(mess.message) + " " + QString::number(val_f, 'f', 1) + "%";
		m_progressBar->setFormat(text);
	}
	else if ( m_valStartStop == 0 )
	{
		m_progressBar->setValue(0);
		m_progressBar->setFormat("");
	}
}

void HorizonAttributComputeDialog::trt_cbSpectrumChange(int val)
{
	m_spectrumParam->setEnabled(m_cbSpectrum->isChecked());
}

void HorizonAttributComputeDialog::trt_cbMeanChange(int val)
{
	m_meanParam->setEnabled(m_cbMean->isChecked());
}

void HorizonAttributComputeDialog::trt_cbGccChange(int val)
{
	m_gccParam->setEnabled(m_cbGcc->isChecked());
}

// ==============================================
HorizonAttributComputeDialogTHREAD::HorizonAttributComputeDialogTHREAD(HorizonAttributComputeDialog *p)
 {
     this->pp = p;
 }

 void HorizonAttributComputeDialogTHREAD::run()
 {
	 fprintf(stderr, "thread start\n");
	 pp->trt_threadRun();
 }

