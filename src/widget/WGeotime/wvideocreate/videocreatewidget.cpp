#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTabWidget>
#include <QScrollArea>
#include <QDebug>
#include <QInputDialog>
#include <QTemporaryFile>
#include <QByteArray>
#include <QDir>

#include <QFileUtils.h>
#include <QProcess>

#include <GeotimeConfiguratorWidget.h>
#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <aviParamWidget.h>
#include <QMessageBox>
#include "sismagedbmanager.h"
#include "smsurvey3D.h"
#include "seismicsurvey.h"
#include "seismic3ddataset.h"
#include "smdataset3D.h"
#include <cuda_rgb2torgb1.h>
#include "globalconfig.h"
#include <rgtToAttribut.h>
#include <cuda_rgt2rgb.h>
#include <spectrumProcessWidget.h>
#include <checkDataSizeMatch.h>
#include <rgtToAttribut.h>
#include <horizonUtils.h>
#include <rawToAviWidget.h>
#include <rgtSpectrumHeader.h>
#include <fileSelectorDialog.h>
#include <freeHorizonQManager.h>
#include <ProjectManager.h>
#include <SurveyManager.h>
#include <workingsetmanager.h>

#include <ihm2.h>
#include <algorithm>
#include <util.h>

#include <isoHorizonManager.h>
#include "collapsablescrollarea.h"
#include <Xt.h>
#include <videocreatewidget.h>


VideoCreateWidget::VideoCreateWidget(ProjectManagerWidget *projectmanager)
{
	QVBoxLayout * mainLayout00 = new QVBoxLayout(this);
	m_processing = new QLabel(".");

	if ( projectmanager == nullptr )
	{
		m_projectmanager = new ProjectManagerWidget;
		// QPushButton *qpb_loadsession = new QPushButton("load session");
		// qvb_programmanager->addWidget(projectmanager);
		m_workingSetManager = new WorkingSetManager(this);
		m_workingSetManager->setManagerWidgetV2(m_projectmanager);
	}
	else
	{
		m_projectmanager = projectmanager;
	}

	QGroupBox *qgbProgramManager = new QGroupBox;
	QVBoxLayout *qvb_programmanager = new QVBoxLayout(qgbProgramManager);
	// QPushButton *qpb_loadsession = new QPushButton("load session");
	qvb_programmanager->addWidget(m_projectmanager);


	QGroupBox *qgbMain2 = new QGroupBox;
	QVBoxLayout *main2 = new QVBoxLayout(qgbMain2);
	qgbMain2->setAlignment(Qt::AlignTop);
	main2->setAlignment(Qt::AlignTop);

	QGroupBox *qgbVideo = new QGroupBox;
	qgbVideo->setAlignment(Qt::AlignTop);
	QVBoxLayout *qvbVideo = new QVBoxLayout(qgbVideo);
	qvbVideo->setAlignment(Qt::AlignTop);


	m_processingType = new QComboBox;
	m_processingType->addItem(m_comboMenuNew);
	m_processingType->addItem(m_comboMenuExist);


//	// QVBoxLayout *qvbSeismic = new QVBoxLayout(this);
//	m_qhbSeismic = new QHBoxLayout;
//	labelSeismic = new QLabel("seismic filename");
//	m_buttonSeismicOpen = new QPushButton("");
//	m_buttonSeismicOpen->setIcon(QIcon(":/slicer/icons/openfile.png"));
//	m_buttonSeismicOpen->setIconSize(QSize(20, 20));
//	m_buttonSeismicOpen->resize(QSize(20, 20));
//	m_seismicLineEdit = new QLineEdit("");
//	m_seismicLineEdit->setReadOnly(true);
//
//	m_qhbSeismic->addWidget(labelSeismic);
//	m_qhbSeismic->addWidget(m_seismicLineEdit);
//	m_qhbSeismic->addWidget(m_buttonSeismicOpen);

	QHBoxLayout *m_qhbIso = new QHBoxLayout;
	labelIso = new QLabel("rgt isovalue");
	m_buttonIsoOpen = new QPushButton("");
	m_buttonIsoOpen->setIcon(QIcon(":/slicer/icons/openfile.png"));
	m_buttonIsoOpen->setIconSize(QSize(20, 20));
	m_buttonIsoOpen->resize(QSize(20, 20));
	m_isoLineEdit = new QLineEdit("");
	m_isoLineEdit->setReadOnly(true);

	m_attributList = new QListWidget();

	m_qhbIso->addWidget(labelIso);
	m_qhbIso->addWidget(m_isoLineEdit);
	m_qhbIso->addWidget(m_buttonIsoOpen);

	m_seismicFileSelectWidget = new FileSelectWidget();
	m_seismicFileSelectWidget->setProjectManager(m_projectmanager);
	m_seismicFileSelectWidget->setLabelText("seismic filename");
	m_seismicFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Seismic);
	m_seismicFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_seismicFileSelectWidget->setLabelDimensionVisible(false);
	m_seismicFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);

	m_rgtFileSelectWidget = new FileSelectWidget();
	m_rgtFileSelectWidget->setProjectManager(m_projectmanager);
	m_rgtFileSelectWidget->setLabelText("rgt filename");
	m_rgtFileSelectWidget->setWorkingSetManager(m_workingSetManager);
	m_rgtFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Rgt);
	m_rgtFileSelectWidget->setLabelDimensionVisible(false);
	m_rgtFileSelectWidget->setFileFormat(FileSelectWidget::FILE_FORMAT::INT16);


	QHBoxLayout *qhbPrefix = new QHBoxLayout;
	QLabel *prefixLabel = new QLabel("prefix: ");
	m_preffix = new QLineEdit("video");
	qhbPrefix->addWidget(prefixLabel);
	qhbPrefix->addWidget(m_preffix);


	QHBoxLayout *qhbIsoStep = new QHBoxLayout;
	QLabel *labelIsoStep = new QLabel("iso step:");
	m_isoStep = new QLineEdit("25");

	m_attributType = new QComboBox;
	m_attributType->addItem("spectrum");
	m_attributType->addItem("mean");
	m_attributType->addItem("gcc");
	m_attributType->setMaximumWidth(200);

	qhbIsoStep->addWidget(m_attributType);
	qhbIsoStep->addWidget(labelIsoStep);
	qhbIsoStep->addWidget(m_isoStep);


	m_spectrumCollapseParam = new CollapsableScrollArea("Parameters");
	// QVBoxLayout *qvbSpectrumParam = new QVBoxLayout;
	QLabel *labelSpectrumWindowSize = new QLabel("windows size:");
	m_spectrumWindowSize = new QLineEdit("64");
	QHBoxLayout *qhbSpectrumWindowSize = new QHBoxLayout;
	qhbSpectrumWindowSize->addWidget(labelSpectrumWindowSize);
	qhbSpectrumWindowSize->addWidget(m_spectrumWindowSize);
	m_spectrumCollapseParam->setContentLayout(*qhbSpectrumWindowSize);
	m_spectrumCollapseParam->setStyleSheet(GeotimeConfigurationWidget::paramColorStyle);

	m_meanCollapseParam = new CollapsableScrollArea("Parameters");
	QLabel *labelMeanWindowSize = new QLabel("windows size:");
	m_meanWindowSize = new QLineEdit("7");
	QHBoxLayout *qhbMeanWindowSize = new QHBoxLayout;
	qhbMeanWindowSize->addWidget(labelMeanWindowSize);
	qhbMeanWindowSize->addWidget(m_meanWindowSize);
	m_meanCollapseParam->setContentLayout(*qhbMeanWindowSize);
	m_meanCollapseParam->setStyleSheet(GeotimeConfigurationWidget::paramColorStyle);

	m_gccCollapseParam = new CollapsableScrollArea("Parameters");
	QLabel *labelGccWindowSize = new QLabel("windows size:");
	m_gccWindowSize = new QLineEdit("7");
	QHBoxLayout *qhbGccWindowSize = new QHBoxLayout;
	qhbGccWindowSize->addWidget(labelGccWindowSize);
	qhbGccWindowSize->addWidget(m_gccWindowSize);

	QLabel *labelGccW = new QLabel("w:");
	m_gccW = new QLineEdit("7");
	QHBoxLayout *qhbGccW = new QHBoxLayout;
	qhbGccW->addWidget(labelGccW);
	qhbGccW->addWidget(m_gccW);

	QLabel *labelGccShift = new QLabel("shift:");
	m_gccShift = new QLineEdit("5");
	QHBoxLayout *qhbGccShift = new QHBoxLayout;
	qhbGccShift->addWidget(labelGccShift);
	qhbGccShift->addWidget(m_gccShift);

	QVBoxLayout *qvbGccParam = new QVBoxLayout;
	qvbGccParam->addLayout(qhbGccWindowSize);
	qvbGccParam->addLayout(qhbGccW);
	qvbGccParam->addLayout(qhbGccShift);
	m_gccCollapseParam->setContentLayout(*qvbGccParam);
	m_gccCollapseParam->setStyleSheet(GeotimeConfigurationWidget::paramColorStyle);


	m_aviParam = new AviParamWidget(this);
	QVBoxLayout *qvbAviParam = new QVBoxLayout;
	qvbAviParam->addWidget(m_aviParam);
	m_collapseAviParam = new CollapsableScrollArea("avi parameters");
	m_collapseAviParam->setContentLayout(*qvbAviParam);
	m_collapseAviParam->setStyleSheet(GeotimeConfigurationWidget::paramColorStyle);

	main2->addWidget(m_seismicFileSelectWidget);
	// main2->addWidget(m_processingType);
	// main2->addLayout(m_qhbSeismic);
	// main2->addLayout(m_qhbIso);
	// main2->addWidget(m_attributList);
	main2->addWidget(m_rgtFileSelectWidget);
	main2->addLayout(qhbPrefix);
	// qvbVideo->addWidget(m_attributType);
	main2->addLayout(qhbIsoStep);
	main2->addWidget(m_spectrumCollapseParam);
	main2->addWidget(m_meanCollapseParam);
	main2->addWidget(m_gccCollapseParam);
	main2->addWidget(m_collapseAviParam);

	QScrollArea *scrollArea = new QScrollArea;
	scrollArea->setWidget(qgbMain2);
	scrollArea->setWidgetResizable(true);

	// qvbVideo->addLayout(main2);
	qvbVideo->addWidget(m_processing);
	qvbVideo->addWidget(scrollArea);

	QGroupBox *qgb_control = new QGroupBox("Run");
	QVBoxLayout *qvb_control = new QVBoxLayout(qgb_control);
	m_textInfo = new QPlainTextEdit("ready");
	m_progress = new QProgressBar();
	// qpbRgtPatchProgress->setGeometry(5, 45, 240, 20);
	m_progress->setMinimum(0);
	m_progress->setMaximum(100);
	m_progress->setValue(0);
	m_progress->setTextVisible(true);
	m_progress->setFormat("");

	m_start = new QPushButton("start");
	m_kill = new QPushButton("stop");
	QHBoxLayout *qhbButtons = new QHBoxLayout;
	qhbButtons->addWidget(m_start);
	qhbButtons->addWidget(m_kill);

	qvb_control->addWidget(m_progress);
	qvb_control->addWidget(m_textInfo);
	qvb_control->addLayout(qhbButtons);

	qvbVideo->addWidget(qgb_control);


	int idx = 0;
	QTabWidget *tabWidgetMain = new QTabWidget();
	tabWidgetMain->insertTab(idx++, qgbVideo, QIcon(":/slicer/icons/VideoRGT.svg"), "Video");

	for (int i=0; i<4; i++)
	{
		QGroupBox *g1 = new QGroupBox;
		tabWidgetMain->insertTab(idx, g1, QIcon(QString("")), "");
		tabWidgetMain->setTabEnabled(idx++, false);
	}
	tabWidgetMain->insertTab(idx++, qgbProgramManager, QIcon(":/slicer/icons/earth.png"), "Manager");
	tabWidgetMain->setStyleSheet("QTabBar::tab { height: 40px; width: 100px; }");
	tabWidgetMain->setIconSize(QSize(40, 40));


//	QScrollArea *scrollArea = new QScrollArea;
//	scrollArea->setWidget(tabWidgetMain);
//	scrollArea->setWidgetResizable(true);

	// mainLayout00->addWidget(scrollArea);

	mainLayout00->addWidget(tabWidgetMain);
	pIhm2 = new Ihm2();

	QTimer *timer = new QTimer();
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
	connect(m_attributType, &QComboBox::currentIndexChanged, this, &VideoCreateWidget::trt_attributTypeChange);
	connect(m_processingType, &QComboBox::currentIndexChanged, this, &VideoCreateWidget::trt_processingTypeChange);
	connect(m_start, SIGNAL(clicked()), this, SLOT(trt_launch()));
	connect(m_kill, SIGNAL(clicked()), this, SLOT(trt_rgt_Kill()));
	// connect(m_buttonSeismicOpen, SIGNAL(clicked()), this, SLOT(trt_seismicOpen()));
	connect(m_buttonIsoOpen, SIGNAL(clicked()), this, SLOT(trt_isoOpen()));
	connect(m_projectmanager->m_projectManager, &ProjectManager::projectChanged, this, &VideoCreateWidget::projectChanged);
	connect(m_projectmanager->m_surveyManager, &SurveyManager::surveyChanged, this, &VideoCreateWidget::surveyChanged);

	connect(m_seismicFileSelectWidget, &FileSelectWidget::filenameChanged, this, &VideoCreateWidget::filenameChanged);

	// processwatcher_rawtoaviprocess = new ProcessWatcherWidget;
	// mainLayout00->addWidget(processwatcher_rawtoaviprocess);

	trt_attributTypeChange();
	trt_processingTypeChange();
	setMinimumSize(QSize(600,800));
	setWindowTitle0();
	// setFixedHeight(500);
	// resize(400, 500);
}


VideoCreateWidget::~VideoCreateWidget()
{

}


void VideoCreateWidget::setWindowTitle0() {
	QString title = "NextVision Processing - ";
	if ( m_projectmanager != nullptr )
	{
		title += m_projectmanager->getProjectName() + " - " + m_projectmanager->getSurveyName();
	}
	this->setWindowTitle(title);
}

void VideoCreateWidget::updateAttributList()
{
//	m_attributList->clear();
//	std::vector<QString> attibutName = getAttributNames(m_isoPath);
//	for (int i=0; i<attributName.size(); i++)
//		m_attributList->addItem(attributName[i]);
}

QString VideoCreateWidget::getSeismicPath()
{
	return m_seismicPath;
}


std::vector<QString> VideoCreateWidget::getAttributNames(QString path0)
{
	QString path = path0 + "/iso_00000";
	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.raw";
	dir.setNameFilters(filters);
	QFileInfoList list = dir.entryInfoList();
	std::vector<QString> out;
	for (int i=0; i<list.size(); i++)
		out.push_back(list[i].fileName());
	return out;
}


QString VideoCreateWidget::getSeimsicNameFromAttributName(QString name0, QString attribut, int w)
{
	QString name = "";
	int pos = name0.indexOf(QChar('_'));
	name = name0.right(name0.size() - pos-1);

	pos = name.lastIndexOf(".");
	name = name.left(name.size()-pos);


	return name;
}

QString VideoCreateWidget::getSeismicNameFromIsoPath(QString path0)
{
	QString attribut = m_attributType->currentText();
	QString path = path0 + "/iso_00000";
	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.raw";
	dir.setNameFilters(filters);
	QFileInfoList list = dir.entryInfoList();
	std::vector<QString> filename;
	for (int i=0; i<list.size(); i++)
	{
		if ( list[i].fileName().contains(attribut) )
		{
			filename.push_back(list[i].fileName());
		}
	}
	if ( filename.size() == 0 ) return "";

	for (int i=0; i<filename.size(); i++)
	{
		QString f = filename[i];
		QString f2 = getSeimsicNameFromAttributName(f, attribut, 64);

	}
	return "";
}

QString VideoCreateWidget::getAttributNameFromIsoPath(QString path0)
{
	QString attribut = m_attributType->currentText();
	QString path = path0 + "/iso_00000";
	QDir dir(path);
	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);
	QStringList filters;
	filters << "*.raw";
	dir.setNameFilters(filters);
	QFileInfoList list = dir.entryInfoList();
	std::vector<QString> filename;
	for (int i=0; i<list.size(); i++)
	{
		if ( list[i].fileName().contains(attribut) )
		{
			filename.push_back(list[i].fileName());
		}
	}
	if ( filename.size() == 0 ) return "";

	for (int i=0; i<filename.size(); i++)
	{
		QString f = filename[i];
		QString f2 = getSeimsicNameFromAttributName(f, attribut, 64);
	}
	return "";
}


void VideoCreateWidget::trt_attributTypeChange()
{
	QString attribut = m_attributType->currentText();
	m_spectrumCollapseParam->setVisible(false);
	m_meanCollapseParam->setVisible(false);
	m_gccCollapseParam->setVisible(false);
	if ( attribut == "spectrum" ) m_spectrumCollapseParam->setVisible(true);
	else if ( attribut == "gcc" ) m_gccCollapseParam->setVisible(true);
	else if ( attribut == "mean" ) m_meanCollapseParam->setVisible(true);
}

void VideoCreateWidget::trt_processingTypeChange()
{
	/*
	QString type = m_processingType->currentText();
	// m_qhbSeismic->setVisible(false);
	labelSeismic->setVisible(false);
	m_seismicLineEdit->setVisible(false);
	m_buttonSeismicOpen->setVisible(false);
	m_rgtFileSelectWidget->setVisible(false);
	labelIso->setVisible(false);
	m_buttonIsoOpen->setVisible(false);
	m_isoLineEdit->setVisible(false);
	m_attributList->setVisible(false);
	if ( type == m_comboMenuNew )
	{
		labelSeismic->setVisible(true);
		m_seismicLineEdit->setVisible(true);
		m_buttonSeismicOpen->setVisible(true);
		m_rgtFileSelectWidget->setVisible(true);
	}
	else if ( type == m_comboMenuExist )
	{
		labelIso->setVisible(true);
		m_buttonIsoOpen->setVisible(true);
		m_isoLineEdit->setVisible(true);
		m_attributList->setVisible(true);
	}
	*/
}

bool VideoCreateWidget::getPropertiesFromDatasetPath(QString filename, int* size, double* steps, double* origins)
{
	if ( filename.compare("") == 0 ) { for (int i=0; i<3;i++) size[i] = 0; return false; }
	inri::Xt xt(filename.toStdString().c_str());
	if (!xt.is_valid()) {
		std::cerr << "xt cube is not valid (" << filename.toStdString() << ")" << std::endl;
		return false;
	}
    size[0] = xt.nRecords();
    size[1] = xt.nSamples();
    size[2] = xt.nSlices();
    steps[0] = xt.stepRecords();
    steps[1] = xt.stepSamples();
    steps[2] = xt.stepSlices();
    if (origins!=nullptr) {
        origins[0] = xt.startRecord();
        origins[1] = xt.startSamples();
        origins[2] = xt.startSlice();
    }

    if ( (size[0] == 0 && size[1] == 0 && size[2] == 0) ||
            steps[0]==0 || steps[1]==0 || steps[2]==0) return false;
    return true;
}


QString VideoCreateWidget::getFFMPEGTime(double _time) {
	QString sign = "";
	if (_time<0) {
		sign = "-";
	}
	double time = std::fabs(_time);
	int timeInt = std::floor(time);
	double ms = time - timeInt;
	int s = timeInt % 60;
	int minutes = (timeInt / 60) % 60;
	int hours = (timeInt / 3600);

	return sign + formatTimeWithMinCharacters(hours) + ":" + formatTimeWithMinCharacters(minutes) + ":" +
			formatTimeWithMinCharacters(s+ms);
}

QString VideoCreateWidget::formatTimeWithMinCharacters(double time, int minCharNumber) {
	int zerosToAdd = minCharNumber;
	double absTime = std::fabs(time);
	double movingTime = absTime;
	while (movingTime>=1) {
		movingTime = movingTime / 10.0f;
		zerosToAdd--;
	}

	if (minCharNumber==zerosToAdd) {
		zerosToAdd = zerosToAdd - 1; // because QString::number will start by a zero
	}

	QStringList zeros;
	for (int i=0; i<zerosToAdd; i++) {
		zeros << "0";
	}
	int precision = 3;
	double decimals = std::floor(absTime) - absTime;
	if (qFuzzyIsNull(decimals)) {
		precision = 0;
	}

	return zeros.join("") + QString::number(time, 'f', precision);
}



std::tuple<double, bool, double> VideoCreateWidget::getAngleFromFilename(QString seismicFullName)
{
	bool topoExist;
	double angle = 0.0;
	bool swapV = true; // swap by default
	std::string surveyPath = SismageDBManager::survey3DPathFromDatasetPath(seismicFullName.toStdString());
	SmSurvey3D survey(surveyPath);

	SeismicSurvey baseSurvey(nullptr, "", survey.inlineDim(), survey.xlineDim(), QString::fromStdString(surveyPath));
	baseSurvey.setInlineXlineToXYTransfo(survey.inlineXlineToXYTransfo());
	baseSurvey.setIJToXYTransfo(survey.ijToXYTransfo());

	Seismic3DDataset seismic(&baseSurvey, "", nullptr, Seismic3DDataset::CUBE_TYPE::Seismic);
	seismic.loadFromXt(seismicFullName.toStdString());
	SmDataset3D d3d(seismicFullName.toStdString());
	seismic.setIJToInlineXlineTransfo(d3d.inlineXlineTransfo());
	seismic.setIJToInlineXlineTransfoForInline(
			d3d.inlineXlineTransfoForInline());
	seismic.setIJToInlineXlineTransfoForXline(
			d3d.inlineXlineTransfoForXline());
	seismic.setSampleTransformation(d3d.sampleTransfo());

	const Affine2DTransformation& transform = *(seismic.ijToXYTransfo());
	double oriX, oriY, endX, endY, vectX, vectY;
	transform.imageToWorld(0, 0, oriX, oriY);
	transform.imageToWorld(0, 1, endX, endY);

	vectX = endX - oriX;
	vectY = endY - oriY;
	double refVectX = 1;
	double refVectY = 0;
	double normVect = std::sqrt(std::pow(vectX, 2) + std::pow(vectY, 2));
	double normRefVect = std::sqrt(std::pow(refVectX, 2) + std::pow(refVectY, 2));

	angle = std::atan(vectY / vectX) - M_PI / 2;
	if (vectX<0) {
		angle += M_PI;
	}
	// std::acos((vectX * refVectX + vectY * refVectY) / ( normVect * normRefVect));

	double orthoX, orthoY;
	transform.imageToWorld(1, 0, orthoX, orthoY);
	double vectOrthoX = orthoX - oriX;
	double vectOrthoY = orthoY - oriY;

	double crossProduct = vectY * vectOrthoX - vectX * vectOrthoY;
	if (crossProduct>=0) {
		swapV = true;
	} else {
		swapV = false;
		angle = angle + M_PI;
	}

	double ratioWH = std::sqrt(std::pow(vectOrthoX, 2) + std::pow(vectOrthoY, 2)) / std::sqrt(std::pow(vectX, 2) + std::pow(vectY, 2));

	std::tuple<double, bool, double> angleAndSwapV(angle, swapV, ratioWH);
	return angleAndSwapV;
}


QSizeF VideoCreateWidget::newSizeFromSizeAndAngle(QSizeF oriSize, double angle) {
	QTransform transfo;
	transfo.rotate(angle * 180 / M_PI);
	QPointF p1(0, 0), p2(0, oriSize.height()), p3(oriSize.width(), 0), p4(oriSize.width(), oriSize.height());
	QList<QPointF> newPts;
	newPts << transfo.map(p1);
	newPts << transfo.map(p2);
	newPts << transfo.map(p3);
	newPts << transfo.map(p4);

	double xmin = newPts[0].x(), xmax=newPts[0].x(), ymin=newPts[0].y(), ymax=newPts[0].y();
	for (QPointF pt : newPts) {
		if (pt.x()<xmin) {
			xmin = pt.x();
		}
		if (pt.x()>xmax) {
			xmax = pt.x();
		}
		if (pt.y()<ymin) {
			ymin = pt.y();
		}
		if (pt.y()>ymax) {
			ymax = pt.y();
		}
	}
	return QSizeF(xmax - xmin, ymax - ymin);
}


std::pair<QString, QString> VideoCreateWidget::getPadCropFFMPEG(double wRatio, double hRatio)
{
	double epsilon = 1.0e-30;
	bool wEqual = std::fabs(wRatio - 1.0) < epsilon;
	bool wPad = wRatio>1 && !wEqual;
	bool wCrop = wRatio<1 && !wEqual;
	bool hEqual = std::fabs(hRatio - 1.0) < epsilon;
	bool hPad = hRatio>1 && !hEqual;
	bool hCrop = hRatio<1 && !hEqual;

	QString padCmd, cropCmd;
	if (wPad && hPad) {
		padCmd = "pad=" + QString::number(wRatio) + "*iw:" + QString::number(hRatio) + "*ih:(ow-iw)/2:(oh-ih)/2";
	} else if (wCrop && hCrop) {
		cropCmd = "crop=" + QString::number(wRatio) + "*iw:" + QString::number(hRatio) + "*ih:(iw-ow)/2:(ih-oh)/2";
	} else {
		if (wPad) {
			padCmd = "pad=" + QString::number(wRatio) + "*iw:ih:(ow-iw)/2:(oh-ih)/2";
		} else if (wCrop) {
			cropCmd = "crop=" + QString::number(wRatio) + "*iw:ih:(iw-ow)/2:(ih-oh)/2";
		}
		if (hPad) {
			padCmd = "pad=iw:" + QString::number(hRatio) + "*ih:(ow-iw)/2:(oh-ih)/2";
		} else if (hCrop) {
			cropCmd = "crop=iw:" + QString::number(hRatio) + "*ih:(iw-ow)/2:(ih-oh)/2";
		}
	}

	return std::pair<QString, QString>(padCmd, cropCmd);
}


void VideoCreateWidget::computeoptimalscale_rawToAvi() {
	if (getSeismicPath().isNull() || getSeismicPath().isEmpty() || m_rgtFileSelectWidget->getPath().isNull() ||
			 m_rgtFileSelectWidget->getPath().isEmpty()) {
		return;
	}

	// aviFilenameUpdate();

	int size[3];
	double steps[3];
	bool isValid = getPropertiesFromDatasetPath(getSeismicPath(), size, steps);
	if (!isValid) {
		return;
	}
	QString logoPath = GlobalConfig::getConfig().logoPath();
	std::tuple<double, bool, double> angleAndSwapV = getAngleFromFilename(getSeismicPath());
	double ratioWH = std::get<2>(angleAndSwapV);// * ((double) size[0]) / size[2];
	double scaleX=1.0, scaleY=1.0;
	if (ratioWH>1) {
		scaleX = ratioWH;
	} else if (ratioWH!=0.0) {
		scaleY = 1.0 / ratioWH;
	}

	double wOri = size[0] * scaleX;
	double hOri = size[2] * scaleY;
	QSizeF wAndH = newSizeFromSizeAndAngle(QSizeF(wOri, hOri), std::get<0>(angleAndSwapV));
	double w = wAndH.width();
	double h = wAndH.height();

//	double wRatio = w / wOri;
//	double hRatio = h / hOri;

	// total video width has 1/4 for section and 3/4 for map
	// remove from limit the padding 20 top, bottom, left, right
	// padding 10 for the separation
	int borderPadding = 20;
	int centerPadding = 10;
	double ratioWLimit = std::ceil(w * 4.0/3.0) / (7680.0 - borderPadding*2 - centerPadding); // test map width taking into account added section to the limit check
	double ratioHLimit = h / (4320.0 - borderPadding*2);

	double worstRatio = std::max(ratioWLimit, ratioHLimit);

	if (worstRatio>0.75) {
		m_aviParam->setVideoScale(0.75 / worstRatio);
	}
	else{
		m_aviParam->setVideoScale(1);
	}
}

QString VideoCreateWidget::formatColorForFFMPEG(const QColor& color) {
	QString qtStyledColor = color.name();
	// remove "#" and add "0x"
	QString ffmpegColor = qtStyledColor;
	ffmpegColor.remove(0, 1);
	ffmpegColor.prepend("0x");
	return ffmpegColor;
}


std::vector<std::string> VideoCreateWidget::getIsoPath(QString path)
{
	// todo
	// if ( m_attributDirectoty->getPath().isEmpty() ) return std::vector<std::string>();
	std::vector<std::string> out;
	int firstIso = m_aviParam->getFirstIso();
	int lastIso = m_aviParam->getLastIso();

	QDir *dir = new QDir(path);
	dir->setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	// dir->setSorting(QDir::Name);
	QFileInfoList list = dir->entryInfoList();
	for (int n=0; n<list.size(); n++)
	{
		QString path0 = list.at(n).fileName();
		if ( path0.contains("iso_") )
		{
			int iso = FreeHorizonQManager::getIsoFromDirectory(path0);
			fprintf(stderr, "iso %d\n", iso);

			QString tmp = path + "/" + path0 + "/";
			out.push_back(tmp.toStdString());
		}
	}
	return out;
}

/*
QString VideoCreateWidget::getAviPath()
{
	QString type = m_processingType->currentText();
	QString ImportExportPath = m_projectmanager->getImportExportPath();
	QString IJKPath = m_projectmanager->getIJKPath();

	std::string attributFilename="";
	if ( m_param.attribut == "spectrum" ) attributFilename = IsoHorizonManager::spectrumFilenameWithoutExtensionCreate(m_param.seismicName.toStdString(), m_param.spectrumWindowSize);
	else if ( m_param.attribut == "mean" ) attributFilename = IsoHorizonManager::meanFilenameWithoutExtensionCreate(m_param.seismicName.toStdString(), m_param.meanWindowSize);
	else if ( m_param.attribut == "gcc" ) attributFilename = IsoHorizonManager::gccFilenameWithoutExtensionCreate(m_param.seismicName.toStdString(), m_param.gccWindowSize, m_param.gccW,  m_param.gccShift);

	if ( type == m_comboMenuNew )
	{
		QString seismicName = m_seismicName;
		QString tinyName = m_rgtFileSelectWidget->getLineEditText();
		QString aviFullName = IJKPath + "/" + seismicName + "/cubeRgt2RGB/" + QString::fromStdString(attributFilename) +
				"_" + m_rgtFileSelectWidget->getLineEditText() + "_" + m_preffix->text() + ".avi";
		return aviFullName;
	}
	else
	{
		QString path0 = m_isoPath;
		QString seismicName = getSeismicNameFromIsoPath(path0);
		return "";
	}
}
*/

QString VideoCreateWidget::getAviPath()
{
	/*
	QString ImportExportPath = m_projectmanager->getImportExportPath();
	QString IJKPath = m_projectmanager->getIJKPath();
	// QString seismicName = m_seismicFileSelectWidget->getLineEditText();
	QString seismicName = m_seismicFileSelectWidget->getLineEditText(); // m_seismicLineEdit->text();
	QString tinyName = m_rgtFileSelectWidget->getLineEditText();
	QString aviFullName = IJKPath + "/" + seismicName + "/cubeRgt2RGB/" + tinyName + "_" + m_preffix->text() + ".avi";
	*/
	QString tinyName = m_rgtFileSelectWidget->getLineEditText();
	return m_projectmanager->getVideoPath() + tinyName + "_" + m_preffix->text() + ".avi";
}



void VideoCreateWidget::trt_launch()
{
	if ( pStatus != 0 ) return;

	QString type = m_processingType->currentText();
	if ( type == m_comboMenuNew )
	{
		QString seismicPath = getSeismicPath();
		if ( seismicPath.isEmpty() )
		{
			QMessageBox::warning(this, tr("Video create"), tr("seismic filename is empty")); return;
		}
		QString rgtPath = m_rgtFileSelectWidget->getPath();
		if ( rgtPath.isEmpty() )
		{
			QMessageBox::warning(this, tr("Video create"), tr("rgt filename is empty")); return;
		}

		inri::Xt xts((char*)seismicPath.toStdString().c_str());
		inri::Xt xtr((char*)rgtPath.toStdString().c_str());
		int dimxs = xts.nSamples();
		int dimys = xts.nRecords();
		int dimzs = xts.nSlices();
		int dimxr = xtr.nSamples();
		int dimyr = xtr.nRecords();
		int dimzr = xtr.nSlices();
		if ( dimxs != dimxr || dimys != dimyr || dimzs != dimzr )
		{
			QMessageBox::warning(this, tr("Video create"), tr("seismic and rgt size incompatible")); return;
		}
	}
	else
	{



	}

	bool onlyFirstImage = m_aviParam->getOnlyFirstImage();
	QString aviFullName = getAviPath();

	QDir d = QFileInfo(aviFullName).absoluteDir();
	QString path = d.absolutePath();

	// QDir dir(aviFullName);
	// QString path = dir.absolutePath();
	QDir dir2(path);
	dir2.mkpath(".");

	if (!onlyFirstImage && QFileInfo(aviFullName).exists()) {
		QStringList overWriteOptions;
		QString noStr = "no";
		overWriteOptions << noStr << "yes";
		bool ok;
		QString val = QInputDialog::getItem(this, "Overwrite ?", "Output file already exists, do you want to overwrite it ?",
				overWriteOptions, 0, false, &ok);

		// no -> abort function
		if (!ok || val.compare(noStr)==0) {
			return;
		}
	}

	if ( !paramInit() ) return;
	VideoCreateWidget_Thread *thread = new VideoCreateWidget_Thread(this);
	thread->start();
	// fprintf(stderr, "ok\n");
	// trt_run();
}

bool VideoCreateWidget::paramInit()
{
	eraseTmpFiles();
	bool onlyFirstImage = m_aviParam->getOnlyFirstImage();

	m_param.seismicName = m_seismicName;
	m_param.seismicPath = getSeismicPath();
	m_param.rgtName = m_rgtFileSelectWidget->getLineEditText();
	m_param.rgtPath = m_rgtFileSelectWidget->getPath();
	m_param.isoStep = m_isoStep->text().toInt();
	m_param.ImportExportPath = m_projectmanager->getImportExportPath();

	m_param.IJKPath = m_projectmanager->getIJKPath();
	m_param.horizonPath = m_projectmanager->getHorizonsPath();
	m_param.isoValPath = m_projectmanager->getHorizonsIsoValPath();
	m_param.rgtPath0 = m_param.isoValPath + m_rgtFileSelectWidget->getLineEditText() + "/";
	m_param.rgb2tmpfilename = getTmpFilename(m_param.rgtPath0, "tmp_", ".raw");
	m_param.outMainDirectory = m_param.rgtPath0;
	m_param.attribut = m_attributType->currentText();
	m_param.attributDirectory = m_param.rgtPath0;

	m_param.rgb1TmpFilename = getTmpFilename(m_param.rgtPath0, "rgb1_", ".rgb");

	m_param.prefix = m_param.seismicName;
	if (  m_param.attribut == "spectrum" )
	{
		m_param.prefix += "_w_" + m_spectrumWindowSize->text();
		m_param.dataType = SPECTRUM_NAME;
	}
	else if (m_param.attribut == "mean" )
	{
		m_param.prefix += "_w_" + m_meanWindowSize->text();
		m_param.dataType = MEAN_NAME;
	}
	else if (m_param.attribut == "gcc" )
	{
		m_param.dataType = GCC_NAME;
	}
	m_param.attributFilename = m_param.dataType + "_" + m_param.prefix + ".amp";

	// mkdirPathIfNotExist(m_param.ImportExportPath);
	// mkdirPathIfNotExist(m_param.IJKPath);
	// mkdirPathIfNotExist(m_param.horizonPath);
	// mkdirPathIfNotExist(m_param.isoValPath);
	// mkdirPathIfNotExist(m_param.rgtPath0);
	QDir d;
	d.mkpath(m_param.rgtPath0);

	m_param.spectrumWindowSize = m_spectrumWindowSize->text().toInt();
	m_param.meanWindowSize = m_meanWindowSize->text().toInt();
	m_param.gccWindowSize = m_gccWindowSize->text().toInt();

	/*
	GlobalConfig& config = GlobalConfig::getConfig();
	QDir tempDir = QDir(config.tempDirPath());
	m_param.outSectionVideoRawFile.setFileTemplate(tempDir.absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	// outSectionVideoRawFile.setFileTemplate(QDir("/data/PLI/NKDEEP/jacques/").absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	m_param.outSectionVideoRawFile.setAutoRemove(false);
	m_param.outSectionVideoRawFile.open();
	m_param.outSectionVideoRawFile.close();
	*/
	m_param.outSectionVideoRawFile = getTmpFilename(m_param.rgtPath0, "NextVision_section_", ".rgb");

	if (onlyFirstImage) {
		/*
		QTemporaryFile outJpgImageFile;
		outJpgImageFile.setFileTemplate(tempDir.absoluteFilePath("NextVision_display_XXXXXX.jpg"));
		outJpgImageFile.setAutoRemove(false);
		outJpgImageFile.open();
		outJpgImageFile.close();
		m_param.jpgTmpPath = outJpgImageFile.fileName();
		*/
		m_param.jpgTmpPath = getTmpFilename(m_param.rgtPath0, "NextVision_display_", ".rgb");
	}

	m_param.axisValue = m_aviParam->getSectionPosition();
	m_param.axisValueIndex = (m_param.axisValue - m_aviParam->getSectionPositionMinimum()) / m_aviParam->getSectionPositionSingleStep();

	QVariant dirVariant = m_aviParam->directionCurrentData();
	bool ok;
	int dirData = dirVariant.toInt(&ok);
	if (!ok) {
		return false;
	}

	if (dirData==0) {
		m_param.direction = SliceDirection::Inline;
		m_param.sectionName = "Inline ";
	} else {
		m_param.direction = SliceDirection::XLine;
		m_param.sectionName = "XLine ";
	}
	m_param.sectionName += QString::number(m_param.axisValue);

	m_vTmpPath.push_back(m_param.rgb1TmpFilename);
	m_vTmpPath.push_back(m_param.outSectionVideoRawFile);
	m_vTmpPath.push_back(m_param.rgb2tmpfilename);
	m_vTmpPath.push_back(m_param.jpgTmpPath);

	qDebug() << "seismicName: " << m_param.seismicName;
	qDebug() << "ImportExportPath: " << m_param.ImportExportPath;
	qDebug() << "IJKPath: " << m_param.IJKPath;
	qDebug() << "horizonPath: " << m_param.horizonPath;
	qDebug() << "isoValPath: " << m_param.isoValPath;
	qDebug() << "rgtPath: " << m_param.rgtPath0;
	qDebug() << "rgb2tmpfilename: " << m_param.rgb2tmpfilename;

	return true;
}

/*
void VideoCreateWidget::trt_run()
{
	computeoptimalscale_rawToAvi();
	int ret = 0;
	m_vTmpPath.push_back(m_param.rgb2tmpfilename);

	QString type = m_processingType->currentText();
	RgtToAttribut *p = new RgtToAttribut();
	p->setSeismicFilename(m_param.seismicPath.toStdString());
	p->setRgtFilename(m_param.rgtPath.toStdString());
	p->setOutRawFilename(m_param.rgb2tmpfilename.toStdString());
	p->setOutMainDirectory(m_param.rgtPath0.toStdString());
	if ( m_param.attribut == "spectrum" ) p->setAttributType(RgtToAttribut::TYPE::spectrum);
	else if ( m_param.attribut == "mean" ) p->setAttributType(RgtToAttribut::TYPE::mean);
	else if ( m_param.attribut == "gcc" ) p->setAttributType(RgtToAttribut::TYPE::gcc);
	p->setAttributData(RgtToAttribut::DATA::iso);
	p->setIsoStep(m_param.isoStep);
	if ( m_param.attribut == "spectrum" )
	{
		p->setWSize(m_param.spectrumWindowSize);
	}
	else if ( m_param.attribut == "mean" )
	{
		p->setWSize(m_param.meanWindowSize);
	}
	else if ( m_param.attribut == "gcc" )
	{
		p->setWSize(m_param.gccWindowSize);
	}
	p->setRgbSubDirectory("rgb2"); // ?
	p->setDataOutFormat(1);
	p->setMainPrefix(m_param.prefix.toStdString());
	// p->setRgbDirectory(paramInit.strOutRGB2Directory);
	// p->setIsoDirectory(paramInit.strOutHorizonsDirectory);
	p->setIhmMessage(pIhm2);

	pStatus = 1;
	ret = 0;
	ret = p->run();
	pStatus = 0;
	if ( ret == FAIL ) return;
	pStatus = 1;
	int size[3];
	double steps[3];
	bool isValid = getPropertiesFromDatasetPath(m_param.seismicPath, size, steps);
	std::vector<std::string> isoPath = getIsoPath(m_param.rgtPath0);
	ret = cuda_rgb2torgb1ByDirectories(isoPath,
			size[0], size[2],
			isoPath.size(), .0001f, 1.0f, (char*)m_param.rgb1TmpFilename.toStdString().c_str(), m_param.dataType.toStdString(),
			m_param.prefix.toStdString(), pIhm2);
	if ( ret == 0 ) return;

	sectionVideoCreate();
	// if ( rawToAviRunFfmegPreprocessing(false) ) pStatus = 2;
	pStatus = 2;
	// fileRemove(rgb2tmpfilename);
	// FREE(tab_gpu)
	// setStartStopStatus(STATUS_STOP);
	// if ( m_selectorWidget ) m_selectorWidget->RgbRawUpdateNames();
	// video
	// rawToAviRunFfmeg(false);
}
*/


bool VideoCreateWidget::attributAndIsochroneExist(std::vector<std::string> &attributpath, std::vector<std::string> &isochronepath)
{
	for (std::string str:attributpath)
	{
		QString filename = QString::fromStdString(str);
		QFileInfo fi(filename);
		if ( !fi.exists() ) return false;
	}
	for (std::string str:isochronepath)
	{
		QString filename = QString::fromStdString(str);
		QFileInfo fi(filename);
		if ( !fi.exists() ) return false;
	}
	return true;
}

void VideoCreateWidget::trt_run()
{
	computeoptimalscale_rawToAvi();
	int ret = 0;
	m_vTmpPath.push_back(m_param.rgb2tmpfilename);
	QString type = m_processingType->currentText();

	int isoStep = m_isoStep->text().toInt();
	int iso1 = m_aviParam->getFirstIso();
	int iso2 = m_aviParam->getLastIso();
	std::vector<int> isoVals = IsoHorizonManager::isoValCreate(iso1, iso2, isoStep);
	std::vector<std::string> isoDir0 = IsoHorizonManager::isoValStringCreate(iso1, iso2, isoStep);
	int N = isoDir0.size();
	std::vector<std::string> isoPath0;
	isoPath0.resize(N);
	for (int i=0; i<N; i++)
	{
		isoPath0[i] = m_param.rgtPath0.toStdString() + "/" + isoDir0[i] + "/";
	}

	std::vector<std::string> isoAttribut;
	std::vector<std::string> isoIsochrone;
	isoAttribut.resize(N);
	isoIsochrone.resize(N);
	for (int i=0; i<N; i++)
	{
		isoIsochrone[i] = m_param.rgtPath0.toStdString() + "/" + isoDir0[i] + "/isochrone.iso";
	}
	for (int i=0; i<N; i++)
	{
		isoAttribut[i] = m_param.rgtPath0.toStdString() + "/" + isoDir0[i] + "/" + m_param.attributFilename.toStdString();
	}

	for (std::string str:isoPath0) fprintf(stderr, "--> %s\n", str.c_str());
	for (std::string str:isoAttribut) fprintf(stderr, "--> %s\n", str.c_str());
	bool exist = attributAndIsochroneExist(isoAttribut, isoIsochrone);

	if ( !exist )
	{
		RgtToAttribut *p = new RgtToAttribut();
		p->setSeismicFilename(m_param.seismicPath.toStdString());
		p->setRgtFilename(m_param.rgtPath.toStdString());
		p->setOutRawFilename(m_param.rgb2tmpfilename.toStdString());
		p->setOutMainDirectory(m_param.rgtPath0.toStdString());
		if ( m_param.attribut == "spectrum" ) p->setAttributType(RgtToAttribut::TYPE::spectrum);
		else if ( m_param.attribut == "mean" ) p->setAttributType(RgtToAttribut::TYPE::mean);
		else if ( m_param.attribut == "gcc" ) p->setAttributType(RgtToAttribut::TYPE::gcc);
		p->setAttributData(RgtToAttribut::DATA::iso);
		p->setIsoStep(m_param.isoStep);
		if ( m_param.attribut == "spectrum" )
		{
			p->setWSize(m_param.spectrumWindowSize);
		}
		else if ( m_param.attribut == "mean" )
		{
			p->setWSize(m_param.meanWindowSize);
		}
		else if ( m_param.attribut == "gcc" )
		{
			p->setWSize(m_param.gccWindowSize);
		}
		p->setRgbSubDirectory("rgb2"); // ?
		p->setDataOutFormat(1);
		p->setMainPrefix(m_param.prefix.toStdString());
		// p->setRgbDirectory(paramInit.strOutRGB2Directory);
		// p->setIsoDirectory(paramInit.strOutHorizonsDirectory);
		p->setIhmMessage(pIhm2);

		p->setIsoVals(isoVals);
		p->setIsoPath(isoPath0);
		p->setIsochronePath(isoIsochrone);
		p->setAttributPath(isoAttribut);

		pStatus = 1;
		ret = 0;
		ret = p->run();
		pStatus = 0;
		if ( ret == FAIL ) return;
	}
	pStatus = 1;
	int size[3];
	double steps[3];
	bool isValid = getPropertiesFromDatasetPath(m_param.seismicPath, size, steps);
	// std::vector<std::string> isoPath = getIsoPath(m_param.rgtPath0);
	ret = cuda_rgb2torgb1ByDirectories(isoIsochrone, isoAttribut,
			size[0], size[2],
			isoIsochrone.size(), .0001f, 1.0f, (char*)m_param.rgb1TmpFilename.toStdString().c_str(), m_param.dataType.toStdString(),
			m_param.prefix.toStdString(), pIhm2);
	if ( ret == 0 ) return;

	sectionVideoCreate();
	// if ( rawToAviRunFfmegPreprocessing(false) ) pStatus = 2;
	pStatus = 2;
	// fileRemove(rgb2tmpfilename);
	// FREE(tab_gpu)
	// setStartStopStatus(STATUS_STOP);
	// if ( m_selectorWidget ) m_selectorWidget->RgbRawUpdateNames();
	// video
	// rawToAviRunFfmeg(false);
}

bool VideoCreateWidget::sectionVideoCreate()
{
	fprintf(stderr, "sections create start\n");
	// std::vector<std::string> isoPath = getIsoPath(m_param.rgtPath0);
	int isoStep = m_isoStep->text().toInt();
	int iso1 = m_aviParam->getFirstIso();
	int iso2 = m_aviParam->getLastIso();
	std::vector<std::string> isoDir0 = IsoHorizonManager::isoValStringCreate(iso1, iso2, isoStep);
	std::vector<std::string> isoPath0;
	isoPath0.resize(isoDir0.size());
	for (int i=0; i<isoDir0.size(); i++)
	{
		isoPath0[i] = m_param.rgtPath0.toStdString() + "/" + isoDir0[i] + "/";
	}
	QString inRgb2Path = m_param.attributDirectory;
	bool sectionValid = SectionToVideo::run2(isoPath0, m_param.seismicPath,
			m_param.seismicName,
			m_param.axisValueIndex, m_param.direction, inRgb2Path, m_param.outSectionVideoRawFile);
	fprintf(stderr, "sections create end\n");
	return sectionValid;
}

bool VideoCreateWidget::rawToAviRunFfmegPreprocessing(bool onlyFirstImage)
{
	GlobalConfig& config = GlobalConfig::getConfig();
	QDir tempDir = QDir(config.tempDirPath());

	// aviFilenameUpdate();
	QString tinyName = m_rgtFileSelectWidget->getLineEditText();
	QString aviFullName = getAviPath();
	int size[3];
	double steps[3];
	bool isValid = getPropertiesFromDatasetPath(getSeismicPath(), size, steps);
	if (!isValid) {
		return false;
	}
	int axisSize = size[0];
	if (m_param.direction == SliceDirection::XLine) {
		axisSize = size[2];
	}

	QString logoPath = GlobalConfig::getConfig().logoPath();
	std::tuple<double, bool, double> angleAndSwapV = getAngleFromFilename(getSeismicPath());
	double ratioWH = std::get<2>(angleAndSwapV);// * ((double) size[0]) / size[2];
	double scaleX=1.0, scaleY=1.0;
	if (ratioWH>1) {
		scaleX = ratioWH;
	} else if (ratioWH!=0.0) {
		scaleY = 1.0 / ratioWH;
	}

	double wOri = size[0] * scaleX;
	double hOri = size[2] * scaleY;
	QSizeF wAndH = newSizeFromSizeAndAngle(QSizeF(wOri, hOri), std::get<0>(angleAndSwapV));
	double w = wAndH.width();
	double h = wAndH.height();

	int sectionHeight = std::floor(h);
	int sectionWidth = std::round(w/3.0);

	double wRatio = w / wOri;
	double hRatio = h / hOri;

	QString inlineAngleName = QString::number(std::get<0>(angleAndSwapV));
	std::pair<QString, QString> padAndCrop = getPadCropFFMPEG(wRatio, hRatio);

	int tOrigin = m_aviParam->rgb1TOrigin;
	int tStep = m_aviParam->rgb1TStep; // expect positive int
	bool isReversed = m_aviParam->rgb1IsReversed;

	// std::size_t sz = isoPath.size(); //getFileSize(m_rawToAviRgb1FileSelectWidget->getPath().toStdString().c_str());
	// int numFrames = sz / (size[0] * size[2] * sizeof(char) * 3);
	// std::vector<std::string> isoPath = getIsoPath(m_param.rgtPath0);
	int isoStep = m_isoStep->text().toInt();
	int iso1 = m_aviParam->getFirstIso();
	int iso2 = m_aviParam->getLastIso();
	std::vector<int> isoVals = IsoHorizonManager::isoValCreate(iso1, iso2, isoStep);
	int numFrames = isoVals.size();

	// todo
	// suppress tOrigin & tStep;
	tOrigin = iso1;
	tStep = isoStep;

	int firstIndex, lastIndex, cutOrigin;
	QString tSign;
	if (isReversed) {
		tOrigin = tOrigin + (numFrames-1) * tStep;
		firstIndex = std::min(std::max((tOrigin - m_aviParam->firstIso) / tStep, 0), numFrames-1);
		lastIndex = std::min(std::max((tOrigin - m_aviParam->lastIso) / tStep, 0), numFrames-1);

		cutOrigin = tOrigin - std::min(firstIndex, lastIndex) * tStep;
		tSign = "-";
	} else {
		firstIndex = std::min(std::max((m_aviParam->firstIso - tOrigin) / tStep, 0), numFrames-1);
		lastIndex = std::min(std::max((m_aviParam->lastIso - tOrigin) / tStep, 0), numFrames-1);
		tSign = "+";

		cutOrigin = tOrigin + std::min(firstIndex, lastIndex) * tStep;
	}
	double firstTime = ((double) std::min(firstIndex, lastIndex)) / m_aviParam->framePerSecond;
	double lastTime = ((double) std::max(firstIndex, lastIndex) + 1) / m_aviParam->framePerSecond;
	double duration = lastTime - firstTime;
	QString beginTime = getFFMPEGTime(firstTime);

	QString drawTextExpr = "%{expr_int_format\\:" + QString::number(cutOrigin) + tSign +
			QString::number(tStep) + "*\\n\\:d\\:5}";

	QStringList options;
	options << "-y" << "-ss" << beginTime << "-t" << QString::number(duration);
	options << "-pixel_format" << "rgb24" << "-vcodec" << "rawvideo" << "-video_size";
	options << (QString::number(size[0]) + "x" + QString::number(size[2])) << "-framerate";
	options << QString::number(m_aviParam->framePerSecond) << "-i" << m_param.rgb1TmpFilename; // tmpFilename; // rgb1Filename.fileName(); // m_rawToAviRgb1FileSelectWidget->getPath();
	options << "-pixel_format" << "rgb24" << "-vcodec" << "rawvideo" << "-video_size";
	options << (QString::number(axisSize) + "x" + QString::number(size[1])) << "-framerate";
	options << QString::number(m_aviParam->framePerSecond) << "-i" << m_param.outSectionVideoRawFile;
	if (!onlyFirstImage) {
		options << "-c:v" << "libx264" << "-crf" << "23";
	}
	options << "-filter_complex";
	QString filterOption = "movie=" + logoPath + " [logo]; [logo] scale=100:-1 [logoBis]; [0] ";
	if (!std::get<1>(angleAndSwapV)) {
		filterOption += "vflip,";
	}
	if (!padAndCrop.first.isNull() && !padAndCrop.first.isEmpty()) {
		filterOption += padAndCrop.first + ",";
	}

	double videoScale = m_aviParam->getVideoScale();
	filterOption += "scale=iw*" + QString::number(videoScale * scaleX) + ":ih*" + QString::number(videoScale * scaleY) +
			",rotate=" + inlineAngleName;
	if (!padAndCrop.second.isNull() && !padAndCrop.second.isEmpty()) {
		filterOption += "," + padAndCrop.second;
	}
	filterOption += ",scale='iw-mod(iw,2)':'ih-mod(ih,2)'";
	filterOption += ",vflip";
	filterOption += ",drawtext=fontfile=/usr/share/fonts/gnu-free/FreeSans.ttf:text='RGT time \\: " + drawTextExpr + "':fontsize="+QString::number(m_aviParam->textSize)+":fontcolor="+formatColorForFFMPEG(m_aviParam->textColor)+":x=w/2-tw/2:y=h-th-10";; //":x=h/2-th/2:y=h-th-10";
	filterOption += " [tmp]; [tmp][logoBis] overlay=(main_w-overlay_w-10):(main_h-overlay_h-10) [last]; ";
	filterOption += "[1] scale=" + QString::number(std::floor(videoScale*sectionWidth)-50) + ":" + QString::number(std::floor(videoScale*sectionHeight));
	filterOption += ",drawtext=fontfile=/usr/share/fonts/gnu-free/FreeSans.ttf:text='" + m_param.sectionName + "':fontsize="+QString::number(m_aviParam->textSize)+":fontcolor="+formatColorForFFMPEG(m_aviParam->textColor)+":x=w/2-tw/2:y=h-th-10 [last1]; "; // ":x=h/2-th/2:y=h-th-10 [last1]; ";
	filterOption += "[last] pad=iw+" + QString::number(std::floor(videoScale*sectionWidth)) + ":ih+40:ow-iw-20:20 [last_padded]; [last_padded][last1] overlay=20:20";

	options << filterOption;
	if (onlyFirstImage) {
		options << "-frames:v" << "1" << "-f" << "image2";
		options << m_param.jpgTmpPath;
	} else {
		options << aviFullName;
	}

	//fprintf(stderr, "command: %s\n", cmd.toStdString().c_str());
	qDebug() << "ffmpeg command : " << "ffmpeg " << options.join(" ");
	m_options = options;
	return true;
}


void VideoCreateWidget::trt_rgt_Kill()
{
	if ( pStatus == 0 ) return;

	QMessageBox *msgBox = new QMessageBox(parentWidget());
	msgBox->setText("warning");
	msgBox->setInformativeText("Do you really want to abort the processing ?");
	msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No );
	int ret = msgBox->exec();
	if ( ret == QMessageBox::Yes )
	{
		// ihm_set_trt(IHM_TRT_RGT_GRAPH_STOP);
		if ( pIhm2 )
		{
			pIhm2->setMasterMessage("stop", 0, 0, 1);
		}
	}
	if (m_process!=nullptr) {
		m_process->kill();
	}
}

void VideoCreateWidget::trt_seismicOpen()
{
	std::vector<QString >seismicNames = m_projectmanager->getSeismicAllNames();
	std::vector<QString> seismicPath = m_projectmanager->getSeismicAllPath();

	FileSelectorDialog dialog(&seismicNames, "Select file name");
	dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::seismic);
	int code = dialog.exec();
	if (code==QDialog::Accepted)
	{
		int selectedIdx = dialog.getSelectedIndex();
		if (selectedIdx>=0 && selectedIdx<seismicNames.size())
		{
			m_seismicName = seismicNames[selectedIdx];
			m_seismicPath = seismicPath[selectedIdx];
			m_seismicLineEdit->setText(m_seismicName);
			// updateLabelDimensions(path);
			// fprintf(stderr, "%d %s\n", selectedIdx, dialog.getSelectedString().toStdString().c_str());
			m_aviParam->initSectionSlider();
		}
	}
}

void VideoCreateWidget::filenameChanged()
{
	m_seismicName = m_seismicFileSelectWidget->getLineEditText();
	m_seismicPath = m_seismicFileSelectWidget->getPath();
	// m_seismicLineEdit->setText(m_seismicName);
	// updateLabelDimensions(path);
	// fprintf(stderr, "%d %s\n", selectedIdx, dialog.getSelectedString().toStdString().c_str());
	m_aviParam->initSectionSlider();
}


void VideoCreateWidget::trt_isoOpen()
{
	std::vector<QString >seismicPath = m_projectmanager->getHorizonIsoValueListPath();
	std::vector<QString> seismicNames = m_projectmanager->getHorizonIsoValueListName();
	FileSelectorDialog dialog(&seismicNames, "Select file name");
	dialog.setMainSearchType(FileSelectorDialog::MAIN_SEARCH_LABEL::all);
	int code = dialog.exec();
	if (code==QDialog::Accepted)
	{
		int selectedIdx = dialog.getSelectedIndex();
		if (selectedIdx>=0 && selectedIdx<seismicNames.size())
		{
			m_isoName = seismicNames[selectedIdx];
			m_isoPath = seismicPath[selectedIdx];
			m_isoLineEdit->setText(m_isoName);
		}
	}

}

void VideoCreateWidget::processingDisplay()
{
	if ( pStatus == 0 )
	{
		m_progress->setValue(0); m_progress->setFormat("");
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


void VideoCreateWidget::showTime()
{
	if ( pIhm2 == nullptr ) return;
	processingDisplay();
	if ( pStatus == 0 ) return;
	if ( pStatus == 2 )
	{
		rawToAviRunFfmegPreprocessing(false);
		ffmpegProcessRun();
		pStatus = 1;
	}

	if ( pIhm2->isSlaveMessage() )
	{
		Ihm2Message mess = pIhm2->getSlaveMessage();
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
	std::vector<std::string> mess = pIhm2->getSlaveInfoMessage();
	for (int n=0; n<mess.size(); n++)
	{
		m_textInfo->appendPlainText(QString(mess[n].c_str()));
	}
}


void VideoCreateWidget::eraseTmpFiles()
{
	for (QString path:m_vTmpPath)
	{
		QFile::remove(path);
	}
}

void VideoCreateWidget::processReset()
{
	if (m_process!=nullptr) {
		// disconnect process signals
		QProcess* processPtr = m_process.get();
		disconnect(processPtr, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &VideoCreateWidget::processFinished);
		disconnect(processPtr, &QProcess::errorOccurred, this, &VideoCreateWidget::errorOccured);
		disconnect(processPtr, &QProcess::readyRead, this, &VideoCreateWidget::readyRead);
		m_process.reset(nullptr);
	}
}

void VideoCreateWidget::processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
	// emit processEnded(m_currentProcessId, exitCode, exitStatus);
	eraseTmpFiles();
	pStatus = 0;
	qDebug() << "process finish";
}

void VideoCreateWidget::errorOccured(QProcess::ProcessError error) {
	// emit processGotError(m_currentProcessId, error);
	qDebug() << "error occured: " << error;
}

void VideoCreateWidget::readyRead() {
	// emit processGotError(m_currentProcessId, error);
	if (m_process==nullptr) {
		return;
	}
	QByteArray data = m_process->readAll();
	QString newData(data);
	qDebug() << newData;
	m_textInfo->appendPlainText(newData);
}

void VideoCreateWidget::ffmpegProcessRun()
{
	m_process.reset(new QProcess);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	m_process->setReadChannel(QProcess::StandardOutput);
	m_process->setProgram("ffmpeg");
	m_process->setArguments(m_options);
	m_process->setWorkingDirectory(QString());
	m_process->setProcessEnvironment(QProcessEnvironment::systemEnvironment());
	// QProcessEnvironment env = m_process->processEnvironment();
	//	qDebug() << env.contains("PATH");
	//	qDebug() << env.value("PATH");

	QProcess* processPtr = m_process.get();
	connect(processPtr, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &VideoCreateWidget::processFinished);
	connect(processPtr, &QProcess::errorOccurred, this, &VideoCreateWidget::errorOccured);
	connect(processPtr, &QProcess::readyRead, this, &VideoCreateWidget::readyRead);

	m_process->start();
	m_process->waitForStarted();
	qDebug() << m_process->state();
	pStatus = 1;
}



void VideoCreateWidget::projectChanged()
{
	setWindowTitle0();
}

void VideoCreateWidget::surveyChanged()
{
	setWindowTitle0();
}



VideoCreateWidget_Thread::VideoCreateWidget_Thread(VideoCreateWidget *p)
 {
     this->pp = p;
 }

 void VideoCreateWidget_Thread::run()
 {
	 fprintf(stderr, "thread start\n");
	 pp->trt_run();
 }

