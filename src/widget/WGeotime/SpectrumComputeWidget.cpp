/*
 *
 *
 *  Created on: 11 September 2020
 *      Author: l1000501
 */


#include <QTableView>
#include <QHeaderView>
#include <QStandardItemModel>
#include <QPushButton>
#include <QRadioButton>
#include <QGroupBox>
#include <QLabel>
#include <QPainter>
#include <QChart>
#include <QLineEdit>
#include <QToolButton>
#include <QLineSeries>
#include <QScatterSeries>
#include <QtCharts>
#include <QRandomGenerator>
#include <QTimer>
#include <QList>

#include <QVBoxLayout>

#include <dialog/validator/OutlinedQLineEdit.h>
#include <dialog/validator/SimpleDoubleValidator.h>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <sys/sysinfo.h>


#include <vector>
#include <math.h>
#include <cmath>
#include <iostream>

#include <fileio2.h>
#include <ihm.h>
#include <cuda_rgt2rgb.h>
#include <cuda_rgb2torgb1.h>
#include <rgb1toxt.h>

#include <Xt.h>

// #include "FileConvertionXTCWT.h"
#include "SpectrumComputeWidget.h"
#include "sismagedbmanager.h"
#include "smsurvey3D.h"
#include "affine2dtransformation.h"
#include "globalconfig.h"
#include "processwatcherwidget.h"
#include "seismic3ddataset.h"
#include "seismicsurvey.h"
#include "smdataset3D.h"
#include <rgtToGCC16Widget.h>
#include <rgb2ToXtWidget.h>
#include "MeanSeismicSpectrumWidget.h"
#include "SectionToVideo.h"
#include "globalconfig.h"



// #define __LINUX__
#define EXPORT_LIB __attribute__((visibility("default")))

using namespace std;




SpectrumComputeWidget::SpectrumComputeWidget(QWidget* parent) :
		QWidget(parent) {

	setAttribute(Qt::WA_DeleteOnClose);
	setWindowTitle("Spectrum");

	GLOBAL_RUN_TYPE = 0;
	GLOBAL_RUNRGB2TORGB1_TYPE = 0;
	GLOBAL_RUNRGB1TOXT_TYPE = 0;
	m_functionType = 0;
	thread = nullptr;
	timer = nullptr;

	QHBoxLayout * mainLayout00 = new QHBoxLayout(this);

	QGroupBox *qgbMainLayout01 = new QGroupBox;
	QVBoxLayout * mainLayout01 = new QVBoxLayout(qgbMainLayout01);

	QGroupBox *qgbProgramManager = new QGroupBox;
	QVBoxLayout * mainLayout02 = new QVBoxLayout(qgbProgramManager);
	// m_selectorWidget = new GeotimeProjectManagerWidget(this);
	m_projectManager = new ProjectManagerWidget();

	/*
	// m_selectorWidget->removeTabHorizons();
	m_selectorWidget->removeTabCulturals();
	m_selectorWidget->removeTabWells();
	m_selectorWidget->removeTabNeurons();
	m_selectorWidget->removeTabPicks();
	*/

	QPushButton *pushbutton_sessionload = new QPushButton("load session");
	QPushButton *pushbutton_sessionsave = new QPushButton("save session");
	// mainLayout02->addWidget(pushbutton_sessionload);
	mainLayout02->addWidget(m_projectManager);
	// mainLayout02->addWidget(pushbutton_sessionsave);

	/*
	QGroupBox *qgbProgramManager = new QGroupBox;
	QVBoxLayout *qvb_programmanager = new QVBoxLayout(qgbProgramManager);
	m_projectManager = new ProjectManagerWidget;
	QPushButton *qpb_loadsession = new QPushButton("load session");
	qvb_programmanager->addWidget(m_projectManager);
	*/

	QGroupBox *qgb_rgt2rgb = new QGroupBox();
	QGroupBox *qgb_rgb2torgb1 = new QGroupBox();
	QGroupBox *qgb_rawtoavi = new QGroupBox();
	QGroupBox *qgb_rgb1toxt = new QGroupBox();
	QGroupBox *qgb_aviview = new QGroupBox();

	QVBoxLayout *layout2 = new QVBoxLayout(qgb_rgt2rgb);
	QVBoxLayout *layout3 = new QVBoxLayout(qgb_rgb2torgb1);
	QVBoxLayout *layout4 = new QVBoxLayout(qgb_rawtoavi);
	QVBoxLayout *layout5 = new QVBoxLayout(qgb_rgb1toxt);
	QVBoxLayout *layout6 = new QVBoxLayout(qgb_aviview);

	label_rgb2Name = new QLabel("..");

	/*
	QHBoxLayout *layout21 = new QHBoxLayout;
	QLabel *labelSeismic = new QLabel("seismic filename");
	lineedit_seismicFilename = new QLineEdit;
	lineedit_seismicFilename->setReadOnly(true);
	QPushButton *pushbutton_seismicOpen = new QPushButton("...");
	layout21->addWidget(labelSeismic);
	layout21->addWidget(lineedit_seismicFilename);
	layout21->addWidget(pushbutton_seismicOpen);
	*/
	m_seismicFileSelectWidget = new FileSelectWidget();
	m_seismicFileSelectWidget->setProjectManager(m_projectManager);
	m_seismicFileSelectWidget->setLabelText("seismic filename");
	m_seismicFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Seismic);
	m_seismicFileSelectWidget->setLabelDimensionVisible(false);

/*
	QHBoxLayout *layout22 = new QHBoxLayout;
	QLabel *labelRgt = new QLabel("rgt filename");
	lineedit_rgtFilename = new QLineEdit;
	lineedit_rgtFilename->setReadOnly(true);
	QPushButton *pushbutton_rgtOpen = new QPushButton("...");
	layout22->addWidget(labelRgt);
	layout22->addWidget(lineedit_rgtFilename);
	layout22->addWidget(pushbutton_rgtOpen);
	*/

	m_rgtFileSelectWidget = new FileSelectWidget();
	m_rgtFileSelectWidget->setProjectManager(m_projectManager);
	m_rgtFileSelectWidget->setLabelText("rgt filename");
	m_rgtFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Rgt);
	m_rgtFileSelectWidget->setLabelDimensionVisible(false);


	QHBoxLayout *layout22_1 = new QHBoxLayout;
	QLabel *labelRgt2Prefix = new QLabel("rgb2 prefix");
	lineedit_rgb2prefix = new QLineEdit("spectrum");
	layout22_1->addWidget(labelRgt2Prefix);
	layout22_1->addWidget(lineedit_rgb2prefix);

	QHBoxLayout *layout23 = new QHBoxLayout;
	QLabel *label_wsize = new QLabel("window size:");
	lineedit_wsize = new QLineEdit("64");

	combobox_horizonchoice = new QComboBox;
	combobox_horizonchoice->addItem("Isovalue");
	combobox_horizonchoice->addItem("Horizons");
	combobox_horizonchoice->setMaximumWidth(200);


	QLabel *labelF1 = new QLabel("f1:");
	lineedit_f1 = new QLineEdit("2");
	QLabel *labelF2 = new QLabel("f2:");
	lineedit_f2 = new QLineEdit("4");
	QLabel *labelF3 = new QLabel("f3:");
	lineedit_f3 = new QLineEdit("6");
	layout23->addWidget(label_wsize);
	layout23->addWidget(lineedit_wsize);
	// layout23->addWidget(label_isostep);
	// layout23->addWidget(lineedit_isostep);
	// layout23->addWidget(labelF1);
	// layout23->addWidget(lineedit_f1);
	// layout23->addWidget(labelF2);
	// layout23->addWidget(lineedit_f2);
	// layout23->addWidget(labelF3);
	// layout23->addWidget(lineedit_f3);




	QHBoxLayout *layout23_2 = new QHBoxLayout;
	label_isostep = new QLabel("iso step:");
	lineedit_isostep = new QLineEdit("25");
	label_layerNumber = new QLabel("Layer number:");
	lineedit_layer_number = new QLineEdit("10");
	layout23_2->addWidget(combobox_horizonchoice);
	layout23_2->addWidget(label_isostep);
	layout23_2->addWidget(lineedit_isostep);
	layout23_2->addWidget(label_layerNumber);
	layout23_2->addWidget(lineedit_layer_number);

	horizonWidget = new QWidget();
	QHBoxLayout *layout23_1 = new QHBoxLayout(horizonWidget);
	QLabel *label_horizons = new QLabel("horizons");
	listwidget_horizons = new QListWidget();
	listwidget_horizons->setMaximumHeight(50);
	QVBoxLayout *layout23_1_1 = new QVBoxLayout;
	QPushButton *pushbutton_horizonadd = new QPushButton("add");
	QPushButton *pushbutton_horizonsub = new QPushButton("suppr");
	layout23_1_1->addWidget(pushbutton_horizonadd);
	layout23_1_1->addWidget(pushbutton_horizonsub);
	layout23_1->addWidget(label_horizons);
	layout23_1->addWidget(listwidget_horizons);
	layout23_1->addLayout(layout23_1_1);


	int N = 5;
	tableWidget_freq = new QTableWidget(N, 2);
	tableWidget_freq->setHorizontalHeaderItem(0, new QTableWidgetItem("f2"));
	tableWidget_freq->setHorizontalHeaderItem(1, new QTableWidgetItem("iso value"));
	tableWidget_freq->setItem(0, 0, new QTableWidgetItem(QString::number(10))); tableWidget_freq->setItem(0, 1, new QTableWidgetItem(QString::number(0)));
	tableWidget_freq->setItem(1, 0, new QTableWidgetItem(QString::number(4))); tableWidget_freq->setItem(1, 1, new QTableWidgetItem(QString::number(32000)));
	// tableWidget_freq->setItem(2, 0, new QTableWidgetItem(QString::number(8))); tableWidget_freq->setItem(2, 1, new QTableWidgetItem(QString::number(32000)));

	qpb_progress = new QProgressBar();
	// qpb_progress->setGeometry(5, 45, 240, 20);
	qpb_progress->setMinimum(0);
	qpb_progress->setMaximum(100);
	qpb_progress->setValue(0);
	// qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,0,0)}");
	qpb_progress->setTextVisible(true);
	qpb_progress->setValue(0);
	qpb_progress->setFormat("");

	pushbutton_startstop = new QPushButton("Start");
	layout2->addWidget(label_rgb2Name);
	// layout2->addLayout(layout21);
	layout2->addWidget(m_rgtFileSelectWidget);
	layout2->addLayout(layout22_1);
	layout2->addLayout(layout23);
	// layout2->addWidget(combobox_horizonchoice);
	layout2->addLayout(layout23_2);
	layout2->addWidget(horizonWidget); // layout2->addLayout(layout23_1);

// 	layout2->addWidget(tableWidget_freq);

	layout2->addWidget(qpb_progress);
	layout2->addWidget(pushbutton_startstop);


	label_rgb1Name = new QLabel("..");
	label_rgb1Name->setWordWrap(true);

	/*
	QHBoxLayout *layout31 = new QHBoxLayout;
	QLabel *labelRgb2Filename = new QLabel("RGB2 filename");
	lineedit_rgb2Filename = new QLineEdit;
	lineedit_rgb2Filename->setReadOnly(true);
	QPushButton *pushbutton_rgb2FilenameOpen = new QPushButton("...");
	layout31->addWidget(labelRgb2Filename);
	layout31->addWidget(lineedit_rgb2Filename);
	layout31->addWidget(pushbutton_rgb2FilenameOpen);
	*/
	m_rb2FileSelectWidget = new FileSelectWidget();
	m_rb2FileSelectWidget->setProjectManager(m_projectManager);
	m_rb2FileSelectWidget->setLabelText("rgb2 filename");
	m_rb2FileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::rgtCubeToAttribut);
	m_rb2FileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Raw);
	m_rb2FileSelectWidget->setLabelDimensionVisible(false);



	QHBoxLayout *layout31_1 = new QHBoxLayout;
	QLabel *labelRgt2toRgb1Prefix = new QLabel("rgb1 prefix");
	lineedit_rgb1prefix = new QLineEdit("");
	layout31_1->addWidget(labelRgt2toRgb1Prefix);
	layout31_1->addWidget(lineedit_rgb1prefix);

	QHBoxLayout *layout32 = new QHBoxLayout;
	QLabel *label_ratio = new QLabel("ratio:");
	lineedit_ratio = new QLineEdit(".0001"); // lineedit_ratio->setMaximumWidth(150);
	QLabel *label_alpha = new QLabel("alpha:");
	lineedit_alpha = new QLineEdit("1.0"); // lineedit_alpha->setMaximumWidth(150);
	layout32->addWidget(label_ratio);
	layout32->addWidget(lineedit_ratio);
	layout32->addWidget(label_alpha);
	layout32->addWidget(lineedit_alpha);

	qpb_progress_rgb2torgb1 = new QProgressBar;
	qpb_progress_rgb2torgb1->setMinimum(0);
	qpb_progress_rgb2torgb1->setMaximum(100);
	qpb_progress_rgb2torgb1->setValue(0);
	qpb_progress_rgb2torgb1->setTextVisible(true);
	qpb_progress_rgb2torgb1->setValue(0);
	qpb_progress_rgb2torgb1->setFormat("");

	pushbutton_rgtb2torgb1StartStop = new QPushButton("Start");

	layout3->addWidget(label_rgb1Name);
	layout3->addWidget(m_rb2FileSelectWidget);
	layout3->addLayout(layout31_1);
	layout3->addLayout(layout32);
	layout3->addWidget(qpb_progress_rgb2torgb1);
	layout3->addWidget(pushbutton_rgtb2torgb1StartStop);


	/*
	QHBoxLayout *layout41 = new QHBoxLayout;
	QLabel *labelrawtoaviFilename = new QLabel("Rgb1 filename:");
	lineedit_rawtoaviFilename = new QLineEdit;
	lineedit_rawtoaviFilename->setReadOnly(true);

	QPushButton *pushbutton_rawtoaviFilenameOpen = new QPushButton("...");
	layout41->addWidget(labelrawtoaviFilename);
	layout41->addWidget(lineedit_rawtoaviFilename);
	layout41->addWidget(pushbutton_rawtoaviFilenameOpen);

	QHBoxLayout *layout41_1 = new QHBoxLayout;
	QLabel *labelrgb2toaviFilename = new QLabel("Rgb2 filename:");
	lineedit_rgb2toaviFilename = new QLineEdit;
	lineedit_rgb2toaviFilename->setReadOnly(true);

	QPushButton *pushbutton_rgb2toaviFilenameOpen = new QPushButton("...");
	layout41_1->addWidget(labelrgb2toaviFilename);
	layout41_1->addWidget(lineedit_rgb2toaviFilename);
	layout41_1->addWidget(pushbutton_rgb2toaviFilenameOpen);
	*/


	m_rawToAviRgb1FileSelectWidget = new FileSelectWidget();
	m_rawToAviRgb1FileSelectWidget->setProjectManager(m_projectManager);
	m_rawToAviRgb1FileSelectWidget->setLabelText("rgb1 filename");
	m_rawToAviRgb1FileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::rgtCubeToAttribut);
	m_rawToAviRgb1FileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Raw);
	m_rawToAviRgb1FileSelectWidget->setLabelDimensionVisible(false);

	m_rawToAviRgb2FileSelectWidget = new FileSelectWidget();
	m_rawToAviRgb2FileSelectWidget->setProjectManager(m_projectManager);
	m_rawToAviRgb2FileSelectWidget->setLabelText("rgb2 filename");
	m_rawToAviRgb2FileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::rgtCubeToAttribut);
	m_rawToAviRgb2FileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Raw);
	m_rawToAviRgb2FileSelectWidget->setLabelDimensionVisible(false);



	QHBoxLayout *layout42 = new QHBoxLayout;
	QLabel *labelaviPrefix = new QLabel("video prefix:");
	lineedit_aviPrefix = new QLineEdit("video");
	layout42->addWidget(labelaviPrefix);
	layout42->addWidget(lineedit_aviPrefix);

	QHBoxLayout* layout43 = new QHBoxLayout;
	QLabel* labelvideoScale = new QLabel("Scale factor :");
	spinboxvideoScale = new QDoubleSpinBox;
	spinboxvideoScale->setMinimum(std::numeric_limits<double>::min());
	spinboxvideoScale->setMaximum(std::numeric_limits<double>::max());
	spinboxvideoScale->setValue(videoScale);

	layout43->addWidget(labelvideoScale, 0);
	layout43->addWidget(spinboxvideoScale, 1);

	QHBoxLayout* layout44 = new QHBoxLayout();
	QLabel* labelTOrigin = new QLabel("T origin :");
	spinboxRgb1TOrigin = new QSpinBox;
	spinboxRgb1TOrigin->setMinimum(0);
	spinboxRgb1TOrigin->setMaximum(std::numeric_limits<int>::max());
	spinboxRgb1TOrigin->setValue(rgb1TOrigin);

	layout44->addWidget(labelTOrigin, 0);
	layout44->addWidget(spinboxRgb1TOrigin, 1);

	QHBoxLayout* layout45 = new QHBoxLayout();
	QLabel* labelTStep = new QLabel("T Step :");
	spinboxRgb1TStep = new QSpinBox;
	spinboxRgb1TStep->setMinimum(1);
	spinboxRgb1TStep->setMaximum(std::numeric_limits<int>::max());
	spinboxRgb1TStep->setValue(rgb1TStep);

	layout45->addWidget(labelTStep, 0);
	layout45->addWidget(spinboxRgb1TStep, 1);

	QHBoxLayout* layout46 = new QHBoxLayout();
	QLabel* labelIsReversed = new QLabel("Is time reversed :");
	checkboxRgb1IsReversed = new QCheckBox();
	checkboxRgb1IsReversed->setCheckState((rgb1IsReversed) ? Qt::Checked : Qt::Unchecked);

	layout46->addWidget(labelIsReversed, 0);
	layout46->addWidget(checkboxRgb1IsReversed, 1);

	QHBoxLayout* layout47 = new QHBoxLayout();
	QLabel* labelFps = new QLabel("FPS :");
	spinboxFps = new QSpinBox;
	spinboxFps->setMinimum(1);
	spinboxFps->setMaximum(std::numeric_limits<int>::max());
	spinboxFps->setValue(framePerSecond);

	layout47->addWidget(labelFps, 0);
	layout47->addWidget(spinboxFps, 1);

	QHBoxLayout* layout48 = new QHBoxLayout();
	QLabel* labelFirstIso = new QLabel("First Iso :");
	spinboxFirstIso = new QSpinBox;
	spinboxFirstIso->setMinimum(std::numeric_limits<int>::min());
	spinboxFirstIso->setMaximum(std::numeric_limits<int>::max());
	spinboxFirstIso->setValue(firstIso);

	layout48->addWidget(labelFirstIso, 0);
	layout48->addWidget(spinboxFirstIso, 1);

	QHBoxLayout* layout49 = new QHBoxLayout();
	QLabel* labelLastIso = new QLabel("Last Iso :");
	spinboxLastIso = new QSpinBox;
	spinboxLastIso->setMinimum(std::numeric_limits<int>::min());
	spinboxLastIso->setMaximum(std::numeric_limits<int>::max());
	spinboxLastIso->setValue(lastIso);

	layout49->addWidget(labelLastIso, 0);
	layout49->addWidget(spinboxLastIso, 1);

	QHBoxLayout* layout410 = new QHBoxLayout();

	directionComboBox = new QComboBox();
	directionComboBox->addItem("Inline", 0);
	directionComboBox->addItem("Xline", 1);
	sectionPositionSpinBox = new QSpinBox();
	sectionPositionSpinBox->setMinimum(0);
	sectionPositionSpinBox->setMaximum(1);
	sectionPositionSpinBox->setValue(0);
	sectionPositionSlider = new QSlider(Qt::Horizontal);
	sectionPositionSlider->setMinimum(0);
	sectionPositionSlider->setMaximum(1);
	sectionPositionSlider->setValue(0);

	layout410->addWidget(directionComboBox);
	layout410->addWidget(sectionPositionSpinBox);
	layout410->addWidget(sectionPositionSlider);

	QHBoxLayout* layout411 = new QHBoxLayout;
	QSpinBox* textSizeSpinBox = new QSpinBox;
	textSizeSpinBox->setMinimum(1);
	textSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	textSizeSpinBox->setValue(textSize);
	textSizeSpinBox->setPrefix("Text size : ");
	QLabel* colorLabel = new QLabel("Color :");
	colorHolder = new QPushButton();
	setAviTextColor(textColor);

	layout411->addWidget(textSizeSpinBox);
	layout411->addWidget(colorLabel);
	layout411->addWidget(colorHolder);


	processwatcher_rawtoaviprocess = new ProcessWatcherWidget;

	QPushButton *pushbutton_rawtoaviDisplay = new QPushButton("display");
	QPushButton *pushbutton_rawtoaviRun = new QPushButton("start");


	layout4->addWidget(m_rawToAviRgb1FileSelectWidget);
	layout4->addWidget(m_rawToAviRgb2FileSelectWidget);
	layout4->addLayout(layout42);
	layout4->addLayout(layout43);
	layout4->addLayout(layout44);
	layout4->addLayout(layout45);
	layout4->addLayout(layout46);
	layout4->addLayout(layout47);
	layout4->addLayout(layout48);
	layout4->addLayout(layout49);
	layout4->addLayout(layout410);
	layout4->addLayout(layout411);
	layout4->addWidget(processwatcher_rawtoaviprocess);
	layout4->addWidget(pushbutton_rawtoaviDisplay);
	layout4->addWidget(pushbutton_rawtoaviRun);


	// ================================================================
	/*
	QHBoxLayout *layout51 = new QHBoxLayout;
	QLabel *label_rgb1toxtFilename = new QLabel("RGB1 (8 bits) filename");
	lineedit_rgb1toxtFilename = new QLineEdit;
	lineedit_rgb1toxtFilename->setReadOnly(true);
	QPushButton *pushbutton_rgb1toxtFileOpen = new QPushButton("...");
	layout51->addWidget(label_rgb1toxtFilename);
	layout51->addWidget(lineedit_rgb1toxtFilename);
	layout51->addWidget(pushbutton_rgb1toxtFileOpen);

	QHBoxLayout *layout53 = new QHBoxLayout;
	QLabel *label_rgb1toxtRGB2Filename = new QLabel("RGB2 (16 bits) filename");
	lineedit_rgb1toxtRGB2Filename = new QLineEdit;
	lineedit_rgb1toxtRGB2Filename->setReadOnly(true);
	QPushButton *pushbutton_rgb1toxtRGB2FileOpen = new QPushButton("...");
	layout53->addWidget(label_rgb1toxtRGB2Filename);
	layout53->addWidget(lineedit_rgb1toxtRGB2Filename);
	layout53->addWidget(pushbutton_rgb1toxtRGB2FileOpen);
	*/

	m_rgb8BitsXtRgb1FileSelectWidget = new FileSelectWidget();
	m_rgb8BitsXtRgb1FileSelectWidget->setProjectManager(m_projectManager);
	m_rgb8BitsXtRgb1FileSelectWidget->setLabelText("RGB1 (8 bits) filename");
	m_rgb8BitsXtRgb1FileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::rgtCubeToAttribut);
	m_rgb8BitsXtRgb1FileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Raw);
	m_rgb8BitsXtRgb1FileSelectWidget->setLabelDimensionVisible(false);


	m_rgb8BitsXtRgb2FileSelectWidget = new FileSelectWidget();
	m_rgb8BitsXtRgb2FileSelectWidget->setProjectManager(m_projectManager);
	m_rgb8BitsXtRgb2FileSelectWidget->setLabelText("RGB2 (16 bits) filename");
	m_rgb8BitsXtRgb2FileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::rgtCubeToAttribut);
	m_rgb8BitsXtRgb2FileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Raw);
	m_rgb8BitsXtRgb2FileSelectWidget->setLabelDimensionVisible(false);

	QHBoxLayout *layout52 = new QHBoxLayout;
	QLabel *label_xtPrefix = new QLabel("xt prefix");
	lineedit_xtPrefix = new QLineEdit("xt");
	layout52->addWidget(label_xtPrefix);
	layout52->addWidget(lineedit_xtPrefix);

	qpb_progress_rgb1toxt = new QProgressBar();
	qpb_progress_rgb1toxt->setMinimum(0);
	qpb_progress_rgb1toxt->setMaximum(100);
	qpb_progress_rgb1toxt->setValue(0);
	qpb_progress_rgb1toxt->setTextVisible(true);
	qpb_progress_rgb1toxt->setValue(0);
	qpb_progress_rgb1toxt->setFormat("");

	qpb_rgb1toxtStart = new QPushButton("start");

	layout5->addWidget(m_rgb8BitsXtRgb1FileSelectWidget);
	layout5->addWidget(m_rgb8BitsXtRgb2FileSelectWidget);
	layout5->addLayout(layout52);
	layout5->addWidget(qpb_progress_rgb1toxt);
	layout5->addWidget(qpb_rgb1toxtStart);

	/*
	QHBoxLayout *layout61 = new QHBoxLayout;
	QLabel *label_aviviewAviFilename = new QLabel("AVI filename");
	lineedit_aviviewAviFilename = new QLineEdit;
	lineedit_aviviewAviFilename->setReadOnly(true);
	QPushButton *pushbutton_aviviewAviFilenOpen = new QPushButton("...");
	layout61->addWidget(label_aviviewAviFilename);
	layout61->addWidget(lineedit_aviviewAviFilename);
	layout61->addWidget(pushbutton_aviviewAviFilenOpen);
	*/

	m_aviViewFileSelectWidget = new FileSelectWidget();
	m_aviViewFileSelectWidget->setProjectManager(m_projectManager);
	m_aviViewFileSelectWidget->setLabelText("avi filename");
	m_aviViewFileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::rgtCubeToAttribut);
	m_aviViewFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Avi);
	m_aviViewFileSelectWidget->setLabelDimensionVisible(false);

	QPushButton* pushbutton_aviviewStart = new QPushButton("start");

	layout6->addWidget(m_aviViewFileSelectWidget);
	layout6->addWidget(pushbutton_aviviewStart);


	qgb_rgt2rgb->setMaximumHeight(300);
	qgb_rgb2torgb1->setMaximumHeight(250);
	//qgb_rawtoavi->setMaximumHeight(500);
	qgb_rgb1toxt->setMaximumHeight(250);
	qgb_aviview->setMaximumHeight(250);


	m_meanSeismicSpectrumWidget = new MeanSeismicSpectrumWidget(m_projectManager, this);
	// m_meanSeismicSpectrumWidget->setProjectManagerWidget(m_selectorWidget);
	m_meanSeismicSpectrumWidget->setSpectrumComputeWidget(this);

	m_rgtToGCC16Widget = new RgtToGCC16Widget(m_projectManager, this);
	// m_rgtToGCC16Widget->setProjectManagerWidget(m_selectorWidget);
	m_rgtToGCC16Widget->setSpectrumComputeWidget(this);

	m_rgb2ToXtWidget = new Rgb2ToXtWidget(m_projectManager, this);
	// m_rgb2ToXtWidget->setProjectManagerWidget(m_selectorWidget);
	m_rgb2ToXtWidget->setSpectrumComputeWidget(this);

	tabwidget_table1 = new QTabWidget();
	tabwidget_table1->insertTab(0, qgb_rgt2rgb, QIcon(QString("")), "RGT --> RGB 16 bits");
	tabwidget_table1->insertTab(1, qgb_rgb2torgb1, QIcon(QString("")), "RGB 16 bits --> RGB 8 bits");
	tabwidget_table1->insertTab(2, qgb_rawtoavi, QIcon(QString("")), "raw to avi");
	tabwidget_table1->insertTab(3, qgb_rgb1toxt, QIcon(QString("")), "RGB 8 bits --> XT");
	tabwidget_table1->insertTab(4, qgb_aviview, QIcon(QString("")), "View AVI");
	// tabwidget_table1->insertTab(5, initMeanSeismicSpectrumGroupBox(), QIcon(QString("")), "RGT --> Seismic Mean");
	tabwidget_table1->insertTab(5, m_meanSeismicSpectrumWidget->getGroupBox(), QIcon(QString("")), "RGT --> Seismic Mean");
	tabwidget_table1->insertTab(6, m_rgtToGCC16Widget->getMainGroupBox(), QIcon(QString("")), "RGT --> GCC16");
	tabwidget_table1->insertTab(7, m_rgb2ToXtWidget->getMainGroupBox(), QIcon(QString("")), "RGB2 --> XT");

	//tabwidget_table1->widget(2)->setEnabled(false);

	mainLayout01->addWidget(m_seismicFileSelectWidget);
	mainLayout01->addWidget(tabwidget_table1);

	/*
	this->systemInfo = new GeotimeSystemInfo(this);
	this->systemInfo->setVisible(true);
	systemInfo->setMinimumWidth(350);
	*/

	QGroupBox *qgbSystem = new QGroupBox;
	this->systemInfo = new GeotimeSystemInfo(this);
	this->systemInfo->setVisible(true);
	systemInfo->setMinimumWidth(350);
	QVBoxLayout *layout2s = new QVBoxLayout(qgbSystem);
	layout2s->addWidget(systemInfo);


	// QVBoxLayout *layout2 = new QVBoxLayout;
	// layout2->addWidget(systemInfo);
	// mainLayout00


	// mainLayout00->addLayout(mainLayout02);
	// mainLayout00->addLayout(mainLayout01);
	// mainLayout00->addWidget(systemInfo);

//	mainLayout00->addWidget(tabwidget_table1);


	// ===================================================
	QTabWidget *tabWidgetMain = new QTabWidget();
	tabWidgetMain->insertTab(0, qgbProgramManager, QIcon(QString("")), "Project Manager");
	tabWidgetMain->insertTab(1, qgbMainLayout01, QIcon(QString("")), "Compute");
	tabWidgetMain->insertTab(2, qgbSystem, QIcon(QString("")), "System");

	QScrollArea *scrollArea = new QScrollArea;
	scrollArea->setWidget(tabWidgetMain);
	scrollArea->setWidgetResizable(true);

	mainLayout00->addWidget(scrollArea);







	timer = new QTimer(this);
    timer->start(1000);
    timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));

    connect(combobox_horizonchoice, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_horizonchoiceclick(int)));
	// connect(pushbutton_seismicOpen, SIGNAL(clicked()), this, SLOT(trt_seismic_open()));
	// connect(pushbutton_rgtOpen, SIGNAL(clicked()), this, SLOT(trt_rgt_open()));
	connect(pushbutton_horizonadd, SIGNAL(clicked()), this, SLOT(trt_horizon_add()));
	connect(pushbutton_horizonsub, SIGNAL(clicked()), this, SLOT(trt_horizon_sub()));
	connect(pushbutton_startstop, SIGNAL(clicked()), this, SLOT(trt_launch_thread()));
	// connect(pushbutton_rgb2FilenameOpen, SIGNAL(clicked()), this, SLOT(trt_rgb2_open()));
	connect(pushbutton_rgtb2torgb1StartStop, SIGNAL(clicked()), this, SLOT(trt_launch_rgb2torgb1_thread()));
	connect(pushbutton_sessionload, SIGNAL(clicked()), this, SLOT(trt_loadSession()));
	connect(pushbutton_sessionsave, SIGNAL(clicked()), this, SLOT(trt_saveSession()));

	// connect(pushbutton_rawtoaviFilenameOpen, SIGNAL(clicked()), this, SLOT(trt_raw1Open()));
	// connect(pushbutton_rgb2toaviFilenameOpen, SIGNAL(clicked()), this, SLOT(trt_rgb2AviOpen()));
	connect(pushbutton_rawtoaviRun, SIGNAL(clicked()), this, SLOT(trt_rawToAviRun()));
	connect(pushbutton_rawtoaviDisplay, SIGNAL(clicked()), this, SLOT(trt_rawToAviDisplay()));
	connect(pushbutton_aviviewStart, SIGNAL(clicked()), this, SLOT(trt_aviviewRun()));


	// connect(pushbutton_rgb1toxtFileOpen, SIGNAL(clicked()), this, SLOT(trt_rgb1toxt_open()));
	// connect(pushbutton_rgb1toxtRGB2FileOpen, SIGNAL(clicked()), this, SLOT(trt_rgb1toxtRGB2_open()));
	// connect(pushbutton_aviviewAviFilenOpen, SIGNAL(clicked()), this, SLOT(trt_aviview_open()));
	connect(qpb_rgb1toxtStart, SIGNAL(clicked()), this, SLOT(trt_launch_rgb1toxt_thread()));

	connect(spinboxvideoScale, SIGNAL(valueChanged(double)), this, SLOT(trt_videoScaleChanged(double)));
	connect(spinboxRgb1TOrigin, SIGNAL(valueChanged(int)), this, SLOT(trt_rgb1TOriginChanged(int)));
	connect(spinboxRgb1TStep, SIGNAL(valueChanged(int)), this, SLOT(trt_rgb1TStepChanged(int)));
	connect(checkboxRgb1IsReversed, SIGNAL(stateChanged(int)), this, SLOT(trt_rgb1IsReversedChanged(int)));
	connect(spinboxFps, SIGNAL(valueChanged(int)), this, SLOT(trt_fpsChanged(int)));
	connect(spinboxFirstIso, SIGNAL(valueChanged(int)), this, SLOT(trt_firstIsoChanged(int)));
	connect(spinboxLastIso, SIGNAL(valueChanged(int)), this, SLOT(trt_lastIsoChanged(int)));
	connect(directionComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_directionChanged(int)));
	connect(sectionPositionSlider, SIGNAL(valueChanged(int)), this, SLOT(trt_sectionIndexChanged(int)));
	connect(sectionPositionSpinBox, SIGNAL(valueChanged(int)), this, SLOT(trt_sectionIndexChanged(int)));
	connect(textSizeSpinBox, SIGNAL(valueChanged(int)), this, SLOT(trt_textSizeChanged(int)));
	connect(colorHolder, SIGNAL(clicked()), this, SLOT(trt_changeAviTextColor()));
	connect(processwatcher_rawtoaviprocess, &ProcessWatcherWidget::processEnded, this, &SpectrumComputeWidget::updateRGBD);

	DisplayHorizonType();

	computeoptimalscale_rawToAvi();
	resize(1500*2/3, 900);
}



SpectrumComputeWidget::~SpectrumComputeWidget() {

	if ( thread != nullptr ) delete thread;
	if ( timer != nullptr ) delete timer;
}



QGroupBox *SpectrumComputeWidget::initMeanSeismicSpectrumGroupBox()
{
	QGroupBox *qgb = new QGroupBox();
	QVBoxLayout *layout = new QVBoxLayout(qgb);


	/*
	QHBoxLayout *hlayout1 = new QHBoxLayout;
	QLabel *rgtLabel = new QLabel("rgt filename");
	le_SesismicMeanRgtFilename = new QLineEdit();
	QPushButton *pb_rgtFilename = new QPushButton("...");
	hlayout1->addWidget(rgtLabel);
	hlayout1->addWidget(le_SesismicMeanRgtFilename);
	hlayout1->addWidget(pb_rgtFilename);
	*/
	m_seismicMeanRgtFileSelectWidget = new FileSelectWidget();
	m_seismicMeanRgtFileSelectWidget->setProjectManager(m_projectManager);
	m_seismicMeanRgtFileSelectWidget->setLabelText("rgt filename");
	m_seismicMeanRgtFileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Rgt);
	m_seismicMeanRgtFileSelectWidget->setLabelDimensionVisible(false);

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
	QLabel *isoStepLabel = new QLabel("iso step");
	le_outMeanIsoStep = new QLineEdit("25");
	hlayout4->addWidget(isoStepLabel);
	hlayout4->addWidget(le_outMeanIsoStep);


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
	layout->addWidget(qpb_seismicMean);
	layout->addWidget(qbp_seismicMeanStart);

	qgb->setMaximumHeight(500);

	// connect(pb_rgtFilename, SIGNAL(clicked()), this, SLOT(trt_rgtMeanSeismicOpen()));
	connect(qbp_seismicMeanStart, SIGNAL(clicked()), this, SLOT(trt_lauchRgtMeanSeismicStart()));


	return qgb;
}



QString SpectrumComputeWidget::getSeismicTinyName()
{
	return m_seismicFileSelectWidget->getFilename();
}

QString SpectrumComputeWidget::getSeismicFullName()
{
	return m_seismicFileSelectWidget->getPath();
}

QString SpectrumComputeWidget::getRgtTinyName()
{
	return m_rgtFileSelectWidget->getFilename();
}

QString SpectrumComputeWidget::getRgtFullName()
{
	return m_rgtFileSelectWidget->getPath();
}


void SpectrumComputeWidget::getTraceParameter(char *filename, float *pasech, float *tdeb)
{
	*pasech = 1.0; *tdeb = 0.0;
	if ( !FILEIO2::exist(filename) ) return;
	inri::Xt *_xt = nullptr;
	if ( filename == nullptr ) return;
	_xt = new inri::Xt(filename);
	if ( _xt == nullptr ) return;
	*tdeb = _xt->startSamples();
	*pasech = _xt->stepSamples();
	if ( _xt != nullptr ) delete _xt;
}


void SpectrumComputeWidget::DisplayHorizonType()
{
	int idx = combobox_horizonchoice->currentIndex();
	switch (idx)
	{
		case 0:
			label_isostep->setVisible(true);
			lineedit_isostep->setVisible(true);
			label_layerNumber->setVisible(false);
			lineedit_layer_number->setVisible(false);
			horizonWidget->setVisible(false);
			break;
		case 1:
			label_isostep->setVisible(false);
			lineedit_isostep->setVisible(false);
			label_layerNumber->setVisible(true);
			lineedit_layer_number->setVisible(true);
			horizonWidget->setVisible(true);
			break;
	}
}

void SpectrumComputeWidget::trt_horizonchoiceclick(int idx)
{
	DisplayHorizonType();
}


void SpectrumComputeWidget::trt_loadSession()
{
	// m_selectorWidget->load_session_gui();
}


void SpectrumComputeWidget::trt_saveSession()
{
	// m_selectorWidget->save_session_gui();
}




int SpectrumComputeWidget::getSizeFromFilename(QString filename, int *size)
{
	if ( filename.compare("") == 0 ) { for (int i=0; i<3;i++) size[i] = 0; return 0; }
    char c_filename[10000];
    strcpy(c_filename, (char*)(filename.toStdString().c_str()));
    char *p = c_filename;
    FILEIO2 *pf = new FILEIO2();
    pf->openForRead(p);
    size[0] = pf->get_dimy();
    size[1] = pf->get_dimx();
    size[2] = pf->get_dimz();
    delete pf;
    if ( size[0] == 0 && size[1] == 0 && size[2] == 0 ) return 0;
    return 1;
}

bool SpectrumComputeWidget::getPropertiesFromDatasetPath(QString filename, int* size, double* steps, double* origins)
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

std::tuple<double, bool, double> SpectrumComputeWidget::getAngleFromFilename(QString seismicFullName)
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

QSizeF SpectrumComputeWidget::newSizeFromSizeAndAngle(QSizeF oriSize, double angle) {
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

void SpectrumComputeWidget::rgb2LabelUpdate(QString name)
{
	label_rgb2Name->setText("output rgb2 filename: " + name);
}

void SpectrumComputeWidget::rgb1LabelUpdate(QString name)
{
	label_rgb1Name->setText("output rgb1 filename: " + name);
}





void SpectrumComputeWidget::frequencyTableWidgetRead(int *arrayFreq, int *arrayIso, int *count)
{
	int n = 0;
	int N = tableWidget_freq->rowCount();
	for (int i=0; i<N; i++)
	{
		QTableWidgetItem *itemFreq = tableWidget_freq->item(i, 0);
		QTableWidgetItem *itemIso = tableWidget_freq->item(i, 1);
		if ( itemFreq && itemIso )
		{
			QString freqString = itemFreq->text();
			QString isoString = itemIso->text();
			if ( freqString.compare("") != 0 && isoString.compare("") != 0 )
			{
				arrayFreq[n] = freqString.toInt();
				arrayIso[n] = isoString.toInt();
				n++;
			}
		}
	}
	*count = n;

	for (int i=0; i<n; i++)
		fprintf(stderr, "%d %d %d\n", i, arrayFreq[i], arrayIso[i]);
}






QString SpectrumComputeWidget::filenameToPath(QString fullName)
{
	int lastPoint = fullName.lastIndexOf("/");
	QString path = fullName.left(lastPoint);
	return path;
}

void SpectrumComputeWidget::aviFilenameUpdate()
{

	if ( getSeismicTinyName().compare("") == 0 || m_rawToAviRgb1FileSelectWidget->getFilename().compare("") == 0 ) return;
	QString path = filenameToPath(m_rawToAviRgb1FileSelectWidget->getPath());
	int size[3];

	QString tinyName;
	QStringList list = m_rawToAviRgb1FileSelectWidget->getFilename().split(" ");// because there is no display name...
	if (list.size()>0) {
		tinyName = list[0];
	}

	aviFullName = path + "/" + tinyName + "_" + lineedit_aviPrefix->text() + ".avi";
	QFileInfo info(aviFullName);
	aviTinyName = info.fileName();
}

void SpectrumComputeWidget::rgb2FilenameUpdate(int depth)
{
	if ( getSeismicTinyName().compare("") == 0 )
	{
		rgb2FullName = "";
		rgb2TinyName = "";
		return;
	}

	QString ImportExportPath = m_projectManager->getImportExportPath();
	QString IJKPath = m_projectManager->getIJKPath();
	QString seimsicNamePath = m_projectManager->getIJKPath() + QString(getSeismicTinyName()) + "/";
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

	int size[3];
	getSizeFromFilename(getSeismicFullName(), size);
	int width = size[0];
	int height = size[2];


	QString ret = "";
	QString path = cubeRgt2RgbPath; // filenameToPath(seismicFullName);
	// int f1 = lineedit_f1->text().toInt();
	// int f2 = lineedit_f2->text().toInt();
	// int f3 = lineedit_f3->text().toInt();
	int wsize = lineedit_wsize->text().toInt();
	int isostep = lineedit_isostep->text().toInt();
	QString prefix = lineedit_rgb2prefix->text();

	// rgb2FullName = path + "/" + seismicTinyName + "__" + "rgb2_" + prefix + "_size_" + QString::number(width) + "x" + QString::number(height) + "x" + QString::number(depth) + ".raw";
	rgb2FullName = path + "/" + "rgb2_" + prefix + "_from_" +
			getSeismicTinyName() + "_size_" + QString::number(width) + "x" + QString::number(height) + "x" + QString::number(depth) + ".raw";
	isoFullName = path + "/" + "iso_" + prefix + "_from_" +
			getSeismicTinyName() + "_size_" + QString::number(width) + "x" + QString::number(height) + "x" + QString::number(depth) + ".raw";

	QFileInfo info(rgb2FullName);
	rgb2TinyName = info.fileName();
	QFileInfo info2(isoFullName);
	isoTinyName = info2.fileName();
}

void SpectrumComputeWidget::rgb1FilenameUpdate()
{
	if ( m_rb2FileSelectWidget->getFilename().compare("") == 0 )
	{
		rgb1FullName = "";
		rgb1TinyName = "";
		return;
	}
	int size[3];
	int ret = getSizeFromFilename(getSeismicFullName(), size);
	int dimx = size[2];
	int dimy = size[0];
	QFile file(m_rb2FileSelectWidget->getPath());
	long size0 = file.size();
	int dimz = size0/dimx/dimy/4/2;

	// QString ret = "";
	QString path = filenameToPath(m_rb2FileSelectWidget->getPath());
	float alpha = lineedit_alpha->text().toFloat();
	float ratio = lineedit_ratio->text().toFloat();
	QString prefix = lineedit_rgb1prefix->text();

	rgb1FullName = path + "/" + "rgb1_" + prefix + "_from_" +
			m_rb2FileSelectWidget->getFilename() + "__alpha_" + lineedit_alpha->text().replace(".", "x") + "__ratio_" + lineedit_ratio->text().replace(".", "x")
			+ QString("_size_") + QString::number(dimy) + "x" + QString::number(dimx) + "x" + QString::number(dimz) + ".rgb";

	QFileInfo info(rgb1FullName);
	rgb1TinyName = info.fileName();
}

int SpectrumComputeWidget::getIndexFromVectorString(std::vector<QString> list, QString txt)
{
    for (int i=0; i<list.size(); i++)
    {
        if ( list[i].compare(txt) == 0 )
            return i;
    }
    return -1;
}

void SpectrumComputeWidget::trt_open_file(std::vector<QString> name_list, char *filename_out, bool multiselection)
{
    if ( filename_out ) filename_out[0] = 0;
    if ( name_list.size() == 0 ) return;
    if ( !multiselection )
    {
    	DialogSp *dlg = new DialogSp(name_list, filename_out, this);
    	dlg->setModal(true);
    	dlg->setGeometry(QRect(0, 0, 700, 500));
    	if ( multiselection ) dlg->setMultiselection();
    	dlg->exec();
    }
    else
    {
    	DialogSp *dlg = new DialogSp(name_list, this);
    	dlg->setModal(true);
    	dlg->setGeometry(QRect(0, 0, 700, 500));
    	dlg->setMultiselection();
    	dlg->exec();
    }
}

void SpectrumComputeWidget::trt_rgt_open()
{
	/*
	std::vector<QString> v_seismic_names = this->m_selectorWidget->get_seismic_names();
	std::vector<QString> v_seismic_filenames = this->m_selectorWidget->get_seismic_fullpath_names();
	char buff[10000];
	trt_open_file(v_seismic_names, buff, false);
	if ( buff[0] == 0 ) return;
	int idx = getIndexFromVectorString(v_seismic_names, QString(buff));
	if (idx<0) return;
	inri::Xt xt(v_seismic_filenames[idx].toStdString().c_str());
	if (!xt.is_valid()) return;
	if (xt.type()!=inri::Xt::Signed_16) {
		QMessageBox::warning(this, "Fail to load cube", "Selected cube is not of type : \"signed short\", abort selection");
		return;
	}
	this->rgtTinyName = QString(buff);
	this->lineedit_rgtFilename->setText(this->rgtTinyName);
	if ( idx >= 0 )
		this->rgtFullName = v_seismic_filenames[idx];
	else
		return;
	int size[3];
	// get_size_from_filename(this->seismic_filename, size);
	// update_label_size(size);
	 * */
}



void SpectrumComputeWidget::trt_seismic_open()
{
	/*
	std::vector<QString> v_seismic_names = this->m_selectorWidget->get_seismic_names();
	std::vector<QString> v_seismic_filenames = this->m_selectorWidget->get_seismic_fullpath_names();
	char buff[10000];
	trt_open_file(v_seismic_names, buff, false);
	if ( buff[0] == 0 ) return;
	int idx = getIndexFromVectorString(v_seismic_names, QString(buff));
	if (idx<0) return;
	inri::Xt xt(v_seismic_filenames[idx].toStdString().c_str());
	if (!xt.is_valid()) return;
	if (xt.type()!=inri::Xt::Signed_16) {
		QMessageBox::warning(this, "Fail to load cube", "Selected cube is not of type : \"signed short\", abort selection");
		return;
	}
	this->seismicTinyName = QString(buff);
	this->lineedit_seismicFilename->setText(this->seismicTinyName);
	if ( idx >= 0 ) {
		this->seismicFullName = v_seismic_filenames[idx];
		computeoptimalscale_rawToAvi();
		trt_directionChanged(directionComboBox->currentIndex());
	}
	else
		return;
	// rgb2FilenameUpdate();
	// rgb2LabelUpdate(rgb2TinyName);
	// get_size_from_filename(this->seismic_filename, size);
	// update_label_size(size);
	 *
	 */
}

void SpectrumComputeWidget::trt_rgb2_open()
{
	/*
	std::vector<QString> v_rgb_names = this->m_selectorWidget->get_rgb_names();
	std::vector<QString> v_rgb_filenames = this->m_selectorWidget->get_rgb_fullnames();
	char buff[10000];
	trt_open_file(v_rgb_names, buff, false);
	if ( buff[0] == 0 ) return;
	this->rgb2_2_TinyName = QString(buff);
	this->lineedit_rgb2Filename->setText(this->rgb2_2_TinyName);
	int idx = getIndexFromVectorString(v_rgb_names, this->rgb2_2_TinyName);
	if ( idx >= 0 )
		this->rgb2_2_FullName = v_rgb_filenames[idx];

	fprintf(stderr, "%s\n", v_rgb_filenames[0].toStdString().c_str());
	*/
}

void SpectrumComputeWidget::trt_raw1Open()
{
	/*
	std::vector<QString> v_rgb_names = this->m_selectorWidget->get_rgb_names();
	std::vector<QString> v_rgb_filenames = this->m_selectorWidget->get_rgb_fullnames();

	// filter to only keep .rgb files
	for (long i=v_rgb_filenames.size()-1; i>=0; i--) {
		QFileInfo fileInfo(v_rgb_filenames[i]);
		if (fileInfo.suffix().toLower().compare("rgb")!=0) {
			std::vector<QString>::iterator it = v_rgb_filenames.begin();
			std::advance(it, i);
			v_rgb_filenames.erase(it);

			std::vector<QString>::iterator it2 = v_rgb_names.begin();
			std::advance(it2, i);
			v_rgb_names.erase(it2);
		}
	}


	char buff[10000];
	trt_open_file(v_rgb_names, buff, false);
	if ( buff[0] == 0 ) return;
	this->rgb1_2_TinyName = QString(buff);
	this->lineedit_rawtoaviFilename->setText(this->rgb1_2_TinyName);
	int idx = getIndexFromVectorString(v_rgb_names, this->rgb1_2_TinyName);
	if ( idx >= 0 ) {
		this->rgb1_2_FullName = v_rgb_filenames[idx];
		computeoptimalscale_rawToAvi();
	}
	else
		return;
		*/
}

void SpectrumComputeWidget::trt_rgb2AviOpen() {
	/*
	std::vector<QString> v_rgb_names = this->m_selectorWidget->get_rgb_names();
	std::vector<QString> v_rgb_filenames = this->m_selectorWidget->get_rgb_fullnames();

	// filter to only keep .rgb files
	for (long i=v_rgb_filenames.size()-1; i>=0; i--) {
		QFileInfo fileInfo(v_rgb_filenames[i]);
		if (fileInfo.suffix().toLower().compare("raw")!=0) {
			std::vector<QString>::iterator it = v_rgb_filenames.begin();
			std::advance(it, i);
			v_rgb_filenames.erase(it);

			std::vector<QString>::iterator it2 = v_rgb_names.begin();
			std::advance(it2, i);
			v_rgb_names.erase(it2);
		}
	}

	char buff[10000];
	trt_open_file(v_rgb_names, buff, false);
	if ( buff[0] == 0 ) return;
	this->rgb2_Avi_TinyName = QString(buff);
	this->lineedit_rgb2toaviFilename->setText(this->rgb2_Avi_TinyName);
	int idx = getIndexFromVectorString(v_rgb_names, this->rgb2_Avi_TinyName);
	if ( idx >= 0 )
		this->rgb2_Avi_FullName = v_rgb_filenames[idx];

	fprintf(stderr, "%s\n", v_rgb_filenames[0].toStdString().c_str());
	*/
}

void SpectrumComputeWidget::trt_aviview_open()
{
	/*
	std::vector<QString> v_rgb_names = this->m_selectorWidget->get_rgb_names();
	std::vector<QString> v_rgb_filenames = this->m_selectorWidget->get_rgb_fullnames();

	// filter to only keep .rgb files
	for (long i=v_rgb_filenames.size()-1; i>=0; i--) {
		QFileInfo fileInfo(v_rgb_filenames[i]);
		if (fileInfo.suffix().toLower().compare("avi")!=0) {
			std::vector<QString>::iterator it = v_rgb_filenames.begin();
			std::advance(it, i);
			v_rgb_filenames.erase(it);

			std::vector<QString>::iterator it2 = v_rgb_names.begin();
			std::advance(it2, i);
			v_rgb_names.erase(it2);
		}
	}


	char buff[10000];
	trt_open_file(v_rgb_names, buff, false);
	if ( buff[0] == 0 ) return;
	this->avi_2_TinyName = QString(buff);
	this->lineedit_aviviewAviFilename->setText(this->avi_2_TinyName);
	int idx = getIndexFromVectorString(v_rgb_names, this->avi_2_TinyName);
	if ( idx >= 0 )
		this->avi_2_FullName = v_rgb_filenames[idx];
	else
		return;
		*/
}

void SpectrumComputeWidget::trt_rgb1toxt_open()
{
	/*
	std::vector<QString> v_rgb_names = this->m_selectorWidget->get_rgb_names();
	std::vector<QString> v_rgb_filenames = this->m_selectorWidget->get_rgb_fullnames();
	char buff[10000];
	trt_open_file(v_rgb_names, buff, false);
	if ( buff[0] == 0 ) return;
	this->rgb1_3_TinyName = QString(buff);
	this->lineedit_rgb1toxtFilename->setText(this->rgb1_3_TinyName);
	int idx = getIndexFromVectorString(v_rgb_names, this->rgb1_3_TinyName);
	QStringList list = this->rgb1_3_TinyName.split(" ");
	if (list.size()>0) {
		this->rgb1_3_TinyName = list[0]; // because there is no display name
	}
	if ( idx >= 0 )
		this->rgb1_3_Fullname = v_rgb_filenames[idx];
	else
		return;
		*/
}


void SpectrumComputeWidget::trt_rgb1toxtRGB2_open()
{
	/*
	std::vector<QString> v_rgb_names = this->m_selectorWidget->get_rgb_names();
	std::vector<QString> v_rgb_filenames = this->m_selectorWidget->get_rgb_fullnames();
	char buff[10000];
	trt_open_file(v_rgb_names, buff, false);
	if ( buff[0] == 0 ) return;
	this->rgb2_3_TinyName = QString(buff);
	this->lineedit_rgb1toxtRGB2Filename->setText(this->rgb2_3_TinyName);
	int idx = getIndexFromVectorString(v_rgb_names, this->rgb2_3_TinyName);
	if ( idx >= 0 )
		this->rgb2_3_Fullname = v_rgb_filenames[idx];
	else
		return;
		*/
}


void SpectrumComputeWidget::trt_horizon_add()
{
	/*
	std::vector<QString> v_horizons_names = this->m_selectorWidget->get_horizon_names();
	std::vector<QString> v_horizons_filenames = this->m_selectorWidget->get_horizon_fullpath_names();
	char buff[10000];
	trt_open_file(v_horizons_names, buff, false);
	if ( buff[0] == 0 ) return;
	horizonTinyName.push_back(QString(buff));
	int idx = getIndexFromVectorString(v_horizons_names, QString(buff));
	horizonFullname.push_back(v_horizons_filenames[idx]);
	listwidget_horizons->clear();
	for (int i=0; i<horizonTinyName.size(); i++)
	{
		QListWidgetItem *item = new QListWidgetItem;
		item->setText(horizonTinyName[i]);
		item->setToolTip(horizonTinyName[i]);
		listwidget_horizons->addItem(item);
	}
	*/
}

void SpectrumComputeWidget::trt_horizon_sub()
{
	/*
	horizonTinyName.clear();
	horizonFullname.clear();
	listwidget_horizons->clear();
	*/
}



// start stop
void SpectrumComputeWidget::trt_launch_thread()
{
	// if ( GLOBAL_RUN_TYPE == 1 ) return;
	/*
	if ( parameters_check() == 0 )
	{
		QMessageBox *msgBox = new QMessageBox(parentWidget());
		msgBox->setText("warning");
		msgBox->setInformativeText("Wrong parameters\nPlease fill in all the fields");
		msgBox->setStandardButtons(QMessageBox::Ok );
		int ret = msgBox->exec();
		return;
	}
	*/
	if (!checkSeismicsSizeMatch(getSeismicFullName(), getRgtFullName()))
	{
		QMessageBox::warning(this, "Volumes mismatch",
				"Seismic and RGT volumes do not match, try again with matching volumes");
		return;
	}
	if ( GLOBAL_RUN_TYPE == 0 )
	{
		// GLOBAL_RUN_TYPE = 1;
		if ( thread == nullptr )
			thread = new MyThreadSpectrumCompute(this);
		m_functionType = 1;
		qDebug() << "start thread0";
		thread->start();
		qDebug() << "start thread0 ok";
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

void SpectrumComputeWidget::trt_launch_rgb2torgb1_thread()
{
	if ( GLOBAL_RUNRGB2TORGB1_TYPE == 0 )
	{
		if ( thread == nullptr )
			thread = new MyThreadSpectrumCompute(this);
		m_functionType = 2;
		qDebug() << "start thread1";
		thread->start();
		qDebug() << "start thread1 ok";
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


void SpectrumComputeWidget::trt_launch_rgb1toxt_thread()
{
	if ( GLOBAL_RUNRGB1TOXT_TYPE == 0 )
	{
		if ( thread == nullptr )
			thread = new MyThreadSpectrumCompute(this);
		m_functionType = 3;
		qDebug() << "start thread2";
		thread->start();
		qDebug() << "start thread2 ok";
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


void SpectrumComputeWidget::horizonRead(std::vector<QString> horizonTinyName, std::vector<QString> horizonFullname,
		int dimx, int dimy, int dimz,
		float pasech, float tdeb,
		float **horizon1, float **horizon2)
{
	int err = 0;
	if ( horizonFullname.size() == 2 )
	{
		long size0 = (long)dimy*dimz;
		float *h1 = nullptr, *h2 = nullptr;
		h1 = (float*)calloc(size0, sizeof(float));
		h2 = (float*)calloc(size0, sizeof(float));
		FILE *pFile = nullptr;
		pFile = fopen(horizonFullname[0].toStdString().c_str(), "r");
		if ( pFile != nullptr )
		{
			fread(h1, sizeof(float), size0, pFile);
			fclose(pFile);
		}
		else
		{
			err = 1;
		}
		pFile = fopen(horizonFullname[1].toStdString().c_str(), "r");
		if ( pFile != nullptr )
		{
			fread(h2, sizeof(float), size0, pFile);
			fclose(pFile);
		}
		else
		{
			err = 1;
		}
		if ( err == 0 )
		{
			double h1m = 0.0, h2m = 0.0;
			for (long add=0; add<size0; add++)
			{
				h1[add] = (h1[add]-tdeb)/pasech;
				h2[add] = (h2[add]-tdeb)/pasech;
				if ( h1[add] >= 0 )	h1m += h1[add];
				if ( h2[add] >= 0 ) h2m += h2[add];
			}
			*horizon1 = (float*)calloc(size0, sizeof(float));
			*horizon2 = (float*)calloc(size0, sizeof(float));
			if ( h1m < h2m )
			{
				for (long add=0; add<size0; add++)
				{
					(*horizon1)[add] = h1[add];
					(*horizon2)[add] = h2[add];
				}
			}
			else
			{
				for (long add=0; add<size0; add++)
				{
					(*horizon1)[add] = h2[add];
					(*horizon2)[add] = h1[add];
				}
			}
		}
		if ( h1 ) free(h1);
		if ( h2 ) free(h2);
	}

}

void SpectrumComputeWidget::trt_rgt2rgbStartStop()
{
	float pasech = 1.0f;
	float tdeb = 0.0f;
	float *horizon1 = nullptr, *horizon2 = nullptr;
	int arrayFreq[10], arrayIso[10], Freqcount = 0, nbLayers = 1;
	int size[3];
	int isostep = lineedit_isostep->text().toInt();

	int idxHorizon = combobox_horizonchoice->currentIndex(); // 0: iso 1: 2 horizons
	int depth = 0;
	if ( idxHorizon ==  0 )
	{
		depth = 32000 / isostep;
	}
	else
	{
		depth = nbLayers;
	}


	rgb2FilenameUpdate(depth);
	rgb2LabelUpdate(rgb2TinyName);

	getSizeFromFilename(getSeismicFullName(), size);
	int width = size[0];
	int height = size[2];

	if ( getSeismicFullName().compare("") == 0 || getRgtFullName().compare("") == 0 ||
			rgb2FullName.compare("") == 0  || isoFullName.compare("") == 0 ||
			lineedit_f1->text().compare("") == 0 || lineedit_f2->text().compare("") == 0 || lineedit_f3->text().compare("") == 0 ) return;



	if ( idxHorizon == 1 && horizonFullname.size() != 2 ) return;


	frequencyTableWidgetRead(arrayFreq, arrayIso, &Freqcount);

	char seismicFilename[10000], rgtFilename[10000], rgbFilename[10000], isoFilename[10000];

	nbLayers = lineedit_layer_number->text().toInt();
	strcpy(seismicFilename, getSeismicFullName().toStdString().c_str());
	strcpy(rgtFilename, getRgtFullName().toStdString().c_str());
	strcpy(rgbFilename, rgb2FullName.toStdString().c_str());
	strcpy(isoFilename, isoFullName.toStdString().c_str());
	getTraceParameter(seismicFilename, &pasech, &tdeb);
	int retsize = getSizeFromFilename(seismicFilename, size);

	if ( idxHorizon == 1 )
		horizonRead(horizonTinyName, horizonFullname, size[1], size[0], size[2], pasech, tdeb, &horizon1, &horizon2);

	int wsize = lineedit_wsize->text().toInt();


	if ( FILEIO2::exist(seismicFilename) && FILEIO2::exist(rgtFilename) && retsize )
	{
	    int *tab_gpu = NULL, tab_gpu_size;
	    tab_gpu = (int*)calloc(this->systemInfo->get_gpu_nbre(), sizeof(int));
	    this->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
		GLOBAL_RUN_TYPE = 1;
		pushbutton_startstop->setText("stop");

		// f_rgt2rgb_2(seismicFilename, rgtFilename, tdeb,  pasech,
		//		0, 32000, isostep, horizon1, horizon2, nbLayers, wsize, arrayFreq, arrayIso, Freqcount, rgbFilename, isoFilename);
		// return;

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
		p->setOutputType(0);
		p->run();
		delete p;

		GLOBAL_RUN_TYPE = 0;
		pushbutton_startstop->setText("start");
		// m_selectorWidget->global_rgb_database_update();
	}

	if ( horizon1 ) free(horizon1);
	if ( horizon2 ) free(horizon2);
}


void SpectrumComputeWidget::trt_rgb2rgb1StartStop()
{
	if ( getSeismicFullName().compare("") == 0 || m_rb2FileSelectWidget->getPath().compare("") == 0 || lineedit_alpha->text().compare("") == 0 || lineedit_ratio->text().compare("") == 0 ) return;

	int size[3];
	getSizeFromFilename(getSeismicFullName(), size);
	rgb1FilenameUpdate();
	rgb1LabelUpdate(rgb1TinyName);


	char rgb2Filename[10000], rgb1Filename[10000];

	// rgb1FullName = "/data/PLI/DIR_PROJET/JD_TEST/DATA/3D/JD/DATA/SEISMIC/test.raw";

	// strcpy(seismicFilename, seismicFullName.toStdString().c_str());
	strcpy(rgb2Filename, m_rb2FileSelectWidget->getPath().toStdString().c_str());
	strcpy(rgb1Filename, rgb1FullName.toStdString().c_str());

	float alpha = lineedit_alpha->text().toFloat();
	float ratio = lineedit_ratio->text().toFloat();

	int dimx = size[2];
	int dimy = size[0];
	QFile file(m_rb2FileSelectWidget->getPath());
	long size0 = file.size();
	int dimz = size0/dimx/dimy/4/2;
	fprintf(stderr, "size: %ld %d %d %d\n", size0, dimx, dimy, dimz);

	GLOBAL_RUNRGB2TORGB1_TYPE = 1;
	pushbutton_rgtb2torgb1StartStop->setText("stop");
	cuda_rgb2torgb1(rgb2Filename, dimx, dimy, dimz, ratio, alpha, rgb1Filename);
	GLOBAL_RUNRGB2TORGB1_TYPE = 0;
	pushbutton_rgtb2torgb1StartStop->setText("start");
	// m_selectorWidget->global_rgb_database_update();
}


void SpectrumComputeWidget::trt_rgb1toxtStartStop()
{
	char seismicFilename[10000];
	char rgb1Filename[10000];
	char rgb2Filename[10000];
	char xtFilename[10000];
	int size[3];
	float tdeb, pasech;

	if ( !FILEIO2::exist((char*)getSeismicFullName().toStdString().c_str()) ||
		 !FILEIO2::exist((char*)m_rgb8BitsXtRgb1FileSelectWidget->getPath().toStdString().c_str()) ||
		 !FILEIO2::exist((char*)m_rgb8BitsXtRgb2FileSelectWidget->getPath().toStdString().c_str()) )
	{
		// TODO
		return;
	}

	getSizeFromFilename(getSeismicFullName(), size);
	int dimx = size[1];
	int dimy = size[0];
	int dimz = size[2];

	QFile file(m_rgb8BitsXtRgb2FileSelectWidget->getPath());
	long size0 = file.size();
	int nbRGB1plans = size0/dimz/dimy/4/2;

	QString path = filenameToPath(m_rgb8BitsXtRgb2FileSelectWidget->getPath());
	QString qt_xtFilename = path + "/" + "xt_" + "from_" +
			m_rgb8BitsXtRgb1FileSelectWidget->getFilename() + "_" + lineedit_xtPrefix->text() + "_dims_" + QString::number(dimx) + "x" +
			QString::number(dimy) + "x" + QString::number(dimz) + ".xt";

	strcpy(seismicFilename, getSeismicFullName().toStdString().c_str());
	strcpy(rgb1Filename, m_rgb8BitsXtRgb1FileSelectWidget->getPath().toStdString().c_str());
	strcpy(rgb2Filename, m_rgb8BitsXtRgb2FileSelectWidget->getPath().toStdString().c_str());
	strcpy(xtFilename, qt_xtFilename.toStdString().c_str());

	getTraceParameter(seismicFilename, &pasech, &tdeb);

	GLOBAL_RUNRGB1TOXT_TYPE = 1;
	qpb_rgb1toxtStart->setText("stop");
	f_rgb1toxt(dimx, dimy, dimz, nbRGB1plans, tdeb, pasech, seismicFilename, rgb1Filename, rgb2Filename, xtFilename);
	GLOBAL_RUNRGB1TOXT_TYPE = 0;
	qpb_rgb1toxtStart->setText("start");
	// m_selectorWidget->global_rgb_database_update();
}


void SpectrumComputeWidget::rawToAviRunFfmeg(bool onlyFirstImage)
{
	// if function changed update computeoptimalscale_rawToAvi
	computeoptimalscale_rawToAvi();

	GlobalConfig& config = GlobalConfig::getConfig();
	QDir tempDir = QDir(config.tempDirPath());

	aviFilenameUpdate();
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

	QString jpgTmpPath;
	if (onlyFirstImage) {
		QTemporaryFile outJpgImageFile;
		outJpgImageFile.setFileTemplate(tempDir.absoluteFilePath("NextVision_display_XXXXXX.jpg"));
		outJpgImageFile.setAutoRemove(false);
		outJpgImageFile.open();
		outJpgImageFile.close();
		jpgTmpPath = outJpgImageFile.fileName();
	}

	int size[3];
	double steps[3];
	bool isValid = getPropertiesFromDatasetPath(getSeismicFullName(), size, steps);
	if (!isValid) {
		return;
	}

	QVariant dirVariant = directionComboBox->currentData(Qt::UserRole);
	bool ok;
	int dirData = dirVariant.toInt(&ok);
	if (!ok) {
		return;
	}
	QString sectionName;
	SliceDirection direction;
	if (dirData==0) {
		direction = SliceDirection::Inline;
		sectionName = "Inline ";
	} else {
		direction = SliceDirection::XLine;
		sectionName = "XLine ";
	}

	int axisValue = sectionPositionSpinBox->value();
	int axisValueIndex = (axisValue - sectionPositionSpinBox->minimum()) / sectionPositionSpinBox->singleStep();

	sectionName += QString::number(axisValue);

	QTemporaryFile outSectionVideoRawFile;
	outSectionVideoRawFile.setFileTemplate(tempDir.absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	// outSectionVideoRawFile.setFileTemplate(QDir("/data/PLI/NKDEEP/jacques/").absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	outSectionVideoRawFile.setAutoRemove(false);
	outSectionVideoRawFile.open();
	outSectionVideoRawFile.close();
	QString inRgb2Path = rgb2_Avi_FullName;// "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/HR_NEAR/cubeRgt2RGB/rgb2_spectrum_mzr_from_HR_NEAR_size_1500x700x1280.raw";
	bool sectionValid = SectionToVideo::run(getSeismicFullName(), axisValueIndex, direction, inRgb2Path, outSectionVideoRawFile.fileName());
	int axisSize = size[0];
	if (direction == SliceDirection::XLine) {
		axisSize = size[2];
	}

	QString logoPath = GlobalConfig::getConfig().logoPath();
	std::tuple<double, bool, double> angleAndSwapV = getAngleFromFilename(getSeismicFullName());
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

	int tOrigin = rgb1TOrigin;
	int tStep = rgb1TStep; // expect positive int
	bool isReversed = rgb1IsReversed;

	std::size_t sz = getFileSize(m_rawToAviRgb1FileSelectWidget->getPath().toStdString().c_str());
	int numFrames = sz / (size[0] * size[2] * sizeof(char) * 3);

	int firstIndex, lastIndex, cutOrigin;
	QString tSign;
	if (isReversed) {
		tOrigin = tOrigin + (numFrames-1) * tStep;
		firstIndex = std::min(std::max((tOrigin - firstIso) / tStep, 0), numFrames-1);
		lastIndex = std::min(std::max((tOrigin - lastIso) / tStep, 0), numFrames-1);

		cutOrigin = tOrigin - std::min(firstIndex, lastIndex) * tStep;
		tSign = "-";
	} else {
		firstIndex = std::min(std::max((firstIso - tOrigin) / tStep, 0), numFrames-1);
		lastIndex = std::min(std::max((lastIso - tOrigin) / tStep, 0), numFrames-1);
		tSign = "+";

		cutOrigin = tOrigin + std::min(firstIndex, lastIndex) * tStep;
	}
	double firstTime = ((double) std::min(firstIndex, lastIndex)) / framePerSecond;
	double lastTime = ((double) std::max(firstIndex, lastIndex) + 1) / framePerSecond;
	double duration = lastTime - firstTime;
	QString beginTime = getFFMPEGTime(firstTime);

	QString drawTextExpr = "%{expr_int_format\\:" + QString::number(cutOrigin) + tSign +
			QString::number(tStep) + "*\\n\\:d\\:5}";

	QStringList options;
	options << "-y" << "-ss" << beginTime << "-t" << QString::number(duration);
	options << "-pixel_format" << "rgb24" << "-vcodec" << "rawvideo" << "-video_size";
	options << (QString::number(size[0]) + "x" + QString::number(size[2])) << "-framerate";
	options << QString::number(framePerSecond) << "-i" << m_rawToAviRgb1FileSelectWidget->getPath();
	options << "-pixel_format" << "rgb24" << "-vcodec" << "rawvideo" << "-video_size";
	options << (QString::number(axisSize) + "x" + QString::number(size[1])) << "-framerate";
	options << QString::number(framePerSecond) << "-i" << outSectionVideoRawFile.fileName();
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
	filterOption += "scale=iw*" + QString::number(videoScale * scaleX) + ":ih*" + QString::number(videoScale * scaleY) +
			",rotate=" + inlineAngleName;
	if (!padAndCrop.second.isNull() && !padAndCrop.second.isEmpty()) {
		filterOption += "," + padAndCrop.second;
	}
	filterOption += ",scale='iw-mod(iw,2)':'ih-mod(ih,2)'";
	filterOption += ",vflip";
	filterOption += ",drawtext=fontfile=/usr/share/fonts/gnu-free/FreeSans.ttf:text='RGT time \\: " + drawTextExpr + "':fontsize="+QString::number(textSize)+":fontcolor="+formatColorForFFMPEG(textColor)+":x=h/2-th/2:y=h-th-10";
	filterOption += " [tmp]; [tmp][logoBis] overlay=(main_w-overlay_w-10):(main_h-overlay_h-10) [last]; ";
	filterOption += "[1] scale=" + QString::number(std::floor(videoScale*sectionWidth)-50) + ":" + QString::number(std::floor(videoScale*sectionHeight));
	filterOption += ",drawtext=fontfile=/usr/share/fonts/gnu-free/FreeSans.ttf:text='" + sectionName + "':fontsize="+QString::number(textSize)+":fontcolor="+formatColorForFFMPEG(textColor)+":x=h/2-th/2:y=h-th-10 [last1]; ";
	filterOption += "[last] pad=iw+" + QString::number(std::floor(videoScale*sectionWidth)) + ":ih+40:ow-iw-20:20 [last_padded]; [last_padded][last1] overlay=20:20";

	options << filterOption;
	if (onlyFirstImage) {
		options << "-frames:v" << "1" << "-f" << "image2";
		options << jpgTmpPath;
	} else {
		options << aviFullName;
	}

	//fprintf(stderr, "command: %s\n", cmd.toStdString().c_str());
	qDebug() << "ffmpeg command : " << "ffmpeg " << options.join(" ");

	if (processwatcher_rawtoaviprocess->processState()==QProcess::NotRunning && m_processwatcher_connection==nullptr) {
		QString outSectionVideoRawFilePath = outSectionVideoRawFile.fileName();
		QMetaObject::Connection conn = connect(processwatcher_rawtoaviprocess, &ProcessWatcherWidget::processEnded, [this, outSectionVideoRawFilePath, jpgTmpPath]() {
			QFile::remove(outSectionVideoRawFilePath);
			if (m_processwatcher_connection!=nullptr) {
				QObject::disconnect(*m_processwatcher_connection);
				m_processwatcher_connection.reset(nullptr);
			}

			QProcess* process = new QProcess();
			process->setProgram("display");
			process->setArguments(QStringList() << jpgTmpPath);

			connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), [process, jpgTmpPath]() {
				process->deleteLater();
				QFile::remove(jpgTmpPath);
			});
			connect(process, &QProcess::errorOccurred, [process, jpgTmpPath]() {
				process->deleteLater();
				QFile::remove(jpgTmpPath);
			});

			process->start();
		});
		m_processwatcher_connection.reset(new QMetaObject::Connection(conn));
		processwatcher_rawtoaviprocess->launchProcess("ffmpeg", options);
	}
	//system(cmd.toStdString().c_str());
	// system("ls");
	// system("ffmpeg -f rawvideo -c:v rawvideo -pix_fmt rgb24 -s:v 2000x400 -r 10 -i /data/PLI/DIR_PROJET/JD_TEST/DATA/3D/JD/DATA/SEISMIC/seismic__spectrum_rgb2_wsize_64_f1_2_f2_4_f3_6__rgb1__alpha_1.0__ratio_.0001.raw -c:v libx264 /data/PLI/jacques/idle0/output0.mpeg");
	//m_selectorWidget->global_rgb_database_update();
}

void SpectrumComputeWidget::trt_rawToAviRun() {
	rawToAviRunFfmeg(false);
}

void SpectrumComputeWidget::trt_rawToAviDisplay() {
	rawToAviRunFfmeg(true);
}

void SpectrumComputeWidget::updateRGBD() {
	// m_selectorWidget->global_rgb_database_update();
}

void SpectrumComputeWidget::trt_aviviewRun()
{
	QString avi_2_FullName = m_aviViewFileSelectWidget->getPath();
	QString cmd = "vlc " + avi_2_FullName;
	int returnVal = system(cmd.toStdString().c_str());
	if (returnVal!=0) {
		cmd = "totem " + avi_2_FullName;
		system(cmd.toStdString().c_str());
	}
}

std::pair<QString, QString> SpectrumComputeWidget::getPadCropFFMPEG(double wRatio, double hRatio)
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

void SpectrumComputeWidget::showTime()
{
    // GLOBAL_textInfo->appendPlainText(QString("timer"));
    char txt[1000], txt2[1000];

    /* debug */
    // qDebug() << "Timer in";
    if ( GLOBAL_RUN_TYPE == 0 && GLOBAL_RUNRGB2TORGB1_TYPE == 0 && GLOBAL_RUNRGB1TOXT_TYPE == 0 )
    {
    	qpb_progress->setValue(0);
    	qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
    	qpb_progress->setFormat("");
    	qpb_progress_rgb2torgb1->setValue(0);
    	qpb_progress_rgb2torgb1->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
    	qpb_progress_rgb2torgb1->setFormat("");
    	qpb_progress_rgb1toxt->setValue(0);
    	qpb_progress_rgb1toxt->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
    	qpb_progress_rgb1toxt->setFormat("");
    	return;
    }

    int type = -1;
    long idx, vmax;
    int msg_new = ihm_get_global_msg(&type, &idx, &vmax, txt);
    // qDebug() << "Timer message: " << QString::number(msg_new);
    if ( msg_new == 0 ) return;
    switch ( type )
    {
    	case 1:
    	{
        	float val_f = 100.0*idx/vmax;
        	int val = (int)(val_f);
        	qpb_progress->setValue(val);
        	sprintf(txt2, "run %.1f%%", val_f);
        	qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
        	qpb_progress->setFormat(txt2);
        	break;
    	}
    	case 2:
    	{
    		// GLOBAL_RUN_TYPE = 0;
    		// qpb_progress->setValue(0);
    		// pushbutton_startstop->setText("start");

    		// m_selectorWidget->global_rgb_database_update();

//    		rgb2_2_TinyName = rgb2TinyName;
//   		rgb2_2_FullName = rgb2FullName;
//    		lineedit_rgb2Filename->setText(rgb2_2_TinyName);

    		/*
    		if ( thread != nullptr )
    		{
    			thread->wait();
    			delete thread; thread = nullptr;
    		}
    		*/
    		break;
    	}
    	case 5:
    	{
    		float val_f = 100.0*idx/vmax;
    		int val = (int)(val_f);
    		qpb_progress_rgb2torgb1->setValue(val);
    		sprintf(txt2, "run %.1f%%", val_f);
    		qpb_progress_rgb2torgb1->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
    		qpb_progress_rgb2torgb1->setFormat(txt2);
    		break;
    	}
    	case 6:
    		// GLOBAL_RUNRGB2TORGB1_TYPE = 0;
    		// qpb_progress_rgb2torgb1->setValue(0);
    		// pushbutton_rgtb2torgb1StartStop->setText("start");

    		// m_selectorWidget->global_rgb_database_update();
    		/*
    		if ( thread != nullptr )
    		{
    			thread->wait();
    			delete thread; thread = nullptr;
    		}
    		*/
    		break;


    	case 7:
    	{
    		float val_f = 100.0*idx/vmax;
    		int val = (int)(val_f);
    		qpb_progress_rgb1toxt->setValue(val);
    		sprintf(txt2, "run %.1f%%", val_f);
    		qpb_progress_rgb1toxt->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
    		qpb_progress_rgb1toxt->setFormat(txt2);
    		break;
    	}
    	case 8:
    		// GLOBAL_RUNRGB1TOXT_TYPE = 0;
    		// qpb_progress_rgb1toxt->setValue(0);
    		// qpb_rgb1toxtStart->setText("start");

    	    // m_selectorWidget->global_rgb_database_update();
    		/*
    	    if ( thread != nullptr )
    	    {
    	    	thread->wait();
    	    	delete thread; thread = nullptr;
    	    }
    	    */
    	    break;
    	case 9:
    	{
    		float val_f = 100.0*idx/vmax;
    		int val = (int)(val_f);
    		qpb_progress->setValue(val);
    		sprintf(txt2, "%s %.1f%%", txt, val_f);
    		qpb_progress->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,200,0)}");
    		qpb_progress->setFormat(txt2);
    		break;
    	}
    }

}

void SpectrumComputeWidget::closeEvent (QCloseEvent *event)
{
	/*
    QMessageBox::StandardButton resBtn = QMessageBox::question( this,  tr("My Application"),
                                                                tr("Are you sure?\n"),
                                                                QMessageBox::Cancel | QMessageBox::No | QMessageBox::Yes,
                                                                QMessageBox::Yes);
    if (resBtn != QMessageBox::Yes) {
        event->ignore();
    } else {
        event->accept();
    }*
    /
     *
     *
     */

	if ( GLOBAL_RUN_TYPE || GLOBAL_RUNRGB2TORGB1_TYPE || GLOBAL_RUNRGB1TOXT_TYPE)
	{
		ihm_set_trt(IHM_TRT_DIP_SMOOTHING_STOP);
	}
	if ( thread != nullptr )
	{
		thread->wait();
		delete thread; thread = nullptr;
	}
	if ( time )
	{
		delete timer;
		timer = nullptr;
	}
}

void SpectrumComputeWidget::trt_videoScaleChanged(double val) {
	videoScale = val;
}

void SpectrumComputeWidget::computeoptimalscale_rawToAvi() {
	if (getSeismicFullName().isNull() || getSeismicFullName().isEmpty() || m_rawToAviRgb1FileSelectWidget->getPath().isNull() ||
			m_rawToAviRgb1FileSelectWidget->getPath().isEmpty()) {
		return;
	}

	aviFilenameUpdate();
	int size[3];
	double steps[3];
	bool isValid = getPropertiesFromDatasetPath(getSeismicFullName(), size, steps);
	if (!isValid) {
		return;
	}
	QString logoPath = GlobalConfig::getConfig().logoPath();
	std::tuple<double, bool, double> angleAndSwapV = getAngleFromFilename(getSeismicFullName());
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
		spinboxvideoScale->setValue(0.75 / worstRatio);
	} else {
		spinboxvideoScale->setValue(1);
	}
}

std::size_t SpectrumComputeWidget::getFileSize(const char* filePath) {
	FILE* f = fopen(filePath, "r");
	std::size_t size = 0;
	if (f!=NULL) {
		fseek(f, 0L, SEEK_END);
		size = ftell(f);
		fclose(f);
	}

	return size;
}

void SpectrumComputeWidget::trt_rgb1TOriginChanged(int val) {
	rgb1TOrigin = val;
}

void SpectrumComputeWidget::trt_rgb1TStepChanged(int val) {
	rgb1TStep = val;
}

void SpectrumComputeWidget::trt_rgb1IsReversedChanged(int state) {
	rgb1IsReversed = state == Qt::Checked;
}

void SpectrumComputeWidget::trt_fpsChanged(int val) {
	framePerSecond = val;
}

void SpectrumComputeWidget::trt_firstIsoChanged(int val) {
	firstIso = val;
}

void SpectrumComputeWidget::trt_lastIsoChanged(int val) {
	lastIso = val;
}

void SpectrumComputeWidget::trt_textSizeChanged(int val) {
	textSize = val;
}

QString SpectrumComputeWidget::formatColorForFFMPEG(const QColor& color) {
	QString qtStyledColor = color.name();
	// remove "#" and add "0x"
	QString ffmpegColor = qtStyledColor;
	ffmpegColor.remove(0, 1);
	ffmpegColor.prepend("0x");
	return ffmpegColor;
}
// =====================================================================================================
//
//
// =====================================================================================================
void SpectrumComputeWidget::trt_rgtMeanSeismicOpen()
{
	/*
	std::vector<QString> v_seismic_names = this->m_selectorWidget->get_seismic_names();
	std::vector<QString> v_seismic_filenames = this->m_selectorWidget->get_seismic_fullpath_names();
	char buff[10000];
	trt_open_file(v_seismic_names, buff, false);
	if ( buff[0] == 0 ) return;
	int idx = getIndexFromVectorString(v_seismic_names, QString(buff));
	if (idx<0) return;
	inri::Xt xt(v_seismic_filenames[idx].toStdString().c_str());
	if (!xt.is_valid()) return;
	if (xt.type()!=inri::Xt::Signed_16) {
		QMessageBox::warning(this, "Fail to load cube", "Selected cube is not of type : \"signed short\", abort selection");
		return;
	}
	this->rgtTinyName = QString(buff);
	this->le_SesismicMeanRgtFilename->setText(this->rgtTinyName);
	if ( idx >= 0 )
		this->rgtFullName = v_seismic_filenames[idx];
	else
		return;
		*/
}

void SpectrumComputeWidget::trt_lauchRgtMeanSeismicStart()
{
	if ( GLOBAL_RUN_TYPE == 0 )
	{
		// GLOBAL_RUN_TYPE = 1;
		if ( thread == nullptr )
			thread = new MyThreadSpectrumCompute(this);
		m_functionType = 4;
		qDebug() << "start thread0";
		thread->start();
		qDebug() << "start thread0 ok";
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


void SpectrumComputeWidget::trt_seismicMeanStartStop()
{
	float pasech = 1.0f;
	float tdeb = 0.0f;
	float *horizon1 = nullptr, *horizon2 = nullptr;
	int arrayFreq[10], arrayIso[10], Freqcount = 0, nbLayers = 1;
	int size[3];
	int isostep = lineedit_isostep->text().toInt();

	int idxHorizon = combobox_horizonchoice->currentIndex(); // 0: iso 1: 2 horizons
	int depth = 0;
	if ( idxHorizon ==  0 )
	{
		depth = 32000 / isostep;
	}
	else
	{
		depth = nbLayers;
	}

	char seismicFilename[10000], rgtFilename[10000], rgbFilename[10000], isoFilename[10000];
//	nbLayers = lineedit_layer_number->text().toInt();
	strcpy(seismicFilename, getSeismicFullName().toStdString().c_str());
	// strcpy(rgtFilename, rgtFullName.toStdString().c_str()); *************
	getTraceParameter(seismicFilename, &pasech, &tdeb);
	int retsize = getSizeFromFilename(seismicFilename, size);
	int wsize = le_outMeanWindowSize->text().toInt();
	strcpy(rgbFilename, "/data/PLI/jacques/testMean.raw");
	strcpy(isoFilename, "");

	if ( FILEIO2::exist(seismicFilename) && FILEIO2::exist(rgtFilename) && retsize )
	{
		int *tab_gpu = NULL, tab_gpu_size;
		tab_gpu = (int*)calloc(this->systemInfo->get_gpu_nbre(), sizeof(int));
		this->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
		GLOBAL_RUN_TYPE = 1;
		qbp_seismicMeanStart->setText("stop");

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
		p->setOutputType(1);
		p->run();
		delete p;

		GLOBAL_RUN_TYPE = 0;
		pushbutton_startstop->setText("start");
		// m_selectorWidget->global_rgb_database_update();
	}

/*
	rgb2FilenameUpdate(depth);
	rgb2LabelUpdate(rgb2TinyName);

	if ( seismicFullName.compare("") == 0 || rgtFullName.compare("") == 0 ||
			rgb2FullName.compare("") == 0  || isoFullName.compare("") == 0 ||
			lineedit_f1->text().compare("") == 0 || lineedit_f2->text().compare("") == 0 || lineedit_f3->text().compare("") == 0 ) return;



	if ( idxHorizon == 1 && horizonFullname.size() != 2 ) return;


	frequencyTableWidgetRead(arrayFreq, arrayIso, &Freqcount);

	char seismicFilename[10000], rgtFilename[10000], rgbFilename[10000], isoFilename[10000];

	nbLayers = lineedit_layer_number->text().toInt();
	strcpy(seismicFilename, seismicFullName.toStdString().c_str());
	strcpy(rgtFilename, rgtFullName.toStdString().c_str());
	strcpy(rgbFilename, rgb2FullName.toStdString().c_str());
	strcpy(isoFilename, isoFullName.toStdString().c_str());
	getTraceParameter(seismicFilename, &pasech, &tdeb);
	int retsize = getSizeFromFilename(seismicFilename, size);

	if ( idxHorizon == 1 )
		horizonRead(horizonTinyName, horizonFullname, size[1], size[0], size[2], pasech, tdeb, &horizon1, &horizon2);

	int wsize = lineedit_wsize->text().toInt();


	if ( FILEIO2::exist(seismicFilename) && FILEIO2::exist(rgtFilename) && retsize )
	{
	    int *tab_gpu = NULL, tab_gpu_size;
	    tab_gpu = (int*)calloc(this->systemInfo->get_gpu_nbre(), sizeof(int));
	    this->systemInfo->get_valid_gpu(tab_gpu, &tab_gpu_size);
		GLOBAL_RUN_TYPE = 1;
		pushbutton_startstop->setText("stop");

		// f_rgt2rgb_2(seismicFilename, rgtFilename, tdeb,  pasech,
		//		0, 32000, isostep, horizon1, horizon2, nbLayers, wsize, arrayFreq, arrayIso, Freqcount, rgbFilename, isoFilename);
		// return;

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
		p->run();
		delete p;

		GLOBAL_RUN_TYPE = 0;
		pushbutton_startstop->setText("start");
		m_selectorWidget->global_rgb_database_update();
	}

	if ( horizon1 ) free(horizon1);
	if ( horizon2 ) free(horizon2);
	*/
}

bool SpectrumComputeWidget::checkSeismicsSizeMatch(const QString& cube1, const QString& cube2) {
	if ( cube1.isNull() || cube1.isEmpty() || cube2.isNull() || cube2.isEmpty()) {
		return false;
	}
	inri::Xt xt1(cube1.toStdString().c_str());
	if (!xt1.is_valid()) {
		std::cerr << "xt cube is not valid (" << cube1.toStdString() << ")" << std::endl;
		return false;
	}
	inri::Xt xt2(cube2.toStdString().c_str());
	if (!xt2.is_valid()) {
		std::cerr << "xt cube is not valid (" << cube2.toStdString() << ")" << std::endl;
		return false;
	}

	bool match = xt1.nRecords()==xt2.nRecords() && xt1.nSamples()==xt2.nSamples() && xt1.nSlices()==xt2.nSlices() &&
			xt1.stepRecords()==xt2.stepRecords() && xt1.stepSamples()==xt2.stepSamples() &&
			xt1.stepSlices()==xt2.stepSlices() && xt1.nRecords()==xt2.nRecords() && xt1.nSamples()==xt2.nSamples() &&
			xt1.nSlices()==xt2.nSlices();
	return match;
}

QString SpectrumComputeWidget::getFFMPEGTime(double _time) {
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

QString SpectrumComputeWidget::formatTimeWithMinCharacters(double time, int minCharNumber) {
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

void SpectrumComputeWidget::trt_directionChanged(int newComboBoxIndex) {
	if (getSeismicTinyName().compare("") == 0) {
		return;
	}

	QVariant dirVariant = directionComboBox->itemData(newComboBoxIndex, Qt::UserRole);
	bool ok;
	int direction = dirVariant.toInt(&ok);
	if (!ok) {
		return;
	}

	int size[3];
	double steps[3];
	double origins[3];
	bool isValid = getPropertiesFromDatasetPath(getSeismicFullName(), size, steps, origins);
	if (!isValid) {
		return;
	}

	// inline
	int axisIndex = 2;
	if (direction == 1) {
		// xline
		axisIndex = 0;
	}
	int axisStart = origins[axisIndex];
	int axisStep = steps[axisIndex];
	int axisEnd = axisStart + (size[axisIndex]-1) * axisStep;

	QSignalBlocker bSlider(sectionPositionSlider);
	QSignalBlocker bSpinBox(sectionPositionSpinBox);
	sectionPositionSlider->setMinimum(axisStart);
	sectionPositionSlider->setMaximum(axisEnd);
	sectionPositionSlider->setSingleStep(axisStep);
	sectionPositionSlider->setValue(axisStart);

	sectionPositionSpinBox->setMinimum(axisStart);
	sectionPositionSpinBox->setMaximum(axisEnd);
	sectionPositionSpinBox->setSingleStep(axisStep);
	sectionPositionSpinBox->setValue(axisStart);
}

void SpectrumComputeWidget::trt_sectionIndexChanged(int sectionIndex) {
	QSignalBlocker bSlider(sectionPositionSlider);
	QSignalBlocker bSpinBox(sectionPositionSpinBox);

	sectionPositionSlider->setValue(sectionIndex);
	sectionPositionSpinBox->setValue(sectionIndex);
}

void SpectrumComputeWidget::trt_changeAviTextColor() {
	QColorDialog dialog(textColor);
	int errCode = dialog.exec();
	if (errCode==QDialog::Accepted) {
		setAviTextColor(dialog.selectedColor());
	}
}

void SpectrumComputeWidget::setAviTextColor(const QColor& color) {
	textColor = color;
	colorHolder->setStyleSheet(QString("QPushButton{ color: %1; background-color: %1; }").arg(color.name()));
}

MyThreadSpectrumCompute::MyThreadSpectrumCompute(SpectrumComputeWidget *p)
 {
     this->pp = p;
 }

 void MyThreadSpectrumCompute::run()
 {
	 switch ( pp->m_functionType )
	 {
	 case 0: break;
	 case 1: pp->trt_rgt2rgbStartStop(); break;
	 case 2: pp->trt_rgb2rgb1StartStop();break;
	 case 3: pp->trt_rgb1toxtStartStop(); break;
	 case 4: pp->trt_seismicMeanStartStop(); break;
	 }
	 pp->m_functionType = 0;
 }

QFileInfoList DialogSp::get_dirlist(QString path)
{
    QDir dir(path);
    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);
    QStringList filters;
    filters << "*.xt" << "*.cwt";
    dir.setNameFilters(filters);
    QFileInfoList list = dir.entryInfoList();
    return list;
}

DialogSp::DialogSp(QString path, char *out, QWidget *parent)
 {
	 pparent = NULL;
    this->pout = out;
    if ( this->pout ) this->pout[0] = 0;
   	setWindowTitle(tr("Choose a file"));
    QGroupBox* qgb_textinfo = new QGroupBox(this);
	qgb_textinfo->setTitle("");
	qgb_textinfo->setGeometry(QRect(10, 10, 680, 450));

    QVBoxLayout* qhb_textinfo = new QVBoxLayout(qgb_textinfo);
    // QLabel *label_title = new QLabel("File open");
    QLineEdit *lineedit_search = new QLineEdit();
	textInfo = new QListWidget();

    QHBoxLayout* qhb1 = new QHBoxLayout(qgb_textinfo);//(qgb_orientation);
    qpb_cancel = new QPushButton("cancel");
    qpb_ok = new QPushButton("OK");
    qhb1->addWidget(qpb_cancel);
    qhb1->addWidget(qpb_ok);

    // qhb_textinfo->addWidget(label_title);
    // qhb_textinfo->addWidget(lineedit_search);
	qhb_textinfo->addWidget(textInfo);
    qhb_textinfo->addLayout(qhb1);
	QVBoxLayout* mainLayout4 = new QVBoxLayout(this);
	mainLayout4->addLayout(qhb_textinfo);

    textInfo->clear();
    QFileInfoList list = get_dirlist(path);
    int idx = 0;
    for (int i=0; i<list.size(); i++)
    {
         textInfo->addItem(list[i].fileName());
         // if ( idx%2 == 0 ) textInfo->item(idx++)->setForeground(Qt::white);
         // else textInfo->item(idx++)->setForeground(Qt::gray);
         // else textInfo->item(idx++)->setForeground(QColor(200, 100, 100));
    }
    connect(qpb_ok, SIGNAL(clicked()), this, SLOT(accept()));
    connect(qpb_cancel, SIGNAL(clicked()), this, SLOT(cancel()));
    connect(textInfo, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(listviewdoubleclick(QListWidgetItem*)));
 }

 DialogSp::DialogSp(std::vector<QString> list, char *out, QWidget *parent)
 {
	 pparent = NULL;
    this->pout = out;
    if ( this->pout ) this->pout[0] = 0;
   	setWindowTitle(tr("Choose a file"));
    QGroupBox* qgb_textinfo = new QGroupBox(this);
	qgb_textinfo->setTitle("");
	qgb_textinfo->setGeometry(QRect(10, 10, 680, 450));

    QVBoxLayout* qhb_textinfo = new QVBoxLayout(qgb_textinfo);
    // QLabel *label_title = new QLabel("File open");
    QLineEdit *lineedit_search = new QLineEdit();
	textInfo = new QListWidget();

    QHBoxLayout* qhb1 = new QHBoxLayout(qgb_textinfo);//(qgb_orientation);
    qpb_cancel = new QPushButton("cancel");
    qpb_ok = new QPushButton("OK");
    qhb1->addWidget(qpb_cancel);
    qhb1->addWidget(qpb_ok);

    // qhb_textinfo->addWidget(label_title);
    // qhb_textinfo->addWidget(lineedit_search);
	qhb_textinfo->addWidget(textInfo);
    qhb_textinfo->addLayout(qhb1);
	QVBoxLayout* mainLayout4 = new QVBoxLayout(this);
	mainLayout4->addLayout(qhb_textinfo);

    textInfo->clear();
    // QFileInfoList list = get_dirlist(path);
    int idx = 0;
    for (int i=0; i<list.size(); i++)
    {
         textInfo->addItem(list[i]);
    }
    connect(qpb_ok, SIGNAL(clicked()), this, SLOT(accept()));
    connect(qpb_cancel, SIGNAL(clicked()), this, SLOT(cancel()));
    connect(textInfo, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(listviewdoubleclick(QListWidgetItem*)));
 }


 DialogSp::DialogSp(QString title, QString msg, char *out, QWidget *parent)
 {
	 /*
    this->pout = out;
    if ( this->pout ) this->pout[0] = 0;
   	setWindowTitle(title);
   	setGeometry(QRect(0, 0, 700, 500));


   	QVBoxLayout * mainLayout=new QVBoxLayout(this);



   	QHBoxLayout* sessionLayout = new QHBoxLayout;
   	QDialogButtonBox *buttonBox = new QDialogButtonBox(
   	QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
   	// connect(buttonBox, SIGNAL(accepted()), this, &QDialog::);
   	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
   	sessionLayout->addWidget(buttonBox);

   	mainLayout->addLayout(sessionLayout);
   	*/
	 pparent = NULL;
	 this->pout = out;
	     if ( this->pout ) this->pout[0] = 0;
	    	setWindowTitle(tr("Choose a file"));
	     QGroupBox* qgb_textinfo = new QGroupBox(this);
	 	qgb_textinfo->setTitle("");
	 	qgb_textinfo->setGeometry(QRect(10, 10, 680, 450));

	     QVBoxLayout* qhb_textinfo = new QVBoxLayout(qgb_textinfo);
	     // QLabel *label_title = new QLabel("File open");
	     QLineEdit *lineedit_search = new QLineEdit();
	 	textInfo = new QListWidget();

	     QHBoxLayout* qhb1 = new QHBoxLayout(qgb_textinfo);//(qgb_orientation);
	     qpb_cancel = new QPushButton("cancel");
	     qpb_ok = new QPushButton("OK");
	     qhb1->addWidget(qpb_cancel);
	     qhb1->addWidget(qpb_ok);

	     // qhb_textinfo->addWidget(label_title);
	     // qhb_textinfo->addWidget(lineedit_search);
	 	qhb_textinfo->addWidget(textInfo);
	     qhb_textinfo->addLayout(qhb1);
	 	QVBoxLayout* mainLayout4 = new QVBoxLayout(this);
	 	mainLayout4->addLayout(qhb_textinfo);

//	     textInfo->clear();
	     // QFileInfoList list = get_dirlist(path);
//	     connect(qpb_ok, SIGNAL(clicked()), this, SLOT(accept()));
//	     connect(qpb_cancel, SIGNAL(clicked()), this, SLOT(cancel()));
//	     connect(textInfo, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(listviewdoubleclick(QListWidgetItem*)));
 }



 DialogSp::DialogSp(std::vector<QString> list, QWidget *parent)
  {
	 pparent = (SpectrumComputeWidget*)parent;
	 pout = NULL;

     QGroupBox* qgb_textinfo = new QGroupBox(this);
     qgb_textinfo->setTitle("");
     qgb_textinfo->setGeometry(QRect(10, 10, 680, 450));

     QVBoxLayout* qhb_textinfo = new QVBoxLayout(qgb_textinfo);
     // QLabel *label_title = new QLabel("File open");
     QLineEdit *lineedit_search = new QLineEdit();
     textInfo = new QListWidget();

     QHBoxLayout* qhb1 = new QHBoxLayout(qgb_textinfo);//(qgb_orientation);
     qpb_cancel = new QPushButton("cancel");
     qpb_ok = new QPushButton("OK");
     qhb1->addWidget(qpb_cancel);
     qhb1->addWidget(qpb_ok);

     // qhb_textinfo->addWidget(label_title);
     // qhb_textinfo->addWidget(lineedit_search);
     qhb_textinfo->addWidget(textInfo);
     qhb_textinfo->addLayout(qhb1);
     QVBoxLayout* mainLayout4 = new QVBoxLayout(this);
     mainLayout4->addLayout(qhb_textinfo);

     textInfo->clear();
     // QFileInfoList list = get_dirlist(path);
     int idx = 0;
     for (int i=0; i<list.size(); i++)
     {
          textInfo->addItem(list[i]);
     }
     connect(qpb_ok, SIGNAL(clicked()), this, SLOT(accept()));
     connect(qpb_cancel, SIGNAL(clicked()), this, SLOT(cancel()));
     connect(textInfo, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(listviewdoubleclick(QListWidgetItem*)));
  }




 void DialogSp::setMultiselection()
 {
	 textInfo->setSelectionMode(QAbstractItemView::MultiSelection);
 }



 void DialogSp::listviewdoubleclick(QListWidgetItem *item)
 {
    if ( pout != NULL ) strcpy(pout, item->text().toStdString().c_str());
    QWidget::close();
 }

 void DialogSp::accept()
 {
     if ( pout != NULL ) strcpy(pout, textInfo->currentItem()->text().toStdString().c_str());
     else if ( pparent != NULL )
     {
    	 /*
    	 QList<QListWidgetItem*> list0 = textInfo->selectedItems();
    	 pparent->listQStringDialog.resize(list0.size());
    	 for (int i=0; i<list0.size(); i++)
    	 {
    		 pparent->listQStringDialog[i] = list0[i]->text();
    	 }
    	 */
     }
     QWidget::close();
 }

 void DialogSp::cancel()
 {
     if ( pout != NULL ) pout[0] = 0;
     QWidget::close();
 }
