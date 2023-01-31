/*
 * NextMainWindow.cpp
 *
 *  Created on: 22 juin 2018
 *      Author: l0380577
 */

#include <qapplication.h>
#include "NextMainWindow.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QToolButton>
#include <QSpacerItem>

#include <QMessageBox>
#include <QMainWindow>
#include <QDockWidget>
#include <QScrollArea>
#include <QMenuBar>
#include <QSvgWidget>
#include <QSvgRenderer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QDir>
#include <QCoreApplication>

#include "DataSelectorDialog.h"
#include "multitypegraphicsview.h"
#include "geotimegraphicsview.h"
#include "slicerwindow.h"
#include "processwatcherwidget.h"

#include "./widget/WGeotime/FileConvertionXTCWT.h"
#include "./widget/WGeotime/GeotimeConfiguratorWidget.h"
#include "./widget/WGeotime/GradientMultiscaleWidget.h"
#include "./widget/WGeotime/ProjectManagerDatabaseUpdate.h"
#include "./widget/WGeotime/SpectrumComputeWidget.h"
#include "./widget/WGeotime/ProjectManagerWidget.h"

#include "./widget/WPetrophysics/PetrophysicsTools.h"
#include "./widget/WPetrophysics/PetroLogTools.h"

#include <spectrumProcessWidget.h>

#include "./widget/WGeotime/DebugWidget.h"
#include "datamanager.h"
#include "bnnilauncher.h"
#include "GeotimeSystemInfo.h"
#include "trainingsetparameterwidget.h"
#include <videocreatewidget.h>
#include <nvhorizonconvertion.h>
#include "videomanagerwrapper.h"


class WindowWidgetPoper: public QMainWindow, public WidgetPoperTrait {
public:
	void popWidget(QWidget* widget) override {
		QDockWidget *dock = new QDockWidget(widget->windowTitle(), this);

		QScrollArea* scrollWidget = new QScrollArea(dock);
		scrollWidget->setWidget(widget);
		widget->setParent(scrollWidget);
		scrollWidget->setWidgetResizable(true);

		dock->setAllowedAreas(Qt::RightDockWidgetArea);
		dock->setWidget(scrollWidget);
		dock->setAttribute(Qt::WA_DeleteOnClose);
		addDockWidget(Qt::RightDockWidgetArea, dock);
	}
};

NextMainWindow::NextMainWindow(QWidget* parent) :
		QWidget(parent) {

	QWidget* backgroundHolder = new QWidget;
	QHBoxLayout* backgroundLayout = new QHBoxLayout;
	this->setLayout(backgroundLayout);
	backgroundLayout->addWidget(backgroundHolder);
	backgroundHolder->setStyleSheet("QWidget {border-image: url(:/slicer/icons/mainwindow/Background.svg) 0 0 0 0 stretch stretch;}");
	backgroundLayout->setContentsMargins(0, 0, 0, 0);

	QVBoxLayout* mainLayout = new QVBoxLayout;
	backgroundHolder->setLayout(mainLayout);

	mainLayout->addSpacing(2);

	// grid of buttons
	QHBoxLayout* mainIconsLayout = new QHBoxLayout;
	mainLayout->addLayout(mainIconsLayout, 2);
	mainIconsLayout->setContentsMargins(0, 0, 0, 0);
	mainLayout->addStretch(6);
	QString version = getVersion();
	if (!version.isNull() && !version.isEmpty()) {
		QLabel* versionLabel = new QLabel("VERSION : " + version+"\nSupport : Wecare / Next Vision");
		versionLabel->setAlignment(Qt::AlignHCenter);
		versionLabel->setStyleSheet("QLabel {border-image: none; background-color: rgba(0, 0, 0, 0%); color: black; font-size: 12pt; margin: 0px; padding: 0px;}");

		mainLayout->addWidget(versionLabel, 1);
	}
	mainLayout->addStretch(3);


	QHBoxLayout* bottomLayout = new QHBoxLayout;
	mainLayout->addLayout(bottomLayout, 2);
	mainLayout->addSpacing(10);

	bottomLayout->addStretch(3);

	QHBoxLayout* systemCallsLayout = new QHBoxLayout;
	bottomLayout->addLayout(systemCallsLayout, 4);
	bottomLayout->addStretch(1);

	QHBoxLayout* logosLayout = new QHBoxLayout;
	logosLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));
	QSvgWidget* nkDeepLogoWidget = new QSvgWidget(":/slicer/icons/mainwindow/LOGO-NKDEEP.svg");
	nkDeepLogoWidget->setStyleSheet("QWidget {border-image: none; background-color: rgba(150, 150, 150, 0%); margin: 0px; padding: 0px;}");
	nkDeepLogoWidget->setMaximumSize(70,70);
	nkDeepLogoWidget->setMinimumSize(70,70);
	logosLayout->addWidget(nkDeepLogoWidget);
	logosLayout->addSpacing(10);
	QSvgWidget* totalLogoWidget = new QSvgWidget(":/slicer/icons/mainwindow/LOGO-TOTAL.svg");
	totalLogoWidget->setStyleSheet("QWidget {border-image: none; background-color: rgba(150, 150, 150, 0%); margin: 0px; padding: 0px;}");
	totalLogoWidget->setMaximumSize(70,70);
	totalLogoWidget->setMinimumSize(70,70);
	logosLayout->addWidget(totalLogoWidget);

	logosLayout->addSpacing(20);
	bottomLayout->addLayout(logosLayout, 2);


	mainIconsLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));

	m_systemInfo = new GeotimeSystemInfo();
	m_rgtButton = initToolButton(":/slicer/icons/mainwindow/GeotimeComputation.svg", "Processing");
	mainIconsLayout->addWidget(m_rgtButton/*, 1, 0*/);

	m_viewButton = initToolButton(":/slicer/icons/mainwindow/GeotimeView.svg", "RGT View");
	mainIconsLayout->addWidget(m_viewButton/*, 1, 1*/);

	m_conversionButton = initToolButton(":/slicer/icons/mainwindow/FileConversion.svg", "File Conversion");
	mainIconsLayout->addWidget(m_conversionButton/*, 1, 2*/);

	m_dataBaseButton = initToolButton(":/slicer/icons/mainwindow/DataBase.svg", "NV Data Base");
	mainIconsLayout->addWidget(m_dataBaseButton/*, 2, 0*/);

	m_videoButton = initToolButton(":/slicer/icons/mainwindow/SpectrumCompute.svg", "Spectrum Video");
	mainIconsLayout->addWidget(m_videoButton/*, 2, 1*/);

	m_playVideoButton = initToolButton(":/slicer/icons/mainwindow/VideoPlayer.svg", "Play Video");
	mainIconsLayout->addWidget(m_playVideoButton/*, 2, 2*/);

	m_bnniButton = initToolButton(":/slicer/icons/mainwindow/BnniIcon.svg", "BNNI");
	mainIconsLayout->addWidget(m_bnniButton/*, 3, 0*/);

	m_ccusButton = initToolButton(":/slicer/icons/mainwindow/CCUSIcon.svg", "CCUS");
	mainIconsLayout->addWidget(m_ccusButton/*, 3, 1*/);

	m_petropyButton = initToolButton(":/slicer/icons/mainwindow/PetropyIcon.svg", "PETROPY");
	mainIconsLayout->addWidget(m_petropyButton/*, 3, 2*/);

	m_systemInfoButton = initToolButton(":/slicer/icons/Info.svg", "System Info");
	systemCallsLayout->addWidget(m_systemInfoButton);

	m_screenShotButton = initToolButton(":/slicer/icons/mainwindow/captureicon.svg", "Screen Shot");
	systemCallsLayout->addWidget(m_screenShotButton/*, 0, 1*/);

	m_videoCaptureButton = initToolButton(":/slicer/icons/mainwindow/Videoicon.svg", "Video Capture");
	systemCallsLayout->addWidget(m_videoCaptureButton/*, 0, 2*/);

	m_vncButton = initToolButton(":/slicer/icons/mainwindow/iconvnc.svg", "VNC");
	systemCallsLayout->addWidget(m_vncButton/*, 1, 0*/);

	m_calculatriceButton = initToolButton(":/slicer/icons/mainwindow/calculatriceicon.svg", "Calculator");
	systemCallsLayout->addWidget(m_calculatriceButton/*, 0, 0*/);

	mainIconsLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));

	m_processRelay = new ProcessRelay(QCoreApplication::instance());

	initLaunchers();

	this->setMinimumSize(1012, 683);
	this->setMaximumSize(1012, 683);
}

QToolButton* NextMainWindow::initToolButton(const QString& iconPath, const QString& text) {
	QToolButton* toolButton = new QToolButton;
	toolButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	toolButton->setText(text);
	QFont font("Roboto");
	font.setPointSize(10);
	//qDebug() << "pointsize : " << font.pointSize();
	toolButton->setFont(font);
	QPixmap rgtPixmap(iconPath);
	QIcon rgtIcon(rgtPixmap);
	toolButton->setIcon(rgtIcon);
	toolButton->setIconSize(QSize(70, 70));
	toolButton->setStyleSheet("QToolButton {border-image: none; background-color: rgba(150, 150, 150, 0%); color: black; margin: 0px; padding: 0px}");
	toolButton->setMaximumWidth(300);
	toolButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	return toolButton;
}

void NextMainWindow::initLaunchers() {

	/*
	connect(m_rgtButton, &QToolButton::clicked, [this]() {
		GeotimeConfigurationWidget* GeotimeComputation = new GeotimeConfigurationWidget;
		GeotimeComputation->setAttribute(Qt::WA_DeleteOnClose);
		GeotimeComputation->setVisible(true);
		// GeotimeComputation->resize(1100+330, 880); // 500 * 800 native
		GeotimeComputation->resize(850, 1024);
		GeotimeComputation->setProcessRelay(m_processRelay);
		GeotimeComputation->setSystemInfo(m_systemInfo);
	});
	*/

	connect(m_rgtButton, &QToolButton::clicked, this, &NextMainWindow::geotimeProcess);
	connect(m_viewButton, &QToolButton::clicked, this, &NextMainWindow::geotimeLaunch);
	connect(m_playVideoButton, &QToolButton::clicked, this, &NextMainWindow::playVideoLaunch);


//	QPushButton* SlicerWindowButton = new QPushButton("Slicer Window");
//	launchersLayout->addWidget(SlicerWindowButton);
//	connect(SlicerWindowButton, &QPushButton::clicked, [this]() {
//		//TODO Attentiion jamais libéré...
//		//TODO
//		QString newName = QString("SlicerWindow")+QString::number(m_idForWindows);
//		m_idForWindows++;
//		SlicerWindow* mainWindow = new SlicerWindow(newName);
////		if (argc > 2) {
////			std::cout << "Loading seismic:" << argv[1] << std::endl;
////			std::cout << "Loading rgt:" << " " << argv[2] << std::endl;
////			bool b = false;
////			if (b > 3)
////				b = (std::string(argv[3]) == "1");
////
////			mainWindow.open(argv[1], argv[2], b);
////		} else if (argc > 1) {
////			mainWindow.openSismageProject(argv[1]);
////		}
//		mainWindow->show();
//
////		QTimer t;
////		t.setInterval(300);
////		t.setSingleShot(true);
////		QObject::connect(&t, SIGNAL(timeout()), &mainWindow, SLOT(show()));
////		QObject::connect(&t, SIGNAL(timeout()), &splash, SLOT(close()));
////		t.start();
////		retVal = app.exec();
//
//	});

	connect(m_conversionButton, &QToolButton::clicked, [this]() {
		FileConversionXTCWT *p = new FileConversionXTCWT();
		p->setVisible(true);
	});


//	connect(m_playVideoButton, &QToolButton::clicked, [this]() {
//		VideoManagerWrapper* p = new VideoManagerWrapper;
//		p->setVisible(true);
//	});

	connect(m_videoButton, &QToolButton::clicked, this, &NextMainWindow::spectrumVideoLaunch);

	connect(m_dataBaseButton, &QToolButton::clicked, [this]() {
		DataManager* manager = new DataManager();
		manager->setVisible(true);
	});

	connect(m_bnniButton, &QToolButton::clicked, [this]() {
		BnniLauncher* widget = new  BnniLauncher;
		widget->setVisible(true);
	});

/*
	QPushButton *pushbutton_DebugJD = new QPushButton("DEBUG jd");
	launchersLayout->addWidget(pushbutton_DebugJD);
	connect(pushbutton_DebugJD, &QPushButton::clicked, [this]() {
	DebugWidget *p = new DebugWidget();
	p->setVisible(true);});
	*/




/*
	QPushButton *jd_GeotimeDebug = new QPushButton("GEOTIME debug");
			launchersLayout->addWidget(jd_GeotimeDebug);
			connect(jd_GeotimeDebug, &QPushButton::clicked, [this]() {
				GeotimeDebugWidget *p = new GeotimeDebugWidget();
			   	// p->setModal(false);
				// p->exec();
				p->setVisible(true);
							});
*/


		/*
		QPushButton *projectmanagerDataBaseUpdate = new QPushButton("Project manager database update");
		launchersLayout->addWidget(projectmanagerDataBaseUpdate);
			connect(projectmanagerDataBaseUpdate, &QPushButton::clicked, [this]() {
				ProjectmanagerDatabaseUpdate *p = new ProjectmanagerDatabaseUpdate();
					   		// p->setModal(false);
					   		// p->exec();
							p->setVisible(true);
			});
*/


/*
	QPushButton *jd_debug = new QPushButton("debug");
	launchersLayout->addWidget(jd_debug);
	connect(jd_debug, &QPushButton::clicked, [this]() {
		ProjectManagerWidget *p = new ProjectManagerWidget();
	   	// p->setModal(false);
		// p->exec();
		p->setVisible(true);
					});
					*/
	connect(m_calculatriceButton, &QToolButton::clicked, [this]() {
		QString calculatriceProgram = "NextVision_calculatrice";
		if (QProcessEnvironment::systemEnvironment().contains(calculatriceProgram)) {
			QProcess* process = new QProcess;
			process->setProcessChannelMode(QProcess::MergedChannels);
			process->setReadChannel(QProcess::StandardOutput);
			process->setProgram(QProcessEnvironment::systemEnvironment().value(calculatriceProgram));

			connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), [process]() {
				process->deleteLater();
			});

			process->start();
		}
	});

	connect(m_screenShotButton, &QToolButton::clicked, [this]() {
		QString screenShootProgram = "NextVision_screenshot";
		QString screenShootOptions= "NextVision_screenshot_options";
		if (QProcessEnvironment::systemEnvironment().contains(screenShootProgram)) {
			QProcess* process = new QProcess;
			process->setProcessChannelMode(QProcess::MergedChannels);
			process->setReadChannel(QProcess::StandardOutput);
			process->setProgram(QProcessEnvironment::systemEnvironment().value(screenShootProgram));

			if (QProcessEnvironment::systemEnvironment().contains(screenShootOptions)) {
				QStringList options = QProcessEnvironment::systemEnvironment().value(screenShootOptions).split(" ");
				process->setArguments(options);
			}

			connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), [process]() {
				process->deleteLater();
			});

			process->start();
		}
	});
	connect(m_videoCaptureButton, &QToolButton::clicked, [this]() {
		QString videoCaptureProgram = "NextVision_videocapture";
		if (QProcessEnvironment::systemEnvironment().contains(videoCaptureProgram)) {
			QProcess* process = new QProcess;
			process->setProcessChannelMode(QProcess::MergedChannels);
			process->setReadChannel(QProcess::StandardOutput);
			process->setProgram(QProcessEnvironment::systemEnvironment().value(videoCaptureProgram));

			connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), [process]() {
				process->deleteLater();
			});

			process->start();
		}
	});

	connect(m_vncButton, &QToolButton::clicked, [this]() {
		QString vncProgram = "NextVision_vnc";
		QString vncOptions = "NextVision_vnc_options";
		if (QProcessEnvironment::systemEnvironment().contains(vncProgram)) {
			ProcessWatcherWidget* process = new ProcessWatcherWidget;
			process->setAttribute(Qt::WA_DeleteOnClose);
			process->setWindowTitle("VNC");
			process->show();
			//process->setProcessChannelMode(QProcess::MergedChannels);
			//process->setReadChannel(QProcess::StandardOutput);

			QStringList options;
			if (QProcessEnvironment::systemEnvironment().contains(vncOptions)) {
				options = QProcessEnvironment::systemEnvironment().value(vncOptions).split(" ");
			}

			connect(process, &ProcessWatcherWidget::processEnded, [process]() {
				process->deleteLater();
			});

			connect(process, &ProcessWatcherWidget::processGotError, [process]() {
				process->deleteLater();
			});

			process->launchProcess(QProcessEnvironment::systemEnvironment().value(vncProgram), options,
					QDir::home().absolutePath(), QProcessEnvironment::systemEnvironment());
		}
	});

	connect(m_systemInfoButton, &QToolButton::clicked, this, &NextMainWindow::openSystemInfo);
	connect(m_petropyButton, &QToolButton::clicked, this, [this]() {
		PetrophysicsTools* w = new  PetrophysicsTools;
		w->setVisible(true);
		//PetroLogTools *wp = new PetroLogTools;
		//wp->setVisible(true);
	});
}

NextMainWindow::~NextMainWindow() {
	qApp->exit();
}

QMenuBar* NextMainWindow::menuBar() {
	return innerMenuBar;
}

void NextMainWindow::geotimeLaunch() {
	WorkingSetManager* manager = new WorkingSetManager(this);

	std::vector<QString> horizonPaths;
	std::vector<QString> horizonNames;
	DataSelectorDialog* dialog = new DataSelectorDialog(this, manager);
	dialog->resize(550*2, 950);
	int code = dialog->exec();
	if (code==QDialog::Accepted) {
		// NVHorizonConvertion::convertion(dialog->getSelectorWidget());
		GeotimeGraphicsView *win = new GeotimeGraphicsView(
				manager, QString("GeotimeView")+QString::number(m_idForWindows), this);
		m_idForWindows++;
		win->setWindowTitle("Geotime View : "+dialog->getSelectorWidget()->get_projet_name()+" - "+
				dialog->getSelectorWidget()->get_survey_name());
		win->setVisible(true);
		win->resize(800, 500);

		manager->setParent(win);

		win->setDataSelectorDialog(dialog);
		win->setProcessRelay(m_processRelay);
		dialog->setParent(win, Qt::Dialog);
	}
	//TODO registerWindow(win);
}


void NextMainWindow::geotimeProcess()
{
	GeotimeConfigurationWidget* GeotimeComputation = new GeotimeConfigurationWidget(nullptr);
	GeotimeComputation->setAttribute(Qt::WA_DeleteOnClose);
	GeotimeComputation->setVisible(true);
	// GeotimeComputation->resize(1100+330, 880); // 500 * 800 native
	GeotimeComputation->resize(850, 1024);
	GeotimeComputation->setProcessRelay(m_processRelay);
	GeotimeComputation->setSystemInfo(m_systemInfo);
}


void NextMainWindow::spectrumVideoLaunch()
{
	VideoCreateWidget *p = new VideoCreateWidget(nullptr);
	p->setVisible(true);
	p->setAttribute(Qt::WA_DeleteOnClose);
}

void NextMainWindow::playVideoLaunch()
{
	VideoManagerWrapper* p = new VideoManagerWrapper(nullptr);
	p->setVisible(true);

	/*
	WorkingSetManager* manager = new WorkingSetManager(this);

	std::vector<QString> horizonPaths;
	std::vector<QString> horizonNames;
	DataSelectorDialog* dialog = new DataSelectorDialog(this, manager, 1);
	dialog->resize(550*2, 950);
	int code = dialog->exec();

	if (code==QDialog::Accepted) {
		VideoManagerWrapper *p = new VideoManagerWrapper(dialog->getSelectorWidget2());
		p->setVisible(true);
		p->setAttribute(Qt::WA_DeleteOnClose);
		// manager->setParent(GeotimeComputation);
		dialog->setParent(p, Qt::Dialog);
	}
	*/
}

void NextMainWindow::openSystemInfo() {
	m_systemInfo->show();
}

QString NextMainWindow::getVersion() {
	QString version;

	QString appFile = QGuiApplication::applicationFilePath();
	QDir dir = QFileInfo(appFile).dir();
	bool valid = dir.cdUp();
	if (valid) {
		QString versionPath = dir.absoluteFilePath("version.txt");
		QFile file(versionPath);
		if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			QTextStream stream(&file);
			// ignore first line
			stream.readLine();

			// remove excessive spaces
			QString versionLine = stream.readLine().simplified();

			// Expect line : "NextVision A.B.C"
			QStringList list = versionLine.split(" ");
			if (list.count()>1) {
				QStringList reducedList = list;
				reducedList.removeFirst();
				version = list.join(" ");
			} else if (list.count()==1) {
				version = list[0];
			}
		}
	}
	return version;
}
