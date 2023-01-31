#include "slicerwindow.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <QApplication>
#include <QToolBar>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QElapsedTimer>

#include <QFileDialog>
#include <QOpenGLContext>

#include <QHBoxLayout>
#include <QLabel>
#include <QSplitter>
#include <QToolButton>

#include <QListWidget>
#include <QListWidgetItem>
#include <QGroupBox>
#include <QSystemTrayIcon>
#include <QInputDialog>

#include "sliceutils.h"
#include "basemapview.h"
#include "Xt.h"

#include "seismicsurvey.h"
#include "seismic3ddataset.h"
#include "seismic3dcudadataset.h"

#include "DataSelectorDialog.h"
#include "workingsetmanager.h"
#include "mousetrackingevent.h"
#include "affine2dtransformation.h"
#include "smdataset3D.h"
#include "slicerep.h"
#include "smsurvey3D.h"
#include "stratislice.h"

#include "basemapgraphicsview.h"
#include "slavesectionview.h"
#include "sectiongraphicsview.h"
#include "view3dgraphicsview.h"
#include "multitypegraphicsview.h"

SlicerWindow *SlicerWindow::m_mainWindow = nullptr;

SlicerWindow::SlicerWindow(QString uniqueName, QWidget *parent) :
		QMainWindow(parent), m_uniqueName(uniqueName) {
	m_manager = new WorkingSetManager(this);

	setWindowTitle(tr("RGT seismic slicer"));

	QAction *openSismageAction = new QAction(tr("&Open Sismage Project"), this);
	openSismageAction->setShortcut(tr("Ctrl+O"));
	connect(openSismageAction, SIGNAL(triggered()), this,
			SLOT(openSismageProject()));

	QAction *openAction = new QAction(
			tr("&Open RGT demo (seismic and rgt volumes)"), this);
	openAction->setStatusTip(tr("Open RGT demo"));
	connect(openAction, SIGNAL(triggered()), this, SLOT(open()));

	QAction *openManagerAction = new QAction(tr("&Open Project Data Manager"), this);
	openManagerAction->setStatusTip(tr("Open Project Data Manager"));
	connect(openManagerAction, SIGNAL(triggered()), this, SLOT(openManager()));

	QAction *exitAction = new QAction(tr("E&xit"), this);
	exitAction->setShortcut(tr("Ctrl+Q"));
	exitAction->setStatusTip(tr("Exit the application"));

	QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(openSismageAction);
	fileMenu->addAction(openAction);
	fileMenu->addAction(openManagerAction);
	fileMenu->addSeparator();
	fileMenu->addAction(exitAction);

	QToolBar *tb = addToolBar(tr("Viewers"));
	tb->setIconSize(QSize(32, 32));
	tb->setToolButtonStyle(Qt::ToolButtonIconOnly);
	tb->setStyleSheet("QToolButton:!hover {background-color:#c0c0c0} QToolBar {background: #32414B}");
	//tb->setStyleSheet("QToolBar {background: #32414B}");
	QMenu *menu = menuBar()->addMenu(tr("&Viewers"));

	m_basemapAction = new QAction(QIcon(":/slicer/icons/map_gray.png"),
			tr("BaseMap view"), this);
	connect(m_basemapAction, SIGNAL(triggered()), this,
			SLOT(openBaseMapWindow()));
	m_basemapAction->setToolTip("BaseMap view");
	menu->addAction(m_basemapAction);
	tb->addAction(m_basemapAction);

	m_inlineAction = new QAction(QIcon(":/slicer/icons/inline_gray.png"),
			tr("Inline view"), this);

	connect(m_inlineAction, SIGNAL(triggered()), this,
			SLOT(openInlineWindow()));
	m_inlineAction->setToolTip("Inline view");
	menu->addAction(m_inlineAction);
	tb->addAction(m_inlineAction);

	m_xlineAction = new QAction(QIcon(":/slicer/icons/xline_gray.png"),
			tr("Xline view"), this);


	connect(m_xlineAction, SIGNAL(triggered()), this, SLOT(openXlineWindow()));
	m_xlineAction->setToolTip("Xline view");
	menu->addAction(m_xlineAction);
	tb->addAction(m_xlineAction);

	m_qt3dAction = new QAction(QIcon(":/slicer/icons/3d_gray.png"), tr("3D view"),
			this);
	m_qt3dAction->setToolTip("3D view");
	menu->addAction(m_qt3dAction);
	tb->addAction(m_qt3dAction);
	connect(m_qt3dAction, SIGNAL(triggered()), this, SLOT(openView3DWindow()));


	m_multiViewAction = new QAction(QIcon(":/slicer/icons/multi_gray.png"), tr("Multi view"),
			this);
	m_multiViewAction->setToolTip("Multi view");
	menu->addAction(m_multiViewAction);
	tb->addAction(m_multiViewAction);
	connect(m_multiViewAction, SIGNAL(triggered()), this, SLOT(openMultiViewWindow()));


	connect(m_manager, SIGNAL(dataAdded(IData *)), this,
			SLOT(dataAdded(IData *)));
	connect(m_manager, SIGNAL(dataRemoved(IData *)), this,
			SLOT(dataRemoved(IData *)));

	QWidget *paramWidget = new QGroupBox("Working set", this);
	//paramWidget->setStyleSheet("background-color:#4F4F4F;");
	//paramWidget->setContentsMargins(0, 0, 0, 0);

	QVBoxLayout *layout = new QVBoxLayout(paramWidget);
	layout->addWidget(createWorkingSetView());

	QLabel *iconLabel = new QLabel("", this);
	QIcon logo(":/slicer/icons/logoTitle.png");
	iconLabel->setPixmap(logo.pixmap(QSize(112, 16)));
	layout->addWidget(iconLabel, 0, Qt::AlignRight);

	setCentralWidget(paramWidget);
	setMinimumSize(500, 500);
	m_mainWindow = this;

	createTrayActions();
	createTrayIcon();
	m_trayIcon->show();
}

QVector<AbstractGraphicsView*> SlicerWindow::currentViewerList() const {
	return m_registredViewers;
}

SlicerWindow* SlicerWindow::get() {
	return m_mainWindow;
}


void SlicerWindow::openMultiViewWindow()
{
	MultiTypeGraphicsView *view = new MultiTypeGraphicsView(m_manager, getNewUniqueNameForView(), this);
	view->setVisible(true);
	view->resize(800, 500);

	registerWindow(view);

}
void SlicerWindow::openView3DWindow() {
	View3DGraphicsView *view = new View3DGraphicsView(m_manager, getNewUniqueNameForView(), this);
	view->setVisible(true);
	view->resize(800, 500);

	registerWindow(view);
}

void SlicerWindow::dataAdded(IData *d) {
	QListWidgetItem *item = new QListWidgetItem(d->name(), m_listView);
	item->setData(Qt::UserRole, QVariant::fromValue(d));
	m_listView->addItem(item);
}

void SlicerWindow::dataRemoved(IData *d) {
	for (int i = 0; i < m_listView->count(); i++) {
		QListWidgetItem *item = m_listView->item(i);
		if (d == item->data(Qt::UserRole).value<IData*>()) {
			m_listView->removeItemWidget(item);
			return;
		}
	}
}

QWidget* SlicerWindow::createWorkingSetView() {
	m_listView = new QListWidget(this);
	for (IData *d : m_manager->data()) {
		QListWidgetItem *item = new QListWidgetItem(d->name(), m_listView);
		item->setData(Qt::UserRole, QVariant::fromValue(d));
		m_listView->addItem(item);
	}
	return m_listView;
}

void SlicerWindow::openBaseMapWindow() {
	BaseMapGraphicsView *view = new BaseMapGraphicsView(m_manager,
			getNewUniqueNameForView(), this);
	view->setVisible(true);
	view->resize(800, 500);

	registerWindow(view);
}

void SlicerWindow::openInlineWindow() {
	SectionGraphicsView *view = new SectionGraphicsView(ViewType::InlineView,
			m_manager, getNewUniqueNameForView(), this);
	view->setVisible(true);
	view->resize(800, 500);

	registerWindow(view);
}

void SlicerWindow::openXlineWindow() {
	SectionGraphicsView *view = new SectionGraphicsView(ViewType::XLineView,
			m_manager, getNewUniqueNameForView(), this);
	view->setVisible(true);
	view->resize(800, 500);

	registerWindow(view);
}

void SlicerWindow::viewMouseMoved(MouseTrackingEvent *event) {
	for (AbstractGraphicsView *view : m_registredViewers) {
		if (view == sender() || view->isMinimized())
			continue;
		QApplication::postEvent(view, new MouseTrackingEvent(*event));
	}
}
void SlicerWindow::registerWindowControlers(AbstractGraphicsView *newView,
		AbstractGraphicsView *existingView) {
	QList<DataControler*> controlers = existingView->getControlers();
	for (DataControler *c : controlers)
		newView->addExternalControler(c);

//Link controler events in both direction
	connect(newView, SIGNAL(controlerActivated(DataControler *)), existingView,
			SLOT(addExternalControler(DataControler *)));
	connect(newView, SIGNAL(controlerDesactivated(DataControler *)),
			existingView, SLOT(removeExternalControler(DataControler *)));

	connect(existingView, SIGNAL(controlerActivated(DataControler *)), newView,
			SLOT(addExternalControler(DataControler *)));
	connect(existingView, SIGNAL(controlerDesactivated(DataControler *)),
			newView, SLOT(removeExternalControler(DataControler *)));
}

void SlicerWindow::unregisterWindowControlers(AbstractGraphicsView *toBeDeleted,
		AbstractGraphicsView *existingView) {
	QList<DataControler*> controlers = toBeDeleted->getControlers();
	for (DataControler *c : controlers)
		existingView->removeExternalControler(c);

	disconnect(toBeDeleted, SIGNAL(controlerActivated(DataControler *)),
			existingView, SLOT(addExternalControler(DataControler *)));
	disconnect(toBeDeleted, SIGNAL(controlerDesactivated(DataControler *)),
			existingView, SLOT(removeExternalControler(DataControler *)));

	disconnect(existingView, SIGNAL(controlerActivated(DataControler *)),
			toBeDeleted, SLOT(addExternalControler(DataControler *)));
	disconnect(existingView, SIGNAL(controlerDesactivated(DataControler *)),
			toBeDeleted, SLOT(removeExternalControler(DataControler *)));
}

QVector<AbstractGraphicsView*> SlicerWindow::graphicsView() {
	return m_registredViewers;
}

void SlicerWindow::registerWindow(AbstractGraphicsView *newView) {
	QVector<AbstractGraphicsView*> views = graphicsView();

	for (AbstractGraphicsView *existingView : views)
		registerWindowControlers(newView, existingView);

	connect(newView, SIGNAL(viewMouseMoved(MouseTrackingEvent *)), this,
			SLOT(viewMouseMoved(MouseTrackingEvent *)));

	connect(newView, SIGNAL(isClosing(AbstractGraphicsView *)), this,
			SLOT(isClosing(AbstractGraphicsView *)));
	m_registredViewers.push_back(newView);
}

void SlicerWindow::isClosing(AbstractGraphicsView *newView) {
	m_registredViewers.removeOne(newView);

	QVector<AbstractGraphicsView*> views = graphicsView();
	//unregister the controler
	for (AbstractGraphicsView *existingView : views)
		unregisterWindowControlers(newView, existingView);
	disconnect(newView, SIGNAL(viewMouseMoved(MouseTrackingEvent *)), this,
			SLOT(viewMouseMoved(MouseTrackingEvent *)));
}

SlicerWindow::~SlicerWindow() {

}

void SlicerWindow::open() {
	QString rgtPath = "/home/a/rgt.xt";
	QString seismicPath = "/home/a/seismic3d.xt";
	seismicPath = QFileDialog::getOpenFileName(this, tr("Open a seismic image"),
			"", // current dir
			tr("File(*.*)"));
	if (seismicPath.isEmpty())
		return;

	rgtPath = QFileDialog::getOpenFileName(this, tr("Open a rgt image"), "", // current dir
			tr("File(*.*)"));
	if (rgtPath.isEmpty())
		return;

	QString gpu("GPU"), cpu("CPU");
	QStringList strList;
	strList << gpu << cpu;
	bool forceCPU = QInputDialog::getItem(this, "Select CPU/GPU mode", "Mode : ", strList, 0).compare("GPU");
	open(seismicPath.toStdString(), rgtPath.toStdString(), forceCPU); //false);
}

void SlicerWindow::openManager() {

	DataSelectorDialog* dialog = new DataSelectorDialog(this, m_manager);
	dialog->resize(550, 750);
	dialog->exec();
//		delete dialog;
}

void SlicerWindow::openSismageProject() {
	QString projectPath = "/home/a/SISMAGE/DIR_PROJET/ZZ_MiniNkanga";
	projectPath = QFileDialog::getExistingDirectory(this,
			tr("Open a Sismage Project"), projectPath);
	if (projectPath.isEmpty())
		return;

	openSismageProject(projectPath);
}

void SlicerWindow::open(const std::string &seismicPath,
		const std::string &rgtPath, bool forceCPU) {

	int width, depth, heigth;
	{
		inri::Xt xt(seismicPath.c_str());
		if (!xt.is_valid()) {
			std::cerr << "xt cube is not valid (" << seismicPath << ")"
					<< std::endl;
			return;
		}

		width = xt.nRecords();
		depth = xt.nSlices();
		heigth = xt.nSamples();
	}
	SeismicSurvey *baseSurvey = new SeismicSurvey(m_manager, "Survey", width,
			depth, "", this);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	size_t maxGPUMem = prop.totalGlobalMem;

	size_t cubeSize = width * heigth * depth * sizeof(short);

	QFileInfo seismicInfo(seismicPath.c_str());
	QFileInfo rgtInfo(rgtPath.c_str());
	Seismic3DAbstractDataset *seismic;
	Seismic3DAbstractDataset *rgt;
	if (/*cubeSize < maxGPUMem - maxGPUMem * 10 / 100 &&*/ !forceCPU) {
		seismic = new Seismic3DCUDADataset(baseSurvey,
				seismicInfo.completeBaseName(), m_manager,
				Seismic3DDataset::CUBE_TYPE::Seismic, QString::fromStdString(seismicPath), this);
		rgt = new Seismic3DCUDADataset(baseSurvey, rgtInfo.completeBaseName(),
				m_manager, Seismic3DDataset::CUBE_TYPE::RGT, QString::fromStdString(rgtPath), this);
	} else {
		std::cout << "CPU Loading" << std::endl;
		seismic = new Seismic3DDataset(baseSurvey,
				seismicInfo.completeBaseName(), m_manager,
				Seismic3DDataset::CUBE_TYPE::Seismic, QString::fromStdString(seismicPath), this);
		rgt = new Seismic3DDataset(baseSurvey, rgtInfo.completeBaseName(),
				m_manager, Seismic3DDataset::CUBE_TYPE::RGT, QString::fromStdString(rgtPath), this);
	}

	QElapsedTimer timer;
	timer.start();
	seismic->loadFromXt(seismicPath);
	std::cout << "Time to load seismic into memory:"
			<< timer.elapsed() / 1000.0f << " s" << std::endl;
	timer.restart();
	rgt->loadFromXt(rgtPath);
	std::cout << "Time to load rgt into memory:" << timer.elapsed() / 1000.0f
			<< " s" << std::endl;

	baseSurvey->addDataset(seismic);
	baseSurvey->addDataset(rgt);
	m_manager->addSeismicSurvey(baseSurvey);
}

Seismic3DAbstractDataset* SlicerWindow::appendDataset(SeismicSurvey *baseSurvey,
		const std::string &path, bool forceCPU) {
	int width, depth, heigth;
	{
		inri::Xt xt(path.c_str());
		if (!xt.is_valid()) {
			std::cerr << "xt cube is not valid (" << path << ")" << std::endl;
			return nullptr;
		}

		width = xt.nRecords();
		depth = xt.nSlices();
		heigth = xt.nSamples();
	}

	size_t avail;
	size_t total;
	cudaMemGetInfo(&avail, &total);

	QFileInfo seismicInfo(path.c_str());
	QString name = QFileInfo(seismicInfo.completeSuffix()).baseName();
	size_t cubeSize = width * heigth * depth * sizeof(short);
	Seismic3DAbstractDataset *seismic;

	Seismic3DDataset::CUBE_TYPE cubeType =
			name.contains("rgt", Qt::CaseInsensitive) ?
					Seismic3DDataset::CUBE_TYPE::RGT :
					Seismic3DDataset::CUBE_TYPE::Seismic;
	if (cubeSize < avail - avail * 5 / 100 && !forceCPU) {
		seismic = new Seismic3DCUDADataset(baseSurvey, name, m_manager,
				cubeType, seismicInfo.absoluteFilePath(), this);
	} else {
		std::cout << "CPU Loading" << std::endl;
		seismic = new Seismic3DDataset(baseSurvey, name, m_manager, cubeType,
				seismicInfo.absoluteFilePath(), this);
	}
	if (seismic->dimV()==1) {
		seismic->loadFromXt(path);

		baseSurvey->addDataset(seismic);
	} else {
		seismic->deleteLater();
		seismic = nullptr;
	}
	return seismic;
}

void SlicerWindow::openSismageProject(const QString &path) {
//List survey
	QDir survey3DDirectory(path + "/DATA/3D/");
	QFileInfoList surveyList = survey3DDirectory.entryInfoList(
			QDir::Dirs | QDir::NoDotAndDotDot, QDir::SortFlag::Name);
	for (QFileInfo surveyInfo : surveyList) {
		if (!surveyInfo.isDir())
			continue;

		QString surveyName = surveyInfo.fileName();
		qDebug() << "Found survey" << surveyName << " "
				<< surveyInfo.absoluteFilePath();

		SmSurvey3D survey(surveyInfo.absoluteFilePath().toStdString());
		SeismicSurvey *baseSurvey = new SeismicSurvey(m_manager, surveyName,
				survey.inlineDim(), survey.xlineDim(), surveyInfo.absoluteFilePath(), this);

		baseSurvey->setInlineXlineToXYTransfo(survey.inlineXlineToXYTransfo());
		baseSurvey->setIJToXYTransfo(survey.ijToXYTransfo());

		QDir datasetDir(surveyInfo.absoluteFilePath() + "/DATA/SEISMIC/");
		QFileInfoList listDataset = datasetDir.entryInfoList( { "*.xt" },
				QDir::Files | QDir::NoDotAndDotDot, QDir::SortFlag::Name);
		for (QFileInfo datasetInfo : listDataset) {
			QString datasetName = datasetInfo.fileName();
			qDebug() << "Found dataset" << datasetName;

			std::string datasetPath =
					datasetInfo.absoluteFilePath().toStdString();

			{
				inri::Xt xt(datasetPath);
				if (!xt.is_valid()) {
					std::cerr << "Invalid dataset:" << datasetPath << std::endl;
					continue;
				}
			}

			Seismic3DAbstractDataset *dataset = appendDataset(baseSurvey,
					datasetPath, true);
			if (dataset!=nullptr) {
				SmDataset3D d3d(datasetPath);
				dataset->setIJToInlineXlineTransfo(d3d.inlineXlineTransfo());
				dataset->setIJToInlineXlineTransfoForInline(
						d3d.inlineXlineTransfoForInline());
				dataset->setIJToInlineXlineTransfoForXline(
						d3d.inlineXlineTransfoForXline());
				dataset->setSampleTransformation(d3d.sampleTransfo());
			}
		}
		m_manager->addSeismicSurvey(baseSurvey);
	}
}

void SlicerWindow::createTrayActions() {
	m_minimizeAction = new QAction(tr("Mi&nimize Manager"), this);
	connect(m_minimizeAction, &QAction::triggered, this, &QWidget::hide);

	m_maximizeAction = new QAction(tr("Ma&ximize Manager"), this);
	connect(m_maximizeAction, &QAction::triggered, this,
			&QWidget::showMaximized);

	m_restoreAction = new QAction(tr("&Restore Manager"), this);
	connect(m_restoreAction, &QAction::triggered, this, &QWidget::showNormal);

	m_quitAction = new QAction(tr("&Quit Application"), this);
	connect(m_quitAction, &QAction::triggered, qApp, &QCoreApplication::quit);
}

void SlicerWindow::createTrayIcon() {
	m_trayIconMenu = new QMenu(this);

	m_trayIconMenu->addAction(m_basemapAction);
	m_trayIconMenu->addAction(m_inlineAction);
	m_trayIconMenu->addAction(m_xlineAction);
	m_trayIconMenu->addAction(m_qt3dAction);
	m_trayIconMenu->addAction(m_multiViewAction);



	m_trayIconMenu->addSeparator();
	m_trayIconMenu->addAction(m_minimizeAction);
	m_trayIconMenu->addAction(m_maximizeAction);
	m_trayIconMenu->addAction(m_restoreAction);
	m_trayIconMenu->addSeparator();
	m_trayIconMenu->addAction(m_quitAction);

	m_trayIcon = new QSystemTrayIcon(this);
	m_trayIcon->setContextMenu(m_trayIconMenu);
	m_trayIcon->setIcon(QIcon(":/slicer/icons/logoIcon.png"));
}


QString SlicerWindow::getNewUniqueNameForView() {
	QString out = m_uniqueName + "_view" + QString::number(m_uniqueId);
	m_uniqueId++;
	return out;
}
