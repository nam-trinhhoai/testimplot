#include "basemapwindow.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <QAction>
#include <QFileDialog>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QMenu>
#include <QMenuBar>
#include <QLabel>
#include <QProgressBar>
#include <QStatusBar>
#include <QSplitter>
#include <QDebug>

#include "qglgdalfullimage.h"
#include "qglfullimageitem.h"
#include "qglgdaltiledimage.h"
#include "qgltiledimageitem.h"

#include "basemapqglgraphicsview.h"
#include "palettewidget.h"
#include "ogrloader.h"
#include "qglimagefilledhistogramitem.h"
#include "qglimagegriditem.h"
#include "qglgriditem.h"
#include "qglscalebaritem.h"
#include "qglsymbolitem.h"

#include "qglcolorbar.h"



BaseMapWindow::BaseMapWindow(QWidget *parent) :
		QMainWindow(parent), m_view(0), m_scene(0) {
	m_item = nullptr;
	m_image = nullptr;

	m_palette = new PaletteWidget(this);
	QSplitter *splitter = new QSplitter(this);
	splitter->setStyleSheet("background-color:#19232D;");
	splitter->addWidget(m_palette);

	m_scene = new QGraphicsScene();
	m_view = new BaseMapQGLGraphicsView(this);
	m_view->setStyleSheet("border: 0px solid;");
	m_view->setScene(m_scene);


	splitter->addWidget(m_view);
	splitter->setStretchFactor(0, 0.1);
	splitter->setStretchFactor(1, 100);

	setCentralWidget(splitter);

	createActions();
	createMenus();

	connect(m_view, SIGNAL(mouseMoved(const QRectF&,double ,double )), this,
			SLOT(mouseMoved(const QRectF&,double ,double)));
}

BaseMapWindow::~BaseMapWindow() {
}

void BaseMapWindow::mouseMoved(const QRectF& visibleArea,double worldX, double worldY) {
	if(m_item==nullptr)
		return;

	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) << "World: [" << worldX << ","
			<< worldY << "]";
	if (m_item != nullptr) {
		if (m_item->boundingRect().contains(QPointF(worldX, worldY))) {

			double value;
			int i, j;
			if (((QGLAbstractImage*)m_image)->value(worldX, worldY, i, j, value))
			{
				ss << " [" << std::setprecision(0) << i << "," << j << "] "
						<< value;
			}


		}
	}
	statusBar()->showMessage(tr(ss.str().c_str()));
}

void BaseMapWindow::openSimpleImage() {

	if (m_item != nullptr) {
		m_scene->removeItem(m_item);
		m_item = nullptr;
	}

	if (m_image != nullptr) {
		m_image->close();
		m_image = nullptr;
	}

	QString filePath = QFileDialog::getOpenFileName(this, tr("Open an image"),
			"", // current dir
			tr("File(*.*)"));
	if (!filePath.isEmpty()) {
		m_image = new QGLGDALFullImage(this);
		if (!m_image->open(filePath)) {
			m_image = nullptr;
			return;
		}
		m_item = new QGLFullImageItem(m_image);

//		m_image = new QGLGDALTiledImage(this);
//		if (!m_image->open(filePath)) {
//			m_image = nullptr;
//			return;
//		}
//		m_item = new QGLTiledImageItem(m_image, this);


//		std::cout<<"Initial image size:"<<m_image->width()<<"\t"<<m_image->height()<<std::endl;
//		QGLGridUtil grided(m_image,m_image->width(),m_image->height());
//		grided.dump();


		//std::cout<<m_item->boundingRect().width()<<std::endl;
		QGLGridItem *baseGrid=new QGLGridItem(m_item->boundingRect());
		m_scene->addItem(baseGrid);

		m_scene->addItem(m_item);

		connect(m_palette, SIGNAL(rangeChanged(const QVector2D &)), m_image,
				SLOT(setRange(const QVector2D &)));
		connect(m_palette, SIGNAL(opacityChanged(float)), m_image,
				SLOT(setOpacity(float)));
		connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
				m_image, SLOT(setLookupTable(const LookupTable &)));

		connect(m_palette, SIGNAL(rangeChanged(const QVector2D &)), this,
				SLOT(refresh()));
		connect(m_palette, SIGNAL(opacityChanged(float)), this,
				SLOT(refresh()));
		connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
				this, SLOT(refresh()));


		QGLColorBar *item=new QGLColorBar(m_item->boundingRect());
		item->setLookupTable(m_image->lookupTable());

		connect(m_palette, SIGNAL(rangeChanged(const QVector2D &)), item,
						SLOT(setRange(const QVector2D &)));
		connect(m_palette, SIGNAL(opacityChanged(float)), item,
				SLOT(setOpacity(float)));
		connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
				item, SLOT(setLookupTable(const LookupTable &)));


		QGLImageFilledHistogramItem *histo=new QGLImageFilledHistogramItem(m_image,m_image);
		m_palette->setLookupTable(m_image->lookupTable());
		m_palette->setPaletteHolder(m_image);

		QGLImageGridItem * grid=new QGLImageGridItem(m_image);
		m_scene->addItem(grid);
		m_scene->addItem(histo);
		m_scene->addItem(item);

		QGLScaleBarItem *scale=new QGLScaleBarItem(m_item->boundingRect());
		m_scene->addItem(scale);

		double p0,p1;
		m_image->imageToWorld(0,0,p0,p1);

//		QGLSymbolItem * test=new QGLSymbolItem(QPointF(p0,p1),'O',m_item->boundingRect(),this);
//		m_scene->addItem(test);

		m_view->resetZoom();


	}
}

void BaseMapWindow::openShapefile() {
	QString filePath = QFileDialog::getOpenFileName(this, tr("Open a shapefile"),
			"", // current dir
			tr("File(*.*)"));
	if (filePath.isEmpty())
		return;

	QList<QGraphicsItem *> items;
	OGRLoader::loadFile(filePath.toStdString(),items);
	m_scene->createItemGroup(items);

	m_view->resetZoom();
}


void BaseMapWindow::refresh() {
	if (m_item != nullptr)
		m_item->update();
}

void BaseMapWindow::createActions() {
	m_openAction = new QAction(tr("&Open an image"), this);
	m_openAction->setShortcut(tr("Ctrl+O"));
	m_openAction->setStatusTip(tr("Open an image file"));
	connect(m_openAction, SIGNAL(triggered()), this, SLOT(openSimpleImage()));

	m_openShapefileAction = new QAction(tr("&Open a shapefile"), this);
	m_openShapefileAction->setShortcut(tr("Ctrl+O"));
	m_openShapefileAction->setStatusTip(tr("Open an image file"));
	connect(m_openShapefileAction, SIGNAL(triggered()), this, SLOT(openShapefile()));


	m_exitAction = new QAction(tr("E&xit"), this);
	m_exitAction->setShortcut(tr("Ctrl+Q"));
	m_exitAction->setStatusTip(tr("Exit the application"));
	connect(m_exitAction, SIGNAL(triggered()), this, SLOT(close()));
}

void BaseMapWindow::createMenus(void) {
	m_fileMenu = menuBar()->addMenu(tr("&File"));
	m_fileMenu->addAction(m_openAction);
	m_fileMenu->addAction(m_openShapefileAction);
	m_fileMenu->addSeparator();
	m_fileMenu->addAction(m_exitAction);
}

