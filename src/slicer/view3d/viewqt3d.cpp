#include "viewqt3d.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <QCoreApplication>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QToolButton>
#include <QSplitter>
#include <QQuickView>
#include <QQuickWidget>
#include <QQuickItem>
#include <QMenu>
#include <QLabel>
#include <iostream>
#include <QCamera>
#include <QTimer>
#include <QStyle>
#include <QtMath>




#include <QPhongAlphaMaterial>
#include <QDiffuseSpecularMapMaterial>
#include <QTextureMaterial>
#include <Qt3DRender/QBlendEquation>
#include <QMatrix4x4>



#include "idata.h"
#include "graphicsutil.h"
#include "abstractgraphicrep.h"
#include "graphic3Dlayer.h"
#include "layerrgt3Dlayer.h"
#include "qt3dressource.h"
#include "qt3dhelpers.h"
#include "isampledependantrep.h"
#include "cameracontroller.h"
#include "mtlengthunit.h"
#include "itooltipprovider.h"

#include "dockwidgetsizegrid.h"

#include "nurbswidget.h"





class Quickview3d : public QQuickView
{
public:
	Quickview3d(ViewQt3D* view3D):QQuickView( )
	{
		m_cameraController = nullptr;
		m_view3D = view3D;





	}

	~Quickview3d()
	{

	}



	void setCameraController(CameraController* cameraController)
	{
		m_cameraController  = cameraController;
	}

	/*void setVitesse(float v)
	{
		m_vitesse =v;
	}
	void addVitesse(float v)
	{
		m_vitesse +=v;
	}

	float vitesse() const
	{
		return m_vitesse;
	}
*/


protected:



	void wheelEvent(QWheelEvent* event)
	{
		m_view3D->window()->activateWindow();
		m_view3D->setFocus();
		QQuickView::wheelEvent(event);
	}


	bool event(QEvent *event) override
	{

		if (event->type() == QEvent::Enter)
		{
			//qDebug()<<"******EVENT enter : "<<event->type();
			m_view3D->showLineVert(false);

			///m_view3D->m_quickview->window()->activateWindow();
			//m_view3D->enterEvent(event);
			return false;
		}
	    return QQuickView::event(event);
	}



	void mouseReleaseEvent(QMouseEvent* e)override
	{
		QQuickView::mouseReleaseEvent(e);

		if (m_cameraController != nullptr)
		{
			m_cameraController->setTranslationActive(false);
		}

		/*if (e->button() == Qt::RightButton)
		{
			m_view3D->hideTooltipWell();

		}*/

	}



	 void keyPressEvent(QKeyEvent *event)override
	{
		// qDebug()<<" quick view keyPressEvent";
		if(event->key() == Qt::Key_Up)
		 {

			if (m_cameraController != nullptr)
			{
				if(m_view3D->vitesse() < 5.0f) m_view3D->addVitesse(0.1f);
				//qDebug()<<m_vitesse <<"   keyPressEvent :"<<event->isAutoRepeat();
				m_cameraController->setDecalUpDown(0.1f*m_view3D->vitesse()*m_view3D->speedUpDown());
			}
		 }
		 if(event->key() == Qt::Key_Down)
		 {

			if (m_cameraController != nullptr)
			{
				if(m_view3D->vitesse() < 5.0f) m_view3D->addVitesse(0.1f);
				m_cameraController->setDecalUpDown(-0.1f*m_view3D->vitesse()*m_view3D->speedUpDown());
			}
		 }
		 if(event->key() == Qt::Key_V)
			 {
			 	// qDebug()<<" touche V default ok";
				if(m_view3D->widgetpath != nullptr)
				{
					if(m_view3D->widgetpath->m_modeEdition)
					{
						QVector3D position = m_view3D->m_camera->position();

						QVector3D target =m_view3D->m_camera->position() +m_view3D->m_camera->viewVector() ;

						//qDebug()<<" ==>position : "<<position<<" , target : "<<target <<" view : "<<m_view3D->m_camera->viewVector() ;
						m_view3D->widgetpath->AddPoints(position, target, 2000);
					}
				}
			 }

		 if(event->key() == Qt::Key_T)
		 	{

			 m_view3D->m_tooltipActif = true;

		 	}





		QQuickView::keyPressEvent(event);
	}

	virtual void keyReleaseEvent(QKeyEvent *event)override
	{
		if(event->key() == Qt::Key_Up || event->key() == Qt::Key_Down)
		 {

			if (m_cameraController != nullptr)
			{
				//qDebug()<<"keyReleaseEvent :"<<event->isAutoRepeat();
				if(!event->isAutoRepeat())
				{
					 m_view3D->setVitesse(0.5f);
					m_cameraController->setDecalUpDown(0.0f);
				}
			}
		 }
		if(event->key() == Qt::Key_T)
			{

			m_view3D->m_tooltipActif = false;

			}

		QQuickView::keyReleaseEvent(event);
	}

private :
	CameraController* m_cameraController;
	ViewQt3D* m_view3D;


};

ViewQt3D::ViewQt3D(bool restictToMonoTypeSplit, QString uniqueName) :
		AbstractInnerView(restictToMonoTypeSplit, uniqueName) {

	m_qmlLoaded = false;
	m_animRunning= false;
	m_pointer = nullptr;
	m_coneEntity = nullptr;
	m_viewType = ViewType::View3D;
	m_currentScale = 1;
	m_wordBoundsInitialized = false;
	m_worldBounds = QRect3D(0, 0, 0, 1000, 1000, 1000);
	m_sectionType = SampleUnit::NONE;
	m_depthLengthUnit = &MtLengthUnit::METRE;
	m_lastSelected= nullptr;

	m_nurbsManager = new NurbsManager(this);



	m_visibleGizmo3d= true;
	m_visibleInfos3d = true;
	m_speedUpDown = 25.0f;


	m_quickview = new Quickview3d(this);//this
	m_quickview->engine()->rootContext()->setContextProperty("viewQt", this);
	//m_quickview->setPersistentOpenGLContext(true);
	m_quickview->setPersistentGraphics(true);
	m_quickview->setResizeMode(QQuickView::SizeRootObjectToView);
	m_quickview->setMinimumSize(QSize(300, 300));
	m_quickview->setColor("#19232D");


	int sizeIcon = 32;

	//InfoTooltip* test = new InfoTooltip(this);
	//	test->show();


    connect(m_quickview, &QQuickView::statusChanged, this, &ViewQt3D::onQMLReady, Qt::QueuedConnection);

	m_quickview->setSource(QUrl("qrc:/Main.qml"));

/*	QToolButton *m_toolsButton = new QToolButton;
	m_toolsButton->setToolTip("3D tools");
	m_toolsButton->setIconSize(QSize(sizeIcon, sizeIcon));
	m_toolsButton->setIcon(QIcon(QString(":/slicer/icons/camera_blanc.png")));//parameters*/

	m_syncButton = new QToolButton;
	m_syncButton->setIconSize(QSize(sizeIcon, sizeIcon));
	m_syncButton->setToolTip("Camera synchro");

/*	m_flyButton = new QToolButton();
	m_flyButton->setIcon(QIcon(QString(":/slicer/icons/carrehelico2.svg")));
	m_flyButton->setIconSize(QSize(sizeIcon, sizeIcon));
	m_flyButton->setCheckable(true);
	m_flyButton->setToolTip("Helico mode");
*/
	QLabel* labelzscale = new QLabel("Z scale:");

	m_spinZScale = new QSpinBox();
	m_spinZScale->setRange(1,10000);
	m_spinZScale->setSingleStep(10);
	m_spinZScale->setValue(100);
	m_spinZScale->setToolTip("Z scale");

	QLabel* labelupdown = new QLabel("Up/Down");
	QToolButton* m_downButton = new QToolButton();
	m_downButton->setIcon(style()->standardPixmap(  QStyle::SP_ArrowDown));//QStyle::PE_IndicatorArrowDown));
	m_downButton->setIconSize(QSize(sizeIcon, sizeIcon));
	m_downButton->setAutoRepeat(true);
	m_downButton->setToolTip("Move down camera");

	m_upButton = new QToolButton();
	m_upButton->setAutoRepeat(true);
	m_upButton->setIcon(style()->standardPixmap(  QStyle::SP_ArrowUp));//QStyle::PE_IndicatorArrowUp));
	m_upButton->setIconSize(QSize(sizeIcon, sizeIcon));
	m_upButton->setToolTip("Move Up camera");

/*	QToolButton* m_helpButton = new QToolButton();
	m_helpButton->setIconSize(QSize(sizeIcon, sizeIcon));
	m_helpButton->setIcon(style()->standardPixmap( QStyle::SP_MessageBoxQuestion));// QStyle::SC_TitleBarContextHelpButton));
	m_helpButton->setToolTip("Help shortcut");
	*/
	/*m_recordButton = new QToolButton();
	m_recordButton->setIcon(QIcon(QString(":/slicer/icons/path-cam.png")));
	m_recordButton->setIconSize(QSize(sizeIcon, sizeIcon));
	//m_recordButton->setIcon(style()->standardPixmap( QStyle::SP_DialogCancelButton));// QStyle::SC_TitleBarContextHelpButton));
	m_recordButton->setToolTip("3D path record");*/


/*	QToolButton* m_resetCamButton = new QToolButton();
	m_resetCamButton->setIcon(QIcon(QString(":/slicer/icons/zoomReset.png")));
	//m_resetCamButton->setIcon(style()->standardPixmap( QStyle::SP_ArrowForward));// QStyle::SC_TitleBarContextHelpButton));
	m_resetCamButton->setToolTip("reset camera");
	m_resetCamButton->setIconSize(QSize(sizeIcon, sizeIcon));*/

/*	m_depthUnitToggle = new QToolButton();
//	QAction* depthUnitAction = new QAction("m");
	QAction* depthUnitAction = new QAction() ;//QIcon(QString(":/slicer/icons/regle_m128_blanc.png")));
	depthUnitAction->setIcon(QIcon(QString(":/slicer/icons/regle_m128_blanc.png")));
	m_depthUnitToggle->setCheckable(true);
	m_depthUnitToggle->setIconSize(QSize(sizeIcon, sizeIcon));
	m_depthUnitToggle->setDefaultAction(depthUnitAction);
	m_depthUnitToggle->setToolTip("Toggle between meter and feet");
	m_depthUnitToggle->setStyleSheet("background-color:#32414B;");*/



/*	m_playButton = new QToolButton();
	m_playButton->setIcon(style()->standardPixmap( QStyle::SP_MediaPlay));// QStyle::SC_TitleBarContextHelpButton));
	m_playButton->setToolTip("3D path play");
	m_playButton->setCheckable(true);
*/


/*	QToolButton* m_propertyButton = new QToolButton();
	m_propertyButton->setIcon(style()->standardPixmap( QStyle::SP_ComputerIcon));// QStyle::SC_TitleBarContextHelpButton));
	m_propertyButton->setToolTip("Properties");
*/
	//m_titleLayout->insertWidget(0,m_toolsButton,0,Qt::AlignLeft);
	m_horizontalToolWidget = new QWidget; // to be used in onQMLReady
	m_horizontalToolLayout = new QHBoxLayout;
	m_horizontalToolWidget->setLayout(m_horizontalToolLayout);
	m_horizontalToolLayout->insertWidget(0,m_syncButton,0,Qt::AlignLeft);
	//m_titleLayout->insertWidget(1,m_flyButton,0,Qt::AlignLeft);
	m_horizontalToolLayout->insertWidget(1,labelzscale,0,Qt::AlignLeft);
	m_horizontalToolLayout->insertWidget(2,m_spinZScale,0,Qt::AlignLeft);

	m_horizontalToolLayout->insertWidget(3,m_upButton,0,Qt::AlignLeft);
	m_horizontalToolLayout->insertWidget(4,labelupdown,0,Qt::AlignLeft);
	m_horizontalToolLayout->insertWidget(5,m_downButton,0,Qt::AlignLeft);

	QPushButton* paletteButton = getPaletteButton();
	if (paletteButton) {
		m_horizontalToolLayout->addWidget(paletteButton,0,Qt::AlignRight);
	}
	m_horizontalToolLayout->addWidget(getSplitButton(),0,Qt::AlignRight);
	//m_titleLayout->insertWidget(7,m_helpButton,0,Qt::AlignLeft);
//	m_titleLayout->insertWidget(9,m_recordButton,0,Qt::AlignLeft);
//	m_titleLayout->insertWidget(10,m_resetCamButton,0,Qt::AlignLeft);
//	m_titleLayout->insertWidget(6,m_depthUnitToggle,0,Qt::AlignLeft);
	//m_titleLayout->insertWidget(9,m_playButton,0,Qt::AlignLeft);
//	m_titleLayout->insertWidget(8,m_propertyButton,0,Qt::AlignLeft);

	setDefaultTitle("");
	//connect(m_toolsButton, &QPushButton::clicked, this, &ViewQt3D::show3DTools);
	connect(m_syncButton, &QPushButton::clicked, this, &ViewQt3D::toggleSyncCamera);
	//connect(m_flyButton, &QPushButton::clicked, this, &ViewQt3D::toggleFlyCamera);

	connect(m_spinZScale, SIGNAL(valueChanged(int)), this,SLOT(onZScaleChanged(int)));

	connect(m_downButton, &QPushButton::pressed, this, &ViewQt3D::downCamera);
	connect(m_upButton, &QPushButton::pressed, this, &ViewQt3D::upCamera);
	//connect(m_upButton, &QPushButton::clicked, this, &ViewQt3D::clickcamera);
	connect(m_downButton, &QPushButton::released, this, &ViewQt3D::stopDownUpCamera);
	connect(m_upButton, &QPushButton::released, this, &ViewQt3D::stopDownUpCamera);
//	connect(m_helpButton, &QPushButton::clicked, this, &ViewQt3D::showHelp);

//	connect(m_recordButton, &QPushButton::clicked, this, &ViewQt3D::recordPath3d);
//	connect(m_resetCamButton, &QPushButton::clicked, this, &ViewQt3D::resetCamera);
	//connect(m_playButton, &QPushButton::clicked, this, &ViewQt3D::playPath3d);
//	connect(m_propertyButton, &QPushButton::clicked, this, &ViewQt3D::showProperty);
//	connect(m_depthUnitToggle->defaultAction(), &QAction::triggered, this, &ViewQt3D::toggleDepthUnit);
	updateSyncButton();





	   connect(this, &ViewQt3D::isFocusedChanged,
	                     this, &ViewQt3D::etatViewChanged);
	//connect(this,SIGNAL(parentCanged ()),this,SLOT(etatViewChanged()));

//    m_camerhudview = new QQuickView();
//    m_camerhudview->setPersistentOpenGLContext(true);
//    m_camerhudview->setResizeMode(QQuickView::SizeRootObjectToView);
//    m_camerhudview->setMinimumSize(QSize(300, 300));
//    m_camerhudview->setColor("#19232D");
//    connect(m_camerhudview, &QQuickView::statusChanged, this, &ViewQt3D::onQMLReady, Qt::QueuedConnection);
//    m_camerhudview->setSource(QUrl("qrc:/CameraHUD.qml"));





}
/*
void ViewQt3D::enterEvent(QEvent* event)
{
	event->accept();
	qDebug()<<" mouse ENTER";
}

void ViewQt3D::leaveEvent(QEvent* event)
{
	event->accept();
	qDebug()<<" mouse EXIT";
}

*/


void ViewQt3D::keyPressEvent(QKeyEvent *event)
{
	//QCoreApplication::sendEvent(m_quickview,event);

	//qDebug()<<" viewQt 3d :keyPressEvent";

	 if(event->key() == Qt::Key_Up)
	 {
		CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
		if (cameraCtrl != nullptr)
		{
			cameraCtrl->setDecalUpDown(0.1f*m_speedUpDown);
		}
	 }
	 if(event->key() == Qt::Key_Down)
	 {
		 CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
		if (cameraCtrl != nullptr)
		{
			cameraCtrl->setDecalUpDown(-0.1f*m_speedUpDown);
		}
	 }

	if(event->key() == Qt::Key_V)
	{

		 if(widgetpath != nullptr)
		{
			if(widgetpath->m_modeEdition)
			{
				QVector3D position = m_camera->position();

				QVector3D target =m_camera->position() +m_camera->viewVector() ;

				//qDebug()<<" ==>position : "<<position<<" , target : "<<target <<" view : "<<m_view3D->m_camera->viewVector() ;
				widgetpath->AddPoints(position, target, 2000);
			}
		}
	}

	if(event->key() == Qt::Key_T)
	{

		m_tooltipActif = true;

	}


 }



void ViewQt3D::keyReleaseEvent(QKeyEvent *event)
{
	if(event->key() == Qt::Key_Up || event->key() == Qt::Key_Down)
	{
		CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
		cameraCtrl->setDecalUpDown(0.0f);
	}


	if(event->key() == Qt::Key_T)
	{
		m_tooltipActif = false;

	}
	//QCoreApplication::sendEvent(m_quickview,event);
//	m_quickview->keyReleaseEvent(event);
}

/*

bool ViewQt3D::event(QEvent *event)
{
	if (event->type() == QEvent::Enter)
			{
				//qDebug()<<"EVENT view : "<<event->type();
				//QQuickItem* item = this->contentItem();
				//item->setFocus(true);

			}

	qDebug()<<"EVENT : "<<event->type();

    // handle events that don't match
    return QWidget::event(event);
}
*/


void ViewQt3D::recordPath3d()
{

	widgetpath->show();

/*	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	if (cameraCtrl != nullptr)
	{

		QVector3D posCam= m_camera->position();
		QVector3D targetCam= m_camera->viewCenter();
		//QString positionStr = QString::number(posCam.x())+"|"+QString::number(posCam.y())+"|"+QString::number(posCam.z());
		//QString targetStr = QString::number(targetCam.x())+"|"+QString::number(targetCam.y())+"|"+QString::number(targetCam.z());

		widgetpath->AddPoints(posCam,targetCam,2000);

	}
*/
	/*if( m_recordPathMode==false)
	{
		CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
		if (cameraCtrl != nullptr)
		{
			cameraCtrl->clearPath();
		}
		m_recordPathMode =true;
		m_recordButton->setIcon(style()->standardPixmap( QStyle::SP_MediaStop));
	}
	else
	{
		m_recordPathMode =false;
		m_recordButton->setIcon(style()->standardPixmap( QStyle::SP_DialogCancelButton));
		CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
		if (cameraCtrl != nullptr)
		{
			cameraCtrl->savePath();
		}

	}*/
}

void ViewQt3D::importNurbs(IsoSurfaceBuffer surface,QString nameNurbs)
{
	 QString pathfile = GraphicsLayersDirPath();


	QString directory="Nurbs/";
	QDir dir(pathfile);
	bool res = dir.mkpath("Nurbs");


	QString path = pathfile+directory+nameNurbs;

	if(m_nurbsManager!= nullptr)
	{
		m_nurbsManager->loadNurbs(path,m_camera->position(),m_root,this,surface,m_sceneTransform);

		float coef = 0.0f;
		QVector3D pos3D = m_nurbsManager->getCurrentNurbs()->getPositionDirectrice(coef);
		QVector3D pos =sceneTransform() * pos3D;
		QPointF normal = m_nurbsManager->getCurrentNurbs()->getNormalDirectrice(coef);
		setSliderXsection(coef, pos,normal);
		createNewXSectionClone(coef);

	}

}

void ViewQt3D::createNurbsSimple( QString nameNurbs,IsoSurfaceBuffer buffer)
{
	if(m_nurbsManager!= nullptr)
		m_nurbsManager->createNurbsSimple(nameNurbs,buffer,m_root,this);
}

void ViewQt3D::importNurbsObj(QString path, QString nameNurbs)
{
	/*QString pathfile = GraphicsLayersDirPath();


		QString directory="Nurbs/";
		QDir dir(pathfile);
		bool res = dir.mkpath("Nurbs");
*/

	//	QString path = pathfile+directory+nameNurbs;

		//QString pathObj  = path.replace(".txt",".obj");
		//QString pathObj = path+nameNurbsObj;
	//	qDebug()<<" load nurbs obj..."<<path;
		//Qt3DHelpers::loadObj(pathObj.toStdString().c_str(),m_root);

		if(m_nurbsManager!= nullptr)m_nurbsManager->importNurbsObj(path,nameNurbs,m_camera->position(),m_root,this);
}


void ViewQt3D::playPath3d()
{
	m_playPathMode=!m_playPathMode;
	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	if (cameraCtrl != nullptr)
	{
		cameraCtrl->setModePlayPath(m_playPathMode);
	}
}

void ViewQt3D::stopPath3d()
{
	m_playPathMode=false;
	//m_playButton->setChecked(false);
	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	if (cameraCtrl != nullptr)
	{
		cameraCtrl->setModePlayPath(m_playPathMode);
	}
}
void ViewQt3D::etatViewChanged(bool b)
{

	if(b)
	{
		CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
		if (cameraCtrl != nullptr)
		{
			cameraCtrl->reinitFocus();
		}

		window()->activateWindow();
	}

}

void ViewQt3D::resetCamera()
{
	resetZoom();


}

void ViewQt3D::hideWell()
{
	if(m_lastSelected != nullptr)
	{
		m_lastSelected->deselectLastWell();
		m_lastSelected->hideLastWell();


		if(mToolTip3D != nullptr)
		{
			mToolTip3D->setProperty("visible", false);
			m_currentTooltipProvider = nullptr;
		}

		/*if(m_line != nullptr)
		{
			delete m_line;
			m_line = nullptr;
		}*/
	}
}

void ViewQt3D::deselectWell()
{
	if(m_lastSelected != nullptr)m_lastSelected->deselectLastWell();
	if(mToolTip3D != nullptr)
	{
		mToolTip3D->setProperty("visible", false);
		m_currentTooltipProvider = nullptr;
	}

	/*if(m_line != nullptr)
	{
		delete m_line;
		m_line = nullptr;
	}*/
}

void ViewQt3D::selectWell(WellBoreLayer3D* wellbore)
{
	if(m_lastSelected != nullptr)m_lastSelected->deselectLastWell();
	m_lastSelected = wellbore;
}

void ViewQt3D::onQMLReady(QQuickView::Status status) {
	//qDebug()<<" on qml ready  ==>"<<m_quickview->errors();

	if (status!=QQuickView::Ready || m_camera!=nullptr) {
		return;
	}
	disconnect(m_quickview, &QQuickView::statusChanged, this, &ViewQt3D::onQMLReady);
    //disconnect(m_camerhudview, &QQuickView::statusChanged, this, &ViewQt3D::onQMLReady);

	//qDebug()<<" on qml ready";

	m_camera = m_quickview->findChild<Qt3DRender::QCamera*>("camera");
	if (m_camera == nullptr) {
		std::cerr << "QML Loading:NOT Found object camera!!!" << std::endl;
		return;
	}

	m_root = m_quickview->findChild<Qt3DCore::QEntity*>("sceneRoot");
	if (m_root == nullptr) {
		std::cerr << "QML Loading:NOT Found object ROOT!!!" << std::endl;
		return;
	}

	m_transfoRoot = m_quickview->findChild<Qt3DCore::QTransform*>("transfoGlobal");
	if (m_transfoRoot == nullptr) {
		std::cerr << "QML Loading:NOT Found object transfoGlobal!!!" << std::endl;
		return;
	}

	m_transfoRootFils = m_quickview->findChild<Qt3DCore::QTransform*>("transfoFilsGlobal");
		if (m_transfoRootFils == nullptr) {
			std::cerr << "QML Loading:NOT Found object transfoFilsGlobal!!!" << std::endl;
			return;
		}

	m_controler = m_quickview->findChild<Qt3DCore::QEntity*>("controler");
	if (m_controler == nullptr) {
		std::cerr << "QML Loading:NOT Found object Controler!!!" << std::endl;
		return;
	}

	m_gizmo = m_quickview->findChild<Qt3DCore::QEntity*>("cameraGizmoEntity");
	if (m_gizmo == nullptr) {
		std::cerr << "QML Loading:NOT Found object cameraGizmoEntity!!!" << std::endl;
		return;
	}

	m_gizmo->setEnabled(m_visibleGizmo3d);

	m_infos3d = m_quickview->findChild<QObject*>("infos3d");
	if (m_infos3d == nullptr) {
		std::cerr << "QML Loading:NOT Found object infos3d!!!" << std::endl;
		return;
	}
	m_infos3d->setProperty("visible", m_visibleInfos3d);

	mToolTip = m_quickview->findChild<QObject*>("tooltip");
	if (mToolTip == nullptr) {
		std::cerr << "QML Loading:NOT Found object mToolTip!!!" << std::endl;
		return;
	}
	mToolTip3D = m_quickview->findChild<QObject*>("recttooltip");
	if (mToolTip3D == nullptr) {
		std::cerr << "QML Loading:NOT Found object mToolTip3D!!!" << std::endl;
		return;
	}

	mLineTooltip= m_quickview->findChild<QObject*>("linetooltip");
	if (mLineTooltip == nullptr) {
		std::cerr << "QML Loading:NOT Found object mLineTooltip!!!" << std::endl;
		return;
	}
	m_textToolTip3D= m_quickview->findChild<QObject*>("nametooltip");
	if (m_textToolTip3D == nullptr) {
		std::cerr << "QML Loading:NOT Found object mTextToolTip3D!!!" << std::endl;
		return;
	}
	m_statustooltip= m_quickview->findChild<QObject*>("statustooltip");
	if (m_statustooltip == nullptr) {
		std::cerr << "QML Loading:NOT Found object statustooltip!!!" << std::endl;
		return;
	}
	m_datetooltip= m_quickview->findChild<QObject*>("datetooltip");
	if (m_datetooltip == nullptr) {
			std::cerr << "QML Loading:NOT Found object datetooltip!!!" << std::endl;
		return;
	}
	m_uwitooltip= m_quickview->findChild<QObject*>("uwitooltip");
	if (m_uwitooltip == nullptr) {
		std::cerr << "QML Loading:NOT Found object uwitooltip!!!" << std::endl;
		return;
	}
	m_domaintooltip= m_quickview->findChild<QObject*>("domaintooltip");
	if (m_domaintooltip == nullptr) {
		std::cerr << "QML Loading:NOT Found object domaintooltip!!!" << std::endl;
		return;
	}
	m_elevtooltip= m_quickview->findChild<QObject*>("elevtooltip");
	if (m_elevtooltip == nullptr) {
		std::cerr << "QML Loading:NOT Found object elevtooltip!!!" << std::endl;
		return;
	}
	m_datumtooltip= m_quickview->findChild<QObject*>("datumtooltip");
	if (m_datumtooltip == nullptr) {
		std::cerr << "QML Loading:NOT Found object datumtooltip!!!" << std::endl;
	return;
	}
	m_velocitytooltip= m_quickview->findChild<QObject*>("velocitytooltip");
	if (m_velocitytooltip == nullptr) {
		std::cerr << "QML Loading:NOT Found object velocitytooltip!!!" << std::endl;
	return;
	}
	m_ihstooltip= m_quickview->findChild<QObject*>("ihstooltip");
	if (m_ihstooltip == nullptr) {
			std::cerr << "QML Loading:NOT Found object ihstooltip!!!" << std::endl;
		return;
	}
	m_deselecttooltip= m_quickview->findChild<QObject*>("deselecttooltip");
	if (m_deselecttooltip == nullptr) {
			std::cerr << "QML Loading:NOT Found object deselecttooltip!!!" << std::endl;
		return;
	}



	connect(m_nurbsManager,SIGNAL(sendCurveData(std::vector<QVector3D>,bool)),this, SLOT(receiveCurveData(std::vector<QVector3D>,bool)));
	connect(m_nurbsManager,SIGNAL(sendCurveDataTangent(QVector<PointCtrl>,bool,QPointF)),this, SLOT(receiveCurveData(QVector<PointCtrl>,bool,QPointF)));
	connect(m_nurbsManager,SIGNAL(sendCurveDataTangentOpt(GraphEditor_ListBezierPath*)),this, SLOT(receiveCurveDataOpt(GraphEditor_ListBezierPath*)));
	connect(m_nurbsManager,SIGNAL(sendCurveDataTangent2(QVector<QVector3D>,QVector<QVector3D>,bool,QPointF,QString )),this, SLOT(receiveCurveData2(QVector<QVector3D>,QVector<QVector3D>,bool,QPointF,QString)));
	connect(m_nurbsManager,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int ,QVector3D)),Qt::QueuedConnection);
	connect(m_nurbsManager,SIGNAL(sendNurbsY(QVector3D,QVector3D)),this, SLOT(receiveNurbsY(QVector3D,QVector3D)));



	//std::vector<QVector3D>

	m_GPURes = m_quickview->findChild<Qt3DRessource*>("qt3DRessource");
	if (m_GPURes == nullptr) {
		std::cerr << "QML Loading:NOT Found object QT3D Ressource!!!"
				<< std::endl;
		return;
	} else {
		//m_GPURes->setzScale(m_currentScale*100);
		m_GPURes->setSectionType(m_sectionType);
		m_GPURes->setDepthLengthUnit(m_depthLengthUnit);
	}

	connect(m_camera,SIGNAL(positionChanged(QVector3D)), this, SLOT(positionCameraChanged(QVector3D)));
	connect(m_camera,SIGNAL(viewCenterChanged(QVector3D)), this, SLOT(viewCenterCameraChanged(QVector3D)));
	connect(m_camera,SIGNAL(upVectorChanged(QVector3D)), this, SLOT(upVectorCameraChanged(QVector3D)));
//	connect(this,SIGNAL(cameraMoved(int,QVector3D)), this, SLOT(positionCameraChanged(int, QVector3D)));




	//Define the scene

	QWidget *mainWidget = new QWidget(this);
	QVBoxLayout *mainWidgetLayout = new QVBoxLayout(mainWidget);
	mainWidgetLayout->setContentsMargins(0,0,0,0);
	mainWidgetLayout->addWidget(m_horizontalToolWidget);



	QHBoxLayout* layH = new QHBoxLayout();

	QVBoxLayout* layV = new QVBoxLayout();

	layV->setAlignment(Qt::AlignTop);

	int sizeIcon = 32;



		QToolButton* m_resetCamButton = new QToolButton();
		m_resetCamButton->setIcon(QIcon(QString(":/slicer/icons/zoomReset.png")));
		//m_resetCamButton->setIcon(style()->standardPixmap( QStyle::SP_ArrowForward));// QStyle::SC_TitleBarContextHelpButton));
		m_resetCamButton->setToolTip("reset camera");
		m_resetCamButton->setIconSize(QSize(sizeIcon, sizeIcon));

		m_flyButton = new QToolButton();
			m_flyButton->setIcon(QIcon(QString(":/slicer/icons/carrehelico2.svg")));
			m_flyButton->setIconSize(QSize(sizeIcon, sizeIcon));
			m_flyButton->setCheckable(true);
			m_flyButton->setToolTip("Helico mode");

	QToolButton *m_toolsButton = new QToolButton;
	m_toolsButton->setToolTip("3D tools");
	m_toolsButton->setIconSize(QSize(sizeIcon, sizeIcon));
	m_toolsButton->setIcon(QIcon(QString(":/slicer/icons/camera_blanc.png")));//parameters


	m_recordButton = new QToolButton();
	m_recordButton->setIcon(QIcon(QString(":/slicer/icons/path-cam.png")));
	m_recordButton->setIconSize(QSize(sizeIcon, sizeIcon));
		//m_recordButton->setIcon(style()->standardPixmap( QStyle::SP_DialogCancelButton));// QStyle::SC_TitleBarContextHelpButton));
	m_recordButton->setToolTip("3D path record");


	QToolButton* m_helpButton = new QToolButton();
	m_helpButton->setIconSize(QSize(sizeIcon, sizeIcon));
	m_helpButton->setIcon(style()->standardPixmap( QStyle::SP_MessageBoxQuestion));// QStyle::SC_TitleBarContextHelpButton));
	m_helpButton->setToolTip("Help shortcut");

/*	m_depthUnitToggle = new QToolButton();
	//	QAction* depthUnitAction = new QAction("m");
		QAction* depthUnitAction = new QAction(m_depthUnitToggle) ;//QIcon(QString(":/slicer/icons/regle_m128_blanc.png")));
		depthUnitAction->setIcon(QIcon(QString(":/slicer/icons/regle_m128_blanc.png")));
		m_depthUnitToggle->setCheckable(true);
		m_depthUnitToggle->setIconSize(QSize(sizeIcon, sizeIcon));
		m_depthUnitToggle->setDefaultAction(depthUnitAction);
		m_depthUnitToggle->setToolTip("Toggle between meter and feet");
		m_depthUnitToggle->setStyleSheet("background-color:#32414B;");*/

	layV->addWidget( m_resetCamButton);
	layV->addWidget( m_flyButton);
	layV->addWidget( m_toolsButton);
	layV->addWidget( m_recordButton);
//	layV->addWidget( m_depthUnitToggle);
	layV->addWidget( m_helpButton);

	layH->addLayout(layV);
	layH->addWidget(
			QWidget::createWindowContainer(m_quickview, mainWidget), 1);

	connect(m_flyButton, &QPushButton::clicked, this, &ViewQt3D::toggleFlyCamera);
	connect(m_resetCamButton, &QPushButton::clicked, this, &ViewQt3D::resetCamera);
	connect(m_toolsButton, &QPushButton::clicked, this, &ViewQt3D::show3DTools);
	connect(m_recordButton, &QPushButton::clicked, this, &ViewQt3D::recordPath3d);
	connect(m_helpButton, &QPushButton::clicked, this, &ViewQt3D::showHelp);
	//connect(m_depthUnitToggle->defaultAction(), &QAction::triggered, this, &ViewQt3D::toggleDepthUnit);

	mainWidgetLayout->addLayout(layH,1);

	mainWidgetLayout->addWidget(generateSizeGrip(), 0, Qt::AlignRight);



	setWidget(mainWidget);



	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	static_cast<Quickview3d*>( m_quickview)->setCameraController(cameraCtrl);

	connect(cameraCtrl,SIGNAL(distanceChanged(float, QVector3D)),this,SLOT(distanceTargetChanged(float,QVector3D)));
	connect(cameraCtrl,SIGNAL(zoomDistanceChanged(float)),this,SLOT(changeDistanceForcing(float)));

	connect(cameraCtrl,SIGNAL(stopModePlay()),this,SLOT(stopPath3d()));

	mainWidget->setMinimumSize(380, 300);
	m_qmlLoaded = true;
	m_animation = new QPropertyAnimation(m_camera,"position", this);
	m_animationCenter= new QPropertyAnimation(m_camera,"viewCenter", this);

	connect(m_animation, &QPropertyAnimation::finished, this, &ViewQt3D::onAnimationFinished);
	connect(m_animation, &QPropertyAnimation::stateChanged, this, &ViewQt3D::onAnimationStopped);

	while (m_waitingReps.size()!=0) {
		showRep(m_waitingReps.dequeue());
	}



	NurbsWidget::clearCombo();
	int nbNurbs =NurbsWidget::getNbNurbs();



	for(int i=0;i<nbNurbs;i++)
	{
		QString path = NurbsWidget::getPath(i);
		QString nameNurbs = NurbsWidget::getName(i).replace(".txt","");


		if(m_nurbsManager!= nullptr)m_nurbsManager->importNurbsObj(path,nameNurbs,m_camera->position(),m_root,this);
		NurbsWidget::addCombo(nameNurbs);
	}


}

void ViewQt3D::showTooltipPick(const IToolTipProvider* tooltipProvider, QString name, int posX, int posY,QVector3D posGlobal)
{
	if(mToolTip3D != nullptr)
	{
		m_lastPos3DWeel = posGlobal;

		/*if(m_line != nullptr)
		{
			delete m_line;
			m_line = nullptr;
		}
*/

		if(mLineTooltip != nullptr)
		{

			int width2 = m_quickview->width();
			int height2 = m_quickview->height();

			int widthtest =( width2-300)*0.5f ;

			float signe = 1.0f;
			if( posY > height2*0.5 )signe = -1.0f;

			int widthX =  qFabs(widthtest - posX);
			int widthY =  qFabs(posY  - height2*0.5);

			int width = qSqrt(widthX *widthX + widthY*widthY);
			QVector2D p1(posX,posY);
			QVector2D p2(widthtest,height2*0.5);



			QVector2D dir1(1.0f,0.0f);

			QVector2D dir2((p2-p1).normalized());

			float angle = signe * 180.0f /3.14159f * qAcos(QVector2D::dotProduct(dir1,dir2));




			mLineTooltip->setProperty("x", posX);
			mLineTooltip->setProperty("y", posY);
			mLineTooltip->setProperty("width", width);
			mLineTooltip->setProperty("rotation", angle);
			mLineTooltip->setProperty("visible", false);

		}


		mToolTip3D->setProperty("visible", true);
		m_currentTooltipProvider = tooltipProvider;
		m_tooltipProviderIsWell = false;

		refreshPickTooltip(name);

	} else {
		m_currentTooltipProvider = nullptr;
	}
}

void ViewQt3D::refreshPickTooltip(QString name)
{
	if(m_textToolTip3D != nullptr)
	{
		//qDebug()<<" information:"<<name;
		int height=5;
		QStringList list = name.split("|");

		if( list.count() > 0 && list[0].length() > 0){m_textToolTip3D->setProperty("y",height); height+=20; QString str ="Name:"+list[0];m_textToolTip3D->setProperty("text",str);}
		if( list.count() > 1 && list[1].length() > 0)
		{
			m_statustooltip->setProperty("y",height);
			height+=20;
			QString str ="Kind:"+list[1];
			m_statustooltip->setProperty("text",str);
		}
		if( list.count() > 2 && list[2].length() > 0){
			m_datetooltip->setProperty("y",height);height+=20;QString str ="Value:"+list[2];m_datetooltip->setProperty("text",str);
		}
		else{
			m_datetooltip->setProperty("text","");
		}
		if( list.count() > 3 && list[3].length() > 0){
			m_uwitooltip->setProperty("y",height);height+=20;QString str ="UWI:"+list[3];m_uwitooltip->setProperty("text",str);
		}
		else{
			m_uwitooltip->setProperty("text","");
		}
		if( list.count() > 4 && list[4].length() > 0){
			m_domaintooltip->setProperty("y",height);height+=20;QString str ="Domain:"+list[4];m_domaintooltip->setProperty("text",str);
		}
		else{
			m_domaintooltip->setProperty("text","");
		}
		if( list.count() > 5 && list[5].length() > 0){
			m_elevtooltip->setProperty("y",height);height+=20;QString str ="Elev:"+list[5];m_elevtooltip->setProperty("text",str);
		}
		else{
			m_elevtooltip->setProperty("text","");
		}
		if( list.count() > 6 && list[6].length() > 0){
			m_datumtooltip->setProperty("y",height);height+=20;QString str ="Datum:"+list[6];m_datumtooltip->setProperty("text",str);
		}
		else{
			m_datumtooltip->setProperty("text","");
		}
		if( list.count() > 7 && list[7].length() > 0){
			m_velocitytooltip->setProperty("y",height);height+=20;QString str ="Velocity:"+list[7];m_velocitytooltip->setProperty("text",str);
		}
		else{
			m_velocitytooltip->setProperty("text","");
		}
		if( list.count() > 8 && list[8].length() > 0){
			m_ihstooltip->setProperty("y",height);height+=20;QString str ="IHS:"+list[8];m_ihstooltip->setProperty("text",str);
		}
		else{
			m_ihstooltip->setProperty("text","");
		}

		m_deselecttooltip->setProperty("y",height);
		height+=30;

		mToolTip3D->setProperty("height", height);
	}
}

void ViewQt3D::refreshWellTooltip(QString name)
{
	if(m_textToolTip3D != nullptr)
	{
		//qDebug()<<" information:"<<name;
		int height=5;
		QStringList list = name.split("|");

		if( list.count() > 0 && list[0].length() > 0){
			m_textToolTip3D->setProperty("y",height); height+=20; QString str ="Name:"+list[0];m_textToolTip3D->setProperty("text",str);
		}
		else{
			m_textToolTip3D->setProperty("text","");
		}
		if( list.count() > 1 && list[1].length() > 0)
		{
			m_statustooltip->setProperty("y",height);
			height+=20;
			QString str ="Status:"+list[1];
			m_statustooltip->setProperty("text",str);
		}else{
			m_statustooltip->setProperty("text","");
		}
		if( list.count() > 2 && list[2].length() > 0){
			m_datetooltip->setProperty("y",height);height+=20;QString str ="Date:"+list[2];m_datetooltip->setProperty("text",str);
		}else{
			m_datetooltip->setProperty("text","");
		}
		if( list.count() > 3 && list[3].length() > 0){
			m_uwitooltip->setProperty("y",height);height+=20;QString str ="UWI:"+list[3];m_uwitooltip->setProperty("text",str);
		}else{
			m_uwitooltip->setProperty("text","");
		}
		if( list.count() > 4 && list[4].length() > 0){
			m_domaintooltip->setProperty("y",height);height+=20;QString str ="Domain:"+list[4];m_domaintooltip->setProperty("text",str);
		}else{
			m_domaintooltip->setProperty("text","");
		}
		if( list.count() > 5 && list[5].length() > 0){
			m_elevtooltip->setProperty("y",height);height+=20;QString str ="Elev:"+list[5];m_elevtooltip->setProperty("text",str);
		}else{
			m_elevtooltip->setProperty("text","");
		}
		if( list.count() > 6 && list[6].length() > 0){
			m_datumtooltip->setProperty("y",height);height+=20;QString str ="Datum:"+list[6];m_datumtooltip->setProperty("text",str);
		}else{
			m_datumtooltip->setProperty("text","");
		}
		if( list.count() > 7 && list[7].length() > 0){
			m_velocitytooltip->setProperty("y",height);height+=20;QString str ="Velocity:"+list[7];m_velocitytooltip->setProperty("text",str);
		}else{
			m_velocitytooltip->setProperty("text","");
		}
		if( list.count() > 8 && list[8].length() > 0){
			m_ihstooltip->setProperty("y",height);height+=20;QString str ="IHS:"+list[8];m_ihstooltip->setProperty("text",str);
		}
		else{
			m_ihstooltip->setProperty("text","");
		}
		m_deselecttooltip->setProperty("y",height);
		height+=30;

		mToolTip3D->setProperty("height", height);
	}
}

void ViewQt3D::showTooltipWell(const IToolTipProvider* tooltipProvider, QString name, int posX, int posY, QVector3D posGlobal)
{
	if(mToolTip3D != nullptr)
	{

			m_lastPos3DWeel = posGlobal;


		/*	if(m_line != nullptr)
				{
				delete m_line;
				m_line = nullptr;
				}*/


			//QVector3D pos3D = Qt3DHelpers::screenToWorld( m_camera->viewMatrix(), m_camera->projectionMatrix(),QVector2D(widthtest,height2*0.5),width2, height2);
			//m_line  = Qt3DHelpers::drawLine(posGlobal, pos3D,  QColor(255,127,0), m_root);

			if(mLineTooltip != nullptr)
			{

				int width2 = m_quickview->width();
				int height2 = m_quickview->height();

				int widthtest =( width2-300)*0.5f ;

				float signe = 1.0f;
				if( posY > height2*0.5 )signe = -1.0f;

				int widthX =  qFabs(widthtest - posX);
				int widthY =  qFabs(posY  - height2*0.5);

				int width = qSqrt(widthX *widthX + widthY*widthY);
				QVector2D p1(posX,posY);
				QVector2D p2(widthtest,height2*0.5);

				QVector2D dir1(1.0f,0.0f);

				QVector2D dir2((p2-p1).normalized());

				float angle = signe * 180.0f /3.14159f * qAcos(QVector2D::dotProduct(dir1,dir2));

				mLineTooltip->setProperty("x", posX);
				mLineTooltip->setProperty("y", posY);
				mLineTooltip->setProperty("width", width);
				mLineTooltip->setProperty("rotation", angle);
				mLineTooltip->setProperty("visible", false);

			}


			mToolTip3D->setProperty("visible", true);

			m_currentTooltipProvider = tooltipProvider;
			m_tooltipProviderIsWell = true;

			refreshWellTooltip(name);

		} else {
			m_currentTooltipProvider = nullptr;
		}


}



void ViewQt3D::show3DTools()
{
	emit signalShowTools(true);
}

void ViewQt3D::setAnimationCamera(int button, QVector3D pos)
{

	if(button ==2)//create tooltip
	{
		QVector3D posWithScale = pos / m_coefGlobal;
		setPositionTooltip(posWithScale);
	}
	if(button ==1) //animation camera
	{
		QVector3D posWithScale = pos / m_coefGlobal;

		float coefZoom = 0.35f;
		QVector3D dirOriginal = m_camera->position()- m_camera->viewCenter();
		QVector3D  newpos =posWithScale + dirOriginal*coefZoom;


		QPropertyAnimation* animation = new QPropertyAnimation(m_camera,"viewCenter");
		animation->setDuration(1000);
		animation->setStartValue(m_camera->viewCenter());
		animation->setEndValue(posWithScale);
		animation->start();

	//	float coefZoom = 0.35f;
	//	QVector3D dirDest = (posWithScale - m_camera->position()) * coefZoom;
	//	QVector3D  newpos = m_camera->position() + dirDest;

		QPropertyAnimation* animation2 = new QPropertyAnimation(m_camera,"position");
		animation2->setDuration(1000);
		animation2->setStartValue(m_camera->position());
		animation2->setEndValue(newpos);
		animation2->start();

	/*	QVector3D up = m_camera->upVector();

		QPropertyAnimation* animation3 = new QPropertyAnimation(m_camera,"upVector");
		animation3->setDuration(1000);
		animation3->setStartValue(up);
		animation3->setEndValue(QVector3D(0,-1,0));
		animation3->start();*/
	}
}


void ViewQt3D::updateWidthRandomView(QString nameView,QVector<QVector3D> listepts,float width)
{

	float height =m_heightBox;
		for(int i=0;i<listepts.count();i++)
		{
			listepts[i].setY(-0.5f *height);
		}
	if(m_randomViews.count()>0)
	{

		m_currentIndexRandom= getIndexRandomView(nameView);
		if( m_currentIndexRandom>=0)
		{

			//m_randomViews[m_currentIndexRandom]->setParam( width, height);//, cudaTexture,range);
			m_randomViews[m_currentIndexRandom]->refreshWidth(listepts,width);

		}
	}
}

void ViewQt3D::updateRandomView(QString nameView,QVector<QVector3D> listepoints,CudaImageTexture* cudaTexture,QVector2D range,bool followCam, float distance, float altitude,float inclinaison,int width,bool withTangent)
{

	float height =m_heightBox;
	for(int i=0;i<listepoints.count();i++)
	{
		listepoints[i].setY(-0.5f *height);
	}
	if(m_randomViews.count()>0)
	{
		//float height =m_heightBox;
		m_currentIndexRandom= getIndexRandomView(nameView);
		if( m_currentIndexRandom>=0)
		{

			m_randomViews[m_currentIndexRandom]->setParam( width, height);//, cudaTexture,range);
			m_randomViews[m_currentIndexRandom]->updateMaterial( cudaTexture,range);

			if(withTangent == true)
			{


				QVector3D position = (listepoints[0]+listepoints[1])*0.5f;
				QVector3D dir1 = (listepoints[1]-listepoints[0]).normalized();
				//QVector3D normal =QVector3D::crossProduct(dir1,QVector3D(0.0f,-1.0f,0.0f));
				QVector3D normal =QVector3D::crossProduct(QVector3D(0.0f,-1.0f,0.0f),dir1);
				m_randomViews[m_currentIndexRandom]->update(position,normal);
			}
		}
	}


	m_followCam = followCam;
	m_distanceCam = distance;
	//m_altitudeCam = altitude;
	//float alt = getAltitude(position);
		//	qDebug()<<" altitude ==>"<<alt;
		//	position = position +QVector3D(0.0f,alt,0.0f);
	m_inclinaisonCam = inclinaison;
	QVector3D tmpY = (listepoints[0] + listepoints[1]) *0.5f;
	m_targetY = m_currentScale* (tmpY.y()+ altitude);
}



void ViewQt3D::createRandomView(bool isOrtho,QString nameView, QVector<QVector3D> listepoints,CudaImageTexture* cudaTexture
		,QVector2D range, RandomLineView* random,GraphEditor_LineShape* line,QVector3D position,QVector3D normal)
{
	float height =m_heightBox;

	for(int i=0;i<listepoints.count();i++)
	{
		listepoints[i].setY(-0.5f *height);
	}

	float width = (listepoints[1] - listepoints[0]).length();

	RandomView3D* view = new RandomView3D(m_working,random,line,nameView/*,getLayerOpaque()*/,m_root);
	connect(view,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int, QVector3D)),Qt::QueuedConnection);
	connect(view,SIGNAL(destroy(RandomView3D*)),this,SLOT( deleteRandom(RandomView3D*)));
	view->init(listepoints, width, height);//, cudaTexture,range);
	view->initMaterial(cudaTexture,range);
	view->update(position,normal);
	m_randomViews.push_back(view);

	if(isOrtho)
	{
		if(m_nurbsManager!= nullptr)m_nurbsManager->setCurrentRandom(view);
		//float alt = getAltitude(position);

		//position = position +QVector3D(0.0f,alt,0.0f);

		//setSliderXsection(0.0f,position,QPointF(normal.x(),normal.z()));
	}

}

void ViewQt3D::createRandomView(bool isOrtho,QString nameView, QVector<QVector3D> listepoints,QVector<CudaImageTexture*> cudaTextures
		,QVector<QVector2D> ranges, RandomLineView* random,GraphEditor_LineShape* line)
{

	//qDebug()<<" CreateRandomview 1   ==>"<<cudaTextures.count();
	float height =m_heightBox;

	for(int i=0;i<listepoints.count();i++)
	{
		listepoints[i].setY(-0.5f *height);
	}

	float width = (listepoints[1] - listepoints[0]).length();



	RandomView3D* view = new RandomView3D(m_working,random,line,nameView,cudaTextures,ranges/*,getLayerOpaque()*/,m_root);
	connect(view,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int, QVector3D)),Qt::QueuedConnection);
	connect(view,SIGNAL(destroy(RandomView3D*)),this,SLOT( deleteRandom(RandomView3D*)));
	view->init(listepoints, width, height);//, cudaTexture,range);
	view->initMaterial(cudaTextures[0],ranges[0]);
	m_randomViews.push_back(view);

	if(isOrtho)
	{
		if(m_nurbsManager!= nullptr)m_nurbsManager->setCurrentRandom(view);
		setSliderXsection(0.0f);
	}


}
/*
void ViewQt3D::createRandomView(bool isOrtho,QString nameView, QVector<QVector3D> listepoints,CudaImageTexture* cudaTexture,QVector2D range, RandomLineView* random,QVector3D position,
			QVector3D normal, float width)
{
	float height =m_heightBox;

	for(int i=0;i<listepoints.count();i++)
	{
		listepoints[i].setY(-0.5f *height);
	}

	//float width = qAbs(listepoints[1].x() - listepoints[0].x());

	RandomView3D* view = new RandomView3D(m_working,random,line,nameView,m_root);
	connect(view,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int, QVector3D)),Qt::QueuedConnection);
	view->init(listepoints, width, height);//, cudaTexture,range);
	view->initMaterial(cudaTexture,range);
	m_randomViews.push_back(view);

	if(isOrtho)
	{
		m_nurbsManager->setCurrentRandom(view);
		setSliderXsection(0.0f);
	}
}*/


void ViewQt3D::createRandomView(bool isOrtho,QString nameView, QVector<QVector3D> listepoints,CudaImageTexture* cudaTexture
		,QVector2D range, RandomLineView* random,GraphEditor_LineShape* line)
{


	float height =m_heightBox;

	for(int i=0;i<listepoints.count();i++)
	{
		listepoints[i].setY(-0.5f *height);
	}

	float width = (listepoints[1] - listepoints[0]).length();

	RandomView3D* view = new RandomView3D(m_working,random,line,nameView,/*getLayerOpaque(),*/m_root);
	connect(view,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int, QVector3D)),Qt::QueuedConnection);
	connect(view,SIGNAL(destroy(RandomView3D*)),this,SLOT( deleteRandom(RandomView3D*)));
	view->init(listepoints, width, height);//, cudaTexture,range);
	view->initMaterial(cudaTexture,range);
	m_randomViews.push_back(view);

	if(isOrtho)
	{
		if(m_nurbsManager!= nullptr)m_nurbsManager->setCurrentRandom(view);
		setSliderXsection(0.0f);
	}
}


void ViewQt3D::createRandomView(bool isOrtho,QString nameView, QVector<QVector3D> listepoints,CudaImageTexture* cudaTexture,QVector2D range, RandomLineView* random)
{

	float height =m_heightBox;

	for(int i=0;i<listepoints.count();i++)
	{
		listepoints[i].setY(-0.5f *height);
	}

	float width = (listepoints[1] - listepoints[0]).length();



	RandomView3D* view = new RandomView3D(m_working,random,nullptr,nameView/*,getLayerOpaque()*/,m_root);
	connect(view,SIGNAL(sendAnimationCam(int,QVector3D)),this, SLOT(setAnimationCamera(int, QVector3D)),Qt::QueuedConnection);
	connect(view,SIGNAL(destroy(RandomView3D*)),this,SLOT( deleteRandom(RandomView3D*)));
	view->init(listepoints, width, height);//, cudaTexture,range);
	view->initMaterial(cudaTexture,range);
	m_randomViews.push_back(view);

	if(isOrtho)
	{
		qDebug()<<" obsolete create randomView";
		if(m_nurbsManager!= nullptr)m_nurbsManager->setCurrentRandom(view);
		setSliderXsection(0.0f);
	}
}

void ViewQt3D::deleteRandom(RandomView3D* r)
{
	deleteCurrentRandomView(r);
}



void ViewQt3D::selectRandomView(QString nameView)
{
	if(m_randomViews.count()==0) return;
	if(m_lastSelectedViews>=0) m_randomViews[m_lastSelectedViews]->setSelected(false);
	int index = getIndexRandomView(nameView);
	if( index>=0)
	{

		m_randomViews[index]->setSelected(true);
		m_lastSelectedViews = index;
	}
}

void ViewQt3D::selectRandomView(int index)
{
	if(m_randomViews.count()==0) return;
	//qDebug()<<" m_randomViews index:"<<index;
//	qDebug()<<" m_lastSelectedViews:"<<m_lastSelectedViews;
	if(m_lastSelectedViews>=0)
	{
		if(m_randomViews[m_lastSelectedViews] != nullptr)
			m_randomViews[m_lastSelectedViews]->setSelected(false);
	}

/*	if( index>=0 && index <m_randomViews.count())
	{
		if(m_randomViews[index] != nullptr) m_randomViews[index]->setSelected(true);
		m_lastSelectedViews = index;
	}*/
}

void ViewQt3D::deleteRandomView(QString nameView)
{
	int index = getIndexRandomView(nameView);
	if(index >=0)
	{

		Manager* nurbs = m_nurbsManager->getNurbsFromRandom(m_randomViews[index]);
		if(nurbs != nullptr) nurbs->setRandom3D(nullptr);
		m_randomViews[index]->destroyRandom();
		delete m_randomViews[index];
		m_randomViews[index]=nullptr;
		m_randomViews.remove(index);

		m_lastSelectedViews = -1;
	}
}

void ViewQt3D::deleteCurrentRandomView(RandomView3D*  rand3d)
{
	if(rand3d != nullptr )
	{
		m_randomViews.removeAll(rand3d);
		rand3d->destroyRandom();

		rand3d->deleteLater();
		//delete rand3d;
		rand3d = nullptr;

		m_lastSelectedViews = -1;
	}
}

int ViewQt3D::getIndexRandomView(QString nameView)
{

	for(int i=0;i<m_randomViews.count();i++)
	{
		if(m_randomViews[i]!= nullptr)
		{
			if(m_randomViews[i]->getName() == nameView)
				return i;
		}
	}
	return -1;
}


int ViewQt3D::destroyRandomView(RandomLineView* random)
{
	//qDebug()<<"===> destroyRandomView ";
	int index = findRandom3d(random);

	if(index>=0)
	{

		Manager* nurbs = m_nurbsManager->getNurbsFromRandom(m_randomViews[index]);
		if(nurbs != nullptr) nurbs->setRandom3D(nullptr);
		m_randomViews[index]->destroyRandom();
		delete m_randomViews[index];
		m_randomViews[index] = nullptr;
		m_randomViews.remove(index);

		m_lastSelectedViews = -1;
	}

	return index;
}

int ViewQt3D::findRandom3d(RandomLineView* random)
{
	for(int i=0;i<m_randomViews.count() ;i++)
	{
		if( m_randomViews[i] != nullptr)
		{
			RandomLineView* rand = m_randomViews[i]->getRandomLineView();
			if( rand!= nullptr && rand == random ) return i;
		}
	}
	return -1;
}



QVector2D ViewQt3D::getPos2D(QVector3D pos,float zScale)
{
	int width2 = m_quickview->width();
	int height2 = m_quickview->height();

	QVector3D posScale = pos;

	float coef =(float)(m_currentScale / (float)(zScale));
	posScale.setY(pos.y()* coef);

	QVector2D pos2D = Qt3DHelpers::worldToScreen( posScale, m_camera->viewMatrix(), m_camera->projectionMatrix(),width2, height2);

	return pos2D;

	/*widgetName* widget = new widgetName(this);
	if ( widget->exec() == QDialog::Accepted)
	{
		QString nom = widget->getName();
		qDebug()<<" name :"<<nom;
		if(nom!="" )
		{
			m_listeTooltips.push_back(new InfoTooltip(nom,pos3D));
			emit addTooltip(posx,posy,nom);
		}
	}*/

}

void ViewQt3D::createTooltip(QString nom, QVector3D pos, int sizePolicy,float zScale,QString family,bool bold, bool italic,QColor color)
{


	int width2 = m_quickview->width();
	int height2 = m_quickview->height();

	int size = sizePolicy;

	QString police =family;
	bool gras=bold;
	bool italique =italic;

	QColor col =color;
	QFont font(police,size);
	font.setBold(bold);
	font.setItalic(italic);


	QVector2D pos2D = Qt3DHelpers::worldToScreen( pos, m_camera->viewMatrix(), m_camera->projectionMatrix(),width2, height2);

	m_listeTooltips.push_back(new InfoTooltip(nom,pos,size,zScale,col,font));
	emit addTooltip(pos2D.x(),pos2D.y(),nom,size,police,gras, italique,col);
}

void ViewQt3D::setPositionTooltip(QVector3D pos)
{

	if(m_tooltipActif ==true)
	{
		int width2 = m_quickview->width();
		int height2 = m_quickview->height();


		QVector2D pos2D = Qt3DHelpers::worldToScreen( pos, m_camera->viewMatrix(), m_camera->projectionMatrix(),width2, height2);


		m_tooltipActif = false;

		widgetName* widget = new widgetName(this);
		if ( widget->exec() == QDialog::Accepted)
		{
			QString nom = widget->getName();
			//int size = widget->getSize();
			int size = widget->getFont().pointSize();
			QString police = widget->getFont().family();

			bool gras = widget->getFont().bold();
			bool italique = widget->getFont().italic();

			QColor col = widget->getColor();

			if(nom!="" )
			{

				//qDebug() <<" color selected :"<<col;
				m_listeTooltips.push_back(new InfoTooltip(nom,pos,size,m_currentScale,col,widget->getFont()));
				emit addTooltip(pos2D.x(),pos2D.y(),nom,size,police ,gras ,italique,col);
				emit sendNewTooltip(nom);
			}
		}
	}
}


void ViewQt3D::setSizePolicy(int index,QString name ,int size)
{
	InfoTooltip* infos = findTooltip(name);
	infos->setSizePolicy(size);
	emit sendUpdateTooltip(index,size);

}

void ViewQt3D::setColorTooltip(int index ,QString name ,QColor color)
{
	InfoTooltip* infos = findTooltip(name);
	infos->setColor(color);
	emit sendColorTooltip(index,color);
}

void ViewQt3D::setFontTooltip(int index ,QString name ,QFont font)
{
	InfoTooltip* infos = findTooltip(name);
	infos->setFont(font);

	QString police = font.family();
	int size = font.pointSize();
	bool italic = font.italic();
	bool bold = font.bold();

	emit sendFontTooltip(index,police,size,italic,bold);
}

void ViewQt3D::destroyAllTooltip()
{

	int count = m_listeTooltips.count();

	m_listeTooltips.remove(0,count);

	emit removeAllTooltip();
}

void ViewQt3D::deleteTooltip(QString s)
{
	InfoTooltip* infos = findTooltip(s);
	if(infos != nullptr) m_listeTooltips.removeOne(infos);
	emit sendDestroyTooltip(s);
}

void ViewQt3D::showCamTooltip(int index)
{
	if(index >=0 && index < m_listeTooltips.count())
	{
		m_camera->setUpVector(QVector3D(0,-1,0));
		setAnimationCamera(1,m_listeTooltips[index]->position());
	}
}

QVector<InfoTooltip*> ViewQt3D::getAllTooltip()
{
	return m_listeTooltips;
}

void ViewQt3D::createNurbs(QVector<QVector3D> listepoints, bool withTangent,IsoSurfaceBuffer surface,GraphEditor_ListBezierPath* path,QString nameNurbs,QColor col)
{

	if(withTangent)
	{
		if(m_nurbsManager!= nullptr)m_nurbsManager->addNurbsFromTangent(m_camera->position(),m_root,this,listepoints,surface,nameNurbs,path,m_sceneTransform,col);
	}
	else
	{
		if(m_nurbsManager!= nullptr)m_nurbsManager->addNurbs(m_camera->position(),m_root,this,listepoints,surface);
	}
	emit sendNurbsName(m_nurbsManager->getCurrentName());
	/*m_managerNurbs->resetNurbs();
	for(int i=0;i<listepoints.count();i++)
	{
		m_managerNurbs->curveDrawMouseBtnDownGeomIsect(listepoints[i]);

	}
	m_managerNurbs->setDirectriceOk();*/
}

/*
void ViewQt3D::createNurbsFromTangent(QVector<QVector3D> listepoints)
{

	emit sendNurbsName(m_nurbsManager->getCurrentName());
}*/

void ViewQt3D::updateNurbs(QVector<QVector3D> listepoints,bool withTangent,QColor col )
{

	qDebug()<<"obsolete ...";
	if(withTangent)
	{
		//m_nurbsManager->updateDirectriceWithTangent(listepoints);
		emit refreshOrthoFromBezier();

	}
	else
		if(m_nurbsManager!= nullptr)m_nurbsManager->updateDirectrice(listepoints);
	/*m_managerNurbs->updateDirectriceCurve(listepoints);
	m_managerNurbs->setDirectriceOk();*/
}

void ViewQt3D::updateNurbs(GraphEditor_ListBezierPath* path,QColor col)
{



	if(m_nurbsManager!= nullptr)m_nurbsManager->updateDirectriceWithTangent(path,m_sceneTransform,col);
	emit refreshOrthoFromBezier();


}

void ViewQt3D::deleteGeneratriceNurbs(QString name )
{
	if(m_nurbsManager!= nullptr)m_nurbsManager->deleteCurrentGeneratrice(name);
}

void ViewQt3D::destroyNurbs(QString name)
{
	if(m_nurbsManager!= nullptr)m_nurbsManager->destroyNurbs(name);
	m_lastSelectedViews = -1;
}

void ViewQt3D::deleteNurbs(QString name)
{
	if( name !="")
	{

		if(m_nurbsManager!= nullptr)
		{
			RandomView3D*  rand3d = m_nurbsManager->getRandom(name);
			if(rand3d != nullptr)
			{
				deleteCurrentRandomView(rand3d);
				emit sendDeleteRandom3D(rand3d);
			}
			m_nurbsManager->deleteNurbs(name);
		}
		return;
	}
	if(m_nurbsManager!= nullptr)
	{
	RandomView3D*  rand3d = m_nurbsManager->getCurrentRandom();
	if(rand3d != nullptr)
	{
		deleteCurrentRandomView(rand3d);
		emit sendDeleteRandom3D(rand3d);
	}
	m_nurbsManager->deleteCurrentNurbs();
	m_lastSelectedViews = -1;
	}
	//m_managerNurbs->resetNurbs();
}

void ViewQt3D::selectNurbs(int index)
{
	selectRandomView(index);
	if(m_nurbsManager!= nullptr)m_nurbsManager->SelectNurbs(index);
}

void ViewQt3D::selectNurbs(QString n)
{
	if(m_nurbsManager!= nullptr)m_nurbsManager->SelectNurbs(n);
}

void ViewQt3D::setPrecisionNurbs(int value)
{
	if(m_nurbsManager!= nullptr)m_nurbsManager->setPrecision(value);
}

void ViewQt3D::setInterpolationNurbs(bool b)
{
	if(m_nurbsManager!= nullptr)m_nurbsManager->setInterpolation(b);
}

void ViewQt3D::setWireframeNurbs(bool b)
{
	if(m_nurbsManager!= nullptr)m_nurbsManager->setWireframe(b);
}

void ViewQt3D::exportNurbsObj(QString nom)
{

	QString pathfile = GraphicsLayersDirPath();


		QString directory="Nurbs/";
		QDir dir(pathfile);
		bool res = dir.mkpath("Nurbs");

	qDebug()<<" Save nurbs : "<<pathfile+directory+nom+".obj";

	if(m_nurbsManager!= nullptr)m_nurbsManager->exportNurbsObj(pathfile+directory+nom+".obj");
}


void ViewQt3D::createNewXSection(float pos)
{
	if(m_nurbsManager!= nullptr)m_nurbsManager->createNewXSection(pos);
	//m_managerNurbs->addinbetweenXsection(pos);
}

void ViewQt3D::createNewXSectionClone(float pos)
{
	if(m_nurbsManager!= nullptr)m_nurbsManager->createNewXSectionClone(pos);
	//m_managerNurbs->addinbetweenXsection(pos);
}

void ViewQt3D::createSection(QVector<QVector3D> listepoints,int index,bool isopen,bool withTangent)
{



	/*if(withTangent){

		m_nurbsManager->addGeneratriceFromTangent(listepoints,index,isopen);
	}
	else
	{*/
	if(m_nurbsManager!= nullptr)m_nurbsManager->addGeneratrice(listepoints,index,isopen);
	//}
	//m_managerNurbs->curveDrawSection(listepoints,index);
	//m_managerNurbs->endCurve();
}

void ViewQt3D::createSection(QVector<PointCtrl> listeCtrls,QVector<QVector3D> listepoints,int index,bool isopen,bool withTangent,QPointF cross,QVector<QVector3D>  listeCtrl3D,QVector<QVector3D>  listeTangent3D,QVector3D cross3D)
{


	if(m_nurbsManager!= nullptr)m_nurbsManager->addGeneratrice(listeCtrls,listepoints,index,listeCtrl3D,listeTangent3D,cross3D,isopen,cross);
}


void ViewQt3D::createSection(GraphEditor_ListBezierPath* path,RandomTransformation* transfo,int index)
{

	if(m_nurbsManager!= nullptr)m_nurbsManager->addGeneratrice(path,transfo);
}

void ViewQt3D::setSliderXsection(float pos)
{
	if(m_nurbsManager!= nullptr)m_nurbsManager->setSliderXsection(pos);
//	m_managerNurbs->setSliderXsectPos(pos);

}

void ViewQt3D::setSliderXsection(float pos,QVector3D position,QPointF normal)
{

	//qDebug()<<" position : "<<position;
	//float alt = getAltitude(position);


	//qDebug()<<" position : "<<position;
//	QVector3D pos2 = position;
//	pos2.setY(-m_heightBox*0.5f+ alt);
	QVector3D normal3D(normal.x(),0.0f,normal.y());

	QVector3D posit = m_nurbsManager->setSliderXsection(pos,position,normal3D);

	QVector3D newpos = m_sceneTransformInverse * posit;





//	emit sendCoefNurbsXYZ(newpos);


	if(m_followCam)
	{

		QVector3D newpos =m_transfoRootFils->scale3D()* position;
		QVector3D targetCam = newpos;
		targetCam.setY( m_targetY);
		QVector3D posCam = targetCam - normal3D *m_distanceCam*m_transfoRootFils->scale3D();

		m_camera->setPosition(posCam);
		m_camera->setUpVector(QVector3D(0,-1,0));
		m_camera->setViewCenter(targetCam);

		m_camera->tilt(m_inclinaisonCam);
	}

}


void ViewQt3D::receiveNurbsY(QVector3D pos, QVector3D normal)
{

	float value = pos.y();
	float distY =qFabs( -m_heightBox*0.5f+ value);
	float coef = distY /m_heightBox;
	emit sendCoefNurbsY(coef);

	QVector3D newpos = m_sceneTransformInverse * pos;

	emit sendCoefNurbsXYZ(newpos);

	emit receiveOrtho(newpos, normal);


	if( m_currentIndexRandom >= 0)
	{
		m_randomViews[m_currentIndexRandom]->update(pos,normal);
	}



	if(m_followCam)
	{

		QVector3D newpos =m_transfoRootFils->scale3D()* pos;
		QVector3D targetCam = newpos;
		targetCam.setY( m_targetY);
		QVector3D posCam = targetCam - normal *m_distanceCam*m_transfoRootFils->scale3D();

		m_camera->setPosition(posCam);
		m_camera->setUpVector(QVector3D(0,-1,0));
		m_camera->setViewCenter(targetCam);

		m_camera->tilt(m_inclinaisonCam);
	}
}

void ViewQt3D::receiveCurveData(QVector<PointCtrl> listePts,bool isopen,QPointF cross)
{
	/*std::vector<QVector3D> listePtsTr;
	for(int i=0;i<listePts.size();i++)
	{

		QVector3D postr = m_sceneTransformInverse * listePts[i];
		listePtsTr.push_back(postr);
	}*/



	emit sendCurveChanged(listePts,isopen,cross);
}


void ViewQt3D::receiveCurveDataOpt(GraphEditor_ListBezierPath* path)
{

	emit sendCurveChangedTangentOpt(path);

}
void ViewQt3D::receiveCurveData2(QVector<QVector3D> listePts,QVector<QVector3D> globalTangente3D,bool isopen,QPointF cross,QString nameNurbs)
{

	QVector<QVector3D> listePtsTr;
	for(int i=0;i<listePts.size();i++)
	{

		QVector3D postr = m_sceneTransformInverse * listePts[i];
		listePtsTr.push_back(postr);
	}

	QVector<QVector3D> listeTanTr;
	for(int i=0;i<globalTangente3D.size();i++)
	{

		QVector3D postr = m_sceneTransformInverse * globalTangente3D[i];
		listeTanTr.push_back(postr);
	}


	emit sendCurveChangedTangent(listePtsTr,listeTanTr,isopen,cross,nameNurbs);
}

void ViewQt3D::receiveCurveData(std::vector<QVector3D> listePts,bool isopen)
{
	std::vector<QVector3D> listePtsTr;
	for(int i=0;i<listePts.size();i++)
	{

		QVector3D postr = m_sceneTransformInverse * listePts[i];
		listePtsTr.push_back(postr);
	}


	emit sendCurveChanged(listePtsTr,isopen);
}


void ViewQt3D::hideTooltipWell()
{

	m_lastSelected = nullptr;

	mToolTip3D->setProperty("visible", false);
	mLineTooltip->setProperty("visible", false);
	/*if(m_line != nullptr)
	{
	delete m_line;
	m_line = nullptr;
	}*/
	m_currentTooltipProvider = nullptr;
}

void ViewQt3D::onAnimationFinished()
{
	//m_animRunning  = false;
}
void ViewQt3D::onAnimationStopped()
{
	if(m_animation->state() == QAbstractAnimation::Stopped)
		m_animRunning  = false;
	if(m_animation->state() == QAbstractAnimation::Running)
			m_animRunning  = true;
//	qDebug()<<m_animRunning<<" animation stopped : "<<m_animation->state();
}

void ViewQt3D::updateControler() {
	/*if (m_camera!=nullptr) {
		m_camera->setFarPlane(
            5
					* std::max( m_worldBounds.width(),
							std::max(m_currentScale * m_worldBounds.height(),
									 m_worldBounds.depth())));
	}

	if (m_controler!=nullptr) {
		m_controler->setProperty("linearSpeed",
				std::max(m_worldBounds.width() / 4,
						std::max(m_currentScale * m_worldBounds.height() / 4,
								m_worldBounds.depth() / 4)));

		m_controler->setProperty("lookSpeed", 50.0);
	}

*/


			/*std::max(m_worldBounds.width() / 500,
					std::max(m_currentScale * m_worldBounds.height() / 500,
							m_worldBounds.depth() / 500)));*/
}

void ViewQt3D::onZScaleChanged(int val)
{
	zScaleChanged((double)val);
}

void ViewQt3D::zScaleChanged(double val) {
	m_currentScale = val / 100.0;


	//m_transfoRoot->setScale3D(QVector3D(m_coefGlobal,m_coefGlobal*m_currentScale,m_coefGlobal));
	m_transfoRootFils->setScale3D(QVector3D(1.0,m_currentScale,1.0));
/*	for (AbstractGraphicRep *rep : m_visibleReps) {
		Graphic3DLayer *layer = rep->layer3D(m_quickview, m_root, m_camera);
		layer->zScale(m_currentScale);

	}
	updateControler();

	if(m_line!= nullptr)
	{

		m_lineTransfo->setScale3D(QVector3D(1.0,m_currentScale,1.0f));
	}

	for(int i=0;i<m_randomViews.count();i++)
	{
		m_randomViews[i]->setZScale(m_currentScale);
	}

	if(widgetpath) widgetpath->setZScale((int)(m_currentScale*100));
*/

	refreshTooltip();

/*	for(int i=0;i<m_listeTooltips.count();i++)
	{
		QVector3D dirTooltip =  m_listeTooltips[i]->position()- m_camera->position();
		dirTooltip = dirTooltip.normalized();

		QVector3D dirViewCam = m_camera->viewVector().normalized();
		float dot = QVector3D::dotProduct(dirViewCam,dirTooltip);
		if(dot >0 )
		{
			QVector2D pos2D = getPos2D(m_listeTooltips[i]->position(),m_listeTooltips[i]->getZScale());
			emit updateTooltip(i, (int)(pos2D.x()),(int)(pos2D.y()));
		}


	}*/


	if (m_syncCamera) {
		emit zScaleChangedSignal(m_currentScale);
	}
}



void ViewQt3D::positionCameraChanged(QVector3D pos) {

	for (AbstractGraphicRep *rep : m_visibleReps) {

		m_camera->setPosition(pos);
	}
	if (m_syncCamera)
	{

		float sca = 1.0f;
		if(widgetpath != nullptr && widgetpath->m_useScale) sca =m_coefGlobal;
		emit positionCamChangedSignal(pos/sca);///m_coefGlobal);
	}

	if(m_nurbsManager != nullptr)
	{
		m_nurbsManager->setPositionLight(pos);
	}
	refreshTooltip();

}

void ViewQt3D::refreshTooltip()
{
	for(int i=0;i<m_listeTooltips.count();i++)
	{
		QVector3D dirTooltip =  m_listeTooltips[i]->position()- m_camera->position();
		dirTooltip = dirTooltip.normalized();

		QVector3D dirViewCam = m_camera->viewVector().normalized();
		float dot = QVector3D::dotProduct(dirViewCam,dirTooltip);
		if(dot >0 )
		{

			QVector2D pos2D = getPos2D(m_listeTooltips[i]->position(),m_listeTooltips[i]->getZScale());
			emit updateTooltip(i, (int)(pos2D.x()),(int)(pos2D.y()));
		}


	}
}

void ViewQt3D::viewCenterCameraChanged(QVector3D center) {

	for (AbstractGraphicRep *rep : m_visibleReps) {

		m_camera->setViewCenter(center);
	}
	if (m_syncCamera) {
		float sca = 1.0f;
		if(widgetpath != nullptr && widgetpath->m_useScale) sca =m_coefGlobal;
		emit viewCenterCamChangedSignal(center/sca);///m_coefGlobal);
	}

}

void ViewQt3D::upVectorCameraChanged(QVector3D up) {

	for (AbstractGraphicRep *rep : m_visibleReps) {
		m_camera->setUpVector(up);
	}
	if (m_syncCamera) {
		emit upVectorCamChangedSignal(up);
	}
}

void ViewQt3D::resetZoom() {
	//m_camera->viewAll(); // does not work as expected : bounding volume does not match the current m_worldBounds (off centered)
	float fov = m_camera->fieldOfView() * M_PI / 180;
	float zPosCamera = -std::max(
			m_worldBounds.depth() / std::tan(fov),
			std::max(m_worldBounds.width() / std::tan(fov),
					m_currentScale * m_worldBounds.height()
							/ std::tan(fov)));

	float y = -m_currentScale * m_worldBounds.height();
	m_camera->setViewCenter(QVector3D(0, 0, 0));
	m_camera->setPosition(QVector3D(0, y, +zPosCamera));//*m_coefGlobal));
	m_camera->setUpVector(QVector3D(0,-1,0));

	refreshTooltip();
}

bool ViewQt3D::updateWorldExtent(const QRect3D &worldExtent) {
	bool changed = false;
	if (!m_wordBoundsInitialized) {
		m_worldBounds = worldExtent;
		m_wordBoundsInitialized = true;
		changed = true;
	} else
		changed = m_worldBounds.merge(worldExtent);

	//if (changed) {
	//	updateControler();
	//}
	/*if(changed)
	{
		if(m_cylMesh != nullptr)
		{
			if(m_worldBounds.height() != m_cylMesh->length())
				m_cylMesh->setLength(m_worldBounds.height());
		}
	}*/
	//qDebug()<<"m_worldBounds changed  "<<m_worldBounds.height();
	return changed;
}

void ViewQt3D::showRep(AbstractGraphicRep *rep) {
	if (!m_qmlLoaded) {
		m_waitingReps.enqueue(rep);
		return;
	}




	//Add the graphic object to the scene
	bool isAddedCorrectly = true;
	QStringList errorMsg;

	ISampleDependantRep* sampleRep = dynamic_cast<ISampleDependantRep*>(rep);
	if (sampleRep!=nullptr) {
		QList<SampleUnit> units = sampleRep->getAvailableSampleUnits();
		if (m_sectionType==SampleUnit::NONE && units.count()>0) {
			m_sectionType = units[0];
			if (m_GPURes) {
				m_GPURes->setSectionType(m_sectionType);
				m_GPURes->setDepthLengthUnit(m_depthLengthUnit);
			}
			isAddedCorrectly = sampleRep->setSampleUnit(m_sectionType);
		} else if (m_sectionType!=SampleUnit::NONE && units.contains(m_sectionType)) {
			isAddedCorrectly = sampleRep->setSampleUnit(m_sectionType);
		} else{
			isAddedCorrectly = false;
		}
		if (!isAddedCorrectly && m_sectionType!=SampleUnit::NONE) {
			errorMsg << sampleRep->getSampleUnitErrorMessage(m_sectionType);
		} else if (!isAddedCorrectly) {
			errorMsg << "Display unit unknown";
		}
	}
	if (isAddedCorrectly) {

        Graphic3DLayer *layer = rep->layer3D(m_quickview, m_root, m_camera);

        connect(layer,SIGNAL(sendInfosCam(QVector3D , QVector3D)), this,SLOT(addNewInfosCam(QVector3D , QVector3D)));

        if (m_visibleReps.size()==0)
        {

        	m_working = rep->data()->workingSetManager();
        	m_nurbsManager->setWorkingSetManager(m_working);
        	NurbsWidget::setView3D(this);
        	NurbsWidget::addView3d(this);

        	m_heightBox = layer->boundingRect().height();
        	m_ydepart = layer->boundingRect().y();
            // set transform as an offset in 3D
            m_sceneTransform.setToIdentity(); // reset before asking bbox
            QRect3D rect = layer->boundingRect(); // bbox use new transform
            QVector3D center(-rect.x() - rect.width()/2,
                    -rect.y() - rect.height()/2,
                    -rect.z() - rect.depth()/2);
            m_sceneTransform.translate(center);
            bool inversible;


            //widgetpath->setSceneTransform(m_sceneTransform);
          //  qDebug()<<" tranformation =====>"<<m_sceneTransform;
            m_sceneTransformInverse = m_sceneTransform.inverted(&inversible);
            if(!inversible) qDebug()<<" m_sceneTransform n'est pas inversible";
        }
        updateWorldExtent(layer->boundingRect());
		layer->show();



		//layer->zScale(m_currentScale);

		if (m_visibleReps.size() == 0) {




			float fov = m_camera->fieldOfView() * M_PI / 180;
			float zPosCamera = -std::max(
					m_worldBounds.depth() / std::tan(fov),
					std::max(m_worldBounds.width() / std::tan(fov),
							m_currentScale * m_worldBounds.height()
									/ std::tan(fov)));

			float y = -m_currentScale * m_worldBounds.height();
			m_camera->setViewCenter(QVector3D(0, 0, 0));
			m_camera->setPosition(QVector3D(0, y, +zPosCamera));
			m_camera->setUpVector(QVector3D(0,-1,0));
		//	updateControler();
			emit positionCamChangedSignal(m_camera->position());
			emit viewCenterCamChangedSignal(m_camera->viewCenter());
			emit upVectorCamChangedSignal(m_camera->upVector());


		//	qDebug()<<" m_worldBounds width:"<<m_worldBounds.width();
		//	qDebug()<<" m_worldBounds height:"<<m_worldBounds.height();
		//	qDebug()<<" m_worldBounds depth:"<<m_worldBounds.depth();


			float maxsize = fmax(m_worldBounds.width(),m_worldBounds.height());
			float zoneVisible = 10.0f*maxsize;//10

			m_coefGlobal = 100000.0f/ zoneVisible;


			m_transfoRoot->setScale3D(QVector3D(m_coefGlobal,m_coefGlobal,m_coefGlobal));

		//	qDebug()<<"==============>m_coefGlobal : "<<m_coefGlobal;

			if(m_line == nullptr)
			{

				m_line = new Qt3DCore::QEntity(m_root);

				m_cylMesh = new Qt3DExtras::QCylinderMesh();
				m_cylMesh->setRadius(20.0f);
				m_cylMesh->setLength(m_heightBox);


				m_lineTransfo = new Qt3DCore::QTransform();

				Qt3DExtras::QPhongMaterial* material = new Qt3DExtras::QPhongMaterial(m_root);
				material->setAmbient(QColor(255, 0, 0, 255));
				material->setDiffuse(QColor(255,0, 0, 255));
				material->setSpecular(QColor(0, 0, 0, 0));
				m_line->addComponent(m_cylMesh);
				m_line->addComponent(m_lineTransfo);
				m_line->addComponent(material);


				m_line->setEnabled(true);
			}

			Qt3DCore::QEntity* lightentity = new Qt3DCore::QEntity(m_camera);
			Qt3DRender::QPointLight* light= new Qt3DRender::QPointLight(lightentity);
			light->setColor("white");
			light->setIntensity(0.5f);

			lightentity->addComponent(light);



		}



		CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
		if (cameraCtrl != nullptr)
		{
			SurfaceCollision* surfaceColl = dynamic_cast<SurfaceCollision*>(layer);
			if(surfaceColl != nullptr)cameraCtrl->setSurfaceCollision(surfaceColl);
		}
		if (m_visibleReps.size() == 0) {
			QString pathfile = GraphicsLayersDirPath();

			widgetpath =new WidgetPath3d(cameraCtrl,m_sceneTransform,m_camera,m_coefGlobal,this);
			widgetpath->setPathFiles(pathfile);
			widgetpath->setZScale((int)(m_currentScale*100));
			widgetpath->newPath();

			connect(this,SIGNAL(signalSpeedAnim(int)),widgetpath,SLOT(speedAnimChanged(int)));
			connect(this,SIGNAL(signalAltitudeAnim(int)),widgetpath,SLOT(altitudeAnimChanged(int)));

		}
	}

	//Qt3DCore::QTransform *transfolight = new Qt3DCore::QTransform();
	//transfolight->setTranslation(QVector3D(0, -500, 0));


//	lightentity->addComponent(transfolight);


	if (isAddedCorrectly) {
		AbstractInnerView::showRep(rep);

	} else{
		// fail to add
		qDebug() << "ViewQt3D : fail to add rep " << rep->name() << " error messages : "<< errorMsg;
	}



	if (m_visibleReps.size()==1) {
		Qt3DHelpers::drawLine({ 0, 0, 0 }, { 1000, 0, 0 }, Qt::red, m_root); // X
		Qt3DHelpers::drawLine({ 0, 0, 0 }, { 0, 1000, 0 }, Qt::green, m_root); // Y
		Qt3DHelpers::drawLine({ 0, 0, 0 }, { 0, 0, 1000 }, Qt::blue, m_root);



	}

}

void ViewQt3D::hideRep(AbstractGraphicRep *rep) {
	if (m_waitingReps.size()!=0) {
		bool removed = m_waitingReps.removeOne(rep);
		if (removed) {
			return;
		}// else rep may already be added
	}


	Graphic3DLayer *layer = rep->layer3D(m_quickview, m_root, m_camera);

	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	if (cameraCtrl != nullptr)
	{
		SurfaceCollision* surfaceColl = cameraCtrl->surfaceCollision();
		if(surfaceColl != nullptr)
		{
			SurfaceCollision* coll= dynamic_cast<SurfaceCollision*>(layer);
			if(coll == surfaceColl)
			{
				cameraCtrl->setSurfaceCollision(nullptr);
			}
		}
	}

	// safety measure, a tooltip provider should be in the 3D scene
	if (m_currentTooltipProvider!=nullptr && m_currentTooltipProvider==dynamic_cast<IToolTipProvider*>(layer))
	{
		m_currentTooltipProvider = nullptr;
	}

	layer->hide();
	AbstractInnerView::hideRep(rep);


	if (m_visibleReps.count()==0) {
		m_sectionType = SampleUnit::NONE;
		if (m_GPURes) {
			m_GPURes->setSectionType(m_sectionType);
			m_GPURes->setDepthLengthUnit(m_depthLengthUnit);
		}
	}
}

const QMatrix4x4& ViewQt3D::sceneTransform() const {
	return m_sceneTransform;
}

const QMatrix4x4& ViewQt3D::sceneTransformInverse() const {
	return m_sceneTransformInverse;
}

QRect3D ViewQt3D::worldBounds()
{
	return m_worldBounds;
}

float ViewQt3D::zScale() const
{
	return m_currentScale;
}

void ViewQt3D::addNewInfosCam(QVector3D posCam, QVector3D targetCam)
{
	/*if(widgetpath)
	{

		widgetpath->AddPoints(posCam,targetCam,2000);
	}*/
}

void ViewQt3D::moveLineVert(float posX, float posZ)
{

	if(m_line!= nullptr)
	{
		m_lineTransfo->setTranslation(QVector3D(posX,0.0f,posZ));
	}
}

void ViewQt3D::showLineVert(bool b)
{
	if(m_line!= nullptr)
	{
		if(m_line->isEnabled() != b)
			m_line->setEnabled(b);
	}
}

void ViewQt3D::setZScale(double zScale) {
	if (!m_syncCamera) {
		return;
	}


//	qDebug()<<"==> setZScale : "<<zScale;
	// update layers
	m_currentScale = zScale;
	for (AbstractGraphicRep *rep : m_visibleReps) {
		Graphic3DLayer *layer = rep->layer3D(m_quickview, m_root, m_camera);
		layer->zScale(m_currentScale);
	}
	//


	updateControler();



	m_spinZScale->setValue(m_currentScale*100);
	/*if (m_GPURes!=nullptr) {
		m_GPURes->setzScale(m_currentScale*100);
	}*/
}

void ViewQt3D::setPositionCam(QVector3D pos) {
	if (!m_syncCamera) {
		return;
	}
	// update layers
	//m_currentScale = zScale;
	for (AbstractGraphicRep *rep : m_visibleReps) {
		//Graphic3DLayer *layer = rep->layer3D(m_quickview, m_root, m_camera);
		//layer->zScale(m_currentScale);

	/*	m_animation->stop();

		m_animation->setDuration(200);
		m_animation->setStartValue(m_camera->position());
		m_animation->setEndValue(pos);

		m_animation->start();*/
		m_camera->setPosition(pos);

	}

	/*if(m_nurbsManager != nullptr)
	{
		m_nurbsManager->setPositionLight(pos);
	}*/
	//
	updateControler();

/*	if (m_GPURes!=nullptr) {
		m_GPURes->setzScale(m_currentScale*100);
	}*/
}

bool ViewQt3D::getAnimRunning() const
{

	if(m_animation->state() == QAbstractAnimation::Running ||m_animationCenter->state() == QAbstractAnimation::Running)
		return true;
	return false;
}

void ViewQt3D::setViewCenterCam(QVector3D center) {
	if (!m_syncCamera) {
		return;
	}
	// update layers
	//m_currentScale = zScale;
	for (AbstractGraphicRep *rep : m_visibleReps) {
		//Graphic3DLayer *layer = rep->layer3D(m_quickview, m_root, m_camera);
		//layer->zScale(m_currentScale);
		m_camera->setViewCenter(center);
	/*	m_animationCenter->stop();

		m_animationCenter->setDuration(200);
		m_animationCenter->setStartValue(m_camera->viewCenter());
		m_animationCenter->setEndValue(center);
		m_animationCenter->start();*/
	}
	//
	updateControler();

}

void ViewQt3D::setUpVectorCam(QVector3D up)
{
	if (!m_syncCamera) {
		return;
	}

	for (AbstractGraphicRep *rep : m_visibleReps) {
		m_camera->setUpVector(up);
	}

	updateControler();
}


QVector3D ViewQt3D::positionCam() const
{
	if( m_camera != nullptr)
	return m_camera->position();

	return QVector3D(0.0f,0.0f,0.0f);
}

QVector3D ViewQt3D::targetCam() const
{
	if( m_camera != nullptr)
		return m_camera->viewCenter();

	return QVector3D(0.0f,0.0f,0.0f);
}

void ViewQt3D::updateSyncButton() {
	if (m_syncCamera) {
		m_syncButton->setIcon(QIcon(QString(":/slicer/icons/icons8-epingle-96.png")));
	} else {
		m_syncButton->setIcon(QIcon(QString(":/slicer/icons/icons8-detacher-96.png")));
	}
}

void ViewQt3D::distanceTargetChanged(float distance, QVector3D target)
{

	float ray = tan(M_PI/120.0f)* distance;
	m_coneMesh->setBottomRadius( ray);
	m_coneMesh->setLength(distance);
	m_coneTransfo->setTranslation(QVector3D(0.0f,-10.0f,-m_coneMesh->length()*0.5f -20.0f));
	QVector2D target2D(target.x(),target.z());

	QVector2D position2D(m_camera->position().x(),m_camera->position().z());
	float distance2d = (target2D-position2D).length();

	emit distanceTargetChangedSignal(distance2d);
}

void ViewQt3D::changeDistanceForcing(float d)
{
	emit distanceTargetChangedSignal(d);
}

void ViewQt3D::toggleFlyCamera()
{
	m_flyCamera = ! m_flyCamera;
/*	if (!m_flyCamera) {
		m_flyButton->setIcon(QIcon(QString(":/slicer/icons/helicopter-1770.svg")));
		} else {
			m_flyButton->setIcon(QIcon(QString(":/slicer/icons/helicopter-1770.svg")));
		}*/
	if(/*m_coneEntity == nullptr &&*/ m_flyCamera)
	{
		m_coneEntity = new Qt3DCore::QEntity(m_camera);

		float ray = tan(M_PI/120.0f)* 5000.0f;

		m_coneMesh = new Qt3DExtras::QConeMesh();
		m_coneMesh->setTopRadius(0.001f);
		m_coneMesh->setBottomRadius( ray);
		m_coneMesh->setLength(5000.0f);
		m_coneMesh->setHasBottomEndcap(false);
		m_coneMesh->setHasTopEndcap(false);


		m_coneTransfo = new Qt3DCore::QTransform();
		m_coneTransfo->setTranslation(QVector3D(0.0f,-10.0f,-m_coneMesh->length()*0.5f -20.0f));
		m_coneTransfo->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1.0f, 0.0f, 0.0f), 90.0f));
		Qt3DExtras::QPhongAlphaMaterial* material = new Qt3DExtras::QPhongAlphaMaterial(m_root);
		material->setAmbient(QColor(255, 0, 0, 255));
		material->setDiffuse(QColor(255, 0, 0, 255));
		material->setSpecular(QColor(0, 0, 0, 0));
		material->setSourceRgbArg(Qt3DRender::QBlendEquationArguments::SourceAlpha);
		material->setDestinationRgbArg(Qt3DRender::QBlendEquationArguments::OneMinusSourceAlpha);
		material->setBlendFunctionArg(Qt3DRender::QBlendEquation::Add);
		material->setSourceAlphaArg(Qt3DRender::QBlendEquationArguments::Zero);
		material->setDestinationAlphaArg(Qt3DRender::QBlendEquationArguments::One);
		material->setAlpha(0.4f);
		m_coneEntity->addComponent(m_coneMesh);
		m_coneEntity->addComponent(m_coneTransfo);
		m_coneEntity->addComponent(material);


		/*if(m_helico == nullptr)
		{
			m_helico  = Qt3DHelpers::loadObj("N916MU.obj", m_root);
			Qt3DCore::QTransform* helicoTransform = new Qt3DCore::QTransform();
			helicoTransform->setRotationX(180.0f);

			Qt3DExtras::QTextureMaterial *backgroundMaterial = new Qt3DExtras::QTextureMaterial;
			Qt3DRender::QTexture2D *      backgroundTexture  = new Qt3DRender::QTexture2D(backgroundMaterial);
			Qt3DRender::QTextureImage *   backgroundImage    = new Qt3DRender::QTextureImage(backgroundMaterial);
			backgroundImage->setSource(QUrl::fromLocalFile("Diffuse2.png"));
			backgroundTexture->addTextureImage(backgroundImage);
			backgroundMaterial->setTexture(backgroundTexture);





			if(m_helico != nullptr)
			{
				m_helico->addComponent(helicoTransform);
				m_helico->addComponent(backgroundMaterial);
			}
		}*/


	}

	if(m_coneEntity != nullptr)
	{
		m_coneEntity->setEnabled(m_flyCamera);
	}

/*	if(m_pointer == nullptr && m_flyCamera)
	{
		 m_pointer = Qt3DHelpers::drawLine({ 0, -10, 0 }, { 0, 0, -5000 }, Qt::red, m_camera);
	}
	if(m_pointer != nullptr )m_pointer->setEnabled( m_flyCamera);*/
	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	if (cameraCtrl != nullptr)
	{
		m_coneEntity->addComponent(cameraCtrl->getLayer());
		cameraCtrl->setFlyCamera();
		//emit showHelico2D(m_flyCamera);
	}
}

void ViewQt3D::showHelico( bool visible)
{
	emit showHelico2D(visible);
}

void ViewQt3D::showHelp()
{
	m_shortcut = new WidgetShorcut(this);
	m_shortcut->show();
}
/*
void ViewQt3D::showProperty()
{
	m_properties = new WidgetProperties(this);
	m_properties->show();
}*/


void ViewQt3D::setInfosVisible(bool visible)
{
	m_visibleInfos3d = visible;
	if(m_infos3d != nullptr)
	{
		m_infos3d->setProperty("visible", visible);
	}
}

void ViewQt3D::setGizmoVisible(bool visible)
{
	m_visibleGizmo3d= visible;
	if(m_gizmo != nullptr)
	{
		m_gizmo->setEnabled(visible);
	}
}

void ViewQt3D::setSpeedHelico( float val)
{
	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
		if (cameraCtrl != nullptr)
		{
			cameraCtrl->setCoefSpeed(val);
		}

}

void ViewQt3D::setSpeedRotHelico( float val)
{
	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	if (cameraCtrl != nullptr)
	{
		cameraCtrl->setRotSpeed(val);
	}
}


void ViewQt3D::setSpeedUpDown( float val)
{
	m_speedUpDown = val;

}


void ViewQt3D::stopDownUpCamera()
{
	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	if (cameraCtrl != nullptr)
	{
		m_vitesse = 0.5f;
		cameraCtrl->setDecalUpDown(0.0f);
	}
}


void ViewQt3D::downCamera()
{
	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	if (cameraCtrl != nullptr)
	{
		if(m_vitesse < 5.0f) m_vitesse+=0.1f;
		cameraCtrl->setDecalUpDown(-0.1f*m_vitesse*m_speedUpDown,10.0f);
	}

}

void ViewQt3D::upCamera()
{
	CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
	if (cameraCtrl != nullptr)
	{
		if(m_vitesse < 5.0f) m_vitesse+= 0.1f;
		cameraCtrl->setDecalUpDown(0.1f*m_vitesse*m_speedUpDown,-10.0f);
	}
}

/*
Qt3DRender::QLayer* ViewQt3D::getLayerTransparent() const
{
//	if(m_layerTransparent == nullptr)
		qDebug()<<"m_layerTransparent est NUUUUUUUUUULLLLLL ";
	return m_layerTransparent;
}

Qt3DRender::QLayer* ViewQt3D::getLayerOpaque() const
{
	//if(m_layerOpaque == nullptr)
		qDebug()<<"m_layerOpaque est NUUUUUUUUUULLLLLL ";
	return m_layerOpaque;
}

*/
void ViewQt3D::toggleSyncCamera() {
	m_syncCamera = !m_syncCamera;
	updateSyncButton();
}

const MtLengthUnit* ViewQt3D::depthLengthUnit() const {
	return m_depthLengthUnit;
}

void ViewQt3D::setDepthLengthUnit(const MtLengthUnit* depthLengthUnit) {
	if ((*m_depthLengthUnit)!=(*depthLengthUnit)) {
		m_depthLengthUnit = depthLengthUnit;

		if (m_GPURes) {
			m_GPURes->setDepthLengthUnit(m_depthLengthUnit);
		}

		/*if (*m_depthLengthUnit==MtLengthUnit::METRE) {
			m_depthUnitToggle->defaultAction()->setIcon(QIcon(QString(":/slicer/icons/regle_m128_blanc.png")));
			//m_depthUnitToggle->defaultAction()->setText("m");
		} else if (*m_depthLengthUnit==MtLengthUnit::FEET) {
			m_depthUnitToggle->defaultAction()->setIcon(QIcon(QString(":/slicer/icons/regle_ft128_blanc.png")));
			//m_depthUnitToggle->defaultAction()->setText("ft");
		}*/

		if (m_currentTooltipProvider) {
			QString name = m_currentTooltipProvider->generateToolTipInfo();
			if (m_tooltipProviderIsWell) {
				refreshWellTooltip(name);
			} else {
				refreshPickTooltip(name);
			}
		}
	}
}

void ViewQt3D::toggleDepthUnit() {
	if (*m_depthLengthUnit==MtLengthUnit::METRE) {
		setDepthLengthUnit(&MtLengthUnit::FEET);
	} else if (*m_depthLengthUnit==MtLengthUnit::FEET) {
		setDepthLengthUnit(&MtLengthUnit::METRE);
	}
}

ViewQt3D::~ViewQt3D() {
	NurbsWidget::removeView3d(this);
	m_lastSelectedViews = -1;

	if(m_nurbsManager)
	{
		m_nurbsManager->deleteLater();
		m_nurbsManager = nullptr;
	}

}
