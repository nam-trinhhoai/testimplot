#include "tools3dWidget.h"
#include "qt3dhelpers.h"
#include "singlesectionview.h"
#include "randomlineview.h"
#include "GraphEditor_LineShape.h"
#include "GraphEditor_ListBezierPath.h"

#include "surfacemeshcacheutils.h"
#include <QByteArray>
#include <QColorDialog>

#include "GraphicToolsWidget.h"

#include "stackbasemapview.h"

Tools3dWidget::Tools3dWidget(QWidget* parent):QDialog(parent)
{

	m_view2D = nullptr;
	m_view3D = nullptr;
	m_viewInline = nullptr;
	m_randomRep = nullptr;
	m_indexRandomRep = -1;
	m_max = 1.0f;
	setWindowTitle(Tools3dWidgetTitle);
	setModal(false);
	setMinimumWidth(400);
	setMinimumHeight(450);//300

	m_timerDist = new QTimer(this);
	m_timerAlt = new QTimer(this);


	 connect(m_timerDist,SIGNAL(timeout()), this,SLOT(onTimeout()));
	 connect(m_timerAlt,SIGNAL(timeout()), this,SLOT(onTimeoutAlt()));

	QGridLayout *layout = new QGridLayout();


	//random views
	QGroupBox* boxRandomView = new QGroupBox("RandomView");
	QLabel* labelview= new QLabel("Views");

	m_comboViews = new QComboBox();

	QToolButton* buttonSelect = new QToolButton();
	buttonSelect->setIcon(QIcon(":/slicer/icons/graphic_tools/mouse.png"));// QStyle::SC_TitleBarContextHelpButton));
	buttonSelect->setToolTip("select view");

	QToolButton* buttonSuppr = new QToolButton();
	buttonSuppr->setIcon(QIcon(":/slicer/icons/graphic_tools/delete.png"));// QStyle::SC_TitleBarContextHelpButton));
	buttonSuppr->setToolTip("delete view");

	QGridLayout *layview = new QGridLayout();
	layview->addWidget(labelview, 0, 0, 1, 1);
	layview->addWidget(m_comboViews, 0, 1, 1, 2);
	layview->addWidget(buttonSelect, 0, 3, 1, 1);
	layview->addWidget(buttonSuppr, 0, 4, 1, 1);

		//lay->addWidget(m_editline, 0, 2, 1, 1);
	boxRandomView->setLayout(layview);


	 connect(buttonSelect, &QPushButton::clicked, this, &Tools3dWidget::selectRandomView);
	 connect(buttonSuppr, &QPushButton::clicked, this, &Tools3dWidget::deleteRandomView);




	//les nurbs
	QGroupBox* boxNurbs = new QGroupBox("Nurbs");

	QLabel* labelnurbs= new QLabel("Nurbs");
	m_comboNurbs = new QComboBox();

	QCheckBox* wireframeCheckbox = new QCheckBox("Wireframe");

	QToolButton *loadNurbsButton = new QToolButton();
	loadNurbsButton->setIcon(style()->standardPixmap( QStyle::QStyle::SP_FileDialogNewFolder));// QStyle::SC_TitleBarContextHelpButton));
	loadNurbsButton->setToolTip("Load nurbs");

	QToolButton* saveNurbsButton = new QToolButton();
	saveNurbsButton->setIcon(style()->standardPixmap( QStyle::SP_DialogSaveButton));// QStyle::SC_TitleBarContextHelpButton));
	saveNurbsButton->setToolTip("Save nurbs");


	connect(loadNurbsButton, SIGNAL(clicked()),this, SLOT(loadNurbs()));
	connect(saveNurbsButton, SIGNAL(clicked()),this, SLOT(exportNurbs()));

	QLabel* labelQuality= new QLabel("Precision");
	QSlider *sliderPrecision = new QSlider(Qt::Horizontal);
	sliderPrecision->setMinimum(10);
	sliderPrecision->setMaximum(140);
	sliderPrecision->setValue(40);


	QCheckBox* interpolationCheckbox = new QCheckBox("Linear interpolation");
	interpolationCheckbox->setCheckState(Qt::Checked);

	QLabel* label2= new QLabel("Directrice");
	m_buttonDirectrice = new QPushButton;
	QString namecolor2= "QPushButton {background-color: rgb("+QString::number(m_directriceColor.red())+","+QString::number(m_directriceColor.green())+","+QString::number(m_directriceColor.blue())+")}";
	m_buttonDirectrice->setStyleSheet(namecolor2);
	connect(m_buttonDirectrice,SIGNAL(clicked()),this,SLOT(setDirectriceColor()));

	QLabel* label3= new QLabel("Nurbs");
	m_buttonNurbs = new QPushButton;
	QColor color3(0,0,255,255);
	QString namecolor3= "QPushButton {background-color: rgb("+QString::number(color3.red())+","+QString::number(color3.green())+","+QString::number(color3.blue())+")}";
	m_buttonNurbs->setStyleSheet(namecolor3);
	connect(m_buttonNurbs,SIGNAL(clicked()),this,SLOT(setNurbsColor()));


	connect(interpolationCheckbox,SIGNAL(stateChanged(int)),this,SLOT(setLinearInterpolation(int)));
	connect(sliderPrecision,SIGNAL(valueChanged(int)), this, SLOT(setNurbsPrecision(int)));
//	connect(m_comboNurbs,SIGNAL(currentIndexChanged(int)),this,SLOT(nurbsSelectedChanged(int)));


	connect(wireframeCheckbox,SIGNAL(stateChanged(int)),this,SLOT(setWireframe(int)));


	/*QLabel* label1= new QLabel("Cross section");

	QSlider *m_sliderCross = new QSlider(Qt::Horizontal);
	m_sliderCross->setMinimum(0);
	m_sliderCross->setMaximum(1000);
	m_sliderCross->setValue(0);
	m_editline = new QLineEdit();


	connect(m_sliderCross,SIGNAL(sliderMoved(int)),this,SLOT(moveCrossSection(int)));
	connect(m_editline,SIGNAL(editingFinished()),this,SLOT(moveSection()));
*/
	QGridLayout *lay = new QGridLayout();
	lay->addWidget(labelnurbs, 0, 0, 1, 1);
	lay->addWidget(m_comboNurbs, 0, 1, 1, 1);
	lay->addWidget(loadNurbsButton, 0, 2, 1, 1);
	lay->addWidget(saveNurbsButton, 0, 3, 1, 1);

	lay->addWidget(label2, 1, 0, 1, 1);
	lay->addWidget(m_buttonDirectrice, 1, 1, 1, 1);
	lay->addWidget(label3, 1, 2, 1, 1);
	lay->addWidget(m_buttonNurbs, 1, 3, 1, 1);
	lay->addWidget(labelQuality, 2, 0, 1, 1);
	lay->addWidget(sliderPrecision, 2, 1, 1, 1);
	lay->addWidget(interpolationCheckbox, 2, 2, 1, 2);
	lay->addWidget(wireframeCheckbox,3, 0, 1,1 );
	boxNurbs->setLayout(lay);


	//Tooltip 3D
	QGroupBox* boxTooltip = new QGroupBox("Tooltip 3D");

	QLabel* labelTooltip= new QLabel("Tooltip");
	m_comboTooltip = new QComboBox();

	QPushButton* m_paramButton = new QPushButton("Font",parent);
//	m_paramButton->setIcon(QIcon(QString("slicer/icons/graphic_tools/text.png")));//":/slicer/icons/property.svg")));
//	m_paramButton->setIconSize(QSize(36, 36));
	m_paramButton->setToolTip("Font settings");

	connect(m_paramButton, &QPushButton::clicked, this, &Tools3dWidget::showFont);
	QPushButton* m_paramButton2 = new QPushButton("Color",parent);
	m_paramButton2->setToolTip("Font color");
	connect(m_paramButton2, &QPushButton::clicked, this, &Tools3dWidget::showColor);

	/*m_comboSize= new QComboBox(parent);
	m_comboSize->addItem("12");m_comboSize->addItem("14");m_comboSize->addItem("16");
	m_comboSize->addItem("18");m_comboSize->addItem("20");m_comboSize->addItem("22");
	m_comboSize->addItem("24");m_comboSize->addItem("26");m_comboSize->addItem("28");


	connect(m_comboSize, SIGNAL(currentIndexChanged(int)), this, SLOT(setSize(int)));*/



	connect(m_comboTooltip, SIGNAL(currentIndexChanged(int)), this, SLOT(selectCombo(int)));

	QToolButton *newButton = new QToolButton();
	newButton->setIcon(style()->standardPixmap( QStyle::QStyle::SP_FileDialogNewFolder));// QStyle::SC_TitleBarContextHelpButton));
	newButton->setToolTip("Load tooltip");

	QToolButton* saveButton = new QToolButton();
	saveButton->setIcon(style()->standardPixmap( QStyle::SP_DialogSaveButton));// QStyle::SC_TitleBarContextHelpButton));
	saveButton->setToolTip("Tooltip save");

	QToolButton* buttonShowTooltip = new QToolButton();
	buttonShowTooltip->setIcon(QIcon(":/slicer/icons/mainwindow/GeotimeView.svg"));// QStyle::SC_TitleBarContextHelpButton));
	buttonShowTooltip->setToolTip("Show tooltip");

	QToolButton* buttonSupprTooltip = new QToolButton();
	buttonSupprTooltip->setIcon(QIcon(":/slicer/icons/graphic_tools/delete.png"));// QStyle::SC_TitleBarContextHelpButton));
	buttonSupprTooltip->setToolTip("Delete tooltip");

	//QLabel* labelfont= new QLabel("Size font");
	//QLineEdit* fontEdit = new QLineEdit("12");



	QGridLayout *layTooltip = new QGridLayout();
	layTooltip->addWidget(labelTooltip, 0, 0, 1, 1);
	layTooltip->addWidget(m_comboTooltip, 0, 1, 1, 1);
	layTooltip->addWidget(m_paramButton, 0, 2, 1, 1);
	layTooltip->addWidget(m_paramButton2, 0, 3, 1, 1);
	layTooltip->addWidget(newButton, 0, 4, 1, 1);
	layTooltip->addWidget(saveButton, 0, 5, 1, 1);
	layTooltip->addWidget(buttonShowTooltip, 0, 6, 1, 1);
	layTooltip->addWidget(buttonSupprTooltip, 0,7, 1, 1);
	//layTooltip->addWidget(labelfont, 1, 0, 1, 1);
	//layTooltip->addWidget(fontEdit, 1,2, 1, 1);
	//lay->addWidget(m_editline, 0, 2, 1, 1);

	boxTooltip->setLayout(layTooltip);

	connect(newButton, SIGNAL(clicked()),this, SLOT(loadTooltip()));
	connect(saveButton, SIGNAL(clicked()),this, SLOT(saveTooltip()));
	connect(buttonShowTooltip, SIGNAL(clicked()),this, SLOT(showCamTooltip()));
	connect(buttonSupprTooltip, SIGNAL(clicked()),this, SLOT(deleteTooltip()));


	//Camera
	QGroupBox* boxCam = new QGroupBox("Camera");

	QCheckBox* attachCamCheckbox = new QCheckBox("attach camera/Xsection");


	connect(attachCamCheckbox,SIGNAL(stateChanged(int)),this,SLOT(attachCamera(int)));

/*	QLabel* labelIncl= new QLabel("Inclinaison");

	QSlider *sliderInclinaison = new QSlider(Qt::Horizontal);
	sliderInclinaison->setMinimum(0);
	sliderInclinaison->setMaximum(120);
	sliderInclinaison->setValue(60);

	QLabel* labeldist= new QLabel("Distance");

	QSlider *sliderDist = new QSlider(Qt::Horizontal);
	sliderDist->setMinimum(200);
	sliderDist->setMaximum(10000);
	sliderDist->setValue(1000);

	QLabel* labelAlt= new QLabel("Altitude");

	m_sliderAlt = new QSlider(Qt::Horizontal);
	m_sliderAlt->setMinimum(0);
	m_sliderAlt->setMaximum(1000);
	m_sliderAlt->setValue(500);

	QLabel* labelSpeed= new QLabel("Speed");

	QSlider *m_sliderSpeed = new QSlider(Qt::Horizontal);
	m_sliderSpeed->setMinimum(1);
	m_sliderSpeed->setMaximum(10);
	m_sliderSpeed->setValue(1);*/



	QLabel* labeldistance= new QLabel("Distance");
	m_moletteDist = new QDial();
	m_moletteDist->setMinimum(0);
	m_moletteDist->setMaximum(100);
	m_moletteDist->setValue(50);


	QLabel* m_labelVitesse= new QLabel("Vitesse");
	m_moletteVitesse = new QDial();
	m_moletteVitesse->setMinimum(1);
	m_moletteVitesse->setMaximum(10);
	m_moletteVitesse->setValue(1);

	QLabel* labelAltitude= new QLabel("Altitude");
	m_moletteAltitude = new QDial();
	m_moletteAltitude->setMinimum(0);
	m_moletteAltitude->setMaximum(100);
	m_moletteAltitude->setValue(50);


	QLabel* m_labelInclinaison= new QLabel("Inclinaison");
	m_moletteInclinaison = new QDial();
	m_moletteInclinaison->setMinimum(0);
	m_moletteInclinaison->setMaximum(120);
	m_moletteInclinaison->setValue(60);

/*	connect(sliderDist,SIGNAL(valueChanged(int)), this, SLOT(setDistance(int)));
	connect(m_sliderAlt,SIGNAL(valueChanged(int)), this, SLOT(setAltitude(int)));
	connect(sliderInclinaison,SIGNAL(valueChanged(int)), this, SLOT(setInclinaison(int)));
	connect(m_sliderSpeed,SIGNAL(valueChanged(int)), this, SLOT(setSpeed(int)));*/


	connect(m_moletteDist,SIGNAL(valueChanged(int)), this, SLOT(setChangedDistance(int)));
	connect(m_moletteDist,SIGNAL(sliderPressed()), this, SLOT(refreshChangedDistance()));
	connect(m_moletteDist,SIGNAL(sliderReleased()), this, SLOT(resetChangedDistance()));


	//connect(m_moletteAltitude,SIGNAL(valueChanged(int)), this, SLOT(setChangedAltitude(int)));
	connect(m_moletteAltitude,SIGNAL(sliderPressed()), this, SLOT(refreshChangedAltitude()));
	connect(m_moletteAltitude,SIGNAL(sliderReleased()), this, SLOT(resetChangedAltitude()));


	//connect(m_moletteAltitude,SIGNAL(valueChanged(int)), this, SLOT(setAltitude(int)));
	connect(m_moletteInclinaison,SIGNAL(valueChanged(int)), this, SLOT(setInclinaison(int)));
	connect(m_moletteVitesse,SIGNAL(valueChanged(int)), this, SLOT(setSpeed(int)));

	QGridLayout *layCam = new QGridLayout();
	layCam->addWidget(attachCamCheckbox, 0, 0, 1, 2);
	/*layCam->addWidget(labelIncl, 1, 0, 1, 1);
	layCam->addWidget(sliderInclinaison, 1, 1, 1, 1);
	layCam->addWidget(labeldist, 1, 2, 1, 1);
	layCam->addWidget(sliderDist, 1, 3, 1, 1);
	layCam->addWidget(labelAlt, 2, 0, 1, 1);
	layCam->addWidget(m_sliderAlt, 2, 1, 1, 1);
	layCam->addWidget(labelSpeed, 2, 2, 1, 1);
	layCam->addWidget(m_sliderSpeed, 2, 3, 1, 1);*/

	layCam->addWidget(labeldistance, 1, 0, 1,2);
	layCam->addWidget(m_moletteDist, 2, 0, 3,2);

	layCam->addWidget(m_labelVitesse, 1, 2, 1,2);
	layCam->addWidget(m_moletteVitesse, 2, 2, 3,2);

	layCam->addWidget(labelAltitude, 5, 0, 1,2);
	layCam->addWidget(m_moletteAltitude, 6, 0, 3,2);

	layCam->addWidget(m_labelInclinaison,5, 2, 1,2);
	layCam->addWidget(m_moletteInclinaison, 6, 2, 3,2);

	boxCam->setLayout(layCam);


	layout->addWidget(boxNurbs, 0, 0, 1, 1);
	layout->addWidget(boxRandomView, 1, 0, 1, 1);
	layout->addWidget(boxTooltip, 2, 0, 1, 1);
	layout->addWidget(boxCam, 3, 0, 2, 1);
	setLayout(layout);


	connect(GraphicToolsWidget::getInstance(),SIGNAL(setViewCurrent(Abstract2DInnerView*)),this,SLOT(currentViewSelected(Abstract2DInnerView*)));

}


void Tools3dWidget::showFont()
{
	bool ok;
	m_fontTooltip = QFontDialog::getFont(
					&ok, m_fontTooltip, this);
	if (ok) {

		int index = m_comboTooltip->currentIndex();
		if(index >=0)
		{
		//QVariant (dialog.currentColor()).toString();
			QString name = m_comboTooltip->currentText();
			emit fontTooltipChanged(index,name,m_fontTooltip);
		}
		// the user clicked OK and font is set to the font the user selected
	} else {
		// the user canceled the dialog; font is set to the initial
		// value, in this case Helvetica [Cronyx], 10
	}
}

void Tools3dWidget::showColor()
{
	 QColorDialog dialog;
	dialog.setCurrentColor (m_colorTooltip);
	dialog.setOption (QColorDialog::DontUseNativeDialog);

	/* Get new color */
	if (dialog.exec() == QColorDialog::Accepted)
	{

		int index = m_comboTooltip->currentIndex();
		if(index >=0)
		{
			QString name = m_comboTooltip->currentText();
			m_colorTooltip =dialog.currentColor() ;//QVariant (dialog.currentColor()).toString();
			emit colorTooltipChanged(index,name,m_colorTooltip);
		}
	}
}


void Tools3dWidget::currentViewSelected(Abstract2DInnerView* view2d)
{
	if(m_view2D !=  view2d && view2d->viewType() == e_ViewType::StackBasemapView) m_view2D = view2d;

}

void Tools3dWidget::setDirectriceColor()
{
	 QColor color = QColorDialog::getColor(m_directriceColor, this );
	if( color.isValid() )
	{
		m_directriceColor = color;
		QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_directriceColor.red())+","+QString::number(m_directriceColor.green())+","+QString::number(m_directriceColor.blue())+")}";
		m_buttonDirectrice->setStyleSheet(namecolor);

		if (m_view3D)
		{
			m_view3D->setColorDirectrice(m_directriceColor);
		}
		RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
		if(randomView)randomView->setColorCross(m_directriceColor);
		else qDebug()<<" No find RandomLineView";

	}
}

void Tools3dWidget::setNurbsColor()
{
	 QColor color = QColorDialog::getColor(m_nurbsColor, this );
	if( color.isValid() )
	{
		m_nurbsColor = color;
		QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_nurbsColor.red())+","+QString::number(m_nurbsColor.green())+","+QString::number(m_nurbsColor.blue())+")}";
		m_buttonNurbs->setStyleSheet(namecolor);

		if (m_view3D)
		{
			m_view3D->setColorNurbs(color);
		}
	}
}

void Tools3dWidget::setNurbsPrecision(int val)
{
	if (m_view3D)
	{
		m_view3D->setPrecisionNurbs(val);
	}
}

void Tools3dWidget::setLinearInterpolation(int val)
{
	bool b = false;
	if( val== 2) b=true;
	if (m_view3D)
	{
		m_view3D->setInterpolationNurbs(b);
	}
}

void Tools3dWidget::setWireframe(int val)
{
	bool b = false;
	if( val== 2) b=true;
	if (m_view3D)
	{
		m_view3D->setWireframeNurbs(b);
	}
}

void Tools3dWidget::setDistance(int val)
{
	m_distanceCam = val;

}

void Tools3dWidget::setInclinaison(int val)
{
	m_inclinaisonCam = val-60.0f;
	if(m_animationRunning==false)
	{
		GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()))->getSelectedBezier();
		if(path != nullptr)
		{
			GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
			if(listBezier != nullptr)
			{
				QPointF pos2D = listBezier->getPosition(m_coefPosition);
				QPointF nor2D = -1.0f * listBezier->getNormal(m_coefPosition);
				moveSectionPosition(pos2D,nor2D);
				return;
			}
		}
		if (m_view3D)
		{
			m_view3D->setSliderXsection(m_coefPosition);
		}

	}
}

void Tools3dWidget::setAltitude(int val)
{
	//int maxvalue  = m_sliderAlt->maximum();
/*	int maxvalue = m_moletteAltitude->maximum();
	m_altitudeCam = maxvalue - val;
	if(m_animationRunning==false && m_view3D)
		{
			m_view3D->setSliderXsection(m_coefPosition);
		}*/

}

void Tools3dWidget::setSpeed(int val)
{
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
	if(randomView != nullptr)
	{
		randomView->setSpeedAnimation(val);
	}
}

void Tools3dWidget::setChangedDistance(int val)
{
	/*float incr = val -(m_moletteDist->maximum()- m_moletteDist->minimum())/2;
	qDebug()<<" increment :"<<incr;
	m_distanceCam += incr;*/
}

void Tools3dWidget::resetChangedDistance()
{
	m_timerDist->stop();
	int center = (m_moletteDist->maximum()- m_moletteDist->minimum())/2;
	m_moletteDist->setValue(center);
}

void Tools3dWidget::refreshChangedDistance()
{
	m_timerDist->start(50);
}


void Tools3dWidget::resetChangedAltitude()
{
	m_timerAlt->stop();
	int center = (m_moletteAltitude->maximum()- m_moletteAltitude->minimum())/2;
	m_moletteAltitude->setValue(center);
}

void Tools3dWidget::refreshChangedAltitude()
{
	m_timerAlt->start(50);
}

void Tools3dWidget::onTimeoutAlt()
{
	float incr = m_moletteAltitude->value() -(m_moletteAltitude->maximum()- m_moletteAltitude->minimum())/2;
	m_altitudeCam -= incr*0.2f;

	if(m_animationRunning==false)
	{
		GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()))->getSelectedBezier();
		if(path != nullptr)
		{
			GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
			if(listBezier != nullptr)
			{
				QPointF pos2D = listBezier->getPosition(m_coefPosition);
				QPointF nor2D = -1.0f * listBezier->getNormal(m_coefPosition);
				moveSectionPosition(pos2D,nor2D);
				return;
			}
		}
		if (m_view3D)
		{
			m_view3D->setSliderXsection(m_coefPosition);
		}
		//m_view3D->setSliderXsection(m_coefPosition);
	}
}

void Tools3dWidget::onTimeout()
{
	float incr = m_moletteDist->value() -(m_moletteDist->maximum()- m_moletteDist->minimum())/2;
	m_distanceCam += (incr*0.2f);

	if(m_animationRunning==false)
	{
		GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()))->getSelectedBezier();
		if(path != nullptr)
		{
			GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
			if(listBezier != nullptr)
			{
				QPointF pos2D = listBezier->getPosition(m_coefPosition);
				QPointF nor2D = -1.0f * listBezier->getNormal(m_coefPosition);
				moveSectionPosition(pos2D,nor2D);
				return;
			}
		}
		if (m_view3D)
		{
			m_view3D->setSliderXsection(m_coefPosition);
		}
	}
}

void Tools3dWidget::attachCamera( int etat)
{
	if(etat > 0)
		m_cameraFollow=true;
	else
		m_cameraFollow = false;

	if(m_animationRunning==false)
	{

		GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()))->getSelectedBezier();
		if(path != nullptr)
		{
			GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
			if(listBezier != nullptr)
			{
				QPointF pos2D = listBezier->getPosition(m_coefPosition);
				QPointF nor2D = -1.0f * listBezier->getNormal(m_coefPosition);
				moveSectionPosition(pos2D,nor2D);
				return;
			}
		}

		if (m_view3D)
		{
			m_view3D->setSliderXsection(m_coefPosition);
		}
	}
}


void Tools3dWidget::cameraFollow(int val)
{
	float coef = (float)(val/(float)m_max);
}

void Tools3dWidget::selectRandomView()
{
	QString nameCurrent = m_comboViews->currentText();
	if (m_view3D)
	{
		m_view3D->selectRandomView(nameCurrent);
	}
}

void Tools3dWidget::deleteRandomView()
{
	qDebug()<<" delete randomView3D ";
	QString nameCurrent = m_comboViews->currentText();
	if (m_view3D)
	{
		m_view3D->deleteRandomView(nameCurrent);
	}
	int index = m_comboViews->currentIndex();

	if(m_indexRandomRep > index )m_indexRandomRep = m_indexRandomRep-1;


	m_comboViews->removeItem(index);
}


void Tools3dWidget::setView3D(ViewQt3D* view3d)
{
	m_view3D = view3d;
	if (m_view3D)
	{
		connect(m_view3D,SIGNAL(sendNewTooltip(QString)),this,SLOT(receiveNewTooltip(QString)));
		connect(m_view3D,SIGNAL(sendNurbsName(QString)),this,SLOT(receiveNaneNurbs(QString)));
		connect(m_view3D->getManagerNurbs(),SIGNAL(sendColorNurbs(QColor,QColor,int,bool,int)),this,SLOT(receiveColorNurbs(QColor,QColor,int,bool,int)));

		connect(this,SIGNAL(sendDeleteTooltip(QString)),m_view3D,SLOT(deleteTooltip(QString)));

		connect(this,SIGNAL(colorTooltipChanged(int,QString,QColor)),m_view3D,SLOT(setColorTooltip(int,QString, QColor)));
		connect(this,SIGNAL(fontTooltipChanged(int,QString,QFont)),m_view3D,SLOT(setFontTooltip(int,QString, QFont)));
		connect(this,SIGNAL(showTooltip(int)),m_view3D,SLOT(showCamTooltip(int)));

		connect(m_view3D,SIGNAL(receiveOrtho(QVector3D,QVector3D)),this,SLOT(updateOrthoFrom3D(QVector3D,QVector3D)));

		connect(m_view3D,SIGNAL(sendDeleteRandom3D(RandomView3D*)),this,SLOT(deletedRandom3D(RandomView3D*)));

		connect(m_view3D,SIGNAL(refreshOrthoFromBezier()),this,SLOT(refreshOrthoFromListBezier()));

		listView3D.push_back(view3d);
	}
}


void Tools3dWidget::removeView2D(Abstract2DInnerView* view2d)
{
	if(listView2D.contains(view2d))
	{
		listView2D.removeAll(view2d);
	}

}


void Tools3dWidget::removeView3D(ViewQt3D* view3d)
{
	if(listView3D.contains(view3d))
	{
		listView3D.removeAll(view3d);
		if (m_view3D==view3d)
		{
			disconnect(m_view3D,SIGNAL(sendNewTooltip(QString)),this,SLOT(receiveNewTooltip(QString)));
			disconnect(m_view3D,SIGNAL(sendNurbsName(QString)),this,SLOT(receiveNaneNurbs(QString)));
			disconnect(m_view3D->getManagerNurbs(),SIGNAL(sendColorNurbs(QColor,QColor,int,bool,int)),this,SLOT(receiveColorNurbs(QColor,QColor,int,bool,int)));

			disconnect(this,SIGNAL(sendDeleteTooltip(QString)),m_view3D,SLOT(deleteTooltip(QString)));

			disconnect(this,SIGNAL(colorTooltipChanged(int,QString,QColor)),m_view3D,SLOT(setColorTooltip(int,QString, QColor)));
			disconnect(this,SIGNAL(fontTooltipChanged(int,QString,QFont)),m_view3D,SLOT(setFontTooltip(int,QString, QFont)));
			disconnect(this,SIGNAL(showTooltip(int)),m_view3D,SLOT(showCamTooltip(int)));

			disconnect(m_view3D,SIGNAL(receiveOrtho(QVector3D,QVector3D)),this,SLOT(updateOrthoFrom3D(QVector3D,QVector3D)));

			disconnect(m_view3D,SIGNAL(sendDeleteRandom3D(RandomView3D*)),this,SLOT(deletedRandom3D(RandomView3D*)));

			disconnect(m_view3D,SIGNAL(refreshOrthoFromBezier()),this,SLOT(refreshOrthoFromListBezier()));
			if (listView3D.size()>0)
			{
				m_view3D = listView3D.last();
			}
			else
			{
				m_view3D = nullptr;
			}
		}
	}

}

void Tools3dWidget::setView2D(Abstract2DInnerView* view2d)
{

	connect(view2d,SIGNAL(addNurbsPoints(QVector<QPointF>,bool,GraphEditor_ListBezierPath*,QString ,QColor)),this,SLOT(receivePointsNurbs(QVector<QPointF>,bool,GraphEditor_ListBezierPath*,QString,QColor)));
	connect(view2d,SIGNAL(updateNurbsPoints(QVector<QPointF>,bool,QColor)),this,SLOT(updatePointsNurbs(QVector<QPointF>,bool,QColor)));
	connect(view2d,SIGNAL(updateNurbsPoints(GraphEditor_ListBezierPath*,QColor)),this,SLOT(updatePointsNurbs(GraphEditor_ListBezierPath*,QColor)));

	connect(view2d,SIGNAL(selectedNurbs(QString)),this,SLOT(setSelectNurbs(QString)));
	connect(view2d,SIGNAL(deletedNurbs(QString)),this,SLOT(setDeleteNurbs(QString)));

	connect(view2d,SIGNAL(signalRandomView(bool,QVector<QPointF>)),this,SLOT(showRandomView(bool, QVector<QPointF>)));
	connect(view2d,SIGNAL(signalRandomView(bool,GraphEditor_LineShape*, RandomLineView*,QString )),this,SLOT(showRandomView(bool,GraphEditor_LineShape*, RandomLineView*,QString)));

	connect(view2d,SIGNAL(signalRandomViewDeleted(RandomLineView*)),this,SLOT(destroyRandomView(RandomLineView*)));


	listView2D.push_back(view2d);
	m_view2D = view2d;

	//connect(this,SIGNAL(refreshOrtho(QVector3)),view2d,SLOT(setRefreshOrtho(QVector3)));


	//qDebug()<<" initialisation view 2D ==>"<<view2d->title();
}

void Tools3dWidget::setInlineView(Abstract2DInnerView* viewInline)
{

	m_viewInline = viewInline;
	//m_viewInline = dynamic_cast<SingleSectionView*>(viewInline);
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);

	if(randomView != nullptr )//&& randomView->getRandomType() == eTypeOrthogonal)
	{


	//	if(m_max<randomView->sizeDiscretePolyline()-1) m_max = randomView->sizeDiscretePolyline()-1;
	//			qDebug()<<" set max :"<<m_max;



		randomView->m_isoBuffer =NurbsWidget::getHorizonBufferValid();// m_view2D->getHorizonBuffer();


		connect(randomView,SIGNAL(etatAnimationChanged(bool)),this,SLOT(animationChanged(bool)));
		connect(randomView,SIGNAL(orthogonalSliceMoved(int,int,QString)),this,SLOT(moveCrossSection(int,int,QString)));
		connect(randomView,SIGNAL(moveCamFollowSection(int)),this,SLOT(cameraFollow(int)));

		connect(randomView,SIGNAL(updateOrtho(QPolygonF)),this,SLOT(updateOrthoSection(QPolygonF)));
		connect(randomView,SIGNAL(newWidthOrtho(QPolygonF)),this,SLOT(updateWidthOrtho(QPolygonF)));

		connect(randomView,SIGNAL(nextKeySection(float)),this,SLOT(nextKey(float)));


		connect(randomView,SIGNAL(sendIndexChanged(int)),this,SLOT(setIndexCurrentPts(int)));
		connect(randomView,SIGNAL(createXSection3D()),this,SLOT(addNewXSectionClone()));

		connect(randomView,SIGNAL(destroyed()),this,SLOT(destroyRandomLineView()));

		connect(randomView,SIGNAL(deletedGeneratrice(QString)),this,SLOT(deleteGeneratrice(QString)));

		connect(randomView,SIGNAL(addCrossPoints(QVector<QPointF>,bool)),this,SLOT(receiveCrossPoints(QVector<QPointF>,bool)));

		connect(randomView,SIGNAL(addCrossPoints(GraphEditor_ListBezierPath*)),this,SLOT(receiveCrossPoints(GraphEditor_ListBezierPath*)));

		connect(randomView,SIGNAL(addCrossPoints(QVector<PointCtrl>,QVector<QPointF>,bool, QPointF)),this,SLOT(receiveCrossPoints(QVector<PointCtrl>,QVector<QPointF>,bool,QPointF)));

		if(randomView->getRandomType() == eTypeOrthogonal)
		{
			if (m_view3D)
			{
				connect(m_view3D,SIGNAL(sendCoefNurbsY(float)),randomView,SLOT(nurbYChanged(float)));
				connect(m_view3D,SIGNAL(sendCoefNurbsXYZ(QVector3D)),randomView,SLOT(nurbsXYZChanged(QVector3D)));
				connect(m_view3D,SIGNAL(sendCurveChanged(std::vector<QVector3D>,bool)),randomView,SLOT(curveChanged(std::vector<QVector3D>,bool)));
				connect(m_view3D,SIGNAL(sendCurveChangedTangent(QVector<QVector3D>,QVector<QVector3D>,bool,QPointF,QString)),randomView,SLOT(curveChangedTangent2(QVector<QVector3D>,QVector<QVector3D>,bool,QPointF,QString)));
				connect(m_view3D,SIGNAL(sendCurveChangedTangentOpt(GraphEditor_ListBezierPath*)),randomView,SLOT(curveChangedTangentOpt(GraphEditor_ListBezierPath*)));

				connect(m_view3D,SIGNAL(sendCurveChanged(QVector<PointCtrl>,bool,QPointF)),randomView,SLOT(curveChangedTangent(QVector<PointCtrl>,bool,QPointF)));
			}
			connect(this,SIGNAL(moveOrthoLine(QPointF,QPointF)),randomView,SLOT(moveLineOrtho(QPointF,QPointF)));

		}

	}

	//connect(m_viewInline,SIGNAL(addCrossPoints(QVector<QPointF>,bool)),this,SLOT(receiveCrossPoints(QVector<QPointF>,bool)));

}

 void Tools3dWidget::refreshOrthoFromListBezier()
 {
	 if(m_view2D)//for(int i=0;i<listView2D.count();i++)
	{
		 GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()))->getSelectedBezier();
		 if(path != nullptr)
		 {
			 GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
			 if(listBezier!= nullptr)
			 {

				QPointF position = listBezier->getPosition(m_coefPosition);
				QPointF normal = listBezier->getNormal(m_coefPosition);

				RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
				if(randomView!= nullptr )randomView->moveLineOrtho(position,normal);
			 }
	 	}
		// else
			// qDebug()<<i<< " , path est null";
	}


 }


void Tools3dWidget::updateWidthOrtho(QPolygonF poly)
{
	if (m_view3D==nullptr)
	{
		return;
	}
	m_comboViews->setCurrentIndex(m_indexRandomRep);

	QVector<QVector3D>  listePts3D;
	for(int i=0;i<poly.count();i++)
	{
		QVector3D pos3D(poly[i].x(),0.0,poly[i].y());
		QVector3D pos = m_view3D->sceneTransform() * pos3D;
		pos.setY(0.0f);
		listePts3D.append(pos);

	}
	float width = (listePts3D[1] - listePts3D[0]).length();
	m_view3D->updateWidthRandomView(m_comboViews->currentText(),listePts3D,width);
}


void Tools3dWidget::nextKey(float coef)
{

	m_coefPosition = coef;
	/*if(m_view2D)
	{
		GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()))->getSelectedBezier();
		if(path != nullptr)
		{
			GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
			if(listBezier != nullptr)
			{
				QPointF pos2D = listBezier->getPosition(m_coefPosition);
				QPointF nor2D = -1.0f * listBezier->getNormal(m_coefPosition);
				qDebug()<<"setSliderXsection listbezier ";
				moveSectionPosition(pos2D,nor2D);
				return;
			}
		}
	}

	qDebug()<<"setSliderXsection classique ";*/
	if (m_view3D)
	{
		m_view3D->setSliderXsection(m_coefPosition);
	}
}

void Tools3dWidget::animationChanged(bool b)
{
	m_animationRunning = b;
}


void Tools3dWidget::updateOrthoFrom3D(QVector3D pos, QVector3D normal)
{

	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
	if(randomView != nullptr)
	{
		randomView->refreshOrtho(pos,normal);
	}
}

void Tools3dWidget::destroyRandomLineView()
{
	//qDebug()<<"destroyRandomLineView  TODO";
	//if (m_view3D)
	//{
	//	m_view3D->deleteCurrentRandomView();
	//}

}

void Tools3dWidget::destroyRandomView(RandomLineView* random)
{
	if (m_view3D)
	{
		int index = m_view3D->destroyRandomView(random);
		if(index>=0)m_comboViews->removeItem(index);
	}
}

void Tools3dWidget::deletedRandom3D(RandomView3D* random3d)
{
	int index = m_comboViews->findText(random3d->getName());
	if(index>=0)m_comboViews->removeItem(index);
}


void Tools3dWidget::showRandomView(bool isOrtho,GraphEditor_LineShape* line, RandomLineView * randomOrtho,QString name)
{


	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
	if(randomView != nullptr && m_view3D!=nullptr)
	{
		m_randomRep = randomView->getRandomSismic (); //(firstRandom);


		CUDAImagePaletteHolder* img = m_randomRep->image();
		float maxHeight = m_view3D->getHeightBox();

		QVector<QPointF> listepoints = line->SceneCordinatesPoints();

		QVector<RandomRep*> reps = randomView->getRandomVisible();

		QVector<QVector2D> listerange;
		QVector<CudaImageTexture*> listetexture;
		for(int i=0;i<reps.count();i++)
		{
			CUDAImagePaletteHolder* img = reps[i]->image();

			QVector2D range = img->range();


			CudaImageTexture* cudaTexture = new CudaImageTexture(img->colorFormat(),img->sampleType(), img->width(), img->height());
			updateTexture(cudaTexture,img);
			listerange.push_back(range);
			listetexture.push_back(cudaTexture);
		}


		CudaImageTexture* cudaTexture = new CudaImageTexture(img->colorFormat(),img->sampleType(), img->width(), img->height());
		updateTexture(cudaTexture,img);

		QVector2D range = img->rangeRatio();

		QVector<QVector3D>  listePts3D;
		for(int i=0;i<listepoints.count();i++)
		{
			QVector3D pos3D(listepoints[i].x(),0.0,listepoints[i].y());
			QVector3D pos = m_view3D->sceneTransform() * pos3D;
			pos.setY(0.0f);
			listePts3D.append(pos);

		}
		QVector3D dir1 = (listePts3D[1]- listePts3D[0]).normalized();
		QVector3D dir2(0,-1,0.0f);
		QVector3D normal = QVector3D::crossProduct(dir2,dir1);//dir1 dir2

		QVector3D position = (listePts3D[1]+ listePts3D[0])*0.5f;


		QString nameView =getUniqueNameRandom();
		m_comboViews->addItem(nameView);

		if(isOrtho)
		{
			m_indexRandomRep = m_comboViews->count()-1;
		}
		m_indexCurrentPts =-1;


//		qDebug()<<"COUNT ::"<<listView2D.count();
	//	isListBezierPath(name);

	//	qDebug()<<" m_indexView : "<<m_indexView;
		//m_view3D->createRandomView(isOrtho, nameView,listePts3D,cudaTexture,range,randomView);
		//
	//	qDebug()<<"Tools3dWidget::showRandomView :"<<name;
	/*	GraphEditor_Path* path = isListBezierPath(name);
		if(path != nullptr)
		{
			m_view3D->createRandomView(isOrtho, nameView,listePts3D,cudaTexture,range,randomView,line,position,normal);

			//GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()))->getSelectedBezier();   //getSelectedBezier();
			//if(path != nullptr)
			//{
				GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
				if(listBezier != nullptr)
				{
					QPointF pos2D = listBezier->getPosition(m_coefPosition);
					randomView->setCrossPosition(pos2D);
				}
			//}
		}*/

		isListBezierPath(name);
		if( m_indexView>= 0)
		{



			m_view3D->createRandomView(isOrtho, nameView,listePts3D,cudaTexture,range,randomView,line,position,normal);

			GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (listView2D[m_indexView]->scene()))->getSelectedBezier();   //getSelectedBezier();
			if(path != nullptr)
			{
				GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
				if(listBezier != nullptr)
				{
					QPointF pos2D = listBezier->getPosition(m_coefPosition);
					randomView->setCrossPosition(pos2D);
				}
			}
			else
			{
				qDebug()<<"Tools3dWidget::getSelectedBezier  est nullptr";
			}
		}
		else
		{
			m_view3D->createRandomView(isOrtho, nameView,listePts3D,cudaTexture,range,randomView,line);
		}
		//m_view3D->createRandomView(isOrtho, nameView,listePts3D,listetexture,listerange,randomView,line);

	}


}


void Tools3dWidget::showRandomView(bool isOrtho,QVector<QPointF>  listepoints)
{

	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
	if(randomView != nullptr && m_view3D!=nullptr)
	{

		m_randomRep = randomView->getRandomSismic (); //firstRandom();
		if(m_randomRep== nullptr)return;


		CUDAImagePaletteHolder* img = m_randomRep->image();
		float maxHeight = m_view3D->getHeightBox();




		CudaImageTexture* cudaTexture = new CudaImageTexture(img->colorFormat(),img->sampleType(), img->width(), img->height());
		updateTexture(cudaTexture,img);

		QVector2D range = img->rangeRatio();

		QVector<QVector3D>  listePts3D;
		for(int i=0;i<listepoints.count();i++)
		{
			QVector3D pos3D(listepoints[i].x(),0.0,listepoints[i].y());
			QVector3D pos = m_view3D->sceneTransform() * pos3D;
			pos.setY(0.0f);
			listePts3D.append(pos);

		}

		QString nameView =getUniqueNameRandom();
		m_comboViews->addItem(nameView);

		if(isOrtho)
		{
			m_indexRandomRep = m_comboViews->count()-1;
		}
		m_indexCurrentPts =-1;


		m_view3D->createRandomView(isOrtho, nameView,listePts3D,cudaTexture,range,randomView);

	}

}



void Tools3dWidget::updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img )
{
	if (texture == nullptr)
			return;

		size_t pointerSize = img->internalPointerSize();
		img->lockPointer();
		texture->setData(
				byteArrayFromRawData((const char*) img->backingPointer(),
						pointerSize));


		img->unlockPointer();
}

void Tools3dWidget::setIndexCurrentPts(int index)
{
	m_indexCurrentPts = index;
}

void Tools3dWidget::addNewXSection()
{
	if (m_view3D)
	{
		m_view3D->createNewXSection(m_coefPosition);
	}
}

void Tools3dWidget::addNewXSectionClone()
{
	if (m_view3D)
	{
		m_view3D->createNewXSectionClone(m_coefPosition);
	}
}

void Tools3dWidget::receiveNewTooltip(QString s)
{
	m_comboTooltip->addItem(s);
}


void Tools3dWidget::saveTooltip()
{
	if (m_view3D==nullptr)
	{
		return;
	}

	QString pathfile = m_view3D->GraphicsLayersDirPath();

	QString directory="Tooltip/";
	QDir dir(pathfile);
	bool res = dir.mkpath("Tooltip");

	QFile file(pathfile+directory+"tooltip.txt");
	if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		qDebug()<<" ouverture du fichier impossible tooltip.txt";
		return;
	}
	//qDebug()<<"chemin :"<<m_pathFiles<<directory<<"in2.txt";
	//qDebug()<<"nb element :"<<m_pathCamera.length();
	QTextStream out(&file);

	QVector<InfoTooltip*> listetooltip = m_view3D->getAllTooltip();
	for (int i=0;i<listetooltip.length();i++)
	{

		QFont ft = listetooltip[i]->getFont();
		out<<listetooltip[i]->getName() <<"|"<<listetooltip[i]->position().x()<<"|"<<listetooltip[i]->position().y()<<"|"<<listetooltip[i]->position().z()
				<<"|"<<listetooltip[i]->getSizePolicy()<<"|"<<listetooltip[i]->getZScale()<<"|"<<listetooltip[i]->getColor().red()<<"|"<<listetooltip[i]->getColor().green()<<"|"<<listetooltip[i]->getColor().blue()
				<<"|"<<ft.family()<<"|"<<ft.pointSize()<<"|"<<ft.bold()<<"|"<<ft.italic()<<"<end>";
	}

}

void Tools3dWidget::loadTooltip()
{
	if (m_view3D==nullptr)
	{
		return;
	}

	QString pathfile = m_view3D->GraphicsLayersDirPath();


	QString directory="Tooltip/";
	QDir dir(pathfile);
	bool res = dir.mkpath("Tooltip");

	QString path = pathfile+directory+"tooltip.txt";

	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"Tools3dWidget Load tooltip ouverture du fichier impossible :"<<path;
		return;
	}

	QString alllines;
	m_comboTooltip->clear();
	m_view3D->destroyAllTooltip();
	QTextStream in(&file);
	QString line = in.readAll();


	QStringList linesplit = line.split("<end>");
	for(int i=0;i<linesplit.length();i++)
	{
		QStringList line1 = linesplit[i].split("|");

		if(line1.length() >3)
		{
			QString texte = line1[0];
			QVector3D pos(line1[1].toFloat(),line1[2].toFloat(),line1[3].toFloat());

			int size = 14;
			int size2 = 14;
			float zScale = 1.0f;
			float red = 1.0f;
			float green = 1.0f;
			float blue = 1.0f;
			bool italic = false;
			bool bold = false;
			QString family="";
			if(line1.length() >4)
			{
				size = line1[4].toInt();
			}

			if(line1.length() >5)
			{
				zScale = line1[5].toFloat();
			}
			if(line1.length() >8)
			{
				red = line1[6].toFloat();
				green = line1[7].toFloat();
				blue = line1[8].toFloat();
			}
			if(line1.length() >9)
			{
				family = line1[9];
			}
			if(line1.length() >10)
			{
				size2 = line1[10].toInt();
			}
			if(line1.length() >11)
			{
				QString val = line1[11];
				bold= (val == "true" ? true : false);
			}
			if(line1.length() >12)
			{
				QString val = line1[12];
				italic= (val == "true" ? true : false);
			}

			m_view3D->createTooltip(texte,pos,size,zScale,family,bold, italic,QColor(red,green,blue));
			m_comboTooltip->addItem(texte);


		}
	}
}

void Tools3dWidget::saveNurbs()
{
	if (m_view3D==nullptr)
	{
		return;
	}

	QString pathfile = m_view3D->GraphicsLayersDirPath();
	qDebug()<<" save Nurbs : "<<pathfile;

	QString directory="Nurbs/";
	QDir dir(pathfile);
	bool res = dir.mkpath("Nurbs");

	QFile file(pathfile+directory+"nurbs.txt");
	if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		qDebug()<<" ouverture du fichier impossible nurbs.txt";
		return;
	}

	QTextStream out(&file);

	QVector<Manager*> listenurbs = m_view3D->m_nurbsManager->getListeNurbs();

	qDebug()<<" length : "<<listenurbs.length();
	for (int i=0;i<listenurbs.length();i++)
	{
		std::vector<QVector3D>  ptsDirectrice= listenurbs[i]->getCurveDirectrice()->data();


		QString keyDirectrice="";
		for(int j=0;j<ptsDirectrice.size();j++)
		{
			QString ptsStr=keyDirectrice.append("|");
		}

		//name id, color nurbs, color directrice, point clé directrice , point clé generatrice

		out<<listenurbs[i]->getNameId() <<"|"<<listenurbs[i]->getColorNurbs().red()<<"-"<<listenurbs[i]->getColorNurbs().green()<<"-"<<listenurbs[i]->getColorNurbs().blue()
		<<"|"<<listenurbs[i]->getColorDirectrice().red()<<"-"<<listenurbs[i]->getColorDirectrice().green()<<"-"<<listenurbs[i]->getColorDirectrice().blue()
		<<"\n";

	/*	std::vector<QVector3D> listptsDir =listenurbs[i]->getCurveDirectrice()->data();
		for(int j=0;j< listptsDir.size();j++)
		{
			out<<listptsDir[j].x() <<"|"<<listptsDir[j].y() <<"|"<<listptsDir[j].z();
		}
		out<<"\n";
*/

		 std::vector< std::shared_ptr<CurveModel> > curves = listenurbs[i]->getOtherCurve();

	   for(int j=0;j<curves.size();j++)
	   {
			std::vector<QVector3D> listptscurve =curves[j]->data();
			for(int k=0;k< listptscurve.size();k++)
			{
				if(j==0) out<<"Directrice|";
				else out<<"Generatrice|";
				out<<listptscurve[k].x() <<"|"<<listptscurve[k].y() <<"|"<<listptscurve[k].z();
			}
			out<<"\n";
	   }

	}
}

void Tools3dWidget::exportNurbs()
{
	/*widgetNameForSave* widget = new widgetNameForSave(this);
	if ( widget->exec() == QDialog::Accepted)
	{
		QString nom = widget->getName();

		if(nom!="" && m_view3D)
		{
			m_view3D->exportNurbsObj(nom);
		}
	}
*/

}

void Tools3dWidget::loadNurbs()
{
	if (m_view3D==nullptr)
	{
		return;
	}

	QString pathfile = m_view3D->GraphicsLayersDirPath();


		QString directory="Nurbs/";
		QDir dir(pathfile);
		bool res = dir.mkpath("Nurbs");

		QString path = pathfile+directory+"nurbs.txt";

		QFile file(path);
		if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
		{
			qDebug()<<"Tools3dWidget Load nurbs ouverture du fichier impossible :"<<path;
			return;
		}

		QString alllines;
	//	m_comboTooltip->clear();
	//	m_view3D->destroyAllTooltip();
		QTextStream in(&file);


		/*while(!in.atEnd())
		{
			QString line = fn.readLine()
		}*/



	/*	QString line = in.readAll();


		QStringList linesplit = line.split("<end>");
		for(int i=0;i<linesplit.length();i++)
		{
			QStringList line1 = linesplit[i].split("|");

			if(line1.length() >3)
			{
				QString texte = line1[0];
				QVector3D pos(line1[1].toFloat(),line1[2].toFloat(),line1[3].toFloat());

				m_comboTooltip->addItem(texte);
				m_view3D->createTooltip(texte,pos);


			}
		}*/
}

void Tools3dWidget::selectCombo(int index)
{
	if (m_view3D==nullptr)
	{
		return;
	}

	if(index>=0 && index <m_comboTooltip->count())
	{
		QString name = m_comboTooltip->currentText();

		InfoTooltip *tooltip  = m_view3D->findTooltip(name);
		if(tooltip==nullptr ) qDebug()<<" le tooltip "<<name<<" est introuvable";

		m_fontTooltip = tooltip->getFont();
		m_colorTooltip = tooltip->getColor();
	//	int size = tooltip->getSizePolicy();
	//m_comboSize->setCurrentText(QString::number(size));
	}

}
void Tools3dWidget::setSize(int)
{
	int index = m_comboTooltip->currentIndex();
	if(index >=0)
	{
		QString name = m_comboTooltip->currentText();
		//int size = m_comboSize->currentText().toInt();

		//emit updateSizePolicy(index,name,size);
	}
}

void Tools3dWidget::deleteTooltip()
{
	int index = m_comboTooltip->currentIndex();
	if(index >=0)
	{
		QString name = m_comboTooltip->currentText();
		m_comboTooltip->removeItem(index);
		emit sendDeleteTooltip(name);
	}
}



void Tools3dWidget::showCamTooltip()
{
	int index = m_comboTooltip->currentIndex();
	emit showTooltip(index);
}

void Tools3dWidget::receiveColorNurbs(QColor colorDir,QColor colorNurbs, int precision,bool edit,int timer)
{
	QString namecolor2= "QPushButton {background-color: rgb("+QString::number(colorDir.red())+","+QString::number(colorDir.green())+","+QString::number(colorDir.blue())+")}";
	m_buttonDirectrice->setStyleSheet(namecolor2);

	QString namecolor3= "QPushButton {background-color: rgb("+QString::number(colorNurbs.red())+","+QString::number(colorNurbs.green())+","+QString::number(colorNurbs.blue())+")}";
	m_buttonNurbs->setStyleSheet(namecolor3);
}

void Tools3dWidget::setSelectNurbs(QString s)
{
	if (m_view3D)
	{
		m_view3D->selectNurbs(s);
	}
}

void Tools3dWidget::setDeleteNurbs(QString s)
{
	if (m_view3D)
	{
		m_view3D->deleteNurbs(s);
	}
	int index = m_comboNurbs->findText(s);
	if(index != -1) m_comboNurbs->removeItem(index);
}

void Tools3dWidget::deleteGeneratrice(QString s)
{
	if (m_view3D)
	{
		m_view3D->deleteGeneratriceNurbs(s);
	}
}


void Tools3dWidget::receiveCrossPoints(GraphEditor_ListBezierPath* path)
{


	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
	if(randomView != nullptr)
	{
		randomView->initTransformation();
		if (m_view3D)
		{
			m_view3D->createSection(path,randomView->randomTransform(),m_indexCurrentPts);
		}
	}
}



void Tools3dWidget::receiveCrossPoints(QVector<QPointF>  listepoints,bool isopen)
{
	if (m_view3D==nullptr)
	{
		return;
	}

	SingleSectionView* sectionView = dynamic_cast<SingleSectionView*>(m_viewInline);
	QVector<QVector3D>  listePts3D;
	if(sectionView != nullptr)
	{

		for(int i=0;i<listepoints.count();i++)
		{
			QVector3D posTr = sectionView->viewWorldTo3dWord(listepoints[i]);
			QVector3D swap(posTr.x(),posTr.z(),posTr.y());
			QVector3D pos = m_view3D->sceneTransform() * swap;
			listePts3D.append(pos);

		}
	}
	else
	{
		RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
		if(randomView != nullptr)
		{
			for(int i=0;i<listepoints.count();i++)
			{
				QVector3D posTr = randomView->viewWorldTo3dWordExtended(listepoints[i]);
				QVector3D swap(posTr.x(),posTr.z(),posTr.y());
				QVector3D pos = m_view3D->sceneTransform() * swap;
				listePts3D.append(pos);
			}
		}
	}



	m_view3D->createSection(listePts3D,m_indexCurrentPts,isopen);
}


void Tools3dWidget::receiveCrossPoints(QVector<PointCtrl> listeCtrls,QVector<QPointF>  listepoints,bool isopen,QPointF cross)
{
	if (m_view3D==nullptr)
	{
		return;
	}

	SingleSectionView* sectionView = dynamic_cast<SingleSectionView*>(m_viewInline);
	QVector<QVector3D>  listePts3D;
	QVector<QVector3D>  listeCtrl3D;
	QVector<QVector3D>  listeTangent3D;
	QVector3D posC;
	if(sectionView != nullptr)
	{

		for(int i=0;i<listepoints.count();i++)
		{
			QVector3D posTr = sectionView->viewWorldTo3dWord(listepoints[i]);
			QVector3D swap(posTr.x(),posTr.z(),posTr.y());
			QVector3D pos = m_view3D->sceneTransform() * swap;
			listePts3D.append(pos);

		}
	}
	else
	{

		RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
		if(randomView != nullptr)
		{

			randomView->initTransformation();

			//QPointF decal = randomView->m_position2DCross - cross;

			//QVector<PointCtrl> listepts3dTr;
		/*	for(int i=0;i<listeCtrls.count();i++)
			{
				listeCtrls[i].m_position = listeCtrls[i].m_position -decal;
				listeCtrls[i].m_ctrl1 = listeCtrls[i].m_ctrl1 -decal;
				listeCtrls[i].m_ctrl2 = listeCtrls[i].m_ctrl2 -decal;
			}
*/

			//qDebug()<<" cross :"<<cross;

				QVector3D posTrCroos = randomView->viewWorldTo3dWordExtended(cross);
				QVector3D swapC(posTrCroos.x(),posTrCroos.z(),posTrCroos.y());
				posC= m_view3D->sceneTransform() * swapC;


				for(int i=0;i<listeCtrls.count();i++)
				{
					QVector3D posTr = randomView->viewWorldTo3dWordExtended(listeCtrls[i].m_position);
					QVector3D swap(posTr.x(),posTr.z(),posTr.y());
					QVector3D pos = m_view3D->sceneTransform() * swap;

					listeCtrl3D.append(pos);

					QVector3D posTr1 = randomView->viewWorldTo3dWordExtended(listeCtrls[i].m_ctrl1);
					QVector3D swap1(posTr1.x(),posTr1.z(),posTr1.y());
					QVector3D pos1 = m_view3D->sceneTransform() * swap1;
					listeTangent3D.append(pos1);

					QVector3D posTr2 = randomView->viewWorldTo3dWordExtended(listeCtrls[i].m_ctrl2);
					QVector3D swap2(posTr2.x(),posTr2.z(),posTr2.y());
					QVector3D pos2 = m_view3D->sceneTransform() * swap2;
					listeTangent3D.append(pos2);

				}


			for(int i=0;i<listepoints.count();i++)
			{
				QVector3D posTr = randomView->viewWorldTo3dWordExtended(listepoints[i]);
				QVector3D swap(posTr.x(),posTr.z(),posTr.y());
				QVector3D pos = m_view3D->sceneTransform() * swap;
				//qDebug()<<i<< " , pos :"<<pos;
				listePts3D.append(pos);
			}
		}
	}


	m_view3D->createSection(listeCtrls, listePts3D,m_indexCurrentPts,isopen,true,cross,listeCtrl3D,listeTangent3D,posC);
}


void Tools3dWidget::receivePointsNurbs(QVector<QPointF>  listepoints, bool withTangent,GraphEditor_ListBezierPath* path ,QString nameNurbs,QColor col)
{
	if (m_view3D==nullptr)
	{
		return;
	}

	IsoSurfaceBuffer  isobuffer =NurbsWidget::getHorizonBufferValid();// m_view2D->getHorizonBuffer();

	m_listePts3D.clear();
//	float lastY = 0.0f;
	for(int i=0;i<listepoints.count();i++)
	{
		float altY =  isobuffer.getAltitude(listepoints[i]);



		QVector3D pos3D(listepoints[i].x(),altY,listepoints[i].y());

		QVector3D pos = m_view3D->sceneTransform() * pos3D;
		m_listePts3D.append(pos);
	}

	/*for(int i=0;i<listView3D.size();i++)
	{
		if(listView3D[i] != nullptr)  listView3D[i]->createNurbs(m_listePts3D,withTangent,isobuffer,path,nameNurbs);
	}*/

	m_view3D->createNurbs(m_listePts3D,withTangent,isobuffer,path,nameNurbs,col);

}

void Tools3dWidget::receiveNaneNurbs(QString name)
{
	m_view2D->setNameItem(name);
	m_comboNurbs->addItem(name);
}

void Tools3dWidget::nurbsSelectedChanged(int index)
{
	if (m_view3D)
	{
		m_view3D->selectNurbs(index);
	}
}

void Tools3dWidget::updatePointsNurbs(GraphEditor_ListBezierPath* path, QColor col)
{
	if (m_view3D)
	{
		m_view3D->updateNurbs(path,col);
	}
}

void Tools3dWidget::updatePointsNurbs(QVector<QPointF>  listepoints,bool withTangent,QColor col)
{

	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
	if(randomView != nullptr && m_view3D)
	{


		m_listePts3D.clear();
		for(int i=0;i<listepoints.count();i++)
		{
			float altY = randomView->m_isoBuffer.getAltitude(listepoints[i]);
			QVector3D pos3D(listepoints[i].x(),altY,listepoints[i].y());

			QVector3D pos = m_view3D->sceneTransform() * pos3D;

			//pos.setY(0.0f);

		//	float altitude = m_view3D->getAltitude(pos);
			//pos.setY(pos.y()+altitude);
			m_listePts3D.append(pos);
		}
		m_view3D->updateNurbs(m_listePts3D, withTangent,col);


	}
	else
	{
		qDebug()<<" ======+> randomView est NULL";
	}


	/*RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
	if(randomView != nullptr)
	{
		randomView->refreshOrthoFromListBezier();
	}
	else
	{
		qDebug()<<" randomView est nullllll";
	}
*/

}


void Tools3dWidget::updateOrthoSection(	QPolygonF poly)
{
	if (m_view3D==nullptr)
	{
		return;
	}

	QVector<QVector3D>  listePts3D;
	QVector2D range;

	for(int i=0;i<poly.count();i++)
	{
		QVector3D pos3D(poly[i].x(),0.0,poly[i].y());
		QVector3D pos = m_view3D->sceneTransform() * pos3D;
		pos.setY(0.0f);
		listePts3D.append(pos);


	}
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
	if(randomView != nullptr  && m_randomRep !=nullptr)
	{
		CUDAImagePaletteHolder* img = m_randomRep->image();
		if(img !=nullptr)
		{
			//connect(img,SIGNAL(dataChanged()),this,SLOT(imageDataChanged()));
			CudaImageTexture* cudaTexture = new CudaImageTexture(img->colorFormat(),img->sampleType(), img->width(), img->height());
			updateTexture(cudaTexture,img);

			QVector2D range = img->rangeRatio();

			m_comboViews->setCurrentIndex(m_indexRandomRep);


			float maxHeight = m_view3D->getHeightBox();

			float width = (listePts3D[1] - listePts3D[0]).length();




			m_view3D->updateRandomView(m_comboViews->currentText(),listePts3D,cudaTexture,range,m_cameraFollow,m_distanceCam, m_altitudeCam,m_inclinaisonCam,width,isListBezierPath());//, cudaTexture)
		}
	}

}

bool Tools3dWidget::isListBezierPath()
{

	GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()))->getSelectedBezier();
	if(path != nullptr)
	{
		GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
		if(listBezier != nullptr)
		{

			return true;
		}
	}
	return false;
}

GraphEditor_Path* Tools3dWidget::isListBezierPath(QString name)
{


		for(int i=0;i<listView2D.count();i++)
		{
			if(listView2D[i] != nullptr)
			{
				IsoSurfaceBuffer buffer = listView2D[i]->getHorizonBuffer();
				if(buffer.isValid())
				{
					//qDebug()<<"getHorizonBufferValid : "<<i;
					m_indexView=i;

				}
			}
		}

	/*for(int i=0;i<listView2D.count();i++)
		{
			if(listView2D[i] != nullptr)
			{
				GraphEditor_Path* path  =(dynamic_cast<GraphicSceneEditor *> (listView2D[i]->scene()))->getCurrentBezier(name);
				if(path != nullptr)
				{
					m_indexView=i;
					//qDebug()<<"getHorizonBufferValid : "<<i;
					return path;
				}
			}
		}*/
	return nullptr;


/*	qDebug()<<"Tools3dWidget::isListBezierPath  "<<m_view2D->defaultTitle();
	GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()))->getSelectedBezier();
	if(path != nullptr)
	{
		GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
		if(listBezier != nullptr)
		{

			return true;
		}
	}
	return false;*/
}

void Tools3dWidget::moveSectionPosition(QPointF pos2D,QPointF nor2D)
{

		emit moveOrthoLine(pos2D,nor2D);
	//	QVector3D swap(pos2D.x(),0.0f,pos2D.y());
	//	QVector3D pos = m_view3D->sceneTransform() * swap;


	//	float altitude = m_view3D->getAltitude(pos);
	//	pos.setY(pos.y()+altitude);

		RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
		if(randomView != nullptr && m_view3D)
		{

			randomView->setCrossPosition(pos2D);

			float altY = randomView->m_isoBuffer.getAltitude(pos2D);
			QVector3D pos3D(pos2D.x(),altY,pos2D.y());
			QVector3D pos = m_view3D->sceneTransform() * pos3D;

			m_view3D->setSliderXsection(m_coefPosition,pos,nor2D);

		}
		else qDebug()<<" randomView est null.........";


}

void Tools3dWidget::moveCrossSection( int value, int max,QString nom)
{
	m_max = max;
	m_coefPosition = (float)(value/(float)m_max);

	isListBezierPath(nom);


	if(m_indexView>=0)//for(int i=0;i<listView2D.count();i++)
	{
		GraphEditor_Path* path = (dynamic_cast<GraphicSceneEditor *> (listView2D[m_indexView]->scene()))->getCurrentBezier(nom);   //getSelectedBezier();
		if( path== nullptr)path = (dynamic_cast<GraphicSceneEditor *> (listView2D[m_indexView]->scene()))->getSelectBezier(nom);
		if(path != nullptr)
		{
			GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(path);
			if(listBezier != nullptr)
			{
				QPointF pos2D = listBezier->getPosition(m_coefPosition);
				QPointF nor2D = -1.0f * listBezier->getNormal(m_coefPosition);

				moveSectionPosition(pos2D,nor2D);
				return;
			}
		}
		else
			qDebug()<<"  path est NULLLLL";


	}


	if (m_view3D)
	{
		m_view3D->setSliderXsection(m_coefPosition);
	}



/*	RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
	if(randomView != nullptr)
	{

		QVector<QVector3D>  listePts3D;
		QVector<QPointF>  listepoints  =  randomView->m_orthoLineList[0]->SceneCordinatesPoints();
		//qDebug()<<"init  m_max :"<<m_max;
		for(int i=0;i<listepoints.count();i++)
		{

			QVector3D posTr = randomView->viewWorldTo3dWord(listepoints[i]);
			QVector3D swap(posTr.x(),posTr.z(),posTr.y());
			QVector3D pos = m_view3D->sceneTransform() * swap;

			listePts3D.append(pos);

		}
		m_view3D->updateRandomView(listePts3D,nullptr);//, cudaTexture)
	}*/
}

void Tools3dWidget::moveSection()
{
	int  valeur = m_editline->text().toInt();
	float res = valeur /m_max;
	//qDebug()<<m_max <<"   moveCrossSection valeur :"<<res;
	if (m_view3D)
	{
		m_view3D->setSliderXsection(res);
	}
}


void Tools3dWidget::AddPointsNurbs(QVector3D pos)
{
	//TODO
}



QString Tools3dWidget::getUniqueNameRandom()
{
	int index = 1;
	QString name = "Random view "+QString::number(index);
	while ( m_comboViews->findText(name) != -1)
	{
		index++;
		name = "Random view "+QString::number(index);
	}
	return name;
}


