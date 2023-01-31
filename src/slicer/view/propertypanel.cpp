#include "propertypanel.h"
#include <QValidator>
#include <QColorDialog>
#include <QDebug>
#include <QDialogButtonBox>
#include <QScrollArea>
#include <QSettings>

#include <cmath>

PropertyPanel::PropertyPanel(QWidget* parent):QDialog(parent)
	{
		setWindowTitle("Preferences");
		setMinimumWidth(400);
		setMinimumHeight(500);
		setModal(false);

		QVBoxLayout* mainLayout = new QVBoxLayout;
		setLayout(mainLayout);

		QScrollArea* scrollArea = new QScrollArea;
		scrollArea->setWidgetResizable(true);
		mainLayout->addWidget(scrollArea);

		QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Reset);
		 QObject::connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));

		 connect(buttonBox->button(QDialogButtonBox::Reset), SIGNAL(clicked()),this, SLOT(reset()));

		QHBoxLayout* buttonBoxLayout = new QHBoxLayout;
		buttonBoxLayout->addStretch(1);
		buttonBoxLayout->addWidget(buttonBox);
		mainLayout->addLayout(buttonBoxLayout);

		//Surface

		/*

		QGroupBox* boxSurface = new QGroupBox("Surface");

		QGridLayout *layoutbox0 = new QGridLayout();

		QLabel* labelsurf= new QLabel("Precision terrain");
		m_sliderPrecisionSurface = new QSlider(Qt::Horizontal);
		m_sliderPrecisionSurface->setMinimum(0);
		m_sliderPrecisionSurface->setMaximum(100);
		m_sliderPrecisionSurface->setValue(m_surfacePrecision);
		m_editPrecisionSurface = new QLineEdit;
		m_editPrecisionSurface->setMaximumWidth(60);
		m_editPrecisionSurface->setValidator( new QIntValidator(0, 100, this) );
		m_editPrecisionSurface->setText(QString::number(m_surfacePrecision));
		connect(m_sliderPrecisionSurface,SIGNAL(valueChanged(int)),this,SLOT(setSurfacePrecision(int)));
		connect(m_editPrecisionSurface,SIGNAL(editingFinished()),this,SLOT(setSurfacePrecision()));

		layoutbox0->addWidget(labelsurf, 1, 0, 1, 1);
		layoutbox0->addWidget(m_sliderPrecisionSurface, 1, 1, 1, 1);
		layoutbox0->addWidget(m_editPrecisionSurface, 1, 2, 1, 1);

		boxSurface->setLayout(layoutbox0);
*/
		//wells
		QGroupBox* boxWells = new QGroupBox("Wells");
		QGridLayout *layoutbox1 = new QGridLayout();

		QLabel* labelP1= new QLabel("Precision threshold");
		m_sliderPrecisionWell = new QSlider(Qt::Horizontal);
		m_sliderPrecisionWell->setMinimum(0);
		m_sliderPrecisionWell->setMaximum(100);
		m_sliderPrecisionWell->setValue(m_wellPrecision);
		m_editPrecisionWell = new QLineEdit;
		m_editPrecisionWell->setMaximumWidth(60);
		m_editPrecisionWell->setValidator( new QIntValidator(0, 100, this) );
		m_editPrecisionWell->setText(QString::number(m_wellPrecision));
		connect(m_sliderPrecisionWell,SIGNAL(valueChanged(int)),this,SLOT(setWellPrecision(int)));
		connect(m_editPrecisionWell,SIGNAL(editingFinished()),this,SLOT(setWellPrecision()));

		QLabel* labelP2= new QLabel("Default color");
		m_buttonColorWell1 = new QPushButton;
		QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_wellColor.red())+","+QString::number(m_wellColor.green())+","+QString::number(m_wellColor.blue())+")}";
		m_buttonColorWell1->setStyleSheet(namecolor);
		connect(m_buttonColorWell1,SIGNAL(clicked()),this,SLOT(setWellDefaultColor()));

		QLabel* labelP3= new QLabel("Selected color");
		m_buttonColorWell2 = new QPushButton;
		QString namecolor3= "QPushButton {background-color: rgb("+QString::number(m_wellSelectedColor.red())+","+QString::number(m_wellSelectedColor.green())+","+QString::number(m_wellSelectedColor.blue())+")}";
		m_buttonColorWell2->setStyleSheet(namecolor3);
		connect(m_buttonColorWell2,SIGNAL(clicked()),this,SLOT(setWellSelectedColor()));

		QLabel* labelP4= new QLabel("Diameter ");
		m_sliderDiameterWell = new QSlider(Qt::Horizontal);
		m_sliderDiameterWell->setMinimum(1);
		m_sliderDiameterWell->setMaximum(100);
		m_sliderDiameterWell->setValue(m_wellDiameter);
		m_editDiameterWell = new QLineEdit;
		m_editDiameterWell->setMaximumWidth(60);
		m_editDiameterWell->setText(QString::number(m_wellDiameter));
		connect(m_sliderDiameterWell,SIGNAL(valueChanged(int)),this,SLOT(setWellDiameter(int)));
		connect(m_editDiameterWell,SIGNAL(editingFinished()),this,SLOT(setWellDiameter()));

		QLabel* labelWidth= new QLabel("Map width ");
		m_sliderMapWidthWell = new QSlider(Qt::Horizontal);
		m_sliderMapWidthWell->setMinimum(1);
		m_sliderMapWidthWell->setMaximum(1000);
		m_sliderMapWidthWell->setValue(m_wellMapWidth*10.0);
		m_editMapWidthWell = new QLineEdit;
		m_editMapWidthWell->setMaximumWidth(60);
		m_editMapWidthWell->setText(QString::number(m_wellMapWidth));
		connect(m_sliderMapWidthWell,SIGNAL(valueChanged(int)),this,SLOT(setWellMapWidth(int)));
		connect(m_editMapWidthWell,SIGNAL(editingFinished()),this,SLOT(setWellMapWidth()));

		QLabel* labelSectionWidth= new QLabel("Section width ");
		m_sliderSectionWidthWell = new QSlider(Qt::Horizontal);
		m_sliderSectionWidthWell->setMinimum(1);
		m_sliderSectionWidthWell->setMaximum(1000);
		m_sliderSectionWidthWell->setValue(m_wellSectionWidth*10.0);
		m_editSectionWidthWell = new QLineEdit;
		m_editSectionWidthWell->setMaximumWidth(60);
		m_editSectionWidthWell->setText(QString::number(m_wellSectionWidth));
		connect(m_sliderSectionWidthWell,SIGNAL(valueChanged(int)),this,SLOT(setWellSectionWidth(int)));
		connect(m_editSectionWidthWell,SIGNAL(editingFinished()),this,SLOT(setWellSectionWidth()));




		//Ligne1
		layoutbox1->addWidget(labelP1, 1, 0, 1, 1);
		layoutbox1->addWidget(m_sliderPrecisionWell, 1, 1, 1, 1);
		layoutbox1->addWidget(m_editPrecisionWell, 1, 2, 1, 1);
		//Ligne2
		layoutbox1->addWidget(labelP2, 2, 0, 1, 1);
		layoutbox1->addWidget(m_buttonColorWell1, 2, 1, 1, 1);
		//Ligne3
		layoutbox1->addWidget(labelP3, 3, 0, 1, 1);
		layoutbox1->addWidget(m_buttonColorWell2, 3, 1, 1, 1);
		//Ligne4
		layoutbox1->addWidget(labelP4, 4, 0, 1, 1);
		layoutbox1->addWidget(m_sliderDiameterWell, 4, 1, 1, 1);
		layoutbox1->addWidget(m_editDiameterWell,4, 2, 1, 1);
		//Ligne5
		layoutbox1->addWidget(labelWidth, 5, 0, 1, 1);
		layoutbox1->addWidget(m_sliderMapWidthWell, 5, 1, 1, 1);
		layoutbox1->addWidget(m_editMapWidthWell,5, 2, 1, 1);
		//Ligne6
		layoutbox1->addWidget(labelSectionWidth, 6, 0, 1, 1);
		layoutbox1->addWidget(m_sliderSectionWidthWell, 6, 1, 1, 1);
		layoutbox1->addWidget(m_editSectionWidthWell,6, 2, 1, 1);

		boxWells->setLayout(layoutbox1);


		//logs
		QGroupBox* boxLogs = new QGroupBox("Logs");
		QGridLayout *layoutboxLogs = new QGridLayout();

		QLabel* labellogs1= new QLabel("Precision ");
		m_sliderPrecisionLogs = new QSlider(Qt::Horizontal);
		m_sliderPrecisionLogs->setMinimum(1);
		m_sliderPrecisionLogs->setMaximum(50);
		m_sliderPrecisionLogs->setValue(m_logsPrecision);
		m_editPrecisionLogs = new QLineEdit;
		m_editPrecisionLogs->setMaximumWidth(60);
		m_editPrecisionLogs->setValidator( new QIntValidator(0, 100, this) );
		m_editPrecisionLogs->setText(QString::number(m_logsPrecision));
		connect(m_sliderPrecisionLogs,SIGNAL(valueChanged(int)),this,SLOT(setLogsPrecision(int)));
		connect(m_editPrecisionLogs,SIGNAL(editingFinished()),this,SLOT(setLogsPrecision()));

		//Ligne1
		layoutboxLogs->addWidget(labellogs1, 1, 0, 1, 1);
		layoutboxLogs->addWidget(m_sliderPrecisionLogs, 1, 1, 1, 1);
		layoutboxLogs->addWidget(m_editPrecisionLogs, 1, 2, 1, 1);

		QLabel* labelthicknessLog= new QLabel("thickness ");
		m_sliderThicknessLog= new QSlider(Qt::Horizontal);
		m_sliderThicknessLog->setMinimum(1);
		m_sliderThicknessLog->setMaximum(10);
		m_sliderThicknessLog->setValue(m_logsThickness);
		m_editThicknessLog = new QLineEdit;
		m_editThicknessLog->setMaximumWidth(60);
		m_editThicknessLog->setText(QString::number(m_logsThickness));
		connect(m_sliderThicknessLog,SIGNAL(valueChanged(int)),this,SLOT(setLogThickness(int)));
		connect(m_editThicknessLog,SIGNAL(editingFinished()),this,SLOT(setLogThickness()));

		QLabel* labelColorLog= new QLabel("Default color");
		m_buttonColorLog = new QPushButton;
		QString namecolorLog= "QPushButton {background-color: rgb("+QString::number(m_logsColor.red())+","+QString::number(m_logsColor.green())+","+QString::number(m_logsColor.blue())+")}";
		m_buttonColorLog->setStyleSheet(namecolorLog);
		connect(m_buttonColorLog,SIGNAL(clicked()),this,SLOT(setLogColor()));


		layoutboxLogs->addWidget(labelthicknessLog, 2, 0, 1, 1);
		layoutboxLogs->addWidget(m_sliderThicknessLog, 2, 1, 1, 1);
		layoutboxLogs->addWidget(m_editThicknessLog, 2, 2, 1, 1);

		layoutboxLogs->addWidget(labelColorLog, 3, 0, 1, 1);
		layoutboxLogs->addWidget(m_buttonColorLog, 3, 1, 1, 1);

		boxLogs->setLayout(layoutboxLogs);

		//picks
		QGroupBox* boxPicks = new QGroupBox("Picks");
		QGridLayout *layoutbox2 = new QGridLayout();

		QLabel* labelP5= new QLabel("Diameter ");
		m_sliderDiameterPick = new QSlider(Qt::Horizontal);
		m_sliderDiameterPick->setMinimum(1);
		m_sliderDiameterPick->setMaximum(100);
		m_sliderDiameterPick->setValue(m_pickDiameter);

		m_editDiameterPick = new QLineEdit;
		m_editDiameterPick->setMaximumWidth(60);
		m_editDiameterPick->setText(QString::number(m_pickDiameter));
		connect(m_sliderDiameterPick,SIGNAL(valueChanged(int)),this,SLOT(setPickDiameter(int)));
		connect(m_editDiameterPick,SIGNAL(editingFinished()),this,SLOT(setPickDiameter()));


		QLabel* labelthickness= new QLabel("thickness ");
		m_sliderThickness= new QSlider(Qt::Horizontal);
		m_sliderThickness->setMinimum(1);
		m_sliderThickness->setMaximum(50);
		m_sliderThickness->setValue(m_pickThickness);
		m_editThickness = new QLineEdit;
		m_editThickness->setMaximumWidth(60);
		m_editThickness->setText(QString::number(m_pickThickness));
		connect(m_sliderThickness,SIGNAL(valueChanged(int)),this,SLOT(setPickThickness(int)));
		connect(m_editThickness,SIGNAL(editingFinished()),this,SLOT(setPickThickness()));


		//Ligne1
		layoutbox2->addWidget(labelP5, 1, 0, 1, 1);
		layoutbox2->addWidget(m_sliderDiameterPick, 1, 1, 1, 1);
		layoutbox2->addWidget(m_editDiameterPick, 1, 2, 1, 1);

		layoutbox2->addWidget(labelthickness, 2, 0, 1, 1);
		layoutbox2->addWidget(m_sliderThickness, 2, 1, 1, 1);
		layoutbox2->addWidget(m_editThickness, 2, 2, 1, 1);

		boxPicks->setLayout(layoutbox2);


		//camera
		QGroupBox* boxCam = new QGroupBox("Camera");
		QGridLayout *layoutbox3 = new QGridLayout();

		QLabel* labelP6= new QLabel("Speed altitude ");
		 m_sliderUpDown = new QSlider(Qt::Horizontal);
		m_sliderUpDown->setMinimum(0);
		m_sliderUpDown->setMaximum(100);
		m_sliderUpDown->setValue(m_speedAltitude);
		m_editUpDown= new QLineEdit;
		m_editUpDown->setMaximumWidth(60);
		m_editUpDown->setText(QString::number(m_speedAltitude));
		connect(m_sliderUpDown,SIGNAL(valueChanged(int)),this,SLOT(setSpeedUpDown(int)));
		connect(m_editUpDown,SIGNAL(editingFinished()),this,SLOT(setSpeedUpDown()));


		QLabel* labelP7= new QLabel("Speed helico ");
		m_sliderSpeedHelico = new QSlider(Qt::Horizontal);
		m_sliderSpeedHelico->setMinimum(0);
		m_sliderSpeedHelico->setMaximum(100);
		m_sliderSpeedHelico->setValue(m_speedHelico);
		m_editSpeedHelico= new QLineEdit;
		m_editSpeedHelico->setMaximumWidth(60);
		m_editSpeedHelico->setText(QString::number(m_speedHelico));
		connect(m_sliderSpeedHelico,SIGNAL(valueChanged(int)),this,SLOT(setSpeedHelico(int)));
		connect(m_editSpeedHelico,SIGNAL(editingFinished()),this,SLOT(setSpeedHelico()));

		QLabel* labelSpeedRotate= new QLabel("Speed orientation ",parent);
		m_sliderSpeedRotHelico = new QSlider(Qt::Horizontal,parent);
		m_sliderSpeedRotHelico->setMinimum(1);
		m_sliderSpeedRotHelico->setMaximum(10);
		m_sliderSpeedRotHelico->setValue(m_speedRotHelico);
		m_editSpeedRotHelico= new QLineEdit(parent);
		m_editSpeedRotHelico->setMaximumWidth(60);
		m_editSpeedRotHelico->setText(QString::number(m_speedRotHelico));
		connect(m_sliderSpeedRotHelico,SIGNAL(valueChanged(int)),this,SLOT(setSpeedRotHelico(int)));
		connect(m_editSpeedRotHelico,SIGNAL(editingFinished()),this,SLOT(setSpeedRotHelico()));

		//Ligne1
		layoutbox3->addWidget(labelP6, 1, 0, 1, 1);
		layoutbox3->addWidget(m_sliderUpDown, 1, 1, 1, 1);
		layoutbox3->addWidget(m_editUpDown, 1, 2, 1, 1);

		//Ligne2
		layoutbox3->addWidget(labelP7, 2, 0, 1, 1);
		layoutbox3->addWidget(m_sliderSpeedHelico, 2, 1, 1, 1);
		layoutbox3->addWidget(m_editSpeedHelico, 2, 2, 1, 1);

		//Ligne2
		layoutbox3->addWidget(labelSpeedRotate, 3, 0, 1, 1);
		layoutbox3->addWidget(m_sliderSpeedRotHelico, 3, 1, 1, 1);
		layoutbox3->addWidget(m_editSpeedRotHelico, 3, 2, 1, 1);

		boxCam->setLayout(layoutbox3);

		//view 3D
		QGroupBox* boxView3d = new QGroupBox("View 3D");
		QGridLayout *layoutbox4 = new QGridLayout();
		checkview1= new QCheckBox("Show 3D info");
		checkview1->setCheckState(Qt::Checked);
		checkview2= new QCheckBox("Show 3D gizmo");
		checkview2->setCheckState(Qt::Checked);
		connect(checkview1,SIGNAL(stateChanged(int)),this, SLOT(info3dChecked(int)));
		connect(checkview2,SIGNAL(stateChanged(int)),this, SLOT(gizmo3dChecked(int)));

		checkview3= new QCheckBox("Wireframe well");
		checkview3->setCheckState(Qt::Unchecked);
		checkview4= new QCheckBox("Show Normals well");
		checkview4->setCheckState(Qt::Unchecked);

		checkview5= new QCheckBox("Show Helico");
		checkview5->setCheckState(Qt::Unchecked);

		connect(checkview3,SIGNAL(stateChanged(int)),this, SLOT(wireframeWellChecked(int)));
		connect(checkview4,SIGNAL(stateChanged(int)),this, SLOT(showNormalsWellChecked(int)));

		connect(checkview5,SIGNAL(stateChanged(int)),this, SLOT(showHelicoChecked(int)));

		layoutbox4->addWidget(checkview1,0,0,1,2);
		layoutbox4->addWidget(checkview2,1,0,1,2);
		layoutbox4->addWidget(checkview3,0,2,1,2);
		layoutbox4->addWidget(checkview4,1,2,1,2);
		layoutbox4->addWidget(checkview5,2,0,1,2);
		boxView3d->setLayout(layoutbox4);


		//animation
		QGroupBox* boxViewAnim = new QGroupBox("Animation");
		QGridLayout *layoutboxAnim = new QGridLayout();

		QLabel* labelAnim1= new QLabel("Speed max");
		m_sliderSpeedMax = new QSlider(Qt::Horizontal);
		m_sliderSpeedMax->setMinimum(200);
		m_sliderSpeedMax->setMaximum(1000);
		m_sliderSpeedMax->setValue(m_speedMaxAnim);
		m_editSpeedMax = new QLineEdit;
		m_editSpeedMax->setMaximumWidth(60);
		m_editSpeedMax->setValidator( new QIntValidator(0, 100, this) );
		m_editSpeedMax->setText(QString::number(m_speedMaxAnim));
		connect(m_sliderSpeedMax,SIGNAL(valueChanged(int)),this,SLOT(setSpeedMaxAnim(int)));
		connect(m_editSpeedMax,SIGNAL(editingFinished()),this,SLOT(setSpeedMaxAnim()));

		//Ligne1
		layoutboxAnim->addWidget(labelAnim1, 1, 0, 1, 1);
		layoutboxAnim->addWidget(m_sliderSpeedMax, 1, 1, 1, 1);
		layoutboxAnim->addWidget(m_editSpeedMax, 1, 2, 1, 1);


		QLabel* labelAnim2= new QLabel("Altitude max");
		m_sliderAltitudeMax = new QSlider(Qt::Horizontal);
		m_sliderAltitudeMax->setMinimum(200);
		m_sliderAltitudeMax->setMaximum(1000);
		m_sliderAltitudeMax->setValue(m_altitudeMaxAnim);
		m_editAltitudeMax = new QLineEdit;
		m_editAltitudeMax->setMaximumWidth(60);
		m_editAltitudeMax->setValidator( new QIntValidator(0, 100, this) );
		m_editAltitudeMax->setText(QString::number(m_altitudeMaxAnim));
		connect(m_sliderAltitudeMax,SIGNAL(valueChanged(int)),this,SLOT(setAltitudeMaxAnim(int)));
		connect(m_editAltitudeMax,SIGNAL(editingFinished()),this,SLOT(setAltitudeMaxAnim()));

		//Ligne2
		layoutboxAnim->addWidget(labelAnim2, 2, 0, 1, 1);
		layoutboxAnim->addWidget(m_sliderAltitudeMax, 2, 1, 1, 1);
		layoutboxAnim->addWidget(m_editAltitudeMax, 2, 2, 1, 1);


		boxViewAnim->setLayout(layoutboxAnim);
	/*	QFrame* line = new QFrame();
		line->setFrameShape(QFrame::HLine);
		line->setFrameShadow(QFrame::Sunken);*/

		QGridLayout *layout = new QGridLayout();

	//	layout->addWidget(label1, 0, 0, 1, 1);
	//	layout->addWidget(label2, 0, 1, 1, 2);
	//	layout->addWidget(line,1,0,1,4);
		//layout->addWidget(boxSurface,0,0,1,4);
		layout->addWidget(boxWells,0,0,1,4);
		layout->addWidget(boxLogs,1,0,1,4);
		layout->addWidget(boxPicks,2,0,1,4);
		layout->addWidget(boxCam,3,0,1,4);
		layout->addWidget(boxView3d,4,0,1,4);
		layout->addWidget(boxViewAnim,5,0,1,4);

		QWidget* propertyHolder = new QWidget;
		propertyHolder->setLayout(layout);
		scrollArea->setWidget(propertyHolder);

	}


//slots
void PropertyPanel::setAltitudeMaxAnim(int value)
{
	if( m_altitudeMaxAnim != value)
	{
		m_altitudeMaxAnim = value;
		m_editAltitudeMax->setText(QString::number(m_altitudeMaxAnim));
		emit altitudeMaxAnimChanged(m_altitudeMaxAnim);
	}
}

void PropertyPanel::setAltitudeMaxAnim()
{
	int value = m_editAltitudeMax->text().toInt();
	if( m_altitudeMaxAnim != value)
	{
		m_altitudeMaxAnim = value;
		m_sliderAltitudeMax->setValue(m_altitudeMaxAnim);
		emit altitudeMaxAnimChanged(m_altitudeMaxAnim);
	}

}

void PropertyPanel::setSpeedMaxAnim(int value)
{
	if( m_speedMaxAnim != value)
	{
		m_speedMaxAnim = value;
		m_editSpeedMax->setText(QString::number(m_speedMaxAnim));
		emit speedMaxAnimChanged(m_speedMaxAnim);
	}
}

void PropertyPanel::setSpeedMaxAnim()
{
	int value = m_editSpeedMax->text().toInt();
	if( m_speedMaxAnim != value)
	{
		m_speedMaxAnim = value;
		m_sliderSpeedMax->setValue(m_speedMaxAnim);
		emit speedMaxAnimChanged(m_speedMaxAnim);
	}

}



void PropertyPanel::setWellPrecision(int value)
{
	if( m_wellPrecision != value)
	{
		m_wellPrecision = value;
		m_editPrecisionWell->setText(QString::number(m_wellPrecision));
		emit simplifySeuilWellChanged(m_wellPrecision);
	}
}

void PropertyPanel::setWellPrecision()
{
	int value = m_editPrecisionWell->text().toInt();
	if( m_wellPrecision != value)
	{
		m_wellPrecision = value;
		m_sliderPrecisionWell->setValue(m_wellPrecision);
		emit simplifySeuilWellChanged(m_wellPrecision);
	}

}


void PropertyPanel::setWellDiameter(int value)
{

	if( m_wellDiameter != value)
	{
		m_wellDiameter = value;
		m_editDiameterWell->setText(QString::number(m_wellDiameter));
		emit wellDiameterChanged(m_wellDiameter);
	}
}
void PropertyPanel::setWellDiameter()
{
	int value = m_editDiameterWell->text().toInt();
	if( m_wellDiameter != value)
	{
		m_wellDiameter = value;
		m_sliderDiameterWell->setValue(m_wellDiameter);
		emit wellDiameterChanged(m_wellDiameter);
	}
}


void PropertyPanel::setWellMapWidth(int value)
{

	if( m_wellMapWidth != value/10.0)
	{
		m_wellMapWidth = value / 10.0;
		m_editMapWidthWell->setText(QString::number(m_wellMapWidth));
		emit wellMapWidthChanged(m_wellMapWidth);
	}
}
void PropertyPanel::setWellMapWidth()
{
	double value = m_editMapWidthWell->text().toDouble();
	if( m_wellMapWidth != value)
	{
		m_wellMapWidth = value;
		m_sliderMapWidthWell->setValue(std::round(m_wellMapWidth*10));
		emit wellMapWidthChanged(m_wellMapWidth);
	}
}


void PropertyPanel::setWellSectionWidth(int value)
{

	if( m_wellSectionWidth != value/10.0)
	{
		m_wellSectionWidth = value / 10.0;
		m_editSectionWidthWell->setText(QString::number(m_wellSectionWidth));
		emit wellSectionWidthChanged(m_wellSectionWidth);
	}
}
void PropertyPanel::setWellSectionWidth()
{
	double value = m_editSectionWidthWell->text().toDouble();
	if( m_wellSectionWidth != value)
	{
		m_wellSectionWidth = value;
		m_sliderSectionWidthWell->setValue(std::round(m_wellSectionWidth*10));
		emit wellSectionWidthChanged(m_wellSectionWidth);
	}
}


void PropertyPanel::setLogsPrecision(int value)
{
	if( m_logsPrecision != value)
	{
		m_logsPrecision = value;
		m_editPrecisionLogs->setText(QString::number(m_logsPrecision));
		emit simplifySeuilLogsChanged(m_logsPrecision);
	}
}

void PropertyPanel::setLogsPrecision()
{
	int value = m_editPrecisionLogs->text().toInt();
	if( m_logsPrecision != value)
	{
		m_logsPrecision = value;
		m_sliderPrecisionLogs->setValue(m_logsPrecision);
		emit simplifySeuilLogsChanged(m_logsPrecision);
	}

}


void PropertyPanel::setLogColor()
{
	 QColor color = QColorDialog::getColor(m_logsColor, this );
	if( color.isValid() )
	{
		m_logsColor = color;
		QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_logsColor.red())+","+QString::number(m_logsColor.green())+","+QString::number(m_logsColor.blue())+")}";
		m_buttonColorLog->setStyleSheet(namecolor);

		emit colorLogChanged(m_logsColor);
	}
}

void PropertyPanel::setSurfacePrecision(int value)
{
	if( m_surfacePrecision != value)
	{
		m_surfacePrecision = value;
		m_editPrecisionSurface->setText(QString::number(m_surfacePrecision));
		emit simplifySurfaceChanged(m_surfacePrecision);

	}
}

void PropertyPanel::setSurfacePrecision()
{
	int value = m_editPrecisionSurface->text().toInt();
	if( m_surfacePrecision != value)
	{
		m_surfacePrecision = value;
		m_sliderPrecisionSurface->setValue(m_surfacePrecision);
		emit simplifySurfaceChanged(m_surfacePrecision);
	}

}

void PropertyPanel::setWellDefaultColor()
{
	 QColor color = QColorDialog::getColor(m_wellColor, this );
	if( color.isValid() )
	{
		m_wellColor = color;
		QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_wellColor.red())+","+QString::number(m_wellColor.green())+","+QString::number(m_wellColor.blue())+")}";
		m_buttonColorWell1->setStyleSheet(namecolor);

		emit colorWellChanged(m_wellColor);

	}
}

void PropertyPanel::setWellSelectedColor()
{
	 QColor color = QColorDialog::getColor(m_wellSelectedColor, this );
	if( color.isValid() )
	{
		m_wellSelectedColor = color;
		QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_wellSelectedColor.red())+","+QString::number(m_wellSelectedColor.green())+","+QString::number(m_wellSelectedColor.blue())+")}";
		m_buttonColorWell2->setStyleSheet(namecolor);

		emit colorSelectedWellChanged(m_wellSelectedColor);
	}
}

void PropertyPanel::setPickThickness(int value)
{
	if( m_pickThickness != value)
	{
		m_pickThickness = value;
		m_editThickness->setText(QString::number(m_pickThickness));
		emit pickThicknessChanged(m_pickThickness);
	}
}
void PropertyPanel::setPickThickness()
{
	int value = m_editThickness->text().toInt();
	if( m_pickThickness != value)
	{
		m_pickThickness = value;
		m_sliderThickness->setValue(m_pickThickness);
		emit pickThicknessChanged(m_pickThickness);
	}
}

void PropertyPanel::setLogThickness(int value)
{
	if( m_logsThickness != value)
	{
		m_logsThickness = value;
		m_editThicknessLog->setText(QString::number(m_logsThickness));
		emit logThicknessChanged(m_logsThickness);
	}
}
void PropertyPanel::setLogThickness()
{
	int value = m_editThickness->text().toInt();
	if( m_logsThickness != value)
	{
		m_logsThickness = value;
		m_sliderThicknessLog->setValue(m_logsThickness);
		emit logThicknessChanged(m_logsThickness);
	}
}



void PropertyPanel::setPickDiameter(int value)
{

	if( m_pickDiameter != value)
	{
		m_pickDiameter = value;
		m_editDiameterPick->setText(QString::number(m_pickDiameter));
		emit pickDiameterChanged(m_pickDiameter);
	}
}
void PropertyPanel::setPickDiameter()
{
	int value = m_editDiameterPick->text().toInt();
	if( m_pickDiameter != value)
	{
		m_pickDiameter = value;
		m_sliderDiameterPick->setValue(m_pickDiameter);
		emit pickDiameterChanged(m_pickDiameter);
	}
}

void PropertyPanel::setSpeedUpDown(int value)
{

	if( m_speedAltitude != value)
	{
		m_speedAltitude = value;
		m_editUpDown->setText(QString::number(m_speedAltitude));
		emit speedUpDownChanged(m_speedAltitude);
	}
}

void PropertyPanel::setSpeedUpDown()
{
	int value = m_editUpDown->text().toInt();
	if( m_speedAltitude != value)
	{
		m_speedAltitude = value;
		m_sliderUpDown->setValue(m_speedAltitude);
		emit speedUpDownChanged(m_speedAltitude);
	}

}


void PropertyPanel::setSpeedHelico(int value)
{

	if( m_speedHelico != value)
	{
		m_speedHelico = value;
		m_editSpeedHelico->setText(QString::number(m_speedHelico));
		emit speedHelicoChanged(m_speedHelico);
	}
}

void PropertyPanel::setSpeedHelico()
{
	int value = m_editSpeedHelico->text().toInt();
	if( m_speedHelico != value)
	{
		m_speedHelico = value;
		m_sliderSpeedHelico->setValue(m_speedHelico);
		emit speedHelicoChanged(m_speedHelico);
	}

}

void PropertyPanel::setSpeedRotHelico(int value)
{
	if( m_speedRotHelico != value)
	{
		m_speedRotHelico = value;
		m_editSpeedRotHelico->setText(QString::number(m_speedRotHelico));
		emit speedRotHelicoChanged(m_speedRotHelico);
	}
}

void PropertyPanel::setSpeedRotHelico()
{
	int value = m_editSpeedRotHelico->text().toInt();
	if( m_speedRotHelico != value)
	{
		m_speedRotHelico = value;
		m_sliderSpeedRotHelico->setValue(m_speedRotHelico);
		emit speedRotHelicoChanged(m_speedRotHelico);
	}
}

void PropertyPanel::showHelicoChecked(int etat)
{
	if(m_showHelico != etat)
	{
		m_showHelico =etat;
		emit showHelicoChanged(m_showHelico);
	}
}

void PropertyPanel::info3dChecked(int etat)
{

	m_showInfos3d =etat;
	emit showInfo3DChanged(m_showInfos3d);
}

void PropertyPanel::gizmo3dChecked(int etat)
{
	m_showGizmo3d = etat;
	emit showGizmo3DChanged(m_showGizmo3d);
}

void PropertyPanel::wireframeWellChecked(int etat)
{

	m_wireframeWell =etat;
	emit wireframeWellChanged(m_wireframeWell);
}

void PropertyPanel::showNormalsWellChecked(int etat)
{
	m_showNormalsWell = etat;
	emit showNormalsWellChanged(m_showNormalsWell);
}


void PropertyPanel::openIni()
{
	QSettings settings("NextVision.ini",QSettings::IniFormat);
	settings.beginGroup("Global");

	settings.endGroup();

	m_logsThickness =settings.value("Log/thickness",2).toInt();

	QString colorLogStr = settings.value("Log/defaultColor","0|255|0").toString();
	QStringList listRgb1 = colorLogStr.split("|");
	if(listRgb1.length()== 3)
	{
		int red = listRgb1[0].toInt();
		int green =  listRgb1[1].toInt();
		int blue =  listRgb1[2].toInt();
		m_logsColor = QColor(red,green,blue);
	}

	m_pickThickness =settings.value("Pick/thickness",15.0).toInt();
	m_pickDiameter =settings.value("Pick/diameter",50.0).toInt();


	m_wellPrecision = settings.value("Well/precision",2).toInt();
	m_wellDiameter = settings.value("Well/diameter",25.0).toDouble();
	m_wellMapWidth = settings.value("Well/mapWidth",2.0).toDouble();
	m_wellSectionWidth = settings.value("Well/sectionWidth",2.0).toDouble();

	m_showGizmo3d = settings.value("View3d/showGizmo",true).toBool();
	m_showInfos3d = settings.value("View3d/showInfos",true).toBool();
	//m_showHelico = settings.value("View3d/showHelico",true).toBool();


	m_speedAltitude = settings.value("Camera/speedUpDown",25.0).toDouble();
	m_speedHelico = settings.value("Camera/speedHelico",25.0).toDouble();
	m_speedRotHelico = settings.value("Camera/speedRotHelico",2.0).toDouble();


	QString colorWellsStr = settings.value("Well/defaultColor","255|255|0").toString();

	QStringList listRgb = colorWellsStr.split("|");
	if(listRgb.length()== 3)
	{
		int red = listRgb[0].toInt();
		int green =  listRgb[1].toInt();
		int blue =  listRgb[2].toInt();
		m_wellColor = QColor(red,green,blue);
	}

	QString colorSelWellsStr = settings.value("Well/selectedColor","0|255|255").toString();

	QStringList listRgbsel = colorSelWellsStr.split("|");
	if(listRgb.length()== 3)
	{
		int red = listRgbsel[0].toInt();
		int green =  listRgbsel[1].toInt();
		int blue =  listRgbsel[2].toInt();
		m_wellSelectedColor = QColor(red,green,blue);
	}

	m_speedMaxAnim = settings.value("Animation/speedMax",200).toInt();
	m_altitudeMaxAnim = settings.value("Animation/altitudeMax",400).toInt();



	init();
}

void PropertyPanel::saveIni()
{
	 QSettings settings("NextVision.ini",QSettings::IniFormat);

	settings.beginGroup("Global");
	settings.setValue("posX", 0);
	settings.setValue("posY", 0);
	settings.endGroup();


	settings.setValue("Surface/precision", m_surfacePrecision);



	QString defaultcolorstr= QString::number(m_wellColor.red())+"|"+QString::number(m_wellColor.green())+"|"+QString::number(m_wellColor.blue());
	QString selectcolorstr= QString::number(m_wellSelectedColor.red())+"|"+QString::number(m_wellSelectedColor.green())+"|"+QString::number(m_wellSelectedColor.blue());
	settings.setValue("Well/precision", m_wellPrecision);
	settings.setValue("Well/diameter", m_wellDiameter);
	settings.setValue("Well/mapWidth", m_wellMapWidth);
	settings.setValue("Well/sectionWidth", m_wellSectionWidth);
	settings.setValue("Well/defaultColor", defaultcolorstr);
	settings.setValue("Well/selectedColor", selectcolorstr);

	QString logcolorstr= QString::number(m_logsColor.red())+"|"+QString::number(m_logsColor.green())+"|"+QString::number(m_logsColor.blue());
	settings.setValue("Log/thickness", m_logsThickness);
	settings.setValue("Log/defaultColor", logcolorstr);


	settings.setValue("Pick/thickness", m_pickThickness);
	settings.setValue("Pick/diameter", m_pickDiameter);

	settings.setValue("View3d/showInfos", m_showInfos3d);
	settings.setValue("View3d/showGizmo", m_showGizmo3d);
	settings.setValue("View3d/showHelico", m_showHelico);


	settings.setValue("Camera/speedUpDown", m_speedAltitude);
	settings.setValue("Camera/speedHelico", m_speedHelico);
	settings.setValue("Camera/speedRotHelico", m_speedRotHelico);

	settings.setValue("Animation/altitudeMax", m_altitudeMaxAnim);
	settings.setValue("Animation/speedMax", m_speedMaxAnim);

}

void PropertyPanel::init()
{
	QString namecolorLog= "QPushButton {background-color: rgb("+QString::number(m_logsColor.red())+","+QString::number(m_logsColor.green())+","+QString::number(m_logsColor.blue())+")}";
	m_buttonColorLog->setStyleSheet(namecolorLog);


	QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_wellColor.red())+","+QString::number(m_wellColor.green())+","+QString::number(m_wellColor.blue())+")}";
	m_buttonColorWell1->setStyleSheet(namecolor);

	QString namecolor2= "QPushButton {background-color: rgb("+QString::number(m_wellSelectedColor.red())+","+QString::number(m_wellSelectedColor.green())+","+QString::number(m_wellSelectedColor.blue())+")}";
	m_buttonColorWell2->setStyleSheet(namecolor2);

	emit colorWellChanged(m_wellColor);
	emit colorSelectedWellChanged(m_wellSelectedColor);

	m_sliderDiameterWell->setValue(m_wellDiameter);
	m_sliderDiameterPick->setValue(m_pickDiameter);
	m_sliderThickness->setValue(m_pickThickness);
	m_sliderMapWidthWell->setValue(m_wellMapWidth*10);
	m_sliderSectionWidthWell->setValue(m_wellSectionWidth*10);



	m_editDiameterWell->setText(QString::number(m_wellDiameter));
	m_editDiameterPick->setText(QString::number(m_pickDiameter));
	m_editThickness->setText(QString::number(m_pickThickness));
	m_editMapWidthWell->setText(QString::number(m_wellMapWidth));
	m_editSectionWidthWell->setText(QString::number(m_wellSectionWidth));

	emit wellDiameterChanged(m_wellDiameter);
	emit pickDiameterChanged(m_pickDiameter);
	emit pickThicknessChanged(m_pickThickness);
	emit wellMapWidthChanged(m_wellMapWidth);
	emit wellSectionWidthChanged(m_wellSectionWidth);

	m_editThicknessLog->setText(QString::number(m_logsThickness));
	m_sliderThicknessLog->setValue(m_logsThickness);
	emit logThicknessChanged(m_logsThickness);

	emit colorLogChanged(m_logsColor);

	if(m_showInfos3d) checkview1->setCheckState(Qt::Checked);
	else checkview1->setCheckState(Qt::Unchecked);

	if(m_showGizmo3d)checkview2->setCheckState(Qt::Checked);
	else checkview2->setCheckState(Qt::Unchecked);

	if(m_showHelico)checkview5->setCheckState(Qt::Checked);
	else checkview5->setCheckState(Qt::Unchecked);

	m_sliderUpDown->setValue(m_speedAltitude);
	m_editUpDown->setText(QString::number(m_speedAltitude));

	emit speedUpDownChanged(m_speedAltitude);

	m_editSpeedHelico->setText(QString::number(m_speedHelico));
	m_sliderSpeedHelico->setValue(m_speedHelico);
	emit speedHelicoChanged(m_speedHelico);

	m_editSpeedRotHelico->setText(QString::number(m_speedRotHelico));
	m_sliderSpeedRotHelico->setValue(m_speedRotHelico);
	emit speedHelicoChanged(m_speedRotHelico);


	m_editSpeedMax->setText(QString::number(m_speedMaxAnim));
	m_sliderSpeedMax->setValue(m_speedMaxAnim);
	emit speedMaxAnimChanged(m_speedMaxAnim);

	m_editAltitudeMax->setText(QString::number(m_altitudeMaxAnim));
	m_sliderAltitudeMax->setValue(m_altitudeMaxAnim);
	emit altitudeMaxAnimChanged(m_altitudeMaxAnim);



}

void PropertyPanel::reset()
{
	m_wellPrecision = 2;
	m_logsPrecision = 1;
	m_logsThickness = 3;
	m_logsColor = Qt::green;
	m_wellDiameter=30.0;
	m_wellMapWidth=2.0;
	m_wellSectionWidth=2.0;
	m_wellColor = Qt::yellow;
	m_wellSelectedColor = Qt::cyan;
	m_pickDiameter =50.0f;
	m_pickThickness = 15.0f;
	m_speedAltitude = 25.0f;
	m_speedHelico = 10.0f;
	m_surfacePrecision=10;
	m_showInfos3d = true;
	m_showGizmo3d = true;
	m_wireframeWell = false;
	m_showNormalsWell = false;
	m_showHelico = false;
	m_speedMaxAnim = 200;
	m_altitudeMaxAnim = 400;

	init();
}

