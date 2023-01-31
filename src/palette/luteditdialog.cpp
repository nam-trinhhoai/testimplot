#include "luteditdialog.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSlider>
#include <QMenu>
#include <QMenuBar>
#include <QActionGroup>
#include <QCheckBox>
#include <QPushButton>
#include <QDebug>

#include "lutrenderutil.h"
#include "lutwidget.h"

LUTEditDialog::LUTEditDialog(const QHistogram &histo, const QVector2D & restrictedRange,const LookupTable & table,QWidget* parent): QDialog(parent) {

	QHBoxLayout *mainLayout=new QHBoxLayout(this);
	mainLayout->setSpacing(0);


	//The main graphical component
	QWidget *lutWidget=new QWidget(this);
	QHBoxLayout *lutLayout=new QHBoxLayout(lutWidget);
	lutLayout->setSpacing(0);

	QWidget *centerWidget=new QWidget(this);
	QVBoxLayout *layout=new QVBoxLayout(centerWidget);
	layout->setSpacing(0);

	m_hSlider=new QSlider(Qt::Orientation::Horizontal,this);
	m_vSlider=new QSlider(Qt::Orientation::Vertical,this);

	m_lutEditor=new LUTWidget(this);

	connect(m_lutEditor, SIGNAL(lookupTableChanged(const LookupTable&)), this, SLOT(lookupTableChangedInternal(const LookupTable&)));
	connect(m_lutEditor, SIGNAL(lookupTableFunctionParamsChanged(int, int)), this, SLOT(lookupTableFunctionParamsChanged(int, int)));
	connect(m_lutEditor, SIGNAL(sizeChanged()), this, SLOT(lookupPanelSizeChanged()));


	layout->addWidget(m_lutEditor,1);
	layout->addWidget(m_hSlider,0,Qt::AlignRight);
	lutLayout->addWidget(centerWidget,1);


	QWidget *sliderWidget=new QWidget(this);
	{
		QVBoxLayout *sliderLayout=new QVBoxLayout(sliderWidget);
		sliderLayout->setSpacing(0);
		sliderLayout->addSpacing(10);
		sliderLayout->addWidget(m_vSlider,0,Qt::AlignRight);
	}

	lutLayout->addWidget(sliderWidget,0,Qt::AlignTop);
	connect(m_hSlider, SIGNAL(valueChanged(int)), this, SLOT(param1Changed(int)));
	connect(m_vSlider, SIGNAL(valueChanged(int)), this, SLOT(param2Changed(int)));

	mainLayout->addWidget(lutWidget);

	//The Lateral buttons
	QWidget * lateralWidget=new QWidget(this);
	QVBoxLayout *lateralLayout=new QVBoxLayout(lateralWidget);

	QPushButton * razTransp=new QPushButton(" RAZ transparency");
	connect(razTransp, SIGNAL(clicked()), this, SLOT(razTransp()));

	QPushButton * razFunction=new QPushButton(" RAZ function");
	connect(razFunction, SIGNAL(clicked()), this, SLOT(razFunction()));

	m_inverted=new QCheckBox("Invert");
	connect(m_inverted, SIGNAL(stateChanged(int)), this, SLOT(invert(int)));

	lateralLayout->addWidget(razTransp);
	lateralLayout->addWidget(razFunction);
	lateralLayout->addWidget(m_inverted);

	mainLayout->addWidget(lateralWidget);

	createActions();

	setHistogramAndLookupTable(histo, restrictedRange, table);
}

void LUTEditDialog::razTransp()
{
	m_lutEditor->razTransp();
}

void LUTEditDialog::razFunction()
{
	m_lutEditor->razFunction();
}

void LUTEditDialog::invert(int state)
{
	m_lutEditor->setTransfertInverted(state==Qt::Checked);
}

void LUTEditDialog::functionChanged(QAction * action)
{
	if(action==m_linearFctAction)
	{
		m_lutEditor->setTransfertFunctionType(AbstractFct::FUNCTION_TYPE::LINEAR);
	}else if(action==m_binaryFctAction)
	{
		m_lutEditor->setTransfertFunctionType(AbstractFct::FUNCTION_TYPE::BINARY);
	}else if(action==m_binlinearFctAction)
	{
		m_lutEditor->setTransfertFunctionType(AbstractFct::FUNCTION_TYPE::BINLINEAR);
	}else if(action==m_logFctAction)
	{
		m_lutEditor->setTransfertFunctionType(AbstractFct::FUNCTION_TYPE::LOG);
	}else if(action==m_tri1Action)
	{
		m_lutEditor->setTransfertFunctionType(AbstractFct::FUNCTION_TYPE::TRIANGLE1);
	}else if(action==m_tri2Action)
	{
		m_lutEditor->setTransfertFunctionType(AbstractFct::FUNCTION_TYPE::TRIANGLE2);
	}
}

void LUTEditDialog::createActions()
{
	QMenuBar *menubar = new QMenuBar(this);
	layout()->setMenuBar(menubar);

	QMenu* menu=menubar->addMenu(tr("&Functions"));

	m_functionGroup = new QActionGroup(this);
	m_functionGroup->setEnabled(true);
	m_functionGroup->setExclusive(true);
	connect(m_functionGroup, SIGNAL(triggered(QAction *)), this, SLOT(functionChanged(QAction *)));

	m_linearFctAction = new QAction(tr("&Linear"), this);
	m_linearFctAction->setCheckable(true);
	m_linearFctAction->setChecked(true);
	m_linearFctAction->setIconVisibleInMenu(true);
	m_linearFctAction->setIcon(QIcon(":/colortable/icons/linear.gif"));
	m_functionGroup->addAction(m_linearFctAction);

	m_binaryFctAction = new QAction(tr("&Binary"), this);
	m_binaryFctAction->setCheckable(true);
	m_binaryFctAction->setIconVisibleInMenu(true);
	m_binaryFctAction->setIcon(QIcon(":/colortable/icons/binary.gif"));
	m_functionGroup->addAction(m_binaryFctAction);

	m_binlinearFctAction = new QAction(tr("&Binary Linear"), this);
	m_binlinearFctAction->setCheckable(true);
	m_binlinearFctAction->setIconVisibleInMenu(true);
	m_binlinearFctAction->setIcon(QIcon(":/colortable/icons/binlinear.gif"));
	m_functionGroup->addAction(m_binlinearFctAction);

	m_logFctAction = new QAction(tr("&Log"), this);
	m_logFctAction->setCheckable(true);
	m_logFctAction->setIconVisibleInMenu(true);
	m_logFctAction->setIcon(QIcon(":/colortable/icons/logarithm.gif"));
	m_functionGroup->addAction(m_logFctAction);

	m_tri1Action = new QAction(tr("&Triangle 1"), this);
	m_tri1Action->setCheckable(true);
	m_tri1Action->setIconVisibleInMenu(true);
	m_tri1Action->setIcon(QIcon(":/colortable/icons/triangle1.gif"));
	m_functionGroup->addAction(m_tri1Action);

	m_tri2Action = new QAction(tr("&Triangle 2"), this);
	m_tri2Action->setCheckable(true);
	m_tri2Action->setIconVisibleInMenu(true);
	m_tri2Action->setIcon(QIcon(":/colortable/icons/triangle2.gif"));
	m_functionGroup->addAction(m_tri2Action);

    menu->addActions(m_functionGroup->actions());
}


void LUTEditDialog::setHistogramAndLookupTable(const QHistogram &histo, const QVector2D& restrictedRange, const LookupTable & table)
{
	m_lutEditor->setHistogramAndLookupTable(histo,restrictedRange,table);

	//Update sliders
	disconnect(m_hSlider, SIGNAL(valueChanged(int)), this, SLOT(param1Changed(int)));
	disconnect(m_vSlider, SIGNAL(valueChanged(int)), this, SLOT(param2Changed(int)));

	m_vSlider->setMinimum(0);
	m_vSlider->setMaximum(table.size());
	m_vSlider->setValue(table.getFunctionParam2());

	m_hSlider->setMinimum(0);
	m_hSlider->setMaximum(table.size());
	m_hSlider->setValue(table.getFunctionParam1());

	connect(m_hSlider, SIGNAL(valueChanged(int)), this, SLOT(param1Changed(int)));
	connect(m_vSlider, SIGNAL(valueChanged(int)), this, SLOT(param2Changed(int)));


	//The function
	disconnect(m_functionGroup, SIGNAL(triggered(QAction *)), this, SLOT(functionChanged(QAction *)));

	AbstractFct::FUNCTION_TYPE type = table.getFunctionType();
	switch(type)
	{
	case AbstractFct::FUNCTION_TYPE::BINARY:
		m_binaryFctAction->setChecked(true);
		break;
	case AbstractFct::FUNCTION_TYPE::BINLINEAR:
		m_binlinearFctAction->setChecked(true);
		break;
	case AbstractFct::FUNCTION_TYPE::LOG:
		m_logFctAction->setChecked(true);
		break;
	case AbstractFct::FUNCTION_TYPE::TRIANGLE1:
		m_tri1Action->setChecked(true);
		break;
	case AbstractFct::FUNCTION_TYPE::TRIANGLE2:
		m_tri2Action->setChecked(true);
		break;
	default:
		m_linearFctAction->setChecked(true);
		break;
	}
	connect(m_functionGroup, SIGNAL(triggered(QAction *)), this, SLOT(functionChanged(QAction *)));



	disconnect(m_inverted, SIGNAL(stateChanged(int)), this, SLOT(invert(int)));
	if(table.isFunctionInverted())
		m_inverted->setCheckState(Qt::Checked);
	else
		m_inverted->setCheckState(Qt::Unchecked);

	connect(m_inverted, SIGNAL(stateChanged(int)), this, SLOT(invert(int)));
}

void LUTEditDialog::setLookupTable( const LookupTable & table)
{
	m_lutEditor->setLookupTable(table);

	//Update sliders
	disconnect(m_hSlider, SIGNAL(valueChanged(int)), this, SLOT(param1Changed(int)));
	disconnect(m_vSlider, SIGNAL(valueChanged(int)), this, SLOT(param2Changed(int)));

	m_vSlider->setMinimum(0);
	m_vSlider->setMaximum(table.size());
	m_vSlider->setValue(table.getFunctionParam2());

	m_hSlider->setMinimum(0);
	m_hSlider->setMaximum(table.size());
	m_hSlider->setValue(table.getFunctionParam1());

	connect(m_hSlider, SIGNAL(valueChanged(int)), this, SLOT(param1Changed(int)));
	connect(m_vSlider, SIGNAL(valueChanged(int)), this, SLOT(param2Changed(int)));


	//The function
	disconnect(m_functionGroup, SIGNAL(triggered(QAction *)), this, SLOT(functionChanged(QAction *)));

	AbstractFct::FUNCTION_TYPE type = table.getFunctionType();
	switch(type)
	{
	case AbstractFct::FUNCTION_TYPE::BINARY:
		m_binaryFctAction->setChecked(true);
		break;
	case AbstractFct::FUNCTION_TYPE::BINLINEAR:
		m_binlinearFctAction->setChecked(true);
		break;
	case AbstractFct::FUNCTION_TYPE::LOG:
		m_logFctAction->setChecked(true);
		break;
	case AbstractFct::FUNCTION_TYPE::TRIANGLE1:
		m_tri1Action->setChecked(true);
		break;
	case AbstractFct::FUNCTION_TYPE::TRIANGLE2:
		m_tri2Action->setChecked(true);
		break;
	default:
		m_linearFctAction->setChecked(true);
		break;
	}
	connect(m_functionGroup, SIGNAL(triggered(QAction *)), this, SLOT(functionChanged(QAction *)));

	disconnect(m_inverted, SIGNAL(stateChanged(int)), this, SLOT(invert(int)));
	if(table.isFunctionInverted())
		m_inverted->setCheckState(Qt::Checked);
	else
		m_inverted->setCheckState(Qt::Unchecked);

	connect(m_inverted, SIGNAL(stateChanged(int)), this, SLOT(invert(int)));
}

void LUTEditDialog::lookupTableFunctionParamsChanged(int p1, int p2)
{
	disconnect(m_hSlider, SIGNAL(valueChanged(int)), this, SLOT(param1Changed(int)));
	disconnect(m_vSlider, SIGNAL(valueChanged(int)), this, SLOT(param2Changed(int)));

	m_hSlider->setValue(p1);
	m_vSlider->setValue(p2);

	connect(m_hSlider, SIGNAL(valueChanged(int)), this, SLOT(param1Changed(int)));
	connect(m_vSlider, SIGNAL(valueChanged(int)), this, SLOT(param2Changed(int)));
}


void LUTEditDialog::lookupTableChangedInternal(const LookupTable& colorTable)
{
	emit lookupTableChanged(colorTable);
}


void LUTEditDialog::param1Changed(int val)
{
	m_lutEditor->setLookupTableParam1(val);

}
void LUTEditDialog::param2Changed(int val)
{
	m_lutEditor->setLookupTableParam2(val);
}

void LUTEditDialog::lookupPanelSizeChanged()
{
	qDebug()<<"SizeChanged";
	m_hSlider->setMinimumSize( m_lutEditor->width()-LUTRenderUtil::X_MIN, 20 );
	m_hSlider->setMaximumSize( m_lutEditor->width()-LUTRenderUtil::X_MIN, 20 );
	m_vSlider->setMinimumSize( 20,m_lutEditor->height()-LUTRenderUtil::Y_OFFSET );
	m_vSlider->setMaximumSize(20,m_lutEditor->height()-LUTRenderUtil::Y_OFFSET );
}
LUTEditDialog::~LUTEditDialog() {

}
