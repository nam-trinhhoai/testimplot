#include <QtWidgets>

#include "GraphicToolsWidget.h"
#include "abstract2Dinnerview.h"
#include "geotimegraphicsview.h"
#include "GraphicSceneEditor.h"
#include "sliceqglgraphicsview.h"

#include <QListWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QString>
#include <QPushButton>
#include <QListWidget>
#include <QListWidget>
#include <QDialogButtonBox>
#include <QAction>
#include <QFont>
#include <QSlider>
#include <QProxyStyle>

class SliderStyle : public QProxyStyle
{
public:
	using QProxyStyle::QProxyStyle;

	int styleHint(QStyle::StyleHint hint, const QStyleOption* option = 0, const QWidget* widget = 0, QStyleHintReturn* returnData = 0) const
	{
		if (hint == QStyle::SH_Slider_AbsoluteSetButtons)
			return (Qt::LeftButton | Qt::MiddleButton | Qt::RightButton);
		return QProxyStyle::styleHint(hint, option, widget, returnData);
	}
};

GraphicToolsWidget* GraphicToolsWidget::m_pInstance = nullptr;

template <typename T>
void updateSelectedItem(QVector<T> in_ActionVec, int &old_item_index);

typedef struct {
	QString text;
	QString iconPath;
	QString shortcut;
} stQActionParams;

static QVector<stQActionParams>  ToolsQActionParamsVec = {
		{"default",			":/slicer/icons/graphic_tools/mouse.png",		""},
		{"Free Draw",			":/slicer/icons/graphic_tools/pencil.png",		""},
		{"Erase",			":/slicer/icons/graphic_tools/erase.png",""},
		{"Fill",			":/slicer/icons/graphic_tools/paint_bucket.png",""},
		{"Copy",			":/slicer/icons/graphic_tools/copy.png",""},
		{"Paste",			":/slicer/icons/graphic_tools/paste.png",""},
		{"Cut",				":/slicer/icons/graphic_tools/cut.png",""},
		{"Delete",			":/slicer/icons/graphic_tools/delete.png",""},
		{"Bring to Front",	":/slicer/icons/graphic_tools/bringtofront.png",""},
		{"Send to Back",	":/slicer/icons/graphic_tools/sendtoback.png",""},
		{"Undo",			":/slicer/icons/graphic_tools/undo.png",""},
		{"Redo",			":/slicer/icons/graphic_tools/redo.png",""},
		{"Draw Line",				":/slicer/icons/graphic_tools/line.png",				""},
		{"Draw Rectangle",			":/slicer/icons/graphic_tools/rectangle.png",			""},
		{"Draw Rounded Rectangle",	":/slicer/icons/graphic_tools/rounded-rectangle.png",	""},
		{"Draw Circle",				":/slicer/icons/graphic_tools/circle.png", 				""},
		{"Draw Ellipse",				":/slicer/icons/graphic_tools/ellipse.png", 			""},
		//{"Draw Triangle",			":/slicer/icons/graphic_tools/triangle.png",			""},
		{"Draw Polygone",			":/slicer/icons/graphic_tools/hexagon.png",				""},
		{"Draw PolyLine",			":/slicer/icons/graphic_tools/polygonal-chain.png",		""},
		//		{"Bezier Curve",		":/slicer/icons/graphic_tools/bspline.png",		""},
		//		{"Sub Divide Bezier",	":/slicer/icons/graphic_tools/bspline.png",		""},
		//		{"Cubic B-Spline",		":/slicer/icons/graphic_tools/bspline.png",		""},
		//		{"Subdivide B-Spline",	":/slicer/icons/graphic_tools/bspline.png",		""},
		//		{"Rational Bezier",		":/slicer/icons/graphic_tools/bspline.png",		""},
		//		{"NURBS",				":/slicer/icons/graphic_tools/bspline.png",		""},
		{"Draw Bezier",			":/slicer/icons/graphic_tools/bezier.png",	""},
		{"Draw Cubic B-spline",		":/slicer/icons/graphic_tools/bspline.png",	""},
		{"Draw regular Bezier path",		":/slicer/icons/graphic_tools/pen_bezier.png",	""},
		{"Draw list Bezier path",		":/slicer/icons/graphic_tools/polygonal-chain.png",	""}
};

static QVector<stQActionParams>  ColorsQActionParamsVec = {
		{"white",		":/slicer/icons/graphic_tools/white.png",	""},
		//{"black",		":/slicer/icons/graphic_tools/black.png",	""},
		{"grey",		":/slicer/icons/graphic_tools/grey.png",	""},
		//{"blue",		":/slicer/icons/graphic_tools/blue.png",	""},
		{"red",			":/slicer/icons/graphic_tools/red.png",		""},
		{"green",		":/slicer/icons/graphic_tools/green.png",	""},
		{"yellow",		":/slicer/icons/graphic_tools/yellow.png",	""},
		//{"magenta",		":/slicer/icons/graphic_tools/magneta.png",	""},
		{"more",		":/slicer/icons/graphic_tools/more.png",	""},
		//{"orange", ":/slicer/icons/graphic_tools/orange.png",""},
};

static QVector<stQActionParams> BrushQActionParamsVec = {
		{"No Brush",			":/slicer/icons/graphic_tools/no_brush.png",	""},
		{"Solid",				":/slicer/icons/graphic_tools/brush_4.png",		""},
		{"Horizontal",			":/slicer/icons/graphic_tools/brush_5.png",		""},
		{"Vertical",			":/slicer/icons/graphic_tools/brush_6.png",		""},
		{"Cross",				":/slicer/icons/graphic_tools/brush_7.png",		""},
		{"Backward Diagonal",	":/slicer/icons/graphic_tools/brush_8.png",		""},
		{"Forward Diagonal",	":/slicer/icons/graphic_tools/brush_9.png",		""},
		{"Diagonal Cross",		":/slicer/icons/graphic_tools/brush_10.png",	""},
		//{"Linear Gradient",		":/slicer/icons/graphic_tools/brush_1.png",		""},
		//{"Radial Gradient",		":/slicer/icons/graphic_tools/brush_2.png",		""},
		//{"Conical Gradient",	":/slicer/icons/graphic_tools/brush_3.png",		""},
};

static QStringList Penwidth_Pixmap_IconList=
{
		":/slicer/icons/graphic_tools/width_1.png",
		":/slicer/icons/graphic_tools/width_2.png",
		":/slicer/icons/graphic_tools/width_3.png",
		":/slicer/icons/graphic_tools/width_4.png",
		":/slicer/icons/graphic_tools/width_5.png",
		":/slicer/icons/graphic_tools/width_6.png"
};

static QStringList PenStyle_Pixmap_IconList=
{
		":/slicer/icons/graphic_tools/style_solid.png",
		":/slicer/icons/graphic_tools/style_dash.png",
		":/slicer/icons/graphic_tools/style_dot.png",
		":/slicer/icons/graphic_tools/style_dash_dot.png",
		":/slicer/icons/graphic_tools/style_dash_dot_dot.png"
		//		":/slicer/icons/graphic_tools/none.png"
};

void GraphicToolsWidget::showPalette(QString viewTitle)
{
	if (!m_pInstance)
	{
		m_pInstance = new GraphicToolsWidget();
	}
	m_pInstance->m_GraphicToolsSettings.enabled = true;
	m_pInstance->changeGraphicsViewsDragMode(QGraphicsView::ScrollHandDrag);

	m_pInstance->show();
}

void GraphicToolsWidget::closePalette()
{
	if (m_pInstance)
	{
		m_pInstance->changeGraphicsViewsDragMode(QGraphicsView::ScrollHandDrag);
		m_pInstance->close();
		m_pInstance->deleteLater();
		m_pInstance=nullptr;
	}
}

void GraphicToolsWidget::changeGraphicsViewsDragMode(QGraphicsView::DragMode mode)
{
	foreach(GraphicSceneEditor* p, m_InnverViewsScene)
	{
		if(p->innerView()->view() != nullptr)
		{
			SliceQGLGraphicsView* slice = dynamic_cast<SliceQGLGraphicsView*>(p->innerView()->view());
			if(slice !=nullptr)
				p->innerView()->view()->setDragMode(QGraphicsView::NoDrag);
			else
				p->innerView()->view()->setDragMode(mode);
		}
	}
}

st_GraphicToolsSettings GraphicToolsWidget::getPaletteSettings()
{
	if (!m_pInstance)
	{
		m_pInstance = new GraphicToolsWidget();
	}
	return m_pInstance->m_GraphicToolsSettings;
}

void GraphicToolsWidget::setDefaultAction()
{
	if (m_pInstance)
	{
		if (!m_pInstance->m_ToolsQActionVec[0]->isChecked())
		{
			m_pInstance->m_GraphicToolsSettings.action= eGraphicAction_Default;
			m_pInstance->changeGraphicsViewsDragMode(QGraphicsView::ScrollHandDrag);
			m_pInstance->deselectTools();
		}
		else
		{
			m_pInstance->changeGraphicsViewsDragMode(QGraphicsView::RubberBandDrag);
		}
	}
}

void GraphicToolsWidget::removeInnerView2(Abstract2DInnerView* pInnerview){

	int position = 0;

	for(GraphicSceneEditor *scene:this->m_InnverViewsScene){
		if(scene->innerView() == pInnerview){
			disconnect(this, nullptr, scene, nullptr);
		//	disconnect(scene, nullptr, m_pInstance, nullptr);
			this->m_InnverViewsScene.erase(this->m_InnverViewsScene.begin()+position);
			break;
		}
		position++;
	}
}

void GraphicToolsWidget::removeInnerView(Abstract2DInnerView* pInnerview){

	int position = 0;
	if(m_pInstance ==nullptr)  return;
	for(GraphicSceneEditor *scene:m_pInstance->m_InnverViewsScene){
		if(scene->innerView() == pInnerview){
			disconnect(m_pInstance, nullptr, scene, nullptr);
		//	disconnect(scene, nullptr, m_pInstance, nullptr);
			m_pInstance->m_InnverViewsScene.erase(m_pInstance->m_InnverViewsScene.begin()+position);
			break;
		}
		position++;
	}
}

void GraphicToolsWidget::setActiveInnerView(QString innverView, GraphicSceneEditor *scene)
{
	if (m_pInstance)
	{
		if (!(std::find(m_pInstance->m_InnverViewsScene.begin(),
				m_pInstance->m_InnverViewsScene.end(),scene)!=m_pInstance->m_InnverViewsScene.end()))
		{
			m_pInstance->m_InnverViewsScene.push_back(scene);

			connect(scene,&GraphicSceneEditor::destroyed,m_pInstance,[m_pInstance,scene](){ m_pInstance->removeInnerView(scene->innerView());});

			connect(m_pInstance, &GraphicToolsWidget::graphicOptionsChanged,
					scene, &GraphicSceneEditor::GraphicToolNewAction );
			connect(m_pInstance, &GraphicToolsWidget::penSettingsChanged,
					scene, &GraphicSceneEditor::updateSelectedItemsPen );
			connect(m_pInstance, &GraphicToolsWidget::brushSettingsChanged,
					scene, &GraphicSceneEditor::updateSelectedItemsBrush );
			connect(m_pInstance, &GraphicToolsWidget::textColorChanged,
					scene, &GraphicSceneEditor::updateSelectedTextColor );
			connect(m_pInstance, &GraphicToolsWidget::textFontChanged,
					scene, &GraphicSceneEditor::updateSelectedTextFont );
			connect(m_pInstance->m_SmoothSlider, &QSlider::valueChanged,
					scene, &GraphicSceneEditor::updateSelectedCurveSmooth );
		}
		 emit m_pInstance->setViewCurrent(scene->innerView());
		m_pInstance->m_GraphicToolsSettings.viewName=innverView;
		m_pInstance->m_GraphicToolsSettings.pActiveScene = scene;
	}
}
GraphicToolsWidget::GraphicToolsWidget():
																																																																																						QDialog()
{
	this->setWindowTitle(GraphicToolsWidgetTitle);

	// Tools Group Box
	QGroupBox *toolsGroupBox = new QGroupBox(tr("Tools"),this);
	QGridLayout *toolsGroupBoxLayout = new QGridLayout();
	//toolsGroupBoxLayout->setContentsMargins(5,5,5,5);
	toolsGroupBoxLayout->setSpacing(0);
	int gridLayoutRow = 1;
	int gridLayoutColumn = 0;
	for (int i=0; i< 12; i++)
	{
		QAction *action = new QAction(QIcon(ToolsQActionParamsVec[i].iconPath), ToolsQActionParamsVec[i].text, this);
		action->setShortcut(ToolsQActionParamsVec[i].shortcut);
		action->setCheckable(true);

		QToolButton *button = new QToolButton(this);
		button->setDefaultAction(action);
		toolsGroupBoxLayout->addWidget(button,gridLayoutRow,gridLayoutColumn);
		gridLayoutColumn++;
		if (gridLayoutColumn>3)
		{
			gridLayoutColumn=0;
			gridLayoutRow++;
		}
		m_ToolsQActionVec.push_back(action);
		connect(action, SIGNAL(triggered(bool)), this,
				SLOT(updateToolSelected()));
	}
	toolsGroupBox->setLayout(toolsGroupBoxLayout);

	// Shape Group Box
	QGroupBox *shapeGroupBox = new QGroupBox(tr("Shape"),this);
	QGridLayout *shapeGroupBoxLayout = new QGridLayout();
	//shapeGroupBoxLayout->setContentsMargins(5,5,5,5);
	shapeGroupBoxLayout->setSpacing(0);
	gridLayoutRow = 1;
	gridLayoutColumn = 0;

	for (int i=12; i< 19; i++)
	{
		QAction *action = new QAction(QIcon(ToolsQActionParamsVec[i].iconPath), ToolsQActionParamsVec[i].text, this);
		action->setShortcut(ToolsQActionParamsVec[i].shortcut);
		action->setCheckable(true);

		QToolButton *button = new QToolButton(this);
		button->setDefaultAction(action);
		button->setFixedSize(20,20);
		button->setIconSize(QSize(20, 20));
		shapeGroupBoxLayout->addWidget(button,gridLayoutRow,gridLayoutColumn);
		gridLayoutColumn++;
		if (gridLayoutColumn>1)
		{
			gridLayoutColumn=0;
			gridLayoutRow++;
		}
		m_ToolsQActionVec.push_back(action);
		connect(action, SIGNAL(triggered(bool)), this,
				SLOT(updateToolSelected()));
	}
	shapeGroupBox->setLayout(shapeGroupBoxLayout);

	// Curve Group Box
	QGroupBox *curveGroupBox = new QGroupBox(tr("Curve"),this);
	QGridLayout *curveGroupBoxLayout = new QGridLayout();
	//curveGroupBoxLayout->setContentsMargins(5,5,5,5);
	curveGroupBoxLayout->setSpacing(0);
	gridLayoutRow = 1;
	gridLayoutColumn = 0;

	for (int i=19; i< 23; i++)
	{
		QAction *action = new QAction(QIcon(ToolsQActionParamsVec[i].iconPath), ToolsQActionParamsVec[i].text, this);
		action->setShortcut(ToolsQActionParamsVec[i].shortcut);
		action->setCheckable(true);

		QToolButton *button = new QToolButton(this);
		button->setDefaultAction(action);
		curveGroupBoxLayout->addWidget(button,gridLayoutRow,gridLayoutColumn);
		gridLayoutColumn++;
		if (gridLayoutColumn>3)
		{
			gridLayoutColumn=0;
			gridLayoutRow++;
		}
		m_ToolsQActionVec.push_back(action);
		connect(action, SIGNAL(triggered(bool)), this,
				SLOT(updateToolSelected()));
	}
	QVBoxLayout *curveBlocLayout = new QVBoxLayout();
	curveBlocLayout->addStretch(0);
	//curveBlocLayout->setContentsMargins(5,5,5,5);
	curveBlocLayout->setSpacing(0);
	curveBlocLayout->addLayout(curveGroupBoxLayout);
	QLabel *smoothLabel = new QLabel ("Smooth", curveGroupBox);
	m_SmoothSlider = new QSlider(Qt::Horizontal,curveGroupBox);
	m_SmoothSlider->setMinimum(0);
	m_SmoothSlider->setMaximum(10);
	m_SmoothSlider->setTickInterval(1);
	m_SmoothSlider->setSingleStep(1);
	m_SmoothSlider->setValue(0);
	m_SmoothSlider->setStyle(new SliderStyle(m_SmoothSlider->style()));
	connect (m_SmoothSlider, &QSlider::valueChanged, this, &GraphicToolsWidget::updateCurveSmoothValue);

	curveBlocLayout->addWidget(smoothLabel);
	curveBlocLayout->addWidget(m_SmoothSlider);
	curveGroupBox->setLayout(curveBlocLayout);

	// TEXT
	QGroupBox *textGroupBox = new QGroupBox(tr("Text"),this);
	QHBoxLayout *textGroupBoxLayout = new QHBoxLayout();
	textGroupBoxLayout->addStretch(0);
	//textGroupBoxLayout->setContentsMargins(5,5,5,5);
	textGroupBoxLayout->setSpacing(0);
	{
		QAction *action = new QAction(QIcon(":/slicer/icons/graphic_tools/text.png"), "Insert a text", this);
		action->setCheckable(true);

		QToolButton *button = new QToolButton(this);
		button->setDefaultAction(action);

		textGroupBoxLayout->addWidget(button);
		m_ToolsQActionVec.push_back(action);
		connect(action, SIGNAL(triggered(bool)), this,
				SLOT(updateToolSelected()));

	}

	m_FontSelection = new QToolButton(this);
	m_FontSelection->setCheckable(true);
	//m_FontSelection->setChecked(true);
	m_FontSelection->setIcon(QIcon(":/slicer/icons/graphic_tools/font.png"));
	m_FontSelection->setFixedSize(50,50);
	m_FontSelection->setIconSize(QSize(50, 50));
	//m_FontSelection->setText("Font");
	//QFont textlabelfont ("Tahoma", 8);
	//textlabelfont.setBold(true);
	//m_FontSelection->setFont(textlabelfont);
	//m_FontSelection->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	textGroupBoxLayout->addWidget(m_FontSelection);



	connect(m_FontSelection, SIGNAL(toggled(bool)), this,
			SLOT(selectFont()));

	textGroupBox->setLayout(textGroupBoxLayout);


	// COLORS
	QGroupBox *colorsGroupBox = new QGroupBox(tr("Colors"),this);
	QGridLayout *colorsGroupBoxLayout = new QGridLayout();
	colorsGroupBoxLayout->setContentsMargins(5,5,5,5);
	colorsGroupBoxLayout->setSpacing(0);
	gridLayoutRow = 1;
	gridLayoutColumn = 0;

	for (int i=0; i< ColorsQActionParamsVec.size(); i++)
	{
		QAction *action = new QAction(QIcon(ColorsQActionParamsVec[i].iconPath), ColorsQActionParamsVec[i].text, this);
		action->setShortcut(ColorsQActionParamsVec[i].shortcut);
		action->setCheckable(true);

		QToolButton *button = new QToolButton(this);
		button->setDefaultAction(action);
		colorsGroupBoxLayout->addWidget(button,gridLayoutRow,gridLayoutColumn);
		gridLayoutColumn++;
		if (gridLayoutColumn>2)
		{
			gridLayoutColumn=0;
			gridLayoutRow++;
		}
		m_ColorQActionVec.push_back(action);
		connect(action, SIGNAL(triggered(bool)), this,
				SLOT(updateColorSelected()));
	}

	QColor initColor = Qt::white;
	//m_SelectedColor_Label->setStyleSheet("margin-right: 5px; border: 1px solid white; border-radius: 2px;background-color: #FFFFFF");

	m_PencolorSelection = new QToolButton(this);
	m_PencolorSelection->setCheckable(true);
	m_PencolorSelection->setChecked(true);
	QPixmap px(40, 35);
	px.fill(initColor);
	QFont font ("Tahoma", 8);
	//font.setBold(true);
	m_PencolorSelection->setIcon(px);
	m_PencolorSelection->setFixedSize(30,50);
	m_PencolorSelection->setIconSize(QSize(25, 25));
	m_PencolorSelection->setText("Pen");

	m_PencolorSelection->setFont(font);
	m_PencolorSelection->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	connect(m_PencolorSelection, SIGNAL(toggled(bool)), this,
			SLOT(selectColorForPen(bool)));

	m_BrushcolorSelection = new QToolButton(this);
	m_BrushcolorSelection->setCheckable(true);
	m_BrushcolorSelection->setChecked(false);

	initColor = Qt::gray;
	px.fill(initColor);
	m_BrushcolorSelection->setIcon(px);
	m_BrushcolorSelection->setFixedSize(30,50);
	m_BrushcolorSelection->setIconSize(QSize(25, 25));
	m_BrushcolorSelection->setText("Fill");
	m_BrushcolorSelection->setFont(font);
	m_BrushcolorSelection->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	connect(m_BrushcolorSelection, SIGNAL(toggled(bool)), this,
			SLOT(selectColorForBrush(bool)));

	m_TextcolorSelection = new QToolButton(this);
	m_TextcolorSelection->setCheckable(true);
	m_TextcolorSelection->setChecked(false);

	initColor = Qt::green;
	px.fill(initColor);
	m_TextcolorSelection->setIcon(px);
	m_TextcolorSelection->setFixedSize(30,50);
	m_TextcolorSelection->setIconSize(QSize(25, 25));
	m_TextcolorSelection->setText("Text");
	m_TextcolorSelection->setFont(font);
	m_TextcolorSelection->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	connect(m_TextcolorSelection, SIGNAL(toggled(bool)), this,
			SLOT(selectColorForText(bool)));

	QHBoxLayout *HLayout_1 = new QHBoxLayout();
	HLayout_1->addWidget(m_PencolorSelection);
	HLayout_1->addWidget(m_BrushcolorSelection);
	HLayout_1->addWidget(m_TextcolorSelection);

	HLayout_1->addSpacing(0);
	HLayout_1->addStretch(0);
	HLayout_1->addLayout(colorsGroupBoxLayout);

	colorsGroupBox->setLayout(HLayout_1);

	QGroupBox *PenOptionsGroupBox = new QGroupBox(tr("Pen/Brush Settings"),this);

	// PEN WIDTH
	QToolButton *penWidth=new QToolButton(this);
	penWidth->setFixedSize(50,60);
	penWidth->setIconSize(QSize(45, 35));
	penWidth->setText("Width");
	penWidth->setIcon(QIcon(":/slicer/icons/graphic_tools/pen_width.png"));
	penWidth->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	penWidth->setPopupMode(QToolButton::InstantPopup);
	penWidthMenu=new QMenu(penWidth);
	for (int i=0; i<Penwidth_Pixmap_IconList.count();i++)
	{
		QToolButton *button=new QToolButton(penWidthMenu);
		button->setFixedSize(100,30);
		button->setIconSize(QSize(100, 30));
		button->setIcon(QIcon(Penwidth_Pixmap_IconList[i]));
		button->setCheckable(true);
		m_PenWidthToolButtonVec.push_back(button);
		connect(button, SIGNAL(toggled(bool)), this,
				SLOT(updateWidthSelected()));
		QWidgetAction *action=new QWidgetAction(penWidthMenu);
		action->setDefaultWidget(button);
		penWidthMenu->addAction(action);
		penWidthMenu->addSeparator();
	}
	penWidth->setMenu(penWidthMenu);

	// PEN STYLE
	QToolButton *penStyle=new QToolButton(this);
	penStyle->setFixedSize(50,60);
	penStyle->setIconSize(QSize(45, 35));
	penStyle->setText("Style");
	penStyle->setIcon(QIcon(":/slicer/icons/graphic_tools/pen_style.png"));
	penStyle->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	penStyle->setPopupMode(QToolButton::InstantPopup);
	penStyleMenu=new QMenu(penStyle);
	for (int i=0; i<PenStyle_Pixmap_IconList.count();i++)
	{
		QToolButton *button=new QToolButton(penStyleMenu);
		button->setFixedSize(100,30);
		button->setIconSize(QSize(100, 30));
		button->setIcon(QIcon(PenStyle_Pixmap_IconList[i]));
		button->setCheckable(true);
		m_PenStyleToolButtonVec.push_back(button);
		connect(button, SIGNAL(toggled(bool)), this,
				SLOT(updateStyleSelected()));

		QWidgetAction *action=new QWidgetAction(penStyleMenu);
		action->setDefaultWidget(button);
		penStyleMenu->addAction(action);
		penStyleMenu->addSeparator();
	}
	penStyle->setMenu(penStyleMenu);

	//	// PEN CAP
	//	QToolButton *penCap=new QToolButton(this);
	//	penCap->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
	//	penCap->setFixedSize(50,25);
	//	penCap->setText("Cap");
	//	//penCap->setIcon(QIcon(":/slicer/icons/graphic_tools/pen_cap.png"));
	//	//penCap->setIconSize(QSize(15, 15));
	//	penCap->setPopupMode(QToolButton::InstantPopup);
	//	QMenu *penCapMenu=new QMenu(penCap);
	//	penCapMenu->setMaximumWidth(90);
	//	QStringList penCapOptions = {"Flat", "Square", "Round"};
	//	QActionGroup* actiongroup = new QActionGroup(penCapMenu);
	//	connect(actiongroup, SIGNAL(triggered(QAction*)), SLOT(PenCapChanged(QAction*)));
	//	actiongroup->setExclusive(true);
	//	for (int i=0; i<penCapOptions.count();i++)
	//	{
	//		QAction *act  = new QAction(penCapOptions[i],penCapMenu);
	//		act->setCheckable(true);
	//		if (i==0)
	//		{
	//			act->setChecked(true);
	//		}
	//		actiongroup->addAction(act);
	//		penCapMenu->addAction(act);
	//	}
	//	penCap->setMenu(penCapMenu);

	//	// PEN JOIN
	//	QToolButton *penJoin=new QToolButton(this);
	//	penJoin->setText("Join");
	//	penJoin->setFixedSize(50,25);
	//	penJoin->setToolButtonStyle(Qt::ToolButtonTextOnly);
	//	penJoin->setPopupMode(QToolButton::InstantPopup);
	//	QMenu *penJoinMenu=new QMenu(penJoin);
	//	penJoinMenu->setMaximumWidth(90);
	//	QStringList penJoinOptions = {"Miter", "Bevel", "Round"};
	//	QActionGroup* penJoinActionGroup = new QActionGroup(penJoinMenu);
	//	connect(penJoinActionGroup, SIGNAL(triggered(QAction*)), SLOT(PenJoinChanged(QAction*)));
	//	penJoinActionGroup->setExclusive(true);
	//	for (int i=0; i<penJoinOptions.count();i++)
	//	{
	//		QAction *act  = new QAction(penJoinOptions[i],penJoinMenu);
	//		act->setCheckable(true);
	//		if (i==0)
	//		{
	//			act->setChecked(true);
	//		}
	//		penJoinActionGroup->addAction(act);
	//		penJoinMenu->addAction(act);
	//	}
	//	penJoin->setMenu(penJoinMenu);

	QHBoxLayout *HLayout_2= new QHBoxLayout();
	HLayout_2->addWidget(penWidth);
	HLayout_2->addWidget(penStyle);
	HLayout_2->addStretch(0);
	HLayout_2->setSpacing(0);


	// BRUSH
	QGroupBox *brushGroupBox = new QGroupBox("",this);
	QGridLayout *brushGroupBoxLayout = new QGridLayout();
	gridLayoutRow = 1;
	gridLayoutColumn = 0;

	for (int i=0; i< BrushQActionParamsVec.size(); i++)
	{
		QAction *action = new QAction(QIcon(BrushQActionParamsVec[i].iconPath), BrushQActionParamsVec[i].text, this);
		action->setShortcut(BrushQActionParamsVec[i].shortcut);
		action->setCheckable(true);

		QToolButton *button = new QToolButton(this);
		button->setDefaultAction(action);
		button->setFixedSize(30,30);
		button->setIconSize(QSize(30, 30));

		brushGroupBoxLayout->addWidget(button,gridLayoutRow,gridLayoutColumn);
		gridLayoutColumn++;
		if (gridLayoutColumn>3)
		{
			gridLayoutColumn=0;
			gridLayoutRow++;
		}
		m_BrushQActionVec.push_back(action);
		connect(action, SIGNAL(triggered(bool)), this,
				SLOT(updateBrushSelected()));
	}
	brushGroupBox->setLayout(brushGroupBoxLayout);

	QToolButton *brushButton=new QToolButton(this);
	brushButton->setFixedSize(50,60);
	brushButton->setIconSize(QSize(45, 35));
	brushButton->setText("Brush");
	brushButton->setIcon(QIcon(":/slicer/icons/graphic_tools/brush.png"));
	brushButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	brushButton->setPopupMode(QToolButton::InstantPopup);
	brushMenu=new QMenu(brushButton);
	QWidgetAction *brushWidgetAction = new QWidgetAction(brushMenu);
	brushWidgetAction->setDefaultWidget(brushGroupBox);
	brushMenu->addAction(brushWidgetAction);
	brushButton->setMenu(brushMenu);

	QHBoxLayout *VLayout_2 = new QHBoxLayout();
	VLayout_2->addLayout(HLayout_2);
	VLayout_2->addWidget(brushButton);
	VLayout_2->addStretch(0);
	//VLayout_2->setContentsMargins(5,5,5,5);
	VLayout_2->setSpacing(0);
	PenOptionsGroupBox->setLayout(VLayout_2);

	//	m_ViewCombobox = new QComboBox(this);
	//	connect(m_ViewCombobox, SIGNAL(currentIndexChanged(int)), this, SLOT(updateSelectedView()));
	//
	std::vector<AbstractInnerView*> InnerViweList = GeotimeGraphicsView::getInnerViewsList();
	//	for (int i=0; i< InnerViweList.size(); i++)
	//	{
	//		m_ViewCombobox->addItem(InnerViweList[i]->title());
	//	}

	//	m_AntialiasingCheckBox = new QCheckBox(tr("&Anti\naliasing"));
	//	connect(m_AntialiasingCheckBox, SIGNAL(	stateChanged(int)), this,
	//			SLOT(setAntialiasingOption()));

	QVBoxLayout *mainLayout = new QVBoxLayout();
	QHBoxLayout *HLayout_3 = new QHBoxLayout();
	HLayout_3->addWidget(toolsGroupBox);
	HLayout_3->addWidget(shapeGroupBox);

	mainLayout->addLayout(HLayout_3);

	QHBoxLayout *HLayout_10 = new QHBoxLayout();

	HLayout_10->addWidget(curveGroupBox);
	HLayout_10->addWidget(textGroupBox);

	mainLayout->addLayout(HLayout_10);

	mainLayout->addWidget(PenOptionsGroupBox);
	mainLayout->addWidget(colorsGroupBox);

	setLayout(mainLayout);

	setDefaultSettings();

	m_GraphicToolsSettings.enabled = false;

	this->setWindowFlags(Qt::WindowStaysOnTopHint);
	this->adjustSize();

	//std::vector<AbstractInnerView*> InnerViweList = GeotimeGraphicsView::getInnerViewsList();
	//	for (int i=0; i< InnerViweList.size(); i++)
	//	{
	//		Abstract2DInnerView *view = dynamic_cast<Abstract2DInnerView*> (InnerViweList[i]);
	//		QGraphicsScene *scene_1 = view->scene();
	//		GraphicSceneEditor *scene = dynamic_cast<GraphicSceneEditor*> (scene_1);
	//		connect(this, &GraphicToolsWidget::graphicOptionsChanged,
	//				scene, &GraphicSceneEditor::GraphicToolNewAction );
	//		connect(this, &GraphicToolsWidget::penSettingsChanged,
	//				scene, &GraphicSceneEditor::updateSelectedItemsPen );
	//		connect(this, &GraphicToolsWidget::brushSettingsChanged,
	//				scene, &GraphicSceneEditor::updateSelectedItemsBrush );
	//		connect(this, &GraphicToolsWidget::textColorChanged,
	//				scene, &GraphicSceneEditor::updateSelectedTextColor );
	//		connect(this, &GraphicToolsWidget::textFontChanged,
	//				scene, &GraphicSceneEditor::updateSelectedTextFont );
	//		connect(m_SmoothSlider, &QSlider::valueChanged,
	//				scene, &GraphicSceneEditor::updateSelectedCurveSmooth );
	//	}
}

void GraphicToolsWidget::selectFont()
{
	//	bool ok;
	//	const QFontDialog::FontDialogOptions options = QFlag(QFontDialog::DontUseNativeDialog
	//			|QFontDialog::ScalableFonts|QFontDialog::NonScalableFonts|QFontDialog::MonospacedFonts
	//			|QFontDialog::ProportionalFonts);
	//	QFont font = QFontDialog::getFont(
	//			&ok, m_GraphicToolsSettings.font , this, "Select Font", options);
	//	if (ok)
	//	{
	//		m_GraphicToolsSettings.font = font;
	//	}
	//	else
	//	{
	//		// the user canceled the dialog; font is set to the initial
	//		// value, in this case Helvetica [Cronyx], 10
	//	}
	QFontDialog *diag = new QFontDialog(this);
	connect(diag,&QFontDialog::currentFontChanged,
			this, &GraphicToolsWidget::textFontChanged);
	connect(diag,&QFontDialog::fontSelected,
			this, &GraphicToolsWidget::textFontChanged);
	diag->show();

}

void GraphicToolsWidget::updateSelectedView()
{
	//m_GraphicToolsSettings.viewName = m_ViewCombobox->currentText();
}

void GraphicToolsWidget::updateCurveSmoothValue(int newValue)
{
	m_GraphicToolsSettings.smooth = newValue;
}

void GraphicToolsWidget::selectColorForPen(bool checked)
{
	if (checked)
	{
		m_PencolorSelection->setChecked(true);
		m_BrushcolorSelection->setChecked(false);
		m_TextcolorSelection->setChecked(false);
	}
	else
	{
		if ( (!m_BrushcolorSelection->isChecked()) && (!m_TextcolorSelection->isChecked()) )
		{
			m_PencolorSelection->setChecked(true);
		}
	}
	emit penSettingsChanged(m_GraphicToolsSettings.pen, e_PenColor);
}

void GraphicToolsWidget::selectColorForBrush(bool checked)
{
	if (checked)
	{
		m_BrushcolorSelection->setChecked(true);
		m_PencolorSelection->setChecked(false);
		m_TextcolorSelection->setChecked(false);
	}
	else
	{
		if ( (!m_PencolorSelection->isChecked()) && (!m_TextcolorSelection->isChecked()) )
		{
			m_BrushcolorSelection->setChecked(true);
		}
	}
	emit brushSettingsChanged(m_GraphicToolsSettings.brush, e_BrushColor);
}

void GraphicToolsWidget::selectColorForText(bool checked)
{
	if (checked)
	{
		m_TextcolorSelection->setChecked(true);
		m_BrushcolorSelection->setChecked(false);
		m_PencolorSelection->setChecked(false);
	}
	else
	{
		if ( (!m_PencolorSelection->isChecked()) && (!m_BrushcolorSelection->isChecked()) )
		{
			m_TextcolorSelection->setChecked(true);
		}
	}
	emit textColorChanged(m_GraphicToolsSettings.textColor);
}

void GraphicToolsWidget::setAntialiasingOption()
{
	//	if (m_AntialiasingCheckBox->isChecked())
	//	{
	//		m_GraphicToolsSettings.antialiasingEnabled=true;
	//	}
	//	else
	//	{
	//		m_GraphicToolsSettings.antialiasingEnabled=false;
	//	}
}

void GraphicToolsWidget::PenCapChanged(QAction* action)
{
	Qt::PenCapStyle capStyle;
	if (action->text() == "Flat")
	{
		capStyle = Qt::FlatCap;
	}
	else if (action->text() == "Square")
	{
		capStyle = Qt::SquareCap;
	}
	else if (action->text() == "Round")
	{
		capStyle = Qt::RoundCap;
	}
	m_GraphicToolsSettings.pen.setCapStyle(capStyle);
	emit penSettingsChanged(m_GraphicToolsSettings.pen, e_PenCap);
}

void GraphicToolsWidget::PenJoinChanged(QAction* action)
{
	Qt::PenJoinStyle joinStyle;
	if (action->text() == "Miter")
	{
		joinStyle = Qt::MiterJoin;
	}
	else if (action->text() == "Bevel")
	{
		joinStyle = Qt::BevelJoin;
	}
	else if (action->text() == "Round")
	{
		joinStyle = Qt::RoundJoin;
	}
	m_GraphicToolsSettings.pen.setJoinStyle(joinStyle);
	emit penSettingsChanged(m_GraphicToolsSettings.pen, e_PenJoinStyle);
}


template <typename T>
void updateSelectedItem(QVector<T> in_ActionVec, int &last_item_index)
{
	for (int i=0; i< in_ActionVec.size(); i++)
	{
		if (in_ActionVec[i]->isChecked())
		{
			if (i==last_item_index)
			{
				continue;
			}
			else
			{
				in_ActionVec[last_item_index]->setChecked(false);
				last_item_index=i;
			}
		}
	}
	in_ActionVec[last_item_index]->setChecked(true);
}

void GraphicToolsWidget::setDefaultSettings()
{
	m_ToolsQActionVec[0]->setChecked(false);
	m_GraphicToolsSettings.enabled=false;

	m_ColorQActionVec[0]->setChecked(true);
	m_GraphicToolsSettings.pen.setCosmetic(true);

	m_GraphicToolsSettings.pen.setColor(QColor(Qt::white));

	m_PenWidthToolButtonVec[1]->setChecked(true);
	m_GraphicToolsSettings.pen.setWidth(3);

	m_PenStyleToolButtonVec[0]->setChecked(true);
	m_GraphicToolsSettings.pen.setStyle(Qt::SolidLine);

	m_BrushQActionVec[1]->setChecked(true);
	m_GraphicToolsSettings.brush.setStyle(Qt::SolidPattern);
	QColor clr(Qt::gray);
	clr.setAlpha(150);
	m_GraphicToolsSettings.brush.setColor(clr);

	m_GraphicToolsSettings.textColor = QColor(Qt::green);
	m_GraphicToolsSettings.font = QFont("Helvetica [Cronyx]", 10);
	m_GraphicToolsSettings.smooth = 0;
}

void updateSelectedTool(QVector<QAction *> in_ActionVec, int &last_item_index)
{
	bool isSelectionToolChecked = in_ActionVec[0]->isChecked();
	for (int i=0; i< in_ActionVec.size(); i++)
	{
		if (in_ActionVec[i]->isChecked())
		{
			if (i==last_item_index)
			{
				continue;
			}
			else
			{
				in_ActionVec[last_item_index]->setChecked(false);
				last_item_index=i;
			}
		}
	}
	if (last_item_index != 0)
	{
		in_ActionVec[last_item_index]->setChecked(true);
	}
}

void GraphicToolsWidget::deselectTools()
{
	for (int i=0; i<m_ToolsQActionVec.size(); i++)
	{
		m_ToolsQActionVec[i]->setChecked(false);
	}
}

void GraphicToolsWidget::updateToolSelected()
{
	static int last_QActionChecked =0;
	updateSelectedTool(m_ToolsQActionVec,last_QActionChecked);

	if (last_QActionChecked == (eGraphicAction)eGraphicAction_Text)
	{
		m_GraphicToolsSettings.action = eGraphicAction_Text;
	}
	else if (last_QActionChecked> 11)
	{
		m_GraphicToolsSettings.action= eGraphicAction_Draw;
		m_GraphicToolsSettings.shape= (eShape) (last_QActionChecked-12);
	}
	else if (last_QActionChecked == (eGraphicAction)eGraphicAction_Draw)
	{
		m_GraphicToolsSettings.action= eGraphicAction_Draw;
		m_GraphicToolsSettings.shape= eShape_FreeDraw;
	}
	else
	{
		m_GraphicToolsSettings.action = (eGraphicAction)last_QActionChecked;
	}

	if (m_GraphicToolsSettings.action == eGraphicAction_Default)
	{
		if (m_ToolsQActionVec[0]->isChecked())
		{
			changeGraphicsViewsDragMode(QGraphicsView::RubberBandDrag);
		}
	}
	else if (m_GraphicToolsSettings.action == eGraphicAction_Draw)
	{
		changeGraphicsViewsDragMode(QGraphicsView::NoDrag);
	}
	else
	{
		changeGraphicsViewsDragMode(QGraphicsView::ScrollHandDrag);
	}
	emit graphicOptionsChanged(m_GraphicToolsSettings.action, m_GraphicToolsSettings);
}


//void GraphicToolsWidget::updateShapeSelected()
//{
//	static int last_QActionChecked =0;
//	updateSelectedItem(m_ShapeQActionVec,last_QActionChecked);
//	m_GraphicToolsSettings.shape= (eShape) last_QActionChecked;
//}

void GraphicToolsWidget::updateColorSelected()
{
	static int last_QActionChecked =0;
	updateSelectedItem(m_ColorQActionVec,last_QActionChecked);
	QColor clr;
	if (m_ColorQActionVec[last_QActionChecked]->text() == "more")
	{
		QColor clr;
		if (m_PencolorSelection->isChecked())
		{
			clr = m_GraphicToolsSettings.pen.color();
		}
		else if (m_BrushcolorSelection->isChecked())
		{
			clr = m_GraphicToolsSettings.brush.color();
			if (clr.alpha() == 255)
			{
				clr.setAlpha(150);
			}
		}
		else
		{
			clr = m_GraphicToolsSettings.textColor;
		}
		const QColorDialog::ColorDialogOptions options = QFlag(QColorDialog::ShowAlphaChannel);
		//clr = QColorDialog::getColor(Qt::white, this, "Select Color", options);

		QColorDialog *dialog = new QColorDialog(clr,this);
		dialog->setOptions(options);
		connect(dialog,&QColorDialog::currentColorChanged,
				this, &GraphicToolsWidget::changeColor);
		connect(dialog,&QColorDialog::colorSelected,
				this, &GraphicToolsWidget::changeColor);
		dialog->show();


	}
	else
	{
		clr = QColor(m_ColorQActionVec[last_QActionChecked]->text());
		if (m_BrushcolorSelection->isChecked())
		{
			clr.setAlpha(150);
		}
		changeColor(clr);
	}


}

void GraphicToolsWidget::changeColor(QColor clr)
{
	QPixmap px(40, 35);
	px.fill(clr);
	if (m_PencolorSelection->isChecked())
	{
		m_PencolorSelection->setIcon(px);
		m_GraphicToolsSettings.pen.setColor(clr);
		emit penSettingsChanged(m_GraphicToolsSettings.pen, e_PenColor);
	}
	else if (m_BrushcolorSelection->isChecked())
	{
		//clr.setAlpha(150);
		px.fill(clr);
		m_BrushcolorSelection->setIcon(px);
		m_GraphicToolsSettings.brush.setColor(clr);
		emit brushSettingsChanged(m_GraphicToolsSettings.brush, e_BrushColor);
	}
	else // TEXT
	{
		m_TextcolorSelection->setIcon(px);
		m_GraphicToolsSettings.textColor = clr;
		emit textColorChanged(m_GraphicToolsSettings.textColor);
	}
}
void GraphicToolsWidget::updateWidthSelected()
{
	static int last_QActionChecked =0;
	updateSelectedItem(m_PenWidthToolButtonVec,last_QActionChecked);
	penWidthMenu->close();
	m_GraphicToolsSettings.pen.setWidth((last_QActionChecked*2)+1);
	emit penSettingsChanged(m_GraphicToolsSettings.pen, e_PenWidth);
}

void GraphicToolsWidget::updateStyleSelected()
{
	static int last_QActionChecked =0;
	updateSelectedItem(m_PenStyleToolButtonVec,last_QActionChecked);
	penStyleMenu->close();
	switch (last_QActionChecked)
	{
	case 0 : m_GraphicToolsSettings.pen.setStyle(Qt::SolidLine); break;
	case 1 : m_GraphicToolsSettings.pen.setStyle(Qt::DashLine); break;
	case 2 : m_GraphicToolsSettings.pen.setStyle(Qt::DotLine); break;
	case 3 : m_GraphicToolsSettings.pen.setStyle(Qt::DashDotLine); break;
	case 4 : m_GraphicToolsSettings.pen.setStyle(Qt::DashDotDotLine); break;
	default : m_GraphicToolsSettings.pen.setStyle(Qt::SolidLine); break;
	}
	emit penSettingsChanged(m_GraphicToolsSettings.pen, e_PenStyle);
}

void GraphicToolsWidget::updateBrushSelected()
{
	static int last_QActionChecked =0;
	updateSelectedItem(m_BrushQActionVec,last_QActionChecked);
	brushMenu->close();
	//	Qt::NoBrush	0	No brush pattern.
	//	Qt::SolidPattern	1	Uniform color.
	//	Qt::Dense1Pattern	2	Extremely dense brush pattern.
	//	Qt::Dense2Pattern	3	Very dense brush pattern.
	//	Qt::Dense3Pattern	4	Somewhat dense brush pattern.
	//	Qt::Dense4Pattern	5	Half dense brush pattern.
	//	Qt::Dense5Pattern	6	Somewhat sparse brush pattern.
	//	Qt::Dense6Pattern	7	Very sparse brush pattern.
	//	Qt::Dense7Pattern	8	Extremely sparse brush pattern.
	//	Qt::HorPattern	9	Horizontal lines.
	//	Qt::VerPattern	10	Vertical lines.
	//	Qt::CrossPattern	11	Crossing horizontal and vertical lines.
	//	Qt::BDiagPattern	12	Backward diagonal lines.
	//	Qt::FDiagPattern	13	Forward diagonal lines.
	//	Qt::DiagCrossPattern	14	Crossing diagonal lines.
	//	Qt::LinearGradientPattern	15	Linear gradient (set using a dedicated QBrush constructor).
	//	Qt::ConicalGradientPattern	17	Conical gradient (set using a dedicated QBrush constructor).
	//	Qt::RadialGradientPattern	16	Radial gradient (set using a dedicated QBrush constructor).

	switch (last_QActionChecked)
	{
	case 0 : m_GraphicToolsSettings.brush.setStyle(Qt::NoBrush); break;
	case 1 : m_GraphicToolsSettings.brush.setStyle(Qt::SolidPattern); break;
	case 2 : m_GraphicToolsSettings.brush.setStyle(Qt::HorPattern); break;
	case 3 : m_GraphicToolsSettings.brush.setStyle(Qt::VerPattern); break;
	case 4 : m_GraphicToolsSettings.brush.setStyle(Qt::CrossPattern); break;
	case 5 : m_GraphicToolsSettings.brush.setStyle(Qt::BDiagPattern); break;
	case 6 : m_GraphicToolsSettings.brush.setStyle(Qt::FDiagPattern); break;
	case 7 : m_GraphicToolsSettings.brush.setStyle(Qt::DiagCrossPattern); break;
	//case 8 : m_GraphicToolsSettings.brush.setStyle(Qt::LinearGradientPattern); break;
	//case 9 : m_GraphicToolsSettings.brush.setStyle(Qt::RadialGradientPattern); break;
	//case 10 : m_GraphicToolsSettings.brush.setStyle(Qt::ConicalGradientPattern); break;

	default : m_GraphicToolsSettings.brush.setStyle(Qt::NoBrush); break;
	}
	emit brushSettingsChanged(m_GraphicToolsSettings.brush, e_BrushStyle);
}

void GraphicToolsWidget::closeEvent(QCloseEvent *event)
{
	changeGraphicsViewsDragMode(QGraphicsView::ScrollHandDrag);
	//m_GraphicToolsSettings.enabled=false;
}

GraphicToolsWidget::~GraphicToolsWidget()
{
	while(m_InnverViewsScene.size()>0)
	{
		this->removeInnerView2(m_InnverViewsScene[0]->innerView());

	}

	if(this== m_pInstance) m_pInstance = nullptr;

}


