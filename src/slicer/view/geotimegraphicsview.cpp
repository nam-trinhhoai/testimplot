#include <typeinfo>
#include <iostream>

#include <QFont>
#include <QBrush>
#include "geotimegraphicsview.h"
#include "abstractinnerview.h"
#include "qinnerviewtreewidgetitem.h"
#include "qgraphicsreptreewidgetitem.h"
#include "slicerep.h"
#include "randomrep.h"
#include "cudaimagepaletteholder.h"
#include "graphicsutil.h"
#include "basemapview.h"
#include "stackbasemapview.h"
#include "abstractsectionview.h"
#include "randomlineview.h"
#include "multiseedslicerep.h"
#include "multiseedrandomrep.h"
#include "seismic3ddataset.h"
#include "LayerSpectrumDialog.h"
#include "RgtVolumicDialog.h"
#include "marker.h"
#include "markertreewidgetitem.h"
#include "wellhead.h"
#include "wellbore.h"
#include "wellheadtreewidgetitem.h"
#include "workingsetmanager.h"
#include "folderdata.h"
#include "horizonfolderdata.h"
// #include "folder.h"
#include "folderdatatreewidgetitem.h"
#include "seismicsurvey.h"
//#include "fixedrgblayersfromdataset.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "fixedrgblayersfromdatasetandcubeimplmulti.h"
#include "fixedlayersfromdatasetandcube.h"
//#include "videolayer.h"
#include "stringselectordialog.h"
//#include "KohonenProcess.h"
#include "ijkhorizon.h"
#include "rgbcomputationondataset.h"
#include "scalarComputationOnDataSet.h"
#include "rgbdataset.h"
#include "DataSelectorDialog.h"
#include "smdataset3D.h"
#include "GraphicToolsWidget.h"
#include "viewqt3d.h"
#include "stacksynchronizer.h"
#include "mtlengthunit.h"
#include "nurbswidget.h"
#include "processrelay.h"
#include "ivolumecomputationoperator.h"
#include "computationoperatordataset.h"
#include "computationdatasetrelayconnectioncloser.h"
#include "computereflectivitywidget.h"
#include <horizonAttributComputeDialog.h>
#include <fixedattributimplfromdirectories.h>
#include <fixedattributimplfreehorizonfromdirectories.h>

#include <freeHorizonQManager.h>
#include <freehorizon.h>
#include <isohorizon.h>
#include <fileSelectorDialog.h>
#include <rgtSpectrumHeader.h>

#include "importsismagehorizondialog.h"
#include "managerwidget.h"
#include "horizonanimaggregator.h"
#include "isohorizoninformationaggregator.h"
#include "nextvisionhorizoninformationaggregator.h"
#include "nurbinformationaggregator.h"
#include "pickinformationaggregator.h"
#include "seismicinformationaggregator.h"
#include "wellinformationaggregator.h"

#include <QStack>
#include <QPushButton>
#include <QToolBar>
#include <QTreeWidget>
#include <QMessageBox>
#include <QFileDialog>
#include <QComboBox>
#include <QLabel>
#include <QInputDialog>
#include <QHeaderView>
#include <QComboBox>
#include <iostream>

#include "globalconfig.h"
#include <globalUtil.h>
#include <memory>
#include <algorithm>

std::vector<AbstractInnerView*> GeotimeGraphicsView::m_InnerViewVec;
WorkingSetManager* GeotimeGraphicsView::m_WorkingSetManager;

GeotimeGraphicsView::GeotimeGraphicsView(WorkingSetManager *factory, QString uniqueName, QWidget *parent) :
	MultiTypeGraphicsView(factory, uniqueName, parent) {
	m_depthLengthUnit = &MtLengthUnit::METRE;



	TemplateView _template;
	TemplateInnerView first;
	first.viewId = 1;
	first.viewType = InlineView;
	TemplateInnerView second;
	second.viewId = 2;
	second.targetId = 1;
	second.viewType = BasemapView;
	second.operation = SplitHorizontal;
	second.title = "Propagation";
	TemplateInnerView third;
	third.viewId = 3;
	third.targetId = 2;
	third.viewType = View3D;
	third.operation = SplitVertical;

	TemplateInnerView fourth;
	fourth.viewId = 4;
	fourth.targetId = 2;
	fourth.viewType = StackBasemapView;
	fourth.operation = AddAsTab;
	fourth.title = "Spectrum";

	TemplateInnerView fifth;
	fifth.viewId = 5;
	fifth.targetId = 2;
	fifth.viewType = StackBasemapView;
	fifth.operation = AddAsTab;
	fifth.title = "GCC";

	TemplateInnerView sixth;
	sixth.viewId = 6;
	sixth.targetId = 2;
	sixth.viewType = StackBasemapView;
	sixth.operation = AddAsTab;
	sixth.title = "TMAP";


	TemplateInnerView seventh;
	seventh.viewId = 7;
	seventh.targetId = 2;
	seventh.viewType = StackBasemapView;
	seventh.operation = AddAsTab;
	seventh.title = "Mean";

	TemplateInnerView eigth;
	eigth.viewId = 8;
	eigth.targetId = 2;
	eigth.viewType = BasemapView;
	eigth.operation = AddAsTab;
	eigth.title = "RGB";



	m_InnerViewVec.clear();

	m_WorkingSetManager = m_currentManager;




	_template.push_back(first);
	_template.push_back(second);
	_template.push_back(third);
	_template.push_back(fourth);
	_template.push_back(fifth);
	_template.push_back(sixth);
	_template.push_back(seventh);
	_template.push_back(eigth);


	setTemplate(_template);


	if(m_tools3D == nullptr)m_tools3D =new Tools3dWidget(this);
	//if(m_nurbs3D == nullptr)m_nurbs3D =new NurbsWidget(this);



	QPushButton *reset = GraphicsUtil::generateToobarButton(":/slicer/icons/Reset.svg", "Reset View", toolBar());
	reset->setIconSize(QSize(32, 32));
	reset->setFixedSize(32,32);
	reset->setDefault(false);
	reset->setAutoDefault(false);
	m_toolbar->addWidget(reset);
	connect(reset, SIGNAL(clicked()), this, SLOT(resetInnerViewsPositions()));



	// Deactivated while waiting for a cleanup for changing the project
//	QPushButton *loadManager = GraphicsUtil::generateToobarButton(
//			":/slicer/icons/earth.png", "Manager", toolBar());
//	loadManager->setDefault(false);
//	loadManager->setAutoDefault(false);
//	QList<QAction*> toolBarActions = m_toolbar->actions();
//	if (toolBarActions.size()>0) {
//		m_toolbar->insertWidget(toolBarActions[0], loadManager);
//	} else {
//		m_toolbar->addWidget(loadManager);
//	}

	QPushButton *saveSessionButton = GraphicsUtil::generateToobarButton(
			":/slicer/icons/earth.svg", "Save session", toolBar());
	saveSessionButton->setIconSize(QSize(32, 32));
	saveSessionButton->setFixedSize(32,32);
	saveSessionButton->setDefault(false);
	saveSessionButton->setAutoDefault(false);
	QList<QAction*> toolBarActions = m_toolbar->actions();
	if (toolBarActions.size()>0) {
		m_toolbar->insertWidget(toolBarActions[0], saveSessionButton);
	} else {
		m_toolbar->addWidget(saveSessionButton);
	}

	connect(saveSessionButton, &QPushButton::clicked, this, &GeotimeGraphicsView::saveSession);

	m_toolbar->addSeparator();

	QPushButton *openSeismicInfoAction = new QPushButton("Seismic");
	openSeismicInfoAction->setStyleSheet("QPushButton {background: #5061FB;}");
	openSeismicInfoAction->setDefault(false);
	openSeismicInfoAction->setAutoDefault(false);
	m_toolbar->addWidget(openSeismicInfoAction);
	connect(openSeismicInfoAction, SIGNAL(clicked()), this, SLOT(openSeismicInformation()));

	QPushButton *openHorizonInfoAction = new QPushButton("Horizon");
	openHorizonInfoAction->setStyleSheet("QPushButton {background: #5061FB;}");
	openHorizonInfoAction->setDefault(false);
	openHorizonInfoAction->setAutoDefault(false);
	m_toolbar->addWidget(openHorizonInfoAction);
	connect(openHorizonInfoAction, SIGNAL(clicked()), this, SLOT(openHorizonInformation()));

	QPushButton *openIsoHorizonInfoAction = new QPushButton("RGT Iso");
	openIsoHorizonInfoAction->setStyleSheet("QPushButton {background: #5061FB;}");
	openIsoHorizonInfoAction->setDefault(false);
	openIsoHorizonInfoAction->setAutoDefault(false);
	m_toolbar->addWidget(openIsoHorizonInfoAction);
	connect(openIsoHorizonInfoAction, SIGNAL(clicked()), this, SLOT(openIsoHorizonInformation()));

	QPushButton *openWellsInfoAction = new QPushButton("Wells");
	openWellsInfoAction->setStyleSheet("QPushButton {background: #5061FB;}");
	openWellsInfoAction->setDefault(false);
	openWellsInfoAction->setAutoDefault(false);
	m_toolbar->addWidget(openWellsInfoAction);
	connect(openWellsInfoAction, SIGNAL(clicked()), this, SLOT(openWellsInformation()));

	QPushButton *openPicksInfoAction = new QPushButton("Picks");
	openPicksInfoAction->setStyleSheet("QPushButton {background: #5061FB;}");
	openPicksInfoAction->setDefault(false);
	openPicksInfoAction->setAutoDefault(false);
	m_toolbar->addWidget(openPicksInfoAction);
	connect(openPicksInfoAction, SIGNAL(clicked()), this, SLOT(openPicksInformation()));

	QPushButton *importSismageAction = new QPushButton("Sismage");
	importSismageAction->setStyleSheet("QPushButton {background: #5061FB;}");
	importSismageAction->setDefault(false);
	importSismageAction->setAutoDefault(false);
	m_toolbar->addWidget(importSismageAction);
	connect(importSismageAction, SIGNAL(clicked()), this, SLOT(openImportSismage()));

	QPushButton *moreAction = new QPushButton("More");
	moreAction->setStyleSheet("QPushButton {background: #5061FB;}");
	moreAction->setDefault(false);
	moreAction->setAutoDefault(false);
	m_toolbar->addWidget(moreAction);

	QMenu *menu = new QMenu(moreAction);
	QAction* nurbsAction = menu->addAction(tr("Nurbs"));
	connect(nurbsAction, SIGNAL(triggered()), this, SLOT(openNurbsInformation()));
	QAction* animHorizonAction = menu->addAction(tr("H. Animation"));
	connect(animHorizonAction, SIGNAL(triggered()), this, SLOT(openHorizonAnimationInformation()));

	moreAction->setMenu(menu);

//	QPushButton *createData = new QPushButton("Create Data");
//	QMenu* createDataMenu = new QMenu("Create Data", this);
//	//createDataMenu->addAction("Animated Surfaces from images", this, &GeotimeGraphicsView::createAnimatedSurfacesImages);
//	// createDataMenu->addAction("I N I T", this, &GeotimeGraphicsView::createAnimatedSurfacesInit);
//	createDataMenu->addAction("Animated Surfaces from cube with spectrum", this, &GeotimeGraphicsView::createAnimatedSurfacesCube);
//	createDataMenu->addAction("Animated Surfaces from cube with gcc", this, &GeotimeGraphicsView::createAnimatedSurfacesCubeGcc);
//	createDataMenu->addAction("Animated Surfaces from cube with rgb1", this, &GeotimeGraphicsView::createAnimatedSurfacesCubeRgb1);
//	createDataMenu->addAction("Animated Surfaces from cube with Mean", this, &GeotimeGraphicsView::createAnimatedSurfacesCubeMean);
//	createDataMenu->addAction("RGB Volume from cube and xt", this, &GeotimeGraphicsView::createRGBVolumeFromCubeAndXt);
//	createDataMenu->addAction("Temp computation data", this, &GeotimeGraphicsView::addDataFromComputation);
//	//createDataMenu->addAction("Video layer", this, &GeotimeGraphicsView::createVideoLayer);
//	//createDataMenu->addAction("Tmap", this, &GeotimeGraphicsView::createTmapLayer);
//	createData->setMenu(createDataMenu);
//	//createData->setPopupMode(QToolButton::InstantPopup);
//	m_toolbar->addWidget(createData);


//	connect(loadManager, SIGNAL(clicked()), this, SLOT(openDataManager()));


	QWidget *spacerWidget = new QWidget(this);
	spacerWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	spacerWidget->setVisible(true);
	m_toolbar->addWidget(spacerWidget);

	QPushButton *openGraphicToolsDialog = new QPushButton("Graphic Tools", toolBar());
	openGraphicToolsDialog->setIcon(QIcon(":/slicer/icons/graphic_tools/palette.png"));
	openGraphicToolsDialog->setToolTip("Open Graphic Tools Widget");
	openGraphicToolsDialog->setDefault(false);
	openGraphicToolsDialog->setAutoDefault(false);
	m_toolbar->addWidget(openGraphicToolsDialog);

	connect(openGraphicToolsDialog, SIGNAL(clicked()), this, SLOT(openGraphicToolsDialog()));

	QComboBox *synchroTypeComboBox = new QComboBox;
	synchroTypeComboBox->addItem("None", QVariant(0));
	synchroTypeComboBox->addItem("Identity", QVariant(1));
	synchroTypeComboBox->addItem("Delta", QVariant(2));
	QLabel *synchroTypeLabel = new QLabel("Synchro:");
	m_toolbar->addWidget(synchroTypeLabel);
	m_toolbar->addWidget(synchroTypeComboBox);
	synchroTypeComboBox->setCurrentIndex(0);
	this->synchroMultiView.setSynchroType(0);

	connect(synchroTypeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
//		QVariant var = synchroTypeComboBox->itemData(index);
//		bool ok;
//		int varInt = var.toInt(&ok);
//		if (ok) {
//			this->synchroMultiView.setSynchroType(varInt);
//		}
		this->synchroMultiView.setSynchroType(index);
	});

	QPushButton* stackSynchroButton = new QPushButton("Manage stack");
	m_toolbar->addWidget(stackSynchroButton);

	m_depthUnitButton = new QToolButton;
	m_depthUnitButton->setIcon(QIcon(":/slicer/icons/regle_m128_blanc.png"));
	m_depthUnitButton->setToolTip("Toggle between meter and feet");
	m_toolbar->addWidget(m_depthUnitButton);


	connect(m_depthUnitButton, &QPushButton::clicked, this, &GeotimeGraphicsView::toggleDepthLengthUnit);

	QPushButton *propertiesButton = GraphicsUtil::generateToobarButton(":/slicer/icons/property.svg", "Preferences", toolBar());

	m_toolbar->addWidget(propertiesButton);

	//QPushButton *multiViewButton = GraphicsUtil::generateToobarButton(":/slicer/icons/3d.png", "Multi view", toolBar());
	//m_toolbar->addWidget(multiViewButton);
	//connect(multiViewButton, SIGNAL(clicked()), this,SLOT(showMultiView()));

	//m_multiView = new MultiView(true,"MultiView");



		//QPushButton *propertiesButton = new QPushButton( toolBar());
		//propertiesButton->setIcon(QIcon(":/slicer/icons/property.svg"));
	//m_toolbar->addWidget(propertiesButton);


	connect(stackSynchroButton, &QPushButton::clicked, this, &GeotimeGraphicsView::manageSynchro);


	connect(propertiesButton, SIGNAL(clicked()), this,
				SLOT(showProperties()));


	m_properties = new PropertyPanel(this);

	connect(m_properties,SIGNAL(simplifySurfaceChanged(int)),this,SLOT(simplifySurface(int)));
	connect(m_properties,SIGNAL(simplifySeuilWellChanged(double)),this,SLOT(simplifySeuilWell(double)));
	connect(m_properties,SIGNAL(showInfo3DChanged(bool)),this,SLOT(showInfo3D(bool)));
	connect(m_properties,SIGNAL(showGizmo3DChanged(bool)),this,SLOT(showGizmo3D(bool)));
	connect(m_properties,SIGNAL(speedUpDownChanged(float)),this,SLOT(setspeedUpDown(float)));
	connect(m_properties,SIGNAL(speedHelicoChanged(float)),this,SLOT(setspeedHelico(float)));
	connect(m_properties,SIGNAL(speedRotHelicoChanged(float)),this,SLOT(setspeedRotHelico(float)));
	connect(m_properties,SIGNAL(showNormalsWellChanged(bool)),this,SLOT(showNormalsWell(bool)));
	connect(m_properties,SIGNAL(wireframeWellChanged(bool)),this,SLOT(wireframeWell(bool)));
	connect(m_properties,SIGNAL(simplifySeuilLogsChanged(int)),this,SLOT(simplifySeuilLogs(int)));
	connect(m_properties,SIGNAL(pickDiameterChanged(int)),this,SLOT(setDiameterPick(int)));
	connect(m_properties,SIGNAL(pickThicknessChanged(int)),this,SLOT(setThicknessPick(int)));
	connect(m_properties,SIGNAL(logThicknessChanged(int)),this,SLOT(setThicknessLog(int)));
	connect(m_properties,SIGNAL(colorLogChanged(QColor)),this,SLOT(setColorLog(QColor)));
	connect(m_properties,SIGNAL(colorWellChanged(QColor)),this,SLOT(setColorWell(QColor)));
	connect(m_properties,SIGNAL(colorSelectedWellChanged(QColor)),this,SLOT(setColorSelectedWell(QColor)));
	connect(m_properties,SIGNAL(wellDiameterChanged(int)),this,SLOT(setDiameterWell(int)));
	connect(m_properties,SIGNAL(wellMapWidthChanged(double)),this,SLOT(setWellMapWidth(double)));
	connect(m_properties,SIGNAL(wellSectionWidthChanged(double)),this,SLOT(setWellSectionWidth(double)));
	connect(m_properties,SIGNAL(speedMaxAnimChanged(int)),this,SLOT(setSpeedAnim(int)));
	connect(m_properties,SIGNAL(altitudeMaxAnimChanged(int)),this,SLOT(setAltitudeAnim(int)));
	connect(m_properties,SIGNAL(showHelicoChanged(bool)),this,SLOT(showHelico(bool)));


	m_properties->openIni();

	//m_orderStackHorizon = new OrderStackHorizonWidget(this,m_WorkingSetManager, nullptr);


	// createAnimatedSurfacesInit();
	//m_tools3D->show();
}

void GeotimeGraphicsView::openGraphicToolsDialog()
{
	GraphicToolsWidget::showPalette("");
}

AbstractInnerView* GeotimeGraphicsView::getInnerView3D(int index)
{
	QVector<AbstractInnerView*> innerViews =getInnerViews();
	if( index>=0 && index <innerViews.size())
	{
		return innerViews[index];
	}

	return nullptr;
}


 void GeotimeGraphicsView::SetDataItem(IData *pData,std::size_t index ,Qt::CheckState state){
	//QVector<AbstractInnerView*> innerViews = m_originViewer->getInnerViews();
	// AbstractInnerView* view = getInnerView3D(index);
	// if( view == nullptr) return ;
	QInnerViewTreeWidgetItem* rootItem = getItemFromView(getInnerViews()[index]);
	QStack<QTreeWidgetItem*> stack;
	stack.push(rootItem);

	//qDebug()<<"SetDataItem index:"<<rootItem->text(0);

	QGraphicsRepTreeWidgetItem* itemData = nullptr;

	while (stack.size()>0 && itemData==nullptr) {
		QTreeWidgetItem* item = stack.pop();

		std::size_t N = item->childCount();
		for (std::size_t index=0; index<N; index++) {
			stack.push(item->child(index));
		}
		QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
		if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {

			const IData* data =_item->getRep()->data();
			if (data!=nullptr && data==pData) {
				itemData = _item;
			}
		}
	}

	if (itemData!=nullptr) {

		itemData->setCheckState(0, state);
	}
}


GeotimeGraphicsView::~GeotimeGraphicsView() {
   if(m_layerSpectrumDialog != nullptr){
      m_layerSpectrumDialog->setGeoTimeView(nullptr);
   }
   m_WorkingSetManager=nullptr;
}

const MtLengthUnit* GeotimeGraphicsView::depthLengthUnit() const {
	return m_depthLengthUnit;
}

std::vector<AbstractInnerView*> GeotimeGraphicsView::getInnerViewsList()
{
	return m_InnerViewVec;
}

void GeotimeGraphicsView::showMultiView()
{
	//qDebug()<<"showMultiView";
	//if(m_multiView != nullptr) m_multiView->show();
}

void GeotimeGraphicsView::showProperties()
{
	if(m_properties != nullptr) m_properties->show();


}

void GeotimeGraphicsView::setSpeedAnim(int value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setSpeedAnim(value);
			}
		}
	}
}

void GeotimeGraphicsView::setAltitudeAnim(int value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setAltitudeAnim(value);
			}
		}
	}
}



void GeotimeGraphicsView::simplifySurface(int value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setSimplificationSurface(value);
			}
		}
	}
}


void GeotimeGraphicsView::simplifySeuilWell(double value) {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setSimplificationWell(value);
			}
		}
	}
}

void GeotimeGraphicsView::simplifySeuilLogs(int value) {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setSimplificationLogs(value);
			}
		}
	}
}

void GeotimeGraphicsView::showNormalsWell(bool visible) {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setShowNormalsWell(visible);
			}
		}
	}
}

void GeotimeGraphicsView::wireframeWell(bool wire) {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setWireframeWell(wire);
			}
		}
	}
}

void GeotimeGraphicsView::showInfo3D(bool visible) {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setInfosVisible(visible);
			}
		}
	}
}

void GeotimeGraphicsView::showGizmo3D(bool visible)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setGizmoVisible(visible);
			}
		}
	}
}

void GeotimeGraphicsView::showHelico(bool visible)
{
	QVector<AbstractInnerView*> views = innerViews();
		for (AbstractInnerView *v : views)
		{
			if (v->viewType() == ViewType::View3D)
			{
				ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
				if (view3D!=nullptr) {
					view3D->showHelico(visible);
				}
			}
		}
}

void GeotimeGraphicsView::setspeedUpDown(float value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setSpeedUpDown(value);
			}
		}
	}
}

void GeotimeGraphicsView::setspeedHelico(float value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setSpeedHelico(value);
			}
		}
	}
}

void GeotimeGraphicsView::setspeedRotHelico(float value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setSpeedRotHelico(value);
			}
		}
	}
}
void GeotimeGraphicsView::setDiameterWell(int value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setDiameterWell(value);
			}
		}
	}
}

void GeotimeGraphicsView::setWellMapWidth(double val) {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::StackBasemapView)
		{
			StackBaseMapView* view = dynamic_cast<StackBaseMapView*>(v);
			if (view!=nullptr) {
				view->setWellMapWidth(val);
			}
		}
		else if (v->viewType() == ViewType::BasemapView)
		{
			BaseMapView* view = dynamic_cast<BaseMapView*>(v);
			if (view!=nullptr) {
				view->setWellMapWidth(val);
			}

		}
	}
}

void GeotimeGraphicsView::setWellSectionWidth(double val) {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::InlineView || v->viewType() == ViewType::XLineView)
		{
			AbstractSectionView* view = dynamic_cast<AbstractSectionView*>(v);
			if (view!=nullptr) {
				view->setWellSectionWidth(val);
			}
		}
		else if (v->viewType() == ViewType::RandomView)
		{
			RandomLineView* view = dynamic_cast<RandomLineView*>(v);
			if (view!=nullptr) {
				view->setWellSectionWidth(val);
			}

		}
	}
}


void GeotimeGraphicsView::setDiameterPick(int value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setDiameterPick(value);
			}
		}
	}
}

void GeotimeGraphicsView::setThicknessPick(int value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setThicknessPick(value);
			}
		}
	}
}

void GeotimeGraphicsView::setThicknessLog(int value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setThicknessLog(value);
			}
		}
	}
}


void GeotimeGraphicsView::setColorLog(QColor value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setColorLog(value);
			}
		}
	}
}


void GeotimeGraphicsView::setColorWell(QColor value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setColorWell(value);
			}
		}




	}

	QList<IData*> data  = m_currentManager->folders().wells->data();
	for(int i=0;i< data.size();i++)
	{
		WellHead* head = dynamic_cast<WellHead*>(data[i]);
		if(head != nullptr)
		{
			QList<WellBore*> wellBores  = head->wellBores();
			for(int j=0;j< wellBores.size();j++)
			{

				wellBores[j]->setWellColor(value);
			}
		}
	}

}

void GeotimeGraphicsView::setColorSelectedWell(QColor value)
{
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
	{
		if (v->viewType() == ViewType::View3D)
		{
			ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
			if (view3D!=nullptr) {
				view3D->setColorSelectedWell(value);
			}
		}
	}
}

void GeotimeGraphicsView::viewPortChanged(const QPolygonF &poly) {
	QVector<AbstractInnerView*> views = innerViews();
	QObject* senderView = sender();
	bool isSenderBaseMap = dynamic_cast<BaseMapView*>(senderView)!=nullptr || dynamic_cast<StackBaseMapView*>(senderView)!=nullptr;
	bool isSenderSection = dynamic_cast<AbstractSectionView*>(senderView)!=nullptr;
	bool isSenderRandom = dynamic_cast<RandomLineView*>(senderView)!=nullptr;
	for (AbstractInnerView *v : views) {
		if (v == senderView)
			continue;
		if (isSenderBaseMap) {
			if (m_mutexView.tryLock()) {

				if (BaseMapView *view2D = dynamic_cast<BaseMapView*>(v)) {
					view2D->setViewRect(poly.boundingRect());
				} else if (StackBaseMapView *view2D = dynamic_cast<StackBaseMapView*>(v)) {
					view2D->setViewRect(poly.boundingRect());
				}
				m_mutexView.unlock();
			}
		} else if ((isSenderSection || isSenderRandom) && (dynamic_cast<AbstractSectionView*>(v)!=nullptr ||
				dynamic_cast<RandomLineView*>(v)!=nullptr)) {
			AbstractSectionView* originSection = dynamic_cast<AbstractSectionView*>(senderView);
			AbstractSectionView* viewSection = dynamic_cast<AbstractSectionView*>(v);
			AbstractInnerView* originView = dynamic_cast<AbstractInnerView*>(senderView);
			Abstract2DInnerView* v2D = dynamic_cast<Abstract2DInnerView*>(v);
			if (originView->viewType()==v->viewType()) {
				if (m_mutexView.tryLock()) {
					v2D->setViewRect(poly.boundingRect());
					m_mutexView.unlock();
				}
			} else if (originSection!=nullptr && viewSection!=nullptr) {
				if (m_mutexView.tryLock()) {
					int worldPos =  originSection->getCurrentSliceWorldPosition();
					QRectF senderRect = poly.boundingRect();
					QRectF viewRect = v2D->viewRect().boundingRect();
					QRectF rect(worldPos - viewRect.width()/2, senderRect.y(), viewRect.width(), senderRect.height());
					v2D->setViewRect(rect);
					m_mutexView.unlock();
				}
			} else {
				if (m_mutexView.tryLock()) {
					QRectF senderRect = poly.boundingRect();
					QRectF viewRect = v2D->viewRect().boundingRect();
					QRectF rect(viewRect.x(), senderRect.y(), viewRect.width(), senderRect.height());
					v2D->setViewRect(rect);
					m_mutexView.unlock();
				}
			}
		} else if ((isSenderRandom && (dynamic_cast<AbstractSectionView*>(v)!=nullptr ||
				dynamic_cast<RandomLineView*>(v)!=nullptr)) || (isSenderSection &&
						(dynamic_cast<AbstractSectionView*>(v)!=nullptr ||
								dynamic_cast<RandomLineView*>(v)!=nullptr))) {
			if (m_mutexView.tryLock()) {
				Abstract2DInnerView* view2D = dynamic_cast<Abstract2DInnerView*>(v);
				QRectF senderRect = poly.boundingRect();
				QRectF oriRect = view2D->viewRect().boundingRect();
				QRectF rect(oriRect.x(), senderRect.y(), oriRect.width(), senderRect.height());
				view2D->setViewRect(rect);
				m_mutexView.unlock();
			}
		}
	}
}

void GeotimeGraphicsView::zScaleChanged(double zScale) {
	QVector<AbstractInnerView*> views = innerViews();
	QObject* senderView = sender();

	m_zScale = zScale;
	for (AbstractInnerView *v : views) {
		if (v == senderView) {
			continue;
		}

		ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
		if (view3D!=nullptr) {
			view3D->setZScale(zScale);
		}
	}
}

void GeotimeGraphicsView::positionCamChanged(QVector3D position) {
	QVector<AbstractInnerView*> views = innerViews();
	QObject* senderView = sender();

	m_posCam= position;
	for (AbstractInnerView *v : views) {
		if (v == senderView) {
			continue;
		}

		ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
		if (view3D!=nullptr) {
			view3D->setPositionCam(position);
		}
	}
}

void GeotimeGraphicsView::viewCenterCamChanged(QVector3D center) {
	QVector<AbstractInnerView*> views = innerViews();
	QObject* senderView = sender();

	m_viewCam= center;
	for (AbstractInnerView *v : views) {
		if (v == senderView) {
			continue;
		}

		ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
		if (view3D!=nullptr) {
			view3D->setViewCenterCam(center);
		}
	}
}

void GeotimeGraphicsView::upVectorCamChanged(QVector3D up) {
	QVector<AbstractInnerView*> views = innerViews();
	QObject* senderView = sender();

	m_upCam = up;
	for (AbstractInnerView *v : views) {
		if (v == senderView) {
			continue;
		}

		ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(v);
		if (view3D!=nullptr) {
			view3D->setUpVectorCam(up);
		}
	}
}
void GeotimeGraphicsView::setTemplate(const TemplateView& views) {


	clearInnerViews();

	std::map<long, KDDockWidgets::DockWidget*> idToDockWidget;


	for (const TemplateInnerView& e : views) {
		KDDockWidgets::DockWidget* operationTarget = nullptr;
		if (e.targetId!=-1) {
			operationTarget = idToDockWidget[e.targetId];
		}

		AbstractInnerView* view = generateView(e.viewType, false);
		registerView(view, e.operation, operationTarget);

		if (view->viewType() != View3D)
		{

			m_InnerViewVec.push_back(view);
			connect(view, &QObject::destroyed, [this, view]() {
				std::vector<AbstractInnerView*>::iterator it = std::find(m_InnerViewVec.begin(),
						m_InnerViewVec.end(), view);
				if (it!=m_InnerViewVec.end()) {
					m_InnerViewVec.erase(it);
				}
			});
		}

		view->setDefaultTitle(e.title);
		dynamic_cast<QInnerViewTreeWidgetItem*>(m_rootItem->child(m_rootItem->childCount()-1))->setData(0, Qt::DisplayRole, view->getBaseTitle());
		idToDockWidget[e.viewId] = view;

	}
}

void GeotimeGraphicsView::resetInnerViewsPositions() {
	// custom function for maybe temporary behavior
	AbstractInnerView* firstSection = nullptr;
	AbstractInnerView* firstBasemap = nullptr;
	AbstractInnerView* firstView3D = nullptr;

	QList<AbstractInnerView*> basemapList;
	QList<AbstractInnerView*> sectionList;
	QList<AbstractInnerView*> view3DList;

	// fill lists & remove views from dock area
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView* e : views) {
		switch (e->viewType()) {
		case ViewType::InlineView:
		case ViewType::XLineView:
		case ViewType::RandomView:
			if (firstSection==nullptr) {
				firstSection = e;
			} else {
				sectionList.append(e);
			}
			break;
		case ViewType::StackBasemapView:
		case ViewType::BasemapView:
			if (firstBasemap==nullptr) {
				firstBasemap = e;
			} else {
				basemapList.append(e);
			}
			break;
		case ViewType::View3D:
			if (firstView3D==nullptr) {
				firstView3D = e;
			} else {
				view3DList.append(e);
			}
			break;
		}
		if (!e->isFloating()) {
//			removeDockWidget(e);
		} else {
			e->setFloating(false);
			//e->setParent(this);
			//removeDockWidget(e);
			e->hide();
		}
	}

	// create views if necessary
	if (firstSection==nullptr) {
		registerView(generateView(ViewType::InlineView, false), SplitHorizontal, nullptr);
	} else {
		addDockWidget(firstSection,
					KDDockWidgets::Location_OnRight);
		firstSection->show();
	}

	if (firstBasemap==nullptr) {
		registerView(generateView(ViewType::BasemapView, false), SplitHorizontal, nullptr);
	} else {
		addDockWidget(firstBasemap, KDDockWidgets::Location_OnRight, firstSection);
		firstBasemap->show();
	}

	if (firstView3D==nullptr) {
		registerView(generateView(ViewType::View3D, false), SplitVertical, nullptr);
	} else {
		addDockWidget(firstView3D, KDDockWidgets::Location_OnBottom, firstBasemap);
		firstView3D->show();
	}


	// add tabs
	for (AbstractInnerView* e : basemapList) {
		firstBasemap->addDockWidgetAsTab(e);
		e->show();
	}
	for (AbstractInnerView* e : sectionList) {
		firstSection->addDockWidgetAsTab(e);
		e->show();
	}
	for (AbstractInnerView* e : view3DList) {
		firstView3D->addDockWidgetAsTab(e);
		e->show();
	}

}

void GeotimeGraphicsView::clearInnerViews() {
	QVector<AbstractInnerView *> innerViews = this->innerViews();
	long index = innerViews.size()-1;
	while(index>=0) {
		unregisterView(innerViews[index]);
		innerViews[index]->deleteLater();
	}
}

QInnerViewTreeWidgetItem* GeotimeGraphicsView::getItemFromView(AbstractInnerView* view) {
	std::size_t index = 0;
	QInnerViewTreeWidgetItem* treeItem = nullptr;
	while (index<m_rootItem->childCount() && treeItem==nullptr) {
		QTreeWidgetItem *e=m_rootItem->child(index);
		treeItem = dynamic_cast<QInnerViewTreeWidgetItem*>(e);
		if (treeItem!=nullptr && treeItem->innerView()!=view) {
			treeItem = nullptr;
		}
		index++;
	}
	return treeItem;
}

QVector<AbstractInnerView *> GeotimeGraphicsView::getInnerViews() const { // public version of protected innerViews from AbstractGeaphicsView
	return innerViews();
}

void GeotimeGraphicsView::registerView(AbstractInnerView *newView, TemplateOperation operation, KDDockWidgets::DockWidget* operationTarget) {
	QVector<AbstractInnerView*> views = innerViews();


	// register in views synchro for slice change
	synchroMultiView.registerView( newView);

	QString viewName;//(
	//		(std::string("View ") + std::to_string(views.size())).c_str());
	//view->setViewIndex(views.size());
	switch (newView->viewType()) {
	case ViewType::InlineView:
		m_inlineCount ++;
		newView->setViewIndex(m_inlineCount);
		break;
	case ViewType::XLineView:
		m_xlineCount ++;
		newView->setViewIndex(m_xlineCount);
		break;
	case ViewType::RandomView:
		m_randomCount ++;
		newView->setViewIndex(m_randomCount);
		break;
	case ViewType::BasemapView:
        case ViewType::StackBasemapView:
		m_basemapCount ++;
		newView->setViewIndex(m_basemapCount);
		break;
	case ViewType::View3D:
		m_view3dCount ++;
		newView->setViewIndex(m_view3dCount);
		break;
	default:
		m_otherViewCount ++;
		newView->setViewIndex(m_otherViewCount);
	}
	viewName = newView->getBaseTitle();

	if (innerViews().size()==0 || operationTarget==nullptr) {
		addDockWidget(newView,
				KDDockWidgets::Location_OnRight);
	} else {
		switch (operation) {
		case SplitHorizontal:
			addDockWidget(newView, KDDockWidgets::Location_OnRight, operationTarget);
			break;
		case SplitVertical:
			addDockWidget(newView, KDDockWidgets::Location_OnBottom, operationTarget);
			break;
		default:
			operationTarget->addDockWidgetAsTab(newView);
			break;
		}
	}
	QInnerViewTreeWidgetItem *newItem = new QInnerViewTreeWidgetItem(viewName,
			newView, m_currentManager, m_rootItem);
	m_rootItem->addChild(newItem);

	for (AbstractInnerView *existingView : views)
		registerWindowControlers(newView, existingView);

	//Connect Event to be transmitted at the top level
	connect(newView, SIGNAL(viewMouseMoved(MouseTrackingEvent *)), this,
			SLOT(innerViewMouseMoved(MouseTrackingEvent *)));
	connect(newView, SIGNAL(controlerActivated(DataControler *)), this,
			SLOT(innerViewControlerActivated(DataControler *)));
	connect(newView, SIGNAL(controlerDesactivated(DataControler *)), this,
			SLOT(innerViewControlerDesactivated(DataControler *)));

	//Add already registred controlers
	for (DataControler *c : m_externalControler)
		newView->addExternalControler(c);

	connect(newView, SIGNAL(isClosing(AbstractInnerView * )), this,
			SLOT(unregisterView(AbstractInnerView *)));

	connect(newView, SIGNAL(askToSplit(AbstractInnerView * ,ViewType , bool , Qt::Orientation )),
			this, SLOT(askToSplit(AbstractInnerView * ,ViewType , bool , Qt::Orientation )));

//	connect(newView, SIGNAL(viewEnter(AbstractInnerView *)), this,
//			SLOT(subWindowActivated(AbstractInnerView *)));

//	connect(newView, &AbstractInnerView::viewEnter, newView,
//			&AbstractInnerView::activateWindow);

	connect(newView, &AbstractInnerView::viewEnter, newView,
			QOverload<>::of(&AbstractInnerView::setFocus));

	connect(newView, SIGNAL(askGeometryChanged(AbstractInnerView * ,const QRect & )), this,
				SLOT(geometryChanged(AbstractInnerView *,const QRect & )));

	if (IDepthView* depthView = dynamic_cast<IDepthView*>(newView)) {
		depthView->setDepthLengthUnit(m_depthLengthUnit);
	}

	bool showWells = true;
	if ((newView->viewType()==InlineView || newView->viewType()==XLineView || newView->viewType()==RandomView)) {
		showWells = initSection(newView);
	}

	if (newView->viewType()==View3D) {
		QInnerViewTreeWidgetItem* rootItem = getItemFromView(newView);

		QStack<QTreeWidgetItem*> stack;
		stack.push(rootItem);
		while (stack.size()>0) {
			QTreeWidgetItem* item = stack.pop();

			std::size_t N = item->childCount();
			for (std::size_t index=0; index<N; index++) {
				stack.push(item->child(index));
			}
			QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);

			if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable) && dynamic_cast<Seismic3DAbstractDataset*>(_item->getRep()->data())) {
				if (_item->getRep()->canBeDisplayed()) {
					_item->setCheckState(0, Qt::Checked);
				}
			}

			/*if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable) && dynamic_cast<NurbsDataset*>(_item->getRep()->data())) {
				if (_item->getRep()->canBeDisplayed()) {
					qDebug()<<"registerview Nurbs dataset";
					_item->setCheckState(0, Qt::Checked);
				}
			}*/

		}
	}

	if (newView->viewType()==BasemapView || newView->viewType()==StackBasemapView) {
		QInnerViewTreeWidgetItem* rootItem = getItemFromView(newView);

		QStack<QTreeWidgetItem*> stack;
		stack.push(rootItem);
		while (stack.size()>0) {
			QTreeWidgetItem* item = stack.pop();

			std::size_t N = item->childCount();
			for (std::size_t index=0; index<N; index++) {
				stack.push(item->child(index));
			}
			QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);

			if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable) && dynamic_cast<SeismicSurvey*>(_item->getRep()->data())) {
				if (_item->getRep()->canBeDisplayed()) {
					_item->setCheckState(0, Qt::Checked);
				}
			}

		}
	}


	if (showWells) {
		QInnerViewTreeWidgetItem* rootItem = getItemFromView(newView);

		QStack<QTreeWidgetItem*> stack;
		stack.push(rootItem);
		while (stack.size()>0) {
			QTreeWidgetItem* item = stack.pop();

			std::size_t N = item->childCount();
			for (std::size_t index=0; index<N; index++) {
				stack.push(item->child(index));
			}
			QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
			if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
				if (_item->getRep()->data()->displayPreference(newView->viewType()) && _item->getRep()->canBeDisplayed()) {
					_item->setCheckState(0, Qt::Checked);
				}
			}

		}
	}

	if(m_tools3D == nullptr)m_tools3D =new Tools3dWidget(this);
	//if(m_nurbs3D == nullptr)m_nurbs3D =new NurbsWidget(this);
	if (dynamic_cast<BaseMapView*>(newView)!=nullptr || dynamic_cast<StackBaseMapView*>(newView)!=nullptr ||
			dynamic_cast<AbstractSectionView*>(newView)!=nullptr || dynamic_cast<RandomLineView*>(newView)!=nullptr) {
		connect(newView, SIGNAL(viewAreaChanged(const QPolygonF & )), this,
				SLOT(viewPortChanged(const QPolygonF &)));
	} else if (dynamic_cast<ViewQt3D*>(newView)!=nullptr) {


		ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(newView);
		if( m_tools3D != nullptr) m_tools3D->setView3D(view3D);
	//	if( m_nurbs3D != nullptr) m_nurbs3D->setView3D(view3D);



	//	emit registerAddView3D();


		view3D->setZScale(m_zScale);
		connect(view3D, &ViewQt3D::zScaleChangedSignal, this,
						&GeotimeGraphicsView::zScaleChanged);
		//synchro camera
		view3D->setPositionCam(m_posCam);
		view3D->setViewCenterCam(m_viewCam);
		view3D->setUpVectorCam(m_upCam);

		connect(view3D, &ViewQt3D::positionCamChangedSignal, this,
								&GeotimeGraphicsView::positionCamChanged);
		connect(view3D, &ViewQt3D::viewCenterCamChangedSignal, this,
										&GeotimeGraphicsView::viewCenterCamChanged);
		connect(view3D, &ViewQt3D::upVectorCamChangedSignal, this,
												&GeotimeGraphicsView::upVectorCamChanged);

		connect(view3D, SIGNAL(signalShowTools(bool)), this,SLOT(show3dTools(bool)));

	}

	if (Abstract2DInnerView* innerView2D = dynamic_cast<Abstract2DInnerView*>(newView)) {

		if(innerView2D != nullptr)
		{
			//qDebug()<<"newView->viewType() : "<<newView->viewType();
			if(newView->viewType()== ViewType::StackBasemapView)
			{
				if( m_tools3D != nullptr) m_tools3D->setView2D(innerView2D);
			//	if( m_nurbs3D != nullptr) m_nurbs3D->setView2D(innerView2D);
			}
			if(newView->viewType()== ViewType::InlineView)
			{
				//qDebug()<<" set inlineView !";
				if( m_tools3D != nullptr) m_tools3D->setInlineView(innerView2D);
		//		if( m_nurbs3D != nullptr) m_nurbs3D->setInlineView(innerView2D);
			}
			if(newView->viewType()== ViewType::RandomView)
			{
				//qDebug()<<" set randomView !";
				if( m_tools3D != nullptr) m_tools3D->setInlineView(innerView2D);
		//		if( m_nurbs3D != nullptr) m_nurbs3D->setInlineView(innerView2D);
			}
		}
		connect(innerView2D, &Abstract2DInnerView::contextualMenuSignal, this,
				&GeotimeGraphicsView::contextMenu);
	}

}

bool GeotimeGraphicsView::initSection(AbstractInnerView* newView) {
	AbstractInnerView* reference = nullptr;
	QVector<AbstractInnerView*> currentInnerViews = innerViews();
	int viewIdx = currentInnerViews.size() - 1;
	while (viewIdx>=0 && reference==nullptr) {
		AbstractInnerView* candidate = currentInnerViews[viewIdx];
		AbstractSectionView* sectionCast = dynamic_cast<AbstractSectionView*>(candidate);
		RandomLineView* randomCast = dynamic_cast<RandomLineView*>(candidate);
		if (candidate!=newView) {
			if (sectionCast!=nullptr) {
				reference = sectionCast;
			} else if (randomCast!=nullptr) {
				reference = randomCast;
			}
		}
		viewIdx--;
	}

	bool output;
	if (reference==nullptr) {
		output = initSectionFromScratch(newView);
	} else {
		output = initSectionFromPrevious(newView, reference);
	}
	return output;
}

bool GeotimeGraphicsView::initSectionFromScratch(AbstractInnerView* newView) {
	QInnerViewTreeWidgetItem* rootItem = getItemFromView(newView);

	QStack<QTreeWidgetItem*> stack;
	stack.push(rootItem);
	QGraphicsRepTreeWidgetItem* itemSeismic = nullptr, *itemRgt = nullptr, *itemHorizon = nullptr;
	QGraphicsRepTreeWidgetItem* itemConstrain = nullptr;
	QList<QGraphicsRepTreeWidgetItem*> potentialconstrainList;
	while (stack.size()>0) {// && (itemSeismic==nullptr || itemRgt==nullptr || itemHorizon==nullptr)) {
		QTreeWidgetItem* item = stack.pop();

		std::size_t N = item->childCount();
		for (std::size_t index=0; index<N; index++) {
			stack.push(item->child(index));
		}
		QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
		if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
			const SliceRep* rep = dynamic_cast<const SliceRep*>(_item->getRep());
			const MultiSeedSliceRep* repHorizon = dynamic_cast<const MultiSeedSliceRep*>(_item->getRep());
			const RandomRep* repRandom = dynamic_cast<const RandomRep*>(_item->getRep());
			const MultiSeedRandomRep* repRandomHorizon = dynamic_cast<const MultiSeedRandomRep*>(_item->getRep());
			const FixedLayerFromDataset* layer = dynamic_cast<const FixedLayerFromDataset*>(_item->getRep()->data());
			if (rep!=nullptr && rep->name().toLower().contains("rgt")) {
				itemRgt = _item;
			} else if (rep!=nullptr) {
				itemSeismic = _item;
			} else if (repHorizon!=nullptr) {
				itemHorizon = _item;
			} else if (repRandom!=nullptr && repRandom->name().toLower().contains("rgt")) {
				itemRgt = _item;
			} else if (repRandom!=nullptr) {
				itemSeismic = _item;
			} else if (repRandomHorizon!=nullptr) {
				itemHorizon = _item;
			} else if (layer!=nullptr) {
				potentialconstrainList.push_back(_item);
			}
		}
	}
	if (itemHorizon!=nullptr) {
		FixedLayerFromDataset* constrain = dynamic_cast<MultiSeedHorizon*>(itemHorizon->getRep()->data())->constrainLayer();
		std::size_t idx=0;
		while (idx<potentialconstrainList.count() && itemConstrain==nullptr) {
			if (potentialconstrainList[idx]->getRep()->data()==constrain) {
				itemConstrain = potentialconstrainList[idx];
			}
		}
	}

	if (itemSeismic!=nullptr) {
		itemSeismic->setCheckState(0, Qt::Checked);
	}
	if (itemRgt!=nullptr) {
		itemRgt->setCheckState(0, Qt::Checked);
		if (dynamic_cast<SliceRep*>(itemRgt->getRep())!=nullptr) {
			dynamic_cast<SliceRep*>(itemRgt->getRep())->image()->setOpacity(0.5);
		} else if (dynamic_cast<RandomRep*>(itemRgt->getRep())!=nullptr) {
			if (dynamic_cast<RandomRep*>(itemRgt->getRep())->image()!=nullptr) {
				dynamic_cast<RandomRep*>(itemRgt->getRep())->image()->setOpacity(0.5);
			}
		}
	}
	if (itemHorizon!=nullptr) {
		itemHorizon->setCheckState(0, Qt::Checked);
		if (itemConstrain!=nullptr) {
			itemConstrain->setCheckState(0, Qt::Checked);
		}
	}
	bool dataAdded = itemSeismic!=nullptr || itemRgt!=nullptr;
	return dataAdded;
}

bool GeotimeGraphicsView::initSectionFromPrevious(AbstractInnerView* newView, AbstractInnerView* reference) {
	typedef struct Container {
		float opacity;
		QVector2D range;
		LookupTable lookupTable;
	} Container;

	std::map<Seismic3DAbstractDataset*, Container> datasetToOpacity;

	// scan reference
	QInnerViewTreeWidgetItem* referenceRootItem = getItemFromView(reference);

	QStack<QTreeWidgetItem*> stack;
	stack.push(referenceRootItem);
	while (stack.size()>0) {
		QTreeWidgetItem* item = stack.pop();

		std::size_t N = item->childCount();
		for (std::size_t index=0; index<N; index++) {
			stack.push(item->child(index));
		}
		QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
		if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
			SliceRep* rep = dynamic_cast<SliceRep*>(_item->getRep());
			RandomRep* repRandom = dynamic_cast<RandomRep*>(_item->getRep());
			Container container;
			if (rep!=nullptr && _item->checkState(0)==Qt::Checked) {
				Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(_item->getRep()->data());
				container.opacity = rep->image()->opacity();
				container.range = rep->image()->range();
				container.lookupTable = rep->image()->lookupTable();
				datasetToOpacity[dataset] = container;
			} else if (repRandom!=nullptr && _item->checkState(0)==Qt::Checked) {
				Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(repRandom->data());
				container.opacity = repRandom->image()->opacity();
				container.range = repRandom->image()->range();
				container.lookupTable = repRandom->image()->lookupTable();
				datasetToOpacity[dataset] = container;
			}
		}
	}

	// apply on newView
	QInnerViewTreeWidgetItem* newViewRootItem = getItemFromView(newView);
	stack.push(newViewRootItem);
	QGraphicsRepTreeWidgetItem* itemHorizon = nullptr;
	QGraphicsRepTreeWidgetItem* itemConstrain = nullptr;
	std::vector<QGraphicsRepTreeWidgetItem*> potentialconstrainList;
	std::vector<QGraphicsRepTreeWidgetItem*> visibleDatasetItems;
	while (stack.size()>0) {
		QTreeWidgetItem* item = stack.pop();

		std::size_t N = item->childCount();
		for (std::size_t index=0; index<N; index++) {
			stack.push(item->child(index));
		}
		QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
		if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
			SliceRep* rep = dynamic_cast<SliceRep*>(_item->getRep());
			const MultiSeedSliceRep* repHorizon = dynamic_cast<const MultiSeedSliceRep*>(_item->getRep());
			RandomRep* repRandom = dynamic_cast<RandomRep*>(_item->getRep());
			const MultiSeedRandomRep* repRandomHorizon = dynamic_cast<const MultiSeedRandomRep*>(_item->getRep());
			const FixedLayerFromDataset* layer = dynamic_cast<const FixedLayerFromDataset*>(_item->getRep()->data());
			Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(_item->getRep()->data());

			std::map<Seismic3DAbstractDataset*, Container>::iterator it = datasetToOpacity.find(dataset);
			if ((rep!=nullptr || repRandom!=nullptr) && dataset!=nullptr && it!=datasetToOpacity.end()) {
				if (rep!=nullptr) {
					rep->image()->setOpacity(datasetToOpacity[dataset].opacity);
					rep->image()->setRange(datasetToOpacity[dataset].range);
					rep->image()->setLookupTable(datasetToOpacity[dataset].lookupTable);
				} else {
					if (repRandom->image()!=nullptr) {
						repRandom->image()->setOpacity(datasetToOpacity[dataset].opacity);
						repRandom->image()->setRange(datasetToOpacity[dataset].range);
						repRandom->image()->setLookupTable(datasetToOpacity[dataset].lookupTable);
					}
				}
				visibleDatasetItems.push_back(_item);
			} else if (repHorizon!=nullptr) {
				itemHorizon = _item;
			} else if (repRandomHorizon!=nullptr) {
				itemHorizon = _item;
			} else if (layer!=nullptr) {
				potentialconstrainList.push_back(_item);
			}
		}
	}

	if (itemHorizon!=nullptr) {
		FixedLayerFromDataset* constrain = dynamic_cast<MultiSeedHorizon*>(itemHorizon->getRep()->data())->constrainLayer();
		std::size_t idx=0;
		while (idx<potentialconstrainList.size() && itemConstrain==nullptr) {
			if (potentialconstrainList[idx]->getRep()->data()==constrain) {
				itemConstrain = potentialconstrainList[idx];
			}
		}
	}

	for (QGraphicsRepTreeWidgetItem* item : visibleDatasetItems) {
		item->setCheckState(0, Qt::Checked);
	}

	if (itemHorizon!=nullptr) {
		itemHorizon->setCheckState(0, Qt::Checked);
		if (itemConstrain!=nullptr) {
			itemConstrain->setCheckState(0, Qt::Checked);
		}
	}

	return visibleDatasetItems.size()>0;
}

void GeotimeGraphicsView::registerView(AbstractInnerView* newView) {
	AbstractInnerView* firstSection = nullptr;
	AbstractInnerView* firstBasemap = nullptr;
	AbstractInnerView* firstView3D = nullptr;

	// fill lists & remove views from dock area
	QVector<AbstractInnerView*> views = innerViews();
	long idx = views.size()-1;
	while (idx>=0 && (firstSection==nullptr || firstBasemap==nullptr || firstView3D==nullptr) ) {
		AbstractInnerView* e = views[idx];
		switch (e->viewType()) {
		case ViewType::InlineView:
		case ViewType::XLineView:
		case ViewType::RandomView:
			if (firstSection==nullptr) {
				firstSection = e;
			}
			break;
		case ViewType::StackBasemapView:
		case ViewType::BasemapView:
		{

			if (firstBasemap==nullptr) {
				firstBasemap = e;
			}
			break;
		}
		case ViewType::View3D:
			if (firstView3D==nullptr) {
				firstView3D = e;
			}
			//init preference

			break;
		}
		idx --;
	}

	TemplateOperation operation = SplitHorizontal;
	AbstractInnerView* operationTarget = nullptr;
	switch (newView->viewType()) {
	case ViewType::InlineView:
	case ViewType::XLineView:
	case ViewType::RandomView:
		if (firstSection!=nullptr) {
			operation = AddAsTab;
			operationTarget = firstSection;
		}
		break;
	case ViewType::StackBasemapView:
	case ViewType::BasemapView:
	{

		if (firstBasemap!=nullptr) {
			operation = AddAsTab;
			operationTarget = firstBasemap;
		}
	/*	Abstract2DInnerView* view2d = dynamic_cast<Abstract2DInnerView*>(newView);
		if(view2d != nullptr)
		{
			if( m_tools3D != nullptr) m_tools3D->setView2D(view2d);
		}*/
		break;
	}
	case ViewType::View3D:

		if (firstView3D!=nullptr) {
			operation = AddAsTab;
			operationTarget = firstView3D;
		}
		ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(newView);

				if(view3D != nullptr)
				{
					//if( m_tools3D != nullptr) m_tools3D->setView3D(view3D);

					view3D->setSpeedHelico(m_properties->speedHelico());
					view3D->setSpeedUpDown(m_properties->speedUpDown());
					view3D->setGizmoVisible(m_properties->showGizmo3d());
					view3D->setInfosVisible(m_properties->showInfos3d());
				}

		break;
	}

	registerView(newView, operation, operationTarget);
}

void GeotimeGraphicsView::unregisterView(AbstractInnerView *toBeDeleted) {
	// register in views synchro for slice change


	synchroMultiView.unregisterView( toBeDeleted );

	MultiTypeGraphicsView::unregisterView(toBeDeleted);
	if (dynamic_cast<BaseMapView*>(toBeDeleted)!=nullptr || dynamic_cast<StackBaseMapView*>(toBeDeleted)!=nullptr ||
			dynamic_cast<AbstractSectionView*>(toBeDeleted)!=nullptr) {
		disconnect(toBeDeleted, SIGNAL(viewAreaChanged(const QPolygonF & )), this,
				SLOT(viewPortChanged(const QPolygonF &)));
	}

	disconnect(toBeDeleted, &AbstractInnerView::viewEnter, toBeDeleted,
			QOverload<>::of(&AbstractInnerView::setFocus));
//	disconnect(toBeDeleted, &AbstractInnerView::viewEnter, toBeDeleted,
//			&AbstractInnerView::activateWindow);
	if (Abstract2DInnerView* innerView2D = dynamic_cast<Abstract2DInnerView*>(toBeDeleted)) {
		disconnect(innerView2D, &Abstract2DInnerView::contextualMenuSignal, this,
				&GeotimeGraphicsView::contextMenu);

		m_tools3D->removeView2D(innerView2D);

	}

	ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(toBeDeleted);
	if(view3D != nullptr )m_tools3D->removeView3D(view3D);
}

void GeotimeGraphicsView::contextMenu(Abstract2DInnerView* emitingView, double worldX, double worldY,
			QContextMenuEvent::Reason reason, QMenu& menu) {
	if (emitingView->viewType()==ViewType::InlineView || emitingView->viewType()==ViewType::XLineView || emitingView->viewType()==ViewType::RandomView) {
		QAction *actionSpectrum = menu.addAction(("Layer Spectrum Computation"));
		connect(actionSpectrum, &QAction::triggered, [this, emitingView, worldX, worldY]() {
			spectrumDecomposition(emitingView, worldX, worldY);
		});

		QAction *addSpectrumSectionAction = menu.addAction(("Create Spectrum Section"));
		connect(addSpectrumSectionAction, &QAction::triggered, [this, emitingView]() {
			createSpectrumData(emitingView);
		});

		/*
		QAction *addGenericTreatmentSectionAction = menu.addAction(("Create time gradient Section"));
		connect(addGenericTreatmentSectionAction, &QAction::triggered, [this, emitingView]() {
			createGenericTreatmentData(emitingView);
		});
		*/

		QAction *addGenericTreatmentSectionAction = menu.addAction(("Erase patch"));
		connect(addGenericTreatmentSectionAction, &QAction::triggered, [this, emitingView, worldX, worldY]() {
			erasePatch(emitingView, worldX, worldY);
		});



		/*
		QAction *actionVolumic = menu.addAction(("RGT Volumic"));
		connect(actionVolumic, &QAction::triggered, [this, emitingView, worldX, worldY]() {
			rgtModification(emitingView, worldX, worldY);
		});
		*/
	}
	if (emitingView->viewType()==ViewType::RandomView) {
		QAction *copyAction = menu.addAction(("Copy Random"));
		connect(copyAction, &QAction::triggered, [this, emitingView]() {
			copyRandom(emitingView);
		});
	}
}

void GeotimeGraphicsView::createSpectrumData(Abstract2DInnerView* emitingView) {
	Seismic3DAbstractDataset* dataset = nullptr;
	int channel;
	if (AbstractSectionView* section = dynamic_cast<AbstractSectionView*>(emitingView)) {
		QStringList items;
		QList<SliceRep*> reps;

		QList<AbstractGraphicRep*> visibleReps = section->getVisibleReps();
		for (long i=visibleReps.count()-1; i>=0; i--) {
			AbstractGraphicRep *r = visibleReps[i];
			if (SliceRep *slice = dynamic_cast<SliceRep*>(r)) {
				reps.push_back(slice);
				items.push_back(slice->data()->name());
			}
		}
		QString name = "Datasets";
		int result = QDialog::Rejected;
		int outIndex = 0;
		if (items.count()>1) {
			StringSelectorDialog dialog(&items, name);
			result = dialog.exec();
			outIndex = dialog.getSelectedIndex();
		} else if (items.count()==1) {
			result = QDialog::Accepted;
			outIndex = 0;
		} else {
			QMessageBox::information(this, "Spectrum Data", "Failed to detect any dataset to compute spectrum on");
		}

		if (result==QDialog::Accepted) {
			SliceRep* rep = reps[outIndex];
			dataset = dynamic_cast<Seismic3DAbstractDataset*>(rep->data());
			channel = rep->channel();
		}
	} else if (RandomLineView* random = dynamic_cast<RandomLineView*>(emitingView)) {
		QStringList items;
		QList<RandomRep*> reps;

		QList<AbstractGraphicRep*> visibleReps = random->getVisibleReps();
		for (long i=visibleReps.count()-1; i>=0; i--) {
			AbstractGraphicRep *r = visibleReps[i];
			if (RandomRep *randomRep = dynamic_cast<RandomRep*>(r)) {
				reps.push_back(randomRep);
				items.push_back(randomRep->data()->name());
			}
		}
		QString name = "Datasets";
		int result = QDialog::Rejected;
		int outIndex = 0;
		if (items.count()>1) {
			StringSelectorDialog dialog(&items, name);
			result = dialog.exec();
			outIndex = dialog.getSelectedIndex();
		} else if (items.count()==1) {
			result = QDialog::Accepted;
			outIndex = 0;
		} else {
			QMessageBox::information(this, "Spectrum Data", "Failed to detect any dataset to compute spectrum on");
		}

		if (result==QDialog::Accepted) {
			RandomRep* rep = reps[outIndex];
			dataset = dynamic_cast<Seismic3DAbstractDataset*>(rep->data());
			channel = rep->channel();
		}
	}
	if (dataset) {
		// add data
		RgbComputationOnDataset* newData = new RgbComputationOnDataset(dataset, channel, dataset->workingSetManager());
		dataset->workingSetManager()->addRgbComputationOnDataset(newData);

		RgbDataset* rgbDataset = RgbDataset::createRgbDataset("Spectrum On " + dataset->name(), newData, 8,
				newData, 10, newData, 12, nullptr, 0, dataset->workingSetManager());
		dataset->workingSetManager()->addRgbDataset(rgbDataset);
		QCoreApplication::processEvents(); // to create reps for this data

		// select data
	}
}


void GeotimeGraphicsView::erasePatch(Abstract2DInnerView* emitingView, double worldX, double worldY) {

	// AbstractSectionView* originSection
	// long z = emitingView->getCurrentSliceWorldPosition();

	Seismic3DAbstractDataset* datasetS = nullptr;
	int channelS = 0;
	Seismic3DAbstractDataset* datasetT = nullptr;
	int channelT = 0;
	for (AbstractGraphicRep *rep : emitingView->getVisibleReps()) {
		IData* ds = rep->data();
		QString name = ds->name();
		Seismic3DDataset* dataset = nullptr;

		if ( ! (dataset = dynamic_cast<Seismic3DDataset*>(ds))) {
			continue;
		}
		SliceRep* sliceRep;
		RandomRep* randomRep;
		int channel = 0;
		if (!(sliceRep = dynamic_cast<SliceRep*>(rep)) && ! (randomRep = dynamic_cast<RandomRep*>(rep))) {
			continue;
		} else {
			if (sliceRep) {
				channel = sliceRep->channel();
			} else if (randomRep) {
				channel = randomRep->channel();
			}
		}
		qDebug() << "Data Name: " << name << "Data Type: " << dataset->type();

		// redo with slicerep and randomrep with currentChannel function
		if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::Seismic && dataset->dimV()==1) {
			datasetS = dataset;
			channelS = channel;
		} if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::RGT && dataset->dimV()==1) {
			datasetT = dataset;
			channelT = channel;
		}
	}

	//TODO
	QPointF referencePoint(worldX, worldY);
	if ( m_layerSpectrumDialog == nullptr ) return;
	m_layerSpectrumDialog->eraseParentPatch(emitingView, referencePoint);
}

void GeotimeGraphicsView::createGenericTreatmentData(Abstract2DInnerView* emitingView)
{
	Seismic3DAbstractDataset* dataset = nullptr;
	int channel;
	if (AbstractSectionView* section = dynamic_cast<AbstractSectionView*>(emitingView)) {
		QStringList items;
		QList<SliceRep*> reps;

		QList<AbstractGraphicRep*> visibleReps = section->getVisibleReps();
		for (long i=visibleReps.count()-1; i>=0; i--) {
			AbstractGraphicRep *r = visibleReps[i];
			if (SliceRep *slice = dynamic_cast<SliceRep*>(r)) {
				reps.push_back(slice);
				items.push_back(slice->data()->name());
			}
		}
		QString name = "Datasets";
		int result = QDialog::Rejected;
		int outIndex = 0;
		if (items.count()>1) {
			StringSelectorDialog dialog(&items, name);
			result = dialog.exec();
			outIndex = dialog.getSelectedIndex();
		} else if (items.count()==1) {
			result = QDialog::Accepted;
			outIndex = 0;
		} else {
			QMessageBox::information(this, "Spectrum Data", "Failed to detect any dataset to compute spectrum on");
		}

		if (result==QDialog::Accepted) {
			SliceRep* rep = reps[outIndex];
			dataset = dynamic_cast<Seismic3DAbstractDataset*>(rep->data());
			channel = rep->channel();
		}
	} else if (RandomLineView* random = dynamic_cast<RandomLineView*>(emitingView)) {
		QStringList items;
		QList<RandomRep*> reps;

		QList<AbstractGraphicRep*> visibleReps = random->getVisibleReps();
		for (long i=visibleReps.count()-1; i>=0; i--) {
			AbstractGraphicRep *r = visibleReps[i];
			if (RandomRep *randomRep = dynamic_cast<RandomRep*>(r)) {
				reps.push_back(randomRep);
				items.push_back(randomRep->data()->name());
			}
		}
		QString name = "Datasets";
		int result = QDialog::Rejected;
		int outIndex = 0;
		if (items.count()>1) {
			StringSelectorDialog dialog(&items, name);
			result = dialog.exec();
			outIndex = dialog.getSelectedIndex();
		} else if (items.count()==1) {
			result = QDialog::Accepted;
			outIndex = 0;
		} else {
			QMessageBox::information(this, "Spectrum Data", "Failed to detect any dataset to compute spectrum on");
		}

		if (result==QDialog::Accepted) {
			RandomRep* rep = reps[outIndex];
			dataset = dynamic_cast<Seismic3DAbstractDataset*>(rep->data());
			channel = rep->channel();
		}
	}

	if (dataset) {
		// add data
		ScalarComputationOnDataset* newData = new ScalarComputationOnDataset(dataset, channel, dataset->workingSetManager());
		dataset->workingSetManager()->addScalarComputationOnDataset(newData);

		RgbDataset* rgbDataset = RgbDataset::createRgbDataset("Spectrum On " + dataset->name(), newData, 8,
				newData, 10, newData, 12, nullptr, 0, dataset->workingSetManager());
		dataset->workingSetManager()->addRgbDataset(rgbDataset);
		QCoreApplication::processEvents(); // to create reps for this data

		// select data
	}

}

QString partialJoin(const QStringList& list, int beg, int end, const QString& joinStr) {
	QString out;
	if (beg>=0 && beg<=end && end<list.count()) {
		QStringList newList;
		for (long i=beg; i<=end; i++) {
			newList << list[i];
		}
		out = newList.join(joinStr);
	}
	return out;
}

std::pair<QString, int> getPairFromRandom(RandomLineView* randomView) {
	QString namePrevious = randomView->defaultTitle();
	std::pair<QString, int> out("", 0);
	if (!namePrevious.isNull() && !namePrevious.isEmpty()) {
		QStringList split1 = namePrevious.split("(");
		QString realName;
		int number = 0;
		bool isValid = false;
		if (split1.size()>1) {
			QStringList split2 = split1.last().split(")");
			if (split2.count()==2) {
				int val = split2[0].toInt(&isValid);
				if (isValid) {
					number = val;
					realName = partialJoin(split1, 0, split1.count()-2, "(");
				}
			}
		}
		if (!isValid) {
			realName = namePrevious;
		}
		out = std::pair<QString, int>(realName, number);
	}
	return out;
}

void GeotimeGraphicsView::copyRandom(Abstract2DInnerView* emitingView) {
	RandomLineView* random = dynamic_cast<RandomLineView*>(emitingView);
	if (random==nullptr) {
		return;
	}

	std::pair<QString, int> namePair = getPairFromRandom(random);
	QString realName = namePair.first;
	int number = namePair.second;
	QPolygonF randomLine = random->polyLine();
	QString newUniqueName = uniqueName() + "_view" + QString::number(getNewUniqueId());
	RandomLineView* newRandomView = new RandomLineView(randomLine, ViewType::RandomView, newUniqueName);
	registerView(newRandomView);
	if (!realName.isNull() && !realName.isEmpty()) {
		bool nameExist = true;
		QString newTitle;
		while (nameExist) {
			number++;
			newTitle = realName + "(" + QString::number(number) + ")";
			QVector<AbstractInnerView*> innerViews = this->innerViews();
			std::size_t i = 0;
			nameExist = false;
			while (i<innerViews.size() && !nameExist) {
				nameExist = newTitle.compare(innerViews[i]->defaultTitle())==0;
				i++;
			}
		}
		changeViewName(newRandomView, newTitle);
	}

}

void GeotimeGraphicsView::spectrumDecomposition(Abstract2DInnerView* emitingView, double worldX, double worldY) {
	struct Candidate {
		Seismic3DAbstractDataset* dataset = nullptr;
		int channel = 0;
	};

	std::vector<Candidate> seismicCandidates, rgtCandidates;

	for (AbstractGraphicRep *rep : emitingView->getVisibleReps()) {
		IData* ds = rep->data();
		QString name = ds->name();
		Seismic3DDataset* dataset = nullptr;

		if ( ! (dataset = dynamic_cast<Seismic3DDataset*>(ds))) {
			continue;
		}
		SliceRep* sliceRep;
		RandomRep* randomRep;
		int channel = 0;
		if (!(sliceRep = dynamic_cast<SliceRep*>(rep)) && ! (randomRep = dynamic_cast<RandomRep*>(rep))) {
			continue;
		} else {
			if (sliceRep) {
				channel = sliceRep->channel();
			} else if (randomRep) {
				channel = randomRep->channel();
			}
		}
		qDebug() << "Data Name: " << name <<
				"Data Type: " << dataset->type();

		// redo with slicerep and randomrep with currentChannel function
		if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::Seismic && dataset->dimV()==1) {
			Candidate candidate;
			candidate.dataset = dataset;
			candidate.channel = channel;
			seismicCandidates.push_back(candidate);
		} if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::RGT && dataset->dimV()==1) {
			Candidate candidate;
			candidate.dataset = dataset;
			candidate.channel = channel;
			rgtCandidates.push_back(candidate);
		}
	}

	Seismic3DAbstractDataset* datasetS = nullptr;
	int channelS = 0;
	Seismic3DAbstractDataset* datasetT = nullptr;
	int channelT = 0;

	int rgtIdx = 0;
	while ((datasetT==nullptr || datasetS==nullptr) && rgtIdx<rgtCandidates.size()) {
		datasetT = rgtCandidates[rgtIdx].dataset;
		channelT = rgtCandidates[rgtIdx].channel;

		int seismicIdx = 0;
		while ((datasetT==nullptr || datasetS==nullptr) && seismicIdx<seismicCandidates.size()) {
			Seismic3DAbstractDataset* dataset = seismicCandidates[seismicIdx].dataset;
			if (dataset->isCompatible(datasetT)) {
				datasetS = dataset;
				channelS = seismicCandidates[seismicIdx].channel;
			}
			seismicIdx++;
		}

		rgtIdx++;
	}

	//TODO
	QPointF referencePoint(worldX, worldY);
	if ( m_layerSpectrumDialog == nullptr ) {
		if (datasetS!=nullptr && datasetT!=nullptr) {
			m_layerSpectrumDialog = new LayerSpectrumDialog(datasetS, channelS, datasetT, channelT,this,emitingView);
			m_layerSpectrumDialog->show();
			connect(m_layerSpectrumDialog, &QWidget::destroyed, [this]() {
				m_layerSpectrumDialog = nullptr;
			});

			m_layerSpectrumDialog->setPoint(emitingView, referencePoint);
		} else {
			QMessageBox::warning(this, "Loading failed", "Failed to find compatible rgt and seismic in the selected data.\nPlease check that there is a valid couple of seismic and rgt selected.");
		}
	}
	else {
		m_layerSpectrumDialog->setPoint(emitingView, referencePoint);
//		m_layerSpectrumDialog->updateData();
	}
}


void GeotimeGraphicsView::rgtModification(Abstract2DInnerView* emitingView, double worldX, double worldY)
{
	Seismic3DAbstractDataset* datasetS = nullptr;
	Seismic3DAbstractDataset* datasetT = nullptr;

	for (AbstractGraphicRep *rep : emitingView->getVisibleReps()) {
		IData* ds = rep->data();
		QString name = ds->name();
		Seismic3DDataset* dataset = nullptr;
		if ( ! (dataset = dynamic_cast<Seismic3DDataset*>(ds))) {
			continue;
		}
		qDebug() << "Data Name: " << name <<
				"Data Type: " << dataset->type();

		if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::Seismic) {
			datasetS = dataset;
		} if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::RGT) {
			datasetT = dataset;
		}
	}

	int channelS = 0;
	int channelT = 0;
	if ( m_rgtVolumicDialog == nullptr ) {
			if (datasetS!=nullptr && datasetT!=nullptr) {
				m_rgtVolumicDialog = new RgtVolumicDialog(datasetS, channelS, datasetT, channelT, this);
				m_rgtVolumicDialog->show();
				/*
				connect(m_layerSpectrumDialog, &QWidget::destroyed, [this]() {
					m_layerSpectrumDialog = nullptr;
				});
				*/
				// m_rgtVolumicDialog->setPoint(emitingView, referencePoint);
			}
		}
}


void GeotimeGraphicsView::openDataManager() {
	if (m_dataSelectorDialog) {
		m_dataSelectorDialog->show();
	}
}

void GeotimeGraphicsView::saveSession() {
	if (m_dataSelectorDialog) {
		m_dataSelectorDialog->saveSession();
	}
}

//void GeotimeGraphicsView::createAnimatedSurfacesImages() {
//	WorkingSetManager* manager = m_currentManager;
//
//	Seismic3DAbstractDataset* dataset = getDataset(false);
//
//	FixedRGBLayersFromDataset* animateData;
//	if (dataset!=nullptr) {
//		animateData = FixedRGBLayersFromDataset::createDataFromDatasetWithUI(
//				QString("Animate ")+dataset->name(), manager,
//				dataset);
//	} else {
//		animateData = nullptr;
//	}
//	if (animateData!=nullptr) {
//		manager->addFixedRGBLayersFromDataset(animateData);
//	} else if (dataset!=nullptr) {
//		QMessageBox::information(this, tr("Load Animated Surfaces"), tr("Could not create animated surfaces"));
//	}
//}

WorkingSetManager* GeotimeGraphicsView::getWorkingSetManager()
{
	return m_WorkingSetManager;
}


//void GeotimeGraphicsView::createAnimatedSurfacesInit()
//{
//	SeismicSurvey* survey = getSurvey();
//	qDebug() << survey->idPath();
//	WorkingSetManager* manager = m_currentManager;
//	std::vector<FixedRGBLayersFromDatasetAndCube*> animateData;
//	std::vector<FixedRGBLayersFromDatasetAndCube*> animateDataFreeHorizon;
//	if (survey!=nullptr) {
//		animateData = FixedRGBLayersFromDatasetAndCube::createDataFromDataset(
//				QString("Animate "), manager,
//				survey, false, "spectrum"
//				// ,static_cast<void*>(m_comboAttributType)
//				);
//		for (int i=0; i<animateData.size(); i++) animateData[i]->setOption(static_cast<void*>(m_comboAttributType));
//
//		animateDataFreeHorizon = FixedRGBLayersFromDatasetAndCube::createDataFreeHorizonFromDataset(
//				QString("Animate "), manager,
//				survey, false, "spectrum");
//		for (int i=0; i<animateDataFreeHorizon.size(); i++) animateDataFreeHorizon[i]->setOption(static_cast<void*>(m_comboAttributType));
//	}
//
//	for (int i=0; i<animateData.size(); i++)
//	{
//		FixedRGBLayersFromDatasetAndCube* animateData_ = animateData[i];
//		if (animateData_ != nullptr ) {
//			manager->addHorizonsIsoFromDirectories(animateData_);
//			// selectLayerTypedData(animateData_);
//		} else if (survey!=nullptr) {
//			QMessageBox::information(this, tr("Load Animated Surfaces"), tr("Could not create animated surfaces"));
//		}
//	}
//	for (int i=0; i<animateDataFreeHorizon.size(); i++)
//	{
//		FixedRGBLayersFromDatasetAndCube* animateData_ = animateDataFreeHorizon[i];
//		if (animateData_ != nullptr ) {
//			manager->addHorizonsFreeFromDirectories(animateData_);
//			// selectLayerTypedData(animateData_);
//		} else if (survey!=nullptr) {
//			QMessageBox::information(this, tr("Load Animated Surfaces"), tr("Could not create animated surfaces"));
//		}
//	}
//
//}

//void GeotimeGraphicsView::createAnimatedSurfacesCube() {
//	WorkingSetManager* manager = m_currentManager;
//
//	SeismicSurvey* survey = getSurvey();
//
//	FixedRGBLayersFromDatasetAndCube* animateData;
//	if (survey!=nullptr) {
//		animateData = FixedRGBLayersFromDatasetAndCube::createDataFromDatasetWithUI(
//					QString("Animate "), manager,
//					survey, false);
//	} else {
//		animateData = nullptr;
//	}
//	if (animateData!=nullptr) {
//		manager->addFixedRGBLayersFromDatasetAndCube(animateData);
//		selectLayerTypedData(animateData);
//	} else if (survey!=nullptr) {
//		QMessageBox::information(this, tr("Load Animated Surfaces"), tr("Could not create animated surfaces"));
//	}
//}
//
//void GeotimeGraphicsView::createAnimatedSurfacesCubeGcc() {
//	WorkingSetManager* manager = m_currentManager;
//
//	SeismicSurvey* survey = getSurvey();
//
//	FixedRGBLayersFromDatasetAndCube* animateData;
//	if (survey!=nullptr) {
//		animateData = FixedRGBLayersFromDatasetAndCube::createDataFromDatasetWithUI(
//					QString("Animate "), manager,
//					survey, false, "gcc");
//	} else {
//		animateData = nullptr;
//	}
//	if (animateData!=nullptr) {
//		manager->addFixedRGBLayersFromDatasetAndCube(animateData);
//		selectLayerTypedData(animateData);
//	} else if (survey!=nullptr) {
//		QMessageBox::information(this, tr("Load Animated Surfaces"), tr("Could not create animated surfaces"));
//	}
//}
//
//void GeotimeGraphicsView::createAnimatedSurfacesCubeRgb1() {
//	WorkingSetManager* manager = m_currentManager;
//
//	SeismicSurvey* survey = getSurvey();
//
//	FixedRGBLayersFromDatasetAndCube* animateData;
//	if (survey!=nullptr) {
//		animateData = FixedRGBLayersFromDatasetAndCube::createDataFromDatasetWithUI(
//					QString("Animate "), manager,
//					survey, true);
//	} else {
//		animateData = nullptr;
//	}
//	if (animateData!=nullptr) {
//		manager->addFixedRGBLayersFromDatasetAndCube(animateData);
//		selectLayerTypedData(animateData);
//	} else if (survey!=nullptr) {
//		QMessageBox::information(this, tr("Load Animated Surfaces"), tr("Could not create animated surfaces"));
//	}
//}
//
//void GeotimeGraphicsView::createAnimatedSurfacesCubeMean() {
//	WorkingSetManager* manager = m_currentManager;
//
//	SeismicSurvey* survey = getSurvey();
//
//	FixedLayersFromDatasetAndCube* animateData;
//	if (survey!=nullptr) {
//		animateData = FixedLayersFromDatasetAndCube::createDataFromDatasetWithUI(
//					QString("Animate "), manager,
//					survey);
//	} else {
//		animateData = nullptr;
//	}
//	if (animateData!=nullptr) {
//		manager->addFixedLayersFromDatasetAndCube(animateData);
//		selectLayerTypedData(animateData);
//	} else if (survey!=nullptr) {
//		QMessageBox::information(this, tr("Load Animated Surfaces"), tr("Could not create animated surfaces"));
//	}
//}

//void GeotimeGraphicsView::createVideoLayer() {
//	QString mediaPath = QFileDialog::getOpenFileName(this);
//	Seismic3DAbstractDataset* dataset = getDataset(false);
//
//	if (dataset!=nullptr) {
//		VideoLayer* layer =  new VideoLayer(m_currentManager, mediaPath, dataset);
//		m_currentManager->addVideoLayer(layer);
//	}
//}

//void GeotimeGraphicsView::createTmapLayer() {
//	Seismic3DDataset* datasetRef = nullptr;
//	int channelRef=0;
//
//	long k=0;
//	QVector<AbstractInnerView *> views = innerViews();
//	while (datasetRef==nullptr && k<views.count()) {
//		AbstractInnerView* innerView = views[k];
//
//		const QList<AbstractGraphicRep*>& visibleReps = innerView->getVisibleReps();
//		long i=0;
//		while (datasetRef==nullptr && i<visibleReps.count()) {
//			AbstractGraphicRep *rep = visibleReps[i];
//			IData* ds = rep->data();
//			QString name = ds->name();
//			Seismic3DDataset* dataset = nullptr;
//			if ( ! (dataset = dynamic_cast<Seismic3DDataset*>(ds))) {
//				continue;
//			}
//			SliceRep* sliceRep;
//			RandomRep* randomRep;
//			int channel = 0;
//			if (!(sliceRep = dynamic_cast<SliceRep*>(rep)) && ! (randomRep = dynamic_cast<RandomRep*>(rep))) {
//				continue;
//			} else {
//				if (sliceRep) {
//					channel = sliceRep->channel();
//				} else if (randomRep) {
//					channel = randomRep->channel();
//				}
//			}
//			qDebug() << "Data Name: " << name <<
//					"Data Type: " << dataset->type();
//
//			if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::Seismic) {
//				datasetRef = dataset;
//				channelRef = channel;
//			}
//			i++;
//		}
//		k++;
//	}
//	if (datasetRef==nullptr) {
//		return;
//	}
//
//	std::unique_ptr<AbstractKohonenProcess> process;
//	process.reset(AbstractKohonenProcess::getObjectFromDataset(datasetRef, channelRef));
//
//	IAbstractIsochrone* isochrone = nullptr;
//	QList<IData*> layersData = m_currentManager->folders().horizonsFree->data();
//	QList<IData*> isochroneGivers;
//	QStringList nameList;
//	for (IData* data : layersData) {
//		if (IJKHorizon* ijkHorizon = dynamic_cast<IJKHorizon*>(data)) {
//			isochroneGivers.append(data);
//			nameList.append(data->name());
//		}
//	}
//	if (nameList.count()>0) {
//		StringSelectorDialog dialog(&nameList, "Select isochrone");
//		int code = dialog.exec();
//		if (code==QDialog::Accepted) {
//			long index = dialog.getSelectedIndex();
//			IJKHorizon* ijkHorizon = dynamic_cast<IJKHorizon*>(isochroneGivers[index]);
//			isochrone = ijkHorizon->getIsochrone(); // take ownership of isochrone
//		}
//	}
//
//
//	if (isochrone==nullptr) {
//		return;
//	}
//	process->setExtractionIsochrone(isochrone);
//
//	FixedLayerFromDataset* outputLayer = new FixedLayerFromDataset("Tmap", m_currentManager, datasetRef);
//	process->setOutputHorizonProperties(outputLayer, "Tmap");
//
//	bool success = process->compute(10, 33, 20);
//	if (success) {
//		float* isoTab = isochrone->getTab();
//		outputLayer->writeProperty(isoTab, FixedLayerFromDataset::ISOCHRONE);
//
//		m_currentManager->addFixedLayerFromDataset(outputLayer);
//		delete isochrone;
//	}
//}

//void GeotimeGraphicsView::createRGBVolumeFromCubeAndXt() {
//	SeismicSurvey* survey = getSurvey();
//	if (survey==nullptr) {
//		return;
//	}
//
//	QStringList xtPaths;
//	QStringList xtNames;
//	QString seachDir = survey->idPath() + "/ImportExport/IJK/";
//
//	if (QDir(seachDir).exists()) {
//		QFileInfoList infoList = QDir(seachDir).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Readable);
//		for (const QFileInfo& fileInfo : infoList) {
//			QDir dir(fileInfo.absoluteFilePath());
//			if(dir.cd("cubeRgt2RGB")) {
//
//				QFileInfoList xtInfoList = dir.entryInfoList(QStringList() << "xt_*xt", QDir::Files | QDir::Readable);
//				for (const QFileInfo& xtInfo : xtInfoList) {
//					xtPaths << xtInfo.absoluteFilePath();
//					xtNames << xtInfo.fileName();
//				}
//			}
//		}
//	}
//	bool isValid = xtNames.count()>0;
//	bool errorLogging = false;
//	int xtIndex = 0;
//	if (isValid) {
//		QString selectedRgb2Name = QInputDialog::getItem(nullptr, "Select Xt", "Xt File", xtNames);
//
//		while (xtIndex<xtNames.count() && selectedRgb2Name.compare(xtNames[xtIndex])!=0) {
//			xtIndex++;
//		}
//		isValid = xtIndex<xtNames.count();
//	} else {
//		QMessageBox::information(nullptr, "Layer Creation", "Failed to find any Xt data");
//		errorLogging = true;
//	}
//
//	if (isValid) {
//		QString name = QFileInfo(xtPaths[xtIndex]).baseName();
//		QString path = xtPaths[xtIndex];
//		Seismic3DDataset* dataset = new Seismic3DDataset(survey, name, survey->workingSetManager(), Seismic3DAbstractDataset::Seismic,
//				path);
//		dataset->loadFromXt(path.toStdString(), 3);
//
//		SmDataset3D d3d(path.toStdString(), 3);
//		dataset->setIJToInlineXlineTransfo(d3d.inlineXlineTransfo());
//		dataset->setIJToInlineXlineTransfoForInline(
//				d3d.inlineXlineTransfoForInline());
//		dataset->setIJToInlineXlineTransfoForXline(
//				d3d.inlineXlineTransfoForXline());
//		dataset->setSampleTransformation(d3d.sampleTransfo());
//
//		RgbDataset* rgbDataset;
//		if (dataset->dimV()!=d3d.dimV()) {
//			rgbDataset = nullptr;
//		} else {
//			rgbDataset = RgbDataset::createRgbDataset(QString("Rgb ")+name,
//					dataset, 0, dataset, 1, dataset, 2, nullptr, 0,
//					survey->workingSetManager());
//		}
//		if (rgbDataset!=nullptr) {
//			survey->addDataset(dataset);
//			survey->workingSetManager()->addRgbDataset(rgbDataset);
//		} else {
//			dataset->deleteLater();
//		}
//	}
//
////	// get dataset reference
////	QStringList datasetsNames;
////	QList<Seismic3DAbstractDataset*> datasets;
////
////	// fill lists
////	for (Seismic3DAbstractDataset* data : survey->datasets()) {
////		if ((dynamic_cast<Seismic3DDataset*>(data)!=nullptr)) {
////			datasets.push_back(data);
////			datasetsNames.push_back(data->name());
////		}
////	}
////
////	QString title = tr("Select Dataset");
////	StringSelectorDialog dialog(&datasetsNames, title);
////
////	int code = dialog.exec();
////
////	Seismic3DAbstractDataset* outDataset = nullptr;
////	if (code == QDialog::Accepted && dialog.getSelectedIndex()>=0) {
////		outDataset = datasets[dialog.getSelectedIndex()];
////	}
////	QString cubeFile;
////
////	QString searchDir;
////	if (Seismic3DDataset* data = dynamic_cast<Seismic3DDataset*>(outDataset)) {
////		// get xt file
////		QFileInfo fileInfo(QString::fromStdString(data->path()));
////		searchDir = fileInfo.absoluteDir().absolutePath();
////
////		QString seismicName;
////		QString seismicPathNoExt = fileInfo.completeBaseName();
////		QString seismicDirPath = fileInfo.absoluteDir().absolutePath();
////		QString ext = fileInfo.suffix();
////
////		QString desc_filename = seismicDirPath + "/" + seismicPathNoExt + ".desc";
////		FILE *pfile = fopen(desc_filename.toStdString().c_str(), "r");
////		int ok = 0;
////		if ( pfile != NULL && ext.compare("xt") == 0 )
////		{
////			char buff[1000];
////			fgets(buff, 10000, pfile);
////			fgets(buff, 10000, pfile);
////			fgets(buff, 10000, pfile);
////			buff[0] = 0; fscanf(pfile, "name=%s\n", buff);
////			fclose(pfile);
////			QString tmp = QString(buff);
////			if ( !tmp.isEmpty() )
////			{
////				seismicName = QString(tmp);
////				ok = 1;
////			}
////		}
////		if (ok==0) {
////			QStringList splitDot = fileInfo.fileName().split(".");
////			if (splitDot.size()>2) {
////				seismicName = splitDot[1];
////			} else {
////				seismicName = splitDot[0];
////			}
////		}
////
////		searchDir += "/../../ImportExport/IJK/" + seismicName + "/cubeRgt2RGB/";
////
////
////		cubeFile = QFileDialog::getOpenFileName(nullptr, tr("Iso file"), searchDir, "*.xt");
////
////		if (!cubeFile.isNull() && !cubeFile.isEmpty()) {
////			QString name = QFileInfo(cubeFile).baseName();
////			QString path = cubeFile;
////			Seismic3DDataset* dataset = new Seismic3DDataset(m_survey, name, m_survey->workingSetManager(), Seismic3DAbstractDataset::Seismic,
////					path);
////			dataset->loadFromXt(path.toStdString(), 3);
////
////			SmDataset3D d3d(path.toStdString(), 3);
////			dataset->setIJToInlineXlineTransfo(d3d.inlineXlineTransfo());
////			dataset->setIJToInlineXlineTransfoForInline(
////					d3d.inlineXlineTransfoForInline());
////			dataset->setIJToInlineXlineTransfoForXline(
////					d3d.inlineXlineTransfoForXline());
////			dataset->setSampleTransformation(d3d.sampleTransfo());
////
////			RgbDataset* rgbDataset;
////			if (dataset->dimV()!=d3d.dimV()) {
////				rgbDataset = nullptr;
////			} else {
////				rgbDataset = RgbDataset::createRgbDataset(QString("Rgb ")+name,
////						dataset, 0, dataset, 1, dataset, 2, nullptr, 0,
////						m_survey->workingSetManager());
////			}
////			if (rgbDataset!=nullptr) {
////				m_survey->addDataset(dataset);
////				m_survey->workingSetManager()->addRgbDataset(rgbDataset);
////			} else {
////				dataset->deleteLater();
////			}
////		}
////	}
//}

Seismic3DAbstractDataset* GeotimeGraphicsView::getDataset(bool onlyCpu) {
	QStringList datasetsNames;
	QList<Seismic3DAbstractDataset*> datasets;

	// fill lists
	for (IData* surveyData : m_currentManager->folders().seismics->data()) {
		if (SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(surveyData)) {
			for (Seismic3DAbstractDataset* dataset : survey->datasets()) {
				if ((onlyCpu && dynamic_cast<Seismic3DDataset*>(dataset)!=nullptr) || !onlyCpu) {
					datasets.push_back(dataset);
					datasetsNames.push_back(dataset->name());
				}
			}
		}
	}

	QString title = tr("Select Dataset");
	StringSelectorDialog dialog(&datasetsNames, title);

	int code = dialog.exec();

	Seismic3DAbstractDataset* outDataset = nullptr;
	if (code == QDialog::Accepted && dialog.getSelectedIndex()>=0) {
		outDataset = datasets[dialog.getSelectedIndex()];
	}
	return outDataset;
}

SeismicSurvey* GeotimeGraphicsView::getSurvey() {
	QStringList surveysNames;
	QList<SeismicSurvey*> surveys;

	// fill lists
	for (IData* surveyData : m_currentManager->folders().seismics->data()) {
		if (SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(surveyData)) {
			surveys.push_back(survey);
			surveysNames.push_back(survey->name());
		}
	}

	SeismicSurvey* outSurvey = nullptr;
	if (surveys.count()==1) {
		outSurvey = surveys[0];
	} else if (surveys.count()>1) {
		QString title = tr("Select Survey");
		StringSelectorDialog dialog(&surveysNames, title);

		int code = dialog.exec();

		if (code == QDialog::Accepted && dialog.getSelectedIndex()>=0) {
			outSurvey = surveys[dialog.getSelectedIndex()];
		}
	}
	return outSurvey;
}

void GeotimeGraphicsView::setDataSelectorDialog(DataSelectorDialog* dialog) {
	m_dataSelectorDialog = dialog;

	if(m_currentManager != nullptr){
	   m_currentManager->setManagerWidget(dialog->getSelector());
	}
}

void GeotimeGraphicsView::manageSynchro() {
	bool managerDefined = m_stackSynchronizer!=nullptr;

	StackSynchronizerDialog dialog(m_currentManager, this);
	if (managerDefined) {
		dialog.setSynchronizer(m_stackSynchronizer.get());
	}
	int code = dialog.exec();
	if (code && !managerDefined) {
		m_stackSynchronizer.reset(dialog.newSynchronizer());
	}
}

// This function will have a lot of overhead by it should not cause slow down issues
// Could be simpler by moving some functions from LayerSpectrumDialog in GeotimeGraphicsView
void GeotimeGraphicsView::selectLayerTypedData(IData* pData) {
	QVector<AbstractInnerView*> innerViews = getInnerViews();

	QString titleName("RGB");

	QVector<AbstractInnerView*> selectedViews;

	for (std::size_t index = 0; index<innerViews.size(); index++) {
		bool validView = (dynamic_cast<StackBaseMapView*>(innerViews[index]) != nullptr)
						&& (dynamic_cast<StackBaseMapView*>(innerViews[index])->getBaseTitle().compare(titleName) == 0);

		validView = validView || (dynamic_cast<ViewQt3D*>(innerViews[index]) != nullptr);
		validView = validView || (dynamic_cast<AbstractSectionView*>(innerViews[index]) != nullptr);
		validView = validView || (dynamic_cast<RandomLineView*>(innerViews[index]) != nullptr);
		if (validView){
			selectedViews.push_back(innerViews[index]);
		}
	}

	for (std::size_t index = 0; index<selectedViews.size(); index++) {
		QInnerViewTreeWidgetItem* rootItem = getItemFromView(selectedViews[index]);

		QStack<QTreeWidgetItem*> stack;
		stack.push(rootItem);

		QGraphicsRepTreeWidgetItem* itemData = nullptr;

		while (stack.size()>0 && itemData==nullptr) {
			QTreeWidgetItem* item = stack.pop();

			std::size_t N = item->childCount();
			for (std::size_t index=0; index<N; index++) {
				stack.push(item->child(index));
			}
			QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
			if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
				const IData* data = _item->getRep()->data();
				if (data!=nullptr && data==pData) {
					itemData = _item;
				}
			}
		}

		if (itemData!=nullptr) {
			itemData->setCheckState(0, Qt::Checked);
		}
	}
}

void GeotimeGraphicsView::closeEvent(QCloseEvent *event)
{
	m_properties->saveIni();
	GraphicToolsWidget::closePalette();
	NurbsWidget::closeWidget();
}


void GeotimeGraphicsView::show3dTools(bool b)
{
	m_tools3D->show();
}

void GeotimeGraphicsView::computeReflectivity() {
	ComputeReflectivityWidget* widget = new ComputeReflectivityWidget(m_currentManager->folders().wells);
	widget->show();
}

void GeotimeGraphicsView::setDepthLengthUnit(const MtLengthUnit* unit) {
	if ((*m_depthLengthUnit)!=(*unit)) {
		m_depthLengthUnit = unit;

		if (*m_depthLengthUnit==MtLengthUnit::METRE) {
			m_depthUnitButton->setIcon(QIcon(":/slicer/icons/regle_m128_blanc.png"));
		} else if (*m_depthLengthUnit==MtLengthUnit::FEET) {
			m_depthUnitButton->setIcon(QIcon(":/slicer/icons/regle_ft128_blanc.png"));
		}

		QVector<AbstractInnerView *> views = innerViews();
		for (int i=0; i<views.size(); i++) {
			AbstractInnerView* view = views[i];
			IDepthView* depthView = dynamic_cast<IDepthView*>(view);
			if (depthView) {
				depthView->setDepthLengthUnit(m_depthLengthUnit);
			}
		}

		emit depthLengthUnitChanged(m_depthLengthUnit);
	}
}

void GeotimeGraphicsView::toggleDepthLengthUnit() {
	if (*m_depthLengthUnit==MtLengthUnit::METRE) {
		setDepthLengthUnit(&MtLengthUnit::FEET);
	} else if (*m_depthLengthUnit==MtLengthUnit::FEET) {
		setDepthLengthUnit(&MtLengthUnit::METRE);
	}
}

void GeotimeGraphicsView::setProcessRelay(ProcessRelay* relay) {
	m_processRelay = relay;
	if (m_processRelay) {
		connect(m_processRelay, &ProcessRelay::processAdded, this, &GeotimeGraphicsView::processDataAddedRelay);

		// init
		std::map<std::size_t, IComputationOperator*> relay = m_processRelay->data();
		for (auto it=relay.begin(); it!=relay.end(); it++) {
			processDataAddedRelay(it->first, it->second);
		}
	}
}

void GeotimeGraphicsView::processDataAddedRelay(long id, IComputationOperator* obj) {
	if (IVolumeComputationOperator* op = dynamic_cast<IVolumeComputationOperator*>(obj)) {
		ComputationOperatorDataset* dataset = new ComputationOperatorDataset(op, m_currentManager);
		m_currentManager->addComputationOperatorDataset(dataset);

		ComputationDatasetRelayConnectionCloser* connectionCloser = new ComputationDatasetRelayConnectionCloser(
				m_currentManager, dataset, m_processRelay, m_processRelay);
	}
}

//void GeotimeGraphicsView::addDataFromComputation() {
//	if (!m_processRelay) {
//		return;
//	}
//
//	std::map<std::size_t, IComputationOperator*> relay = m_processRelay->data();
//
//	fprintf(stderr, "relay.size() >> %d\n", relay.size());
//
//	/* TEST
//	 *
//	QList<IData*> datas = m_currentManager->folders().seismics->data();
//	for (int i=0; i<datas.size(); i++) {
//		SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(datas[i]);
//		if (survey) {
//			QList<Seismic3DAbstractDataset*> datasets = survey->datasets();
//			for (int j=0; j<datasets.size(); j++) {
//				Seismic3DDataset* dataset = dynamic_cast<Seismic3DDataset*>(datasets[j]);
//				if (dataset) {
//					relay[1000+i*100+j] = new ToTest(dataset, dataset);
//				}
//			}
//		}
//	}
//
//
//	if (relay.size()==0) {
//		return;
//	}
//	*/
//
//	QStringList processNames;
//	for (auto it = relay.begin(); it!=relay.end(); it++) {
//		processNames.append(it->second->name());
//	}
//
//	StringSelectorDialog dialog(&processNames, "Compatible Processes");
//	int code = dialog.exec();
//	int idx = dialog.getSelectedIndex();
//	if (code!=QDialog::Accepted || idx>=relay.size()) {
//		return;
//	}
//
//
//	auto it = relay.begin();
//	std::advance(it, idx);
//
//
//	if (IVolumeComputationOperator* op = dynamic_cast<IVolumeComputationOperator*>(it->second)) {
//		ComputationOperatorDataset* dataset = new ComputationOperatorDataset(op, m_currentManager);
//		m_currentManager->addComputationOperatorDataset(dataset);
//
//		ComputationDatasetRelayConnectionCloser* connectionCloser = new ComputationDatasetRelayConnectionCloser(
//				m_currentManager, dataset, m_processRelay, m_processRelay);
//	}
//}



// ======================= horizons

void GeotimeGraphicsView::freeHorizonChangeColor(QTreeWidgetItem *item)
{
	if ( item == nullptr ) return;
	WorkingSetManager* manager = m_currentManager;
	QList<IData*> list = manager->folders().horizonsFree->data();
	FixedAttributImplFreeHorizonFromDirectories *data = nullptr;
	for (int n=0; n<list.size(); n++)
	{
		data = dynamic_cast <FixedAttributImplFreeHorizonFromDirectories*>(list[n]);
		if ( data == nullptr ) continue;
		QString name0 = data->dirName();
		if ( name0.compare(item->text(0)) == 0 )
		{
			 QColorDialog dialog;
			 dialog.setCurrentColor(data->getHorizonColor());
			 dialog.setOption (QColorDialog::DontUseNativeDialog);
			 if (dialog.exec() == QColorDialog::Accepted)
			 {
				 QColor color = dialog.currentColor();
				 data->setHorizonColor(color);
				// item->setTextColor(0, color);
				 item->setForeground(0,QBrush(color));
			 }
			break;
		}
	}
	qDebug() <<item->text(0);
}

QTreeWidgetItem * GeotimeGraphicsView::getItemFromTreeWidget(QTreeWidgetItem *item0, QString name)
{
	int idx = 0;
	int count = item0->childCount();
	QTreeWidgetItem *child = item0->child(idx);
	while ( idx<count-1 && child->text(0) != name )
	{
		idx++;
		child = item0->child(idx);
	}
	if ( !child || child->text(0) == name ) return child; // todo
	return nullptr;
}

void GeotimeGraphicsView::openSeismicInformation() {
	SeismicInformationAggregator* aggregator = new SeismicInformationAggregator(m_currentManager);
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}

void GeotimeGraphicsView::openHorizonInformation() {
	NextvisionHorizonInformationAggregator* aggregator = new NextvisionHorizonInformationAggregator(m_currentManager);
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}

void GeotimeGraphicsView::openIsoHorizonInformation() {
	IsoHorizonInformationAggregator* aggregator = new IsoHorizonInformationAggregator(m_currentManager);
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}

void GeotimeGraphicsView::openWellsInformation() {
	WellInformationAggregator* aggregator = new WellInformationAggregator(m_currentManager);
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}

void GeotimeGraphicsView::openPicksInformation() {
	PickInformationAggregator* aggregator = new PickInformationAggregator(m_currentManager);
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}

void GeotimeGraphicsView::openNurbsInformation() {
	NurbInformationAggregator* aggregator = new NurbInformationAggregator(m_currentManager);
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}

void GeotimeGraphicsView::openHorizonAnimationInformation() {
	HorizonAnimAggregator* aggregator = new HorizonAnimAggregator(m_currentManager);
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}

void GeotimeGraphicsView::openImportSismage() {
	ImportSismageHorizonDialog *p = new ImportSismageHorizonDialog(m_currentManager);
	p->show();
}

