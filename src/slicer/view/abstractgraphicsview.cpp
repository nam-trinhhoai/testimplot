#include "abstractgraphicsview.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <cmath>
#include <QDebug>
#include <QVBoxLayout>
#include <QMenu>
#include <QSplitter>
#include <QStack>
#include <QSpacerItem>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QItemSelectionModel>
#include <QPushButton>
#include <QToolButton>
#include <QLabel>
#include <QToolBar>
#include <QTabWidget>
#include <QHeaderView>
#include "slavesectionview.h"

#include "slicerwindow.h"
#include "workingsetmanager.h"
#include "mousetrackingevent.h"
#include "graphicsutil.h"
#include "qinnerviewtreewidgetitem.h"
#include "abstractinnerview.h"
#include "abstractgraphicrep.h"

#include "viewqt3d.h"
#include "basemapview.h"
#include "stackbasemapview.h"
#include "extendedbasemapview.h"
#include "dockwidgetsizegrid.h"
#include "singlesectionview.h"
#include "slavesectionview.h"
#include "randomlineview.h"
#include "wellbore.h"
#include "selectrandomcreationmode.h"
#include "randomrep.h"
#include "slicerep.h"
#include "qgraphicsreptreewidgetitem.h"
#include "cudaimagepaletteholder.h"
#include "nvtreewidget.h"

AbstractGraphicsView::AbstractGraphicsView(WorkingSetManager *factory,
		QString uniqueName, QWidget *parent) :
		KDDockWidgets::MainWindow(uniqueName, KDDockWidgets::MainWindowOption_None, parent) {
	setAttribute(Qt::WA_DeleteOnClose);
	setDockNestingEnabled(true);
	m_currentManager = factory;

	m_toolbar =addToolBar("Main Toolbar");
	m_toolbar->setMinimumHeight(40);
	m_toolbar->setStyleSheet("background-color:#32414B;");

	//setCentralWidget(createLateralButtonWidget());
//	QWidget* lateralButtonWidget = createLateralButtonWidget();
//	QWidget* mainWidget = centralWidget();
//	if (mainWidget->layout()==nullptr) {
//		mainWidget->setLayout(new QHBoxLayout);
//	}
//	mainWidget->layout()->addWidget(lateralButtonWidget);

	m_parameterSplitter = new QSplitter(Qt::Orientation::Vertical, this);

	m_parametersControler = new KDDockWidgets::DockWidget(uniqueName+"_parametersControler");
	m_parametersControler->setWindowFlag(Qt::SubWindow,true);

	m_parametersControler->setOptions(KDDockWidgets::DockWidget::Option_NotClosable);
//	m_parametersControler->setFeatures(
//			QDockWidget::DockWidgetFeature::DockWidgetFloatable
//					| QDockWidget::DockWidgetFeature::DockWidgetMovable);

	QWidget * inner=new QWidget(m_parametersControler);

	QVBoxLayout *layout=new QVBoxLayout(inner);
	layout->setContentsMargins(0,0,0,0);
	layout->addWidget(m_parameterSplitter);

	QWidget * status=new QWidget(inner);
	QHBoxLayout *hlayout=new QHBoxLayout(status);
	hlayout->setContentsMargins(0,0,0,0);
	hlayout->addWidget(new QLabel(" "));
	DockWidgetSizeGrid * sizegrip=new DockWidgetSizeGrid(m_parametersControler);
	connect(sizegrip,SIGNAL(geometryChanged(const QRect &)),this,SLOT(geometryChanged(const QRect &)));
	hlayout->addWidget(sizegrip);
	layout->addWidget(status,0,Qt::AlignRight);
	m_parametersControler->setWidget(inner);

	addDockWidget(m_parametersControler, KDDockWidgets::Location_OnRight);
	m_parametersControler->setWindowTitle("Data View");

	m_parameterSplitter->setMinimumWidth(250);

	m_treeWidget = new NVTreeWidget(m_parameterSplitter);
	m_treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
	m_treeWidget->setHeaderLabel("");
	m_treeWidget->setHeaderHidden(true);
	// https://stackoverflow.com/questions/6625188/qtreeview-horizontal-scrollbar-problems
	m_treeWidget->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
	m_treeWidget->header()->setStretchLastSection(false);


	m_rootItem = m_treeWidget->invisibleRootItem();

	connect(m_treeWidget, SIGNAL(itemSelectionChanged()), this,
			SLOT(itemSelectionChanged()));

	m_model = m_treeWidget->model();
	connect(m_model,
			SIGNAL(
					dataChanged(const QModelIndex &, const QModelIndex &, const QVector<int> &)),
			this,
			SLOT(
					dataChanged(const QModelIndex &, const QModelIndex &, const QVector<int> &)));

	m_controlTabWidget = new QTabWidget;
	m_controlTabWidget->addTab(m_treeWidget, "View");

	m_parameterSplitter->addWidget(m_controlTabWidget);//m_treeWidget);
	m_parameterSplitter->addWidget(getPlaceHolderForUse());//new QWidget(this));

	connect(m_treeWidget, SIGNAL(customContextMenuRequested(const QPoint &)),this, SLOT(onCustomContextMenu(const QPoint &)));
	createToolbar();
}
void AbstractGraphicsView::geometryChanged(const QRect & geom)
{
	m_parametersControler->window()->setGeometry(geom);
}

void AbstractGraphicsView::geometryChanged(AbstractInnerView *newView,const QRect & geom)
{
	newView->window()->setGeometry(geom);
}

void AbstractGraphicsView::closeEvent(QCloseEvent *closeEvent) {
	emit isClosing(this);
	QWidget::closeEvent(closeEvent);
}
QWidget* AbstractGraphicsView::createLateralButtonWidget() {
	QWidget *paramWidget = new QWidget(this);
	paramWidget->setStyleSheet("background-color:#32414B;");
	paramWidget->setContentsMargins(0, 0, 0, 0);
	paramWidget->setMaximumWidth(20);
	paramWidget->setMinimumWidth(20);
	QVBoxLayout *buttonLayout = new QVBoxLayout(paramWidget);
	//buttonLayout->setMargin(0);
	buttonLayout->setContentsMargins(0,0,0,0);

	//Hide show toolBar
	QToolButton *paramButton = new QToolButton(paramWidget);
	buttonLayout->addWidget(paramButton, 0, Qt::AlignmentFlag::AlignTop);
	m_parameterAction = new QAction(QIcon(":slicer/icons/parameters.png"), "",
			this);
	m_parameterAction->setCheckable(true);
	m_parameterAction->setChecked(true);
	connect(m_parameterAction, SIGNAL(triggered()), this,
			SLOT(showParameters()));
	paramButton->setDefaultAction(m_parameterAction);

	//Tile the display
	QToolButton *orgWindowButton = new QToolButton(paramWidget);
	buttonLayout->addWidget(orgWindowButton, 0, Qt::AlignmentFlag::AlignTop);
	QAction *orgWindowAction = new QAction(QIcon(":slicer/icons/viewers.png"),
			"", this);
	connect(orgWindowAction, SIGNAL(triggered()), this, SLOT(arrangeWindows()));
	orgWindowButton->setDefaultAction(orgWindowAction);

	buttonLayout->addSpacerItem(
			new QSpacerItem(0, 0, QSizePolicy::Minimum,
					QSizePolicy::Expanding));
	return paramWidget;
}

void AbstractGraphicsView::arrangeWindows() {
	GraphicsUtil::arrangeWindows();
}

void AbstractGraphicsView::showParameters() {
	if (m_parameterAction->isChecked())
		m_parametersControler->show();
	else
		m_parametersControler->hide();
}

QToolBar* AbstractGraphicsView::toolBar() const {
	return m_toolbar;
}

void AbstractGraphicsView::createToolbar() {

}

void AbstractGraphicsView::resetZoom() {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
		v->resetZoom();
}

bool AbstractGraphicsView::event(QEvent *event) {
	if (event->type() == MouseTrackingEvent::type()) {
		MouseTrackingEvent *myEvent = static_cast<MouseTrackingEvent*>(event);
		externalMouseMoved(myEvent);
		return true;
	}
	return QWidget::event(event);
}

void AbstractGraphicsView::externalMouseMoved(MouseTrackingEvent *event) {
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
		v->externalMouseMoved(event);
}

void AbstractGraphicsView::onCustomContextMenu(const QPoint &point) {
	QTreeWidgetItem *item = m_treeWidget->itemAt(point);
	if (item) {
		QMenu menu;
		AbstractGraphicRep *rep = item->data(0, Qt::UserRole).value<
				AbstractGraphicRep*>();
		if (rep != nullptr) {
			rep->buildContextMenu(&menu);
			if (menu.actions().size()>0) {
				menu.exec(m_treeWidget->mapToGlobal(point));
			}
		}
	}
}

void AbstractGraphicsView::dataChanged(const QModelIndex &topLeft,
		const QModelIndex &bottomRight, const QVector<int> &roles) {
	for (int i : roles) {
		if (i == Qt::CheckStateRole) {
			QVariant var = topLeft.data(Qt::UserRole);
			AbstractGraphicRep *rep = var.value<AbstractGraphicRep*>();
			if (rep == nullptr)
				return;

			if (topLeft.data(Qt::CheckStateRole).toBool()) {
				showRep(rep);
				showPropertyPanel(rep);
			} else {
				hideRep(rep);
				if (rep==m_lastPropPanelRep) {
					releasePropertyPanel();
				}
			}
		}
	}
}

void AbstractGraphicsView::hideRgtRep(AbstractGraphicRep *rep)
{
	bool isRgtRep = false;
	const SliceRep* pSliceRep = dynamic_cast<const SliceRep*>(rep);
	const RandomRep* pRandomRep = dynamic_cast<const RandomRep*>(rep);
	AbstractInnerView *view = nullptr;
	if(pSliceRep != nullptr){
		if(pSliceRep->name().toLower().contains("rgt")){
			isRgtRep = true;
			view = pSliceRep->view();
		}
	} else {
		if(pRandomRep != nullptr){
			if(pRandomRep->name().toLower().contains("rgt")){
				isRgtRep = true;
				view = pRandomRep->view();
			}
		}
	}

	if(isRgtRep == true){
		//for(AbstractInnerView *view:innerViews()){
		if ((view->viewType()==InlineView || view->viewType()==XLineView || view->viewType()==RandomView)) {
			std::size_t index = 0;
			QInnerViewTreeWidgetItem* rootItem = nullptr;
			while (index<m_rootItem->childCount() && rootItem==nullptr) {
				QTreeWidgetItem *e=m_rootItem->child(index);
				rootItem = dynamic_cast<QInnerViewTreeWidgetItem*>(e);
				if (rootItem!=nullptr && rootItem->innerView()!=view) {
					rootItem = nullptr;
				}
				index++;
			}

			QStack<QTreeWidgetItem*> stack;
			stack.push(rootItem);
			QGraphicsRepTreeWidgetItem *itemRgt = nullptr;
			const SliceRep* slicerep;
			const RandomRep* repRandom;
			QString name;
			while (stack.size()>0) {
				QTreeWidgetItem* item = stack.pop();

				std::size_t N = item->childCount();
				for (std::size_t index=0; index<N; index++) {
					stack.push(item->child(index));
				}

				QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);

				if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
					slicerep = dynamic_cast<const SliceRep*>(_item->getRep());
					repRandom = dynamic_cast<const RandomRep*>(_item->getRep());

					if (slicerep!=nullptr && slicerep->name().toLower().contains("rgt")) {
						itemRgt = _item;
						name = slicerep->name();
						//qDebug()<<name;
					} else if (repRandom!=nullptr && repRandom->name().toLower().contains("rgt")) {
						itemRgt = _item;
						name = repRandom->name();
						//qDebug()<<name;
					}
				}

				if(itemRgt!=nullptr){
					//qDebug() << name << rep->name();
					if (name != rep->name()) {
						itemRgt->setCheckState(0, Qt::Unchecked);
					}
					else{
						if (dynamic_cast<SliceRep*>(itemRgt->getRep())!=nullptr) {
							dynamic_cast<SliceRep*>(itemRgt->getRep())->image()->setOpacity(0.5);
						} else if (dynamic_cast<RandomRep*>(itemRgt->getRep())!=nullptr) {
							if (dynamic_cast<RandomRep*>(itemRgt->getRep())->image()!=nullptr) {
								dynamic_cast<RandomRep*>(itemRgt->getRep())->image()->setOpacity(0.5);
							}
						}
					}
				}
			}
		}
	}
}

void AbstractGraphicsView::showRep(AbstractGraphicRep *rep) {
	this->hideRgtRep(rep);
	rep->view()->showRep(rep);
}
void AbstractGraphicsView::hideRep(AbstractGraphicRep *rep) {
	rep->view()->hideRep(rep);
}
void AbstractGraphicsView::showPropertyPanel(AbstractGraphicRep *rep) {
	QWidget *parmPanel = new QWidget(this);
	QWidget *propertyPanel = rep->propertyPanel();
	if (propertyPanel != nullptr) {
		QVBoxLayout *parameterLayout = new QVBoxLayout(parmPanel);
		parameterLayout->addWidget(propertyPanel);
		parameterLayout->addSpacerItem(
				new QSpacerItem(0, 0, QSizePolicy::Minimum,
						QSizePolicy::Expanding));

		m_lastPropPanelRep = rep;
	}
	m_parameterSplitter->replaceWidget(1, parmPanel);
	releasePlaceHolder();
	parmPanel->show();
}

void AbstractGraphicsView::releasePropertyPanel() {
	QWidget* placeHolder = getPlaceHolderForUse();
	if (placeHolder!=m_parameterSplitter->widget(1)) {
		m_parameterSplitter->replaceWidget(1, placeHolder);
		m_lastPropPanelRep = nullptr;
	}
}

void AbstractGraphicsView::itemSelectionChanged() {
	QList<QTreeWidgetItem*> selectedItems = m_treeWidget->selectedItems();
	if (selectedItems.empty()) {
		releasePropertyPanel();
		return;
	}
	QTreeWidgetItem *current = selectedItems.first();
	QVariant var = current->data(0, Qt::UserRole);
	AbstractGraphicRep *rep = var.value<AbstractGraphicRep*>();
	if (rep == nullptr || (!current->data(0, Qt::CheckStateRole).toBool() && !current->data(0, Qt::CheckStateRole).isNull()) || rep->propertyPanel() == nullptr )
		releasePropertyPanel();
	else
		showPropertyPanel(rep);
}
QVector<AbstractInnerView*> AbstractGraphicsView::innerViews() const {
	QVector<AbstractInnerView*> views;
	for (int i = 0; i < m_rootItem->childCount(); i++) {
		if (QInnerViewTreeWidgetItem *it =
				dynamic_cast<QInnerViewTreeWidgetItem*>(m_rootItem->child(i)))
			views.push_back(it->innerView());
	}
	return views;
}

void AbstractGraphicsView::registerView(AbstractInnerView *newView) {
	// register in views synchro for slice change
	synchroMultiView.registerView( newView );

	//register controlers
	QVector<AbstractInnerView*> views = innerViews();
	QString viewName(
			(std::string("View ") + std::to_string(views.size())).c_str());
	newView->setViewIndex(views.size());

	addDockWidget(newView, KDDockWidgets::Location_OnRight);//,
//			Qt::Orientation::Horizontal);

//    // handle floating changes
//    QObject::connect(newView, &QDockWidget::topLevelChanged, [newView] (bool floating)
//    {
//        if (floating)
//        {
//            qDebug()<<"Start to float";
//        }
//    });

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

	connect(newView, SIGNAL(askGeometryChanged(AbstractInnerView * ,const QRect & )), this,
				SLOT(geometryChanged(AbstractInnerView *,const QRect & )));
}

void AbstractGraphicsView::subWindowActivated(AbstractInnerView *window) {
	for (int i = 0; i < m_rootItem->childCount(); i++) {
		if (QInnerViewTreeWidgetItem *it =
				dynamic_cast<QInnerViewTreeWidgetItem*>(m_rootItem->child(i))) {
			if (it->innerView() == window) {
				AbstractGraphicRep *rep = it->innerView()->lastRep();
				if (false/*rep != nullptr*/) {
					QTreeWidgetItem *t = QInnerViewTreeWidgetItem::findRepNode(
							rep, it);
					if (t != nullptr) {
						m_treeWidget->scrollToItem(t,
								QAbstractItemView::ScrollHint::PositionAtCenter);
					} else
						qDebug()
								<< "Should not happen!: not found a selected graphic rep in the tree";

				} else {
					if (it->childCount() == 0 || !it->isExpanded())
						m_treeWidget->scrollToItem(it,
								QAbstractItemView::ScrollHint::PositionAtCenter);
					else
						m_treeWidget->scrollToItem(
								it->child(it->childCount() / 2),
								QAbstractItemView::ScrollHint::PositionAtCenter);
				}
				break;
			}
		}
	}
}

AbstractInnerView* AbstractGraphicsView::generateView(ViewType viewType, bool restrictedToMonoTypeSplit) {
	QString newUniqueName = uniqueName() + "_view" + QString::number(getNewUniqueId());
	if (viewType == ViewType::InlineView || viewType == ViewType::XLineView) {

		AbstractInnerView* abstractInnerView;
		if(restrictedToMonoTypeSplit)
			abstractInnerView = new SlaveSectionView(restrictedToMonoTypeSplit,viewType, newUniqueName);
		else
			abstractInnerView = new SingleSectionView(restrictedToMonoTypeSplit,viewType, newUniqueName);


		//GS synchroMultiView.registerView( abstractInnerView );
		return abstractInnerView;
	} else if (viewType == ViewType::BasemapView) {

#if 1
		return new StackBaseMapView(restrictedToMonoTypeSplit,newUniqueName,eModeStandardView, this);
#else
		return new ExtendedBaseMapView(restrictedToMonoTypeSplit,newUniqueName,
				ExtendedBaseMapView::eTypeStackMode, this);
#endif
	} else if (viewType == ViewType::StackBasemapView) {

		return new StackBaseMapView(restrictedToMonoTypeSplit,newUniqueName,eModeStandardView, this);
	} else if (viewType == ViewType::View3D) {

		return new ViewQt3D(restrictedToMonoTypeSplit,newUniqueName);
	}


}

void AbstractGraphicsView::askToSplit(AbstractInnerView * toSplit,ViewType type, bool restictToMonoTypeSplitChild, Qt::Orientation orientation)
{
	AbstractInnerView *newView;
	if (type==ViewType::RandomView) {
		selectRandomActionWithUI(toSplit, restictToMonoTypeSplitChild, orientation);
	} else {
		newView = generateView(type,restictToMonoTypeSplitChild);
		askToSplitStep2(newView, toSplit, type, restictToMonoTypeSplitChild, orientation);
	}
}

void AbstractGraphicsView::askToSplitStep2(AbstractInnerView* newView, AbstractInnerView * toSplit,ViewType type, bool restictToMonoTypeSplitChild, Qt::Orientation orientation)
{
	bool isToSplitExist = false;
	QVector<AbstractInnerView*> existingViews = innerViews();
	std::size_t idx=0;
	std::size_t N = existingViews.size();
	while(!isToSplitExist && idx<N) {
		isToSplitExist = toSplit == existingViews[idx];
		idx++;
	}
	if (!isToSplitExist) {
		return;
	}
	registerView(newView);
//	splitDockWidget(toSplit, newView, orientation);
	KDDockWidgets::Location location = (orientation==Qt::Horizontal) ? KDDockWidgets::Location_OnRight : KDDockWidgets::Location_OnBottom;

	addDockWidget(newView, location, toSplit);

}
void AbstractGraphicsView::unregisterView(AbstractInnerView *toBeDeleted) {
	// register in views synchro for slice change
	synchroMultiView.unregisterView( toBeDeleted );

	//Remove the tree item
	for (int i = 0; i < m_rootItem->childCount(); i++) {
		if (QInnerViewTreeWidgetItem *it =
				dynamic_cast<QInnerViewTreeWidgetItem*>(m_rootItem->child(i))) {
			if (it->innerView() == toBeDeleted) {
				m_rootItem->removeChild(it);
				it->deleteLater();
				break;
			}
		}
	}

	//register controlers
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *existingView : views)
		unregisterWindowControlers(toBeDeleted, existingView);

	//Connect Event to be transmitted at the top level
	disconnect(toBeDeleted, SIGNAL(viewMouseMoved(MouseTrackingEvent *)), this,
			SLOT(innerViewMouseMoved(MouseTrackingEvent *)));
	disconnect(toBeDeleted, SIGNAL(controlerActivated(DataControler *)), this,
			SLOT(innerViewControlerActivated(DataControler *)));
	disconnect(toBeDeleted, SIGNAL(controlerDesactivated(DataControler *)),
			this, SLOT(innerViewControlerDesactivated(DataControler *)));

	//Add already registred controlers
	for (DataControler *c : m_externalControler)
		toBeDeleted->removeExternalControler(c);
}

void AbstractGraphicsView::innerViewControlerActivated(DataControler *c) {
	emit controlerActivated(c);
}
void AbstractGraphicsView::innerViewControlerDesactivated(DataControler *c) {
	emit controlerDesactivated(c);
}

void AbstractGraphicsView::innerViewMouseMoved(MouseTrackingEvent *event) {

	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views) {
		if (v != sender())
			v->externalMouseMoved(event);
	}
	emit viewMouseMoved(event);
}

void AbstractGraphicsView::registerWindowControlers(AbstractInnerView *newView,
		AbstractInnerView *existingView) {

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

void AbstractGraphicsView::unregisterWindowControlers(
		AbstractInnerView *toBeDeleted, AbstractInnerView *existingView) {
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

void AbstractGraphicsView::addExternalControler(DataControler *controler) {
	m_externalControler.push_back(controler);
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
		v->addExternalControler(controler);
}

void AbstractGraphicsView::removeExternalControler(DataControler *controler) {
	m_externalControler.removeOne(controler);
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
		v->removeExternalControler(controler);
}

QList<DataControler*> AbstractGraphicsView::getControlers() const {
	QList<DataControler*> result;
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views)
		result.append(v->getControlers());
	return result;
}

std::size_t AbstractGraphicsView::getNewUniqueId() {
	std::size_t out = m_uniqueId;
	m_uniqueId++;
	return out;

}

AbstractGraphicsView::~AbstractGraphicsView() {

}

void AbstractGraphicsView::changeViewName(AbstractInnerView* view, QString newName) {
	view->setDefaultTitle(newName);
	QInnerViewTreeWidgetItem* item = nullptr;
	long idx=0;
	while (idx<m_rootItem->childCount() && item==nullptr) {
		QInnerViewTreeWidgetItem* castItem = dynamic_cast<QInnerViewTreeWidgetItem*>(m_rootItem->child(idx));
		if (castItem && castItem->innerView()==view) {
			item = castItem;
		}
		idx++;
	}
	if (item) {
		item->setData(0, Qt::DisplayRole, newName);
	}
}

void AbstractGraphicsView::selectRandomActionWithUI(AbstractInnerView * toSplit, bool restictToMonoTypeSplitChild, Qt::Orientation orientation) {
	// create dialog
	SelectRandomCreationMode dialog(m_currentManager, this);
	int result = dialog.exec();

	if (result==QDialog::Accepted) {
		addRandomFromWellBore(dialog.selectedWellBores(), dialog.wellMargin(), toSplit, restictToMonoTypeSplitChild, orientation);
	}
}

void AbstractGraphicsView::addRandomFromWellBore(QList<WellBore*> wells, double margin, AbstractInnerView * toSplit, bool restictToMonoTypeSplitChild, Qt::Orientation orientation) {
	QPolygonF poly;
	bool isFirstSegmentSet = false;
	for (WellBore* well : wells) {
		const Deviations& deviations = well->deviations();
		for (std::size_t idx=0; idx<deviations.xs.size(); idx++) {
			poly << QPointF(deviations.xs[idx], deviations.ys[idx]);
			if (!isFirstSegmentSet && poly.size()>=2) {
				QPolygonF tmpPoly(poly);

				QPointF second = poly[1];
				QPointF first = poly[0];
				QPointF vect = first - second;
				double dist = std::sqrt(vect.x()*vect.x() + vect.y()*vect.y());
				if (dist!=0.0) {
					vect = vect / dist * margin;
				} else {
					vect = QPointF(-margin, 0);
				}
				QPointF marginPoint = first + vect;

				poly.clear();
				poly << marginPoint;
				poly << tmpPoly;
				isFirstSegmentSet = true;
			}
		}
	}
	if (poly.size()>1) {
		QPointF last = poly[poly.size()-1];
		QPointF beforeLast = poly[poly.size()-2];
		QPointF vect = last - beforeLast;
		double dist = std::sqrt(vect.x()*vect.x() + vect.y()*vect.y());
		if (dist!=0) {
			vect = vect / dist * margin;
		} else {
			vect = QPointF(margin, 0);
		}
		QPointF marginPoint = last + vect;
		poly << marginPoint;
	}
	RandomLineView* random = createRandomView(poly, toSplit, restictToMonoTypeSplitChild, orientation);
	random->setDisplayDistance(0.1);

	// change name
	QStringList wellNames;
	for (WellBore* well : wells) {
		wellNames << well->name();
	}
	QString realName = "rd : " + wellNames.join(", ");
	changeViewName(random, realName);
}

RandomLineView* AbstractGraphicsView::createRandomView(QPolygonF polygon, AbstractInnerView * toSplit, bool restictToMonoTypeSplitChild, Qt::Orientation orientation) {
	QString newUniqueName = uniqueName() + "_view" + QString::number(getNewUniqueId());
	RandomLineView* randomView = new RandomLineView(polygon, ViewType::RandomView, newUniqueName);
	askToSplitStep2(randomView, toSplit, ViewType::RandomView, restictToMonoTypeSplitChild, orientation);
	return randomView;
}

QWidget* AbstractGraphicsView::getPlaceHolderForUse() {
	if (m_placeHolderWidget==nullptr) {
		m_placeHolderWidget = new QWidget(this);
	}
	return m_placeHolderWidget;
}

void AbstractGraphicsView::releasePlaceHolder() {
	if (m_placeHolderWidget!=nullptr) {
		m_placeHolderWidget->deleteLater();
		m_placeHolderWidget = nullptr;
	}
}
