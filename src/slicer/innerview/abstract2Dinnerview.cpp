#include "abstract2Dinnerview.h"
#include <iomanip>
#include <sstream>
#include <iostream>

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGridLayout>
#include <QLabel>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QToolButton>
#include <QAction>
#include <QDir>

#include "GraphicLayerSelectorDialog.h"
#include "qglscalebaritem.h"
#include "qglcrossitem.h"
#include "qgllineitem.h"
#include "baseqglgraphicsview.h"
#include "graphiclayer.h"
#include "imouseimagedataprovider.h"
#include "mousetrackingevent.h"
#include "idatacontrolerholder.h"
#include "idatacontrolerprovider.h"
#include "abstractgraphicrep.h"
#include "pickingtask.h"
#include "idata.h"
#include "statusbar.h"
#include "GraphicSceneEditor.h"
#include "slicerep.h"
#include "GraphicToolsWidget.h"
#include "SaveGraphicLayerDialog.h"
#include "geotimegraphicsview.h"
#include "workingsetmanager.h"
#include "folderdata.h"
#include "seismicsurvey.h"
#include "GraphEditor_Item.h"
#include "singlesectionview.h"
#include "iGraphicToolDataControl.h"
#include "iRepGraphicItem.h"
#include "idata.h"
#include "wellheadreponmap.h"
#include "multiseedslicelayer.h"
#include "wellborelayeronslice.h"
#include "multiseedrandomlayer.h"
#include "wellborelayeronrandom.h"
#include "GraphEditor_LineShape.h"
#include "abstractgraphicsview.h"
#include "GraphEditor_RegularBezierPath.h"
#include "GraphEditor_ListBezierPath.h"
#include "GraphEditor_LineShape.h"

#include "mtlengthunit.h"


int Abstract2DInnerView::PICKING_ITEM_Z = 1000;
int Abstract2DInnerView::SCALE_ITEM_Z = 2000;
int Abstract2DInnerView::CROSS_ITEM_Z = 3000;
int Abstract2DInnerView::COURBE_ITEM_Z = 100;
int Abstract2DInnerView::DATA_ITEM_Z = 1;

int Abstract2DInnerView::HORIZONTAL_AXIS_SIZE = 30;
int Abstract2DInnerView::VERTICAL_AXIS_SIZE = 80;

int Abstract2DInnerView::getPickingItemZ() {
	return PICKING_ITEM_Z;
}

Abstract2DInnerView::Abstract2DInnerView(bool restictToMonoTypeSplit,BaseQGLGraphicsView *view,
		BaseQGLGraphicsView *verticalAxisView,
		BaseQGLGraphicsView *horizontalAxisView, QString uniqueName,eModeView typeView, KDDockWidgets::MainWindow * geoTimeView) :
																						AbstractInnerView(restictToMonoTypeSplit,uniqueName,typeView) {
	m_scaleItem = nullptr;
	m_depthLengthUnit = &MtLengthUnit::METRE;
	m_GeoTimeView = geoTimeView;
	m_scene = new GraphicSceneEditor(this,this);
	m_view = view;
	m_view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	m_view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	m_wordBoundsInitialized = false;
	m_worldBounds = QRectF(0, 0, 1000, 1000);

	m_view->setScene(m_scene);
	m_view->setMinimumSize(100 + VERTICAL_AXIS_SIZE,100 + HORIZONTAL_AXIS_SIZE);
	m_view->setStyleSheet("border: 0px solid;");

	//Vertical Axis
	m_verticalAxisView = verticalAxisView;
	m_verticalAxisView->lockZoom(true);
	m_verticalAxisView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	m_verticalAxisView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	m_verticalAxisView->setStyleSheet("min-width: " + QString::number(VERTICAL_AXIS_SIZE)	+ "px;max-width: " + QString::number(VERTICAL_AXIS_SIZE)+ "px;border: 0px solid;");
	m_verticalAxisScene = new QGraphicsScene();
	m_verticalAxisView->setScene(m_verticalAxisScene);
	m_verticalAxisView->setMinimumSize(VERTICAL_AXIS_SIZE, 100);

	//Horizontal axis
	m_horizontalAxisView = horizontalAxisView;
	m_horizontalAxisView->lockZoom(true);
	m_horizontalAxisView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	m_horizontalAxisView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	m_horizontalAxisView->setStyleSheet("min-height: " + QString::number(HORIZONTAL_AXIS_SIZE)+ "px;max-height: " + QString::number(HORIZONTAL_AXIS_SIZE)	+ "px;border: 0px solid;");
	m_horizontalAxisScene = new QGraphicsScene();
	m_horizontalAxisView->setScene(m_horizontalAxisScene);
	m_horizontalAxisView->setMinimumSize(100, HORIZONTAL_AXIS_SIZE);

	//Define a transparent style to not show scroll bar.
	QFile f(":/qdarkstyle/transScroll.qss");
	f.open(QFile::ReadOnly | QFile::Text);
	QTextStream ts(&f);
	QString style = ts.readAll();
	m_horizontalAxisView->verticalScrollBar()->setStyleSheet(style);
	m_verticalAxisView->horizontalScrollBar()->setStyleSheet(style);
	f.close();

	m_verticalAxisView->setTransformationAnchor(QGraphicsView::ViewportAnchor::NoAnchor);
	m_horizontalAxisView->setTransformationAnchor(QGraphicsView::ViewportAnchor::NoAnchor);

	//setup the view
	QWidget *plotWidget = new QWidget(this);
	plotWidget->setMinimumSize(200, 200);
	plotWidget->setContentsMargins(0, 0, 0, 0);

	m_box = new QGridLayout(plotWidget);
	m_box->setHorizontalSpacing(0);
	m_box->setVerticalSpacing(0);
	//m_box->setMargin(0);
	m_box->setContentsMargins(0, 0, 0, 0);
	m_boxTopCornerPreviousWidget = new QLabel("");
	m_box->addWidget(m_boxTopCornerPreviousWidget, 0, 0);
	m_box->addWidget(m_horizontalAxisView, 0, 1);
	m_box->addWidget(m_verticalAxisView, 1, 0);
	m_box->addWidget(m_view, 1, 1, -1, -1);

	QWidget *mainWidget = new QWidget(this);
	m_mainLayout = new QVBoxLayout(mainWidget);
	//m_mainLayout->setMargin(0);
	m_mainLayout->setContentsMargins(0,0,0,0);

	m_statusBar = new StatusBar(mainWidget);
	m_mainLayout->addWidget(plotWidget);


	QWidget *status = new QWidget(this);
	QHBoxLayout *hlayout = new QHBoxLayout(status);
	hlayout->setContentsMargins(0, 0, 0, 0);
	hlayout->addWidget(m_statusBar,1);
	hlayout->addWidget(generateSizeGrip(),0,Qt::AlignRight);

	m_mainLayout->addWidget(status);

	setWidget(mainWidget);

	m_scaleItem = new QGLScaleBarItem(m_worldBounds);
	m_scaleItem->setZValue(SCALE_ITEM_Z);
	m_scene->addItem(m_scaleItem);

	m_crossHairItem = new QGLCrossItem(m_worldBounds);
	m_crossHairItem->setZValue(CROSS_ITEM_Z);
	m_scene->addItem(m_crossHairItem);



	connect(m_view,
			SIGNAL(
					mouseMoved(double ,double ,Qt::MouseButton ,Qt::KeyboardModifiers )),
					this,
					SLOT(
							mouseMoved(double ,double ,Qt::MouseButton ,Qt::KeyboardModifiers )));
	connect(m_view,
			SIGNAL(
					mousePressed(double ,double ,Qt::MouseButton ,Qt::KeyboardModifiers )),
					this,
					SLOT(
							mousePressed(double ,double ,Qt::MouseButton ,Qt::KeyboardModifiers )));

	connect(m_view,
			SIGNAL(
					mouseRelease(double ,double ,Qt::MouseButton ,Qt::KeyboardModifiers )),
					this,
					SLOT(
							mouseRelease(double ,double ,Qt::MouseButton ,Qt::KeyboardModifiers )));

	connect(m_view,
			SIGNAL(
					mouseDoubleClick(double ,double ,Qt::MouseButton ,Qt::KeyboardModifiers )),
					this,
					SLOT(
							mouseDoubleClick(double ,double ,Qt::MouseButton ,Qt::KeyboardModifiers )));
	connect(m_view, &BaseQGLGraphicsView::contextMenu, this, &Abstract2DInnerView::contextMenu);


	//For the scale synchronization
	connect(m_view, SIGNAL(scaleChanged(double , double )), this,
			SLOT(scaleChanged(double , double )));

	connect(m_view->verticalScrollBar(), SIGNAL(valueChanged(int)),
			m_verticalAxisView->verticalScrollBar(), SLOT(setValue(int)));

	connect(m_view->horizontalScrollBar(), SIGNAL(valueChanged(int)),
			m_horizontalAxisView->horizontalScrollBar(), SLOT(setValue(int)));
	connnectScrollBarToExternalViewAreaChangedSignal();
}


QToolButton* Abstract2DInnerView::createToogleBarButton(const QString &iconPath,
		const QString &tooltip) const {

	QToolButton * pickButton = new QToolButton();
	QAction *pickAction = new QAction(QIcon(iconPath),tr(""), pickButton);
	pickAction->setCheckable(true);
	pickButton->setDefaultAction(pickAction);
	pickButton->setIconSize(QSize(8, 8));
	pickButton->setToolTip(tooltip);
	pickButton->setStyleSheet("background-color:#32414B;");
	return pickButton;
}

void Abstract2DInnerView::showRandomView(bool isOrtho,QVector<QPointF>  listepoints)
{
	if(m_viewType == ViewType::StackBasemapView)
		emit signalRandomView(isOrtho,listepoints);
}

void Abstract2DInnerView::showRandomView(bool isOrtho,GraphEditor_LineShape* line, RandomLineView * randomOrtho,QString name)
{
	if(m_viewType == ViewType::StackBasemapView)
		emit signalRandomView(isOrtho,line,randomOrtho,name);
}


void Abstract2DInnerView::randomLineDeleted(RandomLineView* random)
{
	if(m_viewType == ViewType::StackBasemapView)
		emit signalRandomViewDeleted(random);
}

void Abstract2DInnerView::setNurbsPoints(QVector<PointCtrl> listeCtrls,GraphEditor_ListBezierPath* path , QVector<QPointF>  listepoints,QString name,bool isopen, bool withTangent,QColor col)
{
	if(m_viewType == ViewType::StackBasemapView)
			emit addNurbsPoints(listepoints,withTangent,path,name,col);
	if(m_viewType == ViewType::RandomView)
	{
	//	emit addCrossPoints(listeCtrls, listepoints,isopen,cross);
		emit addCrossPoints(path);
	}
}


void Abstract2DInnerView::setNurbsPoints(QVector<QPointF>  listepoints,bool isopen,bool withTangent,QColor col)
{

	if(m_viewType == ViewType::StackBasemapView)
		emit addNurbsPoints(listepoints,withTangent,nullptr,"",col);
	if(m_viewType == ViewType::InlineView)
		emit addCrossPoints(listepoints,isopen);
	if(m_viewType == ViewType::RandomView)
		emit addCrossPoints(listepoints,isopen);
}

void Abstract2DInnerView::refreshNurbsPoints(QVector<QPointF>  listepoints,bool isopen,bool withTangent,QColor col)
{
	if(m_viewType == ViewType::StackBasemapView)
		emit updateNurbsPoints(listepoints,withTangent,col);
	if(m_viewType == ViewType::InlineView)
		emit addCrossPoints(listepoints,isopen);
	if(m_viewType == ViewType::RandomView)
		emit addCrossPoints(listepoints,isopen);
}

void Abstract2DInnerView::refreshNurbsPoints(QVector<PointCtrl> listeCtrls,QVector<QPointF>  listepoints,bool isopen,bool withTangent,QPointF cross,QColor col)
{
	if(m_viewType == ViewType::StackBasemapView)
		emit updateNurbsPoints(listepoints,withTangent,col);
	if(m_viewType == ViewType::InlineView)
		emit addCrossPoints(listepoints,isopen);
	if(m_viewType == ViewType::RandomView)
		emit addCrossPoints(listeCtrls,listepoints,isopen,cross);
}

void Abstract2DInnerView::refreshNurbsPoints(GraphEditor_ListBezierPath* path,QColor col)
{
	if(m_viewType == ViewType::StackBasemapView)
		emit updateNurbsPoints(path,col);

	if(m_viewType == ViewType::RandomView)
		emit addCrossPoints(path);
	//qDebug()<<" refresh nurbs points ...... ";
	/*if(m_viewType == ViewType::InlineView)
		emit addCrossPoints(listepoints,isopen);
	if(m_viewType == ViewType::RandomView)
		emit addCrossPoints(listeCtrls,listepoints,isopen,cross);*/
}

void Abstract2DInnerView::deleteGeneratrice(QString name)
{
	if(m_viewType == ViewType::RandomView)
		emit deletedGeneratrice(name);
}


void Abstract2DInnerView::connnectScrollBarToExternalViewAreaChangedSignal() {
	connect(m_view->horizontalScrollBar(), SIGNAL(valueChanged(int)), this,
			SLOT(scrollBarPosChanged()));

	connect(m_view->verticalScrollBar(), SIGNAL(valueChanged(int)), this,
			SLOT(scrollBarPosChanged()));
}

void Abstract2DInnerView::disconnnectScrollBarToExternalViewAreaChangedSignal() {
	disconnect(m_view->horizontalScrollBar(), SIGNAL(valueChanged(int)), this,
			SLOT(scrollBarPosChanged()));

	disconnect(m_view->verticalScrollBar(), SIGNAL(valueChanged(int)), this,
			SLOT(scrollBarPosChanged()));
}

StatusBar* Abstract2DInnerView::statusBar() const {
	return m_statusBar;
}

void Abstract2DInnerView::resizeEvent(QResizeEvent *resizeEvent) {
	AbstractInnerView::resizeEvent(resizeEvent);
	m_verticalAxisScene->update();
	m_horizontalAxisScene->update();
	//m_scene->update();
}

void Abstract2DInnerView::scrollBarPosChanged() {
	emit viewAreaChanged(m_view->mapToScene(m_view->viewport()->rect()));
}

void Abstract2DInnerView::scaleChanged(double sx, double sy) {
	m_verticalAxisView->scale(1, sy);
	m_horizontalAxisView->scale(sx, 1);
	//Cool way to resynchronize!
	m_horizontalAxisView->horizontalScrollBar()->setValue(m_view->horizontalScrollBar()->value());
	m_verticalAxisView->verticalScrollBar()->setValue(m_view->verticalScrollBar()->value());

	emit viewAreaChanged(m_view->mapToScene(m_view->viewport()->rect()));
}

void Abstract2DInnerView::resetZoom() {
	std::pair<float, float> r = m_view->resetZoom();
	BaseQGLGraphicsView::resetScale(m_verticalAxisView);
	BaseQGLGraphicsView::resetScale(m_horizontalAxisView);

	m_verticalAxisView->scale(1, r.second);
	m_horizontalAxisView->scale(r.first, 1);

	emit viewAreaChanged(m_view->mapToScene(m_view->viewport()->rect()));
}

void Abstract2DInnerView::setViewRect(const QRectF &viewArea) {
	//Commited from outside we decide not to resend an event
	//disconnnectScrollBarToExternalViewAreaChangedSignal();

	//float r = m_view->setVisibleRect(viewArea);
	std::pair<float, float> r = m_view->setVisibleRect(viewArea);
	BaseQGLGraphicsView::resetScale(m_verticalAxisView);
	BaseQGLGraphicsView::resetScale(m_horizontalAxisView);

	m_verticalAxisView->scale(1, r.second);
	m_horizontalAxisView->scale(r.first, 1);

	//Cool way to resynchronize!
	m_horizontalAxisView->horizontalScrollBar()->setValue(m_view->horizontalScrollBar()->value());
	m_verticalAxisView->verticalScrollBar()->setValue(m_view->verticalScrollBar()->value());

	//connnectScrollBarToExternalViewAreaChangedSignal();
}

QPolygonF Abstract2DInnerView::viewRect() const {
	return m_view->mapToScene(m_view->viewport()->rect());
}

void Abstract2DInnerView::externalMouseMoved(MouseTrackingEvent *event) {
	MouseTrackingEvent tmp(*event);
	if (absoluteWorldToViewWorld(tmp)) {
		m_lastMousePosition.setX(tmp.worldX());
		m_lastMousePosition.setY(tmp.worldY());
		MouseTrackingEvent loc;
		fillStatusBar(tmp.worldX(), tmp.worldY(), loc);
		m_crossHairItem->setPosition(tmp.worldX(), tmp.worldY());
		propagateMouseMoveEvent(tmp.worldX(), tmp.worldY(),
				Qt::MouseButton::NoButton, Qt::NoModifier);
	}
}

void Abstract2DInnerView::mouseMoved(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
	m_crossHairItem->setPosition(worldX, worldY);

	m_lastMousePosition.setX(worldX);
	m_lastMousePosition.setY(worldY);

	//Handling cross viewer event
	MouseTrackingEvent event;
	fillStatusBar(worldX, worldY, event);
	//Convert from view to world
	if (viewWorldToAbsoluteWorld(event))
		emit viewMouseMoved(new MouseTrackingEvent(event));

	propagateMouseMoveEvent(worldX, worldY, button, modifiers);

	//picking task
	for (PickingTask *p : m_pickingTask)
		p->mouseMoved(worldX, worldY, button, modifiers,collectPickInfo(worldX, worldY));
}

void Abstract2DInnerView::propagateMouseMoveEvent(double worldX, double worldY,Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
	//notify displayed layers/controlers
	for (AbstractGraphicRep *rep : m_visibleReps) {
		GraphicLayer *layer = rep->layer(m_scene, getZFromRep(rep), nullptr);
		if (layer!=nullptr) {
			layer->mouseMoved(worldX, worldY, button, modifiers);
		}
		if (IDataControlerHolder *holder = dynamic_cast<IDataControlerHolder*>(rep)) {
			holder->notifyDataControlerMouseMoved(worldX, worldY, button,modifiers);
		}
	}
}
void Abstract2DInnerView::mousePressed(double worldX, double worldY,Qt::MouseButton button, Qt::KeyboardModifiers keys) {

	//graphic rep and controlers
	for (AbstractGraphicRep *rep : m_visibleReps) {
		GraphicLayer *layer = rep->layer(m_scene, getZFromRep(rep), nullptr);
		layer->mousePressed(worldX, worldY, button, keys);
		if (IDataControlerHolder *holder =
				dynamic_cast<IDataControlerHolder*>(rep)) {
			holder->notifyDataControlerMousePressed(worldX, worldY, button,
					keys);
		}
	}

	//Picking task
	for (PickingTask *p : m_pickingTask)
		p->mousePressed(worldX, worldY, button, keys,
				collectPickInfo(worldX, worldY));

}
void Abstract2DInnerView::mouseRelease(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {
	for (AbstractGraphicRep *rep : m_visibleReps) {
		GraphicLayer *layer = rep->layer(m_scene, getZFromRep(rep), nullptr);
		layer->mouseRelease(worldX, worldY, button, keys);
		if (IDataControlerHolder *holder =
				dynamic_cast<IDataControlerHolder*>(rep)) {
			holder->notifyDataControlerMouseRelease(worldX, worldY, button,
					keys);
		}
	}

	//Picking task
	for (PickingTask *p : m_pickingTask)
		p->mouseRelease(worldX, worldY, button, keys,
				collectPickInfo(worldX, worldY));
}

void Abstract2DInnerView::mouseDoubleClick(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {

	//graphic rep and controlers
	for (AbstractGraphicRep *rep : m_visibleReps) {
		GraphicLayer *layer = rep->layer(m_scene, getZFromRep(rep), nullptr);
		layer->mouseDoubleClick(worldX, worldY, button, keys);
		if (IDataControlerHolder *holder =
				dynamic_cast<IDataControlerHolder*>(rep)) {
			holder->notifyDataControlerMouseDoubleClick(worldX, worldY, button, keys);
		}
	}

	//Picking task
	for (PickingTask *p : m_pickingTask)
		p->mouseDoubleClick(worldX, worldY, button, keys,
				collectPickInfo(worldX, worldY));
}

void Abstract2DInnerView::contextMenu(double worldX, double worldY,
		QContextMenuEvent::Reason reason, QMenu& menu) {
	// Contextual Menu
	bool foundGraphicsitem = false;
	foreach (QGraphicsItem *p, m_scene->items())
	{
		if (dynamic_cast<GraphEditor_Item *>(p))
		{

			if ((p->contains(p->mapFromScene(QPointF(worldX,worldY))) && (p->isVisible()))
				|| (dynamic_cast<GraphEditor_LineShape *>(p) && (p->isSelected()))
			//	|| (dynamic_cast<GraphEditor_RegularBezierPath *>(p) && (p->isSelected()))
				|| (dynamic_cast<GraphEditor_Path *>(p) && (p->isSelected())) )
			{


				QPoint mapPos = m_view->mapFromScene(QPointF( worldX, worldY));
				QPoint globalPos = m_view->mapToGlobal(mapPos);

				dynamic_cast<GraphEditor_Item *>(p)->ContextualMenu(globalPos);
				foundGraphicsitem = true;
				break;
				// TO DO : search the item who has the biggest Z value
				break;
			}
		}
	}
	if (!foundGraphicsitem)
	{
		contextualMenuFromGraphics( worldX, worldY, reason, menu);
		menu.addSeparator();
		emit contextualMenuSignal(this, worldX, worldY, reason, menu);
	}
}

QVector<PickingInfo> Abstract2DInnerView::collectPickInfo(double worldX,
		double worldY) {
	QVector<PickingInfo> results;
	for (AbstractGraphicRep *rep : m_visibleReps) {
		if (IMouseImageDataProvider *provider =
				dynamic_cast<IMouseImageDataProvider*>(rep)) {
			IMouseImageDataProvider::MouseInfo info;
			if (provider->mouseData(worldX, worldY, info)) {
				results.push_back(PickingInfo(rep->data()->dataID(), info.values));
			}
		}
	}
	return results;
}

bool Abstract2DInnerView::fillStatusBar(double worldX, double worldY,
		MouseTrackingEvent &event) {
	StatusBar *b = statusBar();
	b->x(worldX);
	b->y(worldY);

	b->clearI();
	b->clearJ();
	b->clearDepth();
	b->clearValue();
	event.setPos(worldX, worldY);
	for (AbstractGraphicRep *rep : m_visibleReps) {
		if (IMouseImageDataProvider *provider =
				dynamic_cast<IMouseImageDataProvider*>(rep)) {
			IMouseImageDataProvider::MouseInfo info;
			if (provider->mouseData(worldX, worldY, info)) {
				b->i(info.i);
				b->j(info.j);
				if (info.depthValue) {
					double depth = info.depth;
					if (info.depthUnit==SampleUnit::DEPTH) {
						depth = MtLengthUnit::convert(MtLengthUnit::METRE, *m_depthLengthUnit, info.depth);
					}
					b->depth(depth);
					event.setPos(worldX, worldY, info.depth, info.depthUnit);
				}
				std::stringstream ss;
				ss << std::fixed << std::setprecision(2);
				if (info.values.size() > 1) {
					ss << "[";
					for (int i = 0; i < info.values.size() - 1; i++) {
						ss << info.values[i] << ",";
					}
					ss << info.values[info.values.size() - 1];
				} else if (info.values.size() == 1) {
					ss << info.values[0];
				}

				b->value(QString(ss.str().c_str()));
				return true;
			}
		}
	}
	return false;
}

bool Abstract2DInnerView::updateWorldExtent(const QRectF &worldExtent) {
	bool changed = false;
	if(worldExtent.isNull()) return false;

	if (!m_wordBoundsInitialized) {
		m_worldBounds = worldExtent;
		m_wordBoundsInitialized = true;
		changed = true;
	} else {
		QRectF newBounds = m_worldBounds;
		newBounds = newBounds.united(worldExtent);
		changed = newBounds != m_worldBounds;
		m_worldBounds = newBounds;
	}

	if (changed) {
		m_scaleItem->updateWorldExtent(m_worldBounds);
		m_crossHairItem->updateWorldExtent(m_worldBounds);
		//m_lineItem->updateWorldExtent(m_worldBounds);

		m_scene->setSceneRect(m_worldBounds);
	}
	return changed;
}

void Abstract2DInnerView::showRep(AbstractGraphicRep *rep) {


	//Add the graphic object to the scene
	GraphicLayer *layer = rep->layer(m_scene, getZFromRep(rep), nullptr);
	if (layer!=nullptr) {
		// Workaround : Do No not update worldExtent for MultiSeedSliceLayer and WellBoreLayerOnSlice , WellBoreLayerOnRandom and MultiSeedRandomLayer
		if((dynamic_cast<MultiSeedSliceLayer*>(layer) == nullptr) && (dynamic_cast<WellBoreLayerOnSlice*>(layer) == nullptr)
			&& (dynamic_cast<MultiSeedRandomLayer*>(layer) == nullptr) && (dynamic_cast<WellBoreLayerOnRandom*>(layer) == nullptr)){

			updateWorldExtent(layer->boundingRect());
		}
		layer->show();
		connect(layer, SIGNAL(boundingRectChanged(QRectF)),this,SLOT(updateWorldExtent(QRectF)));
	}

	for (DataControler *c : m_externalControler)
		showControler(c, rep);

	AbstractInnerView::showRep(rep);
}

void Abstract2DInnerView::hideRep(AbstractGraphicRep *rep) {
	const QList<AbstractGraphicRep*>& visibleReps = getVisibleReps();
	long indexRep = 0;
	bool repNotFound = true;
	while (repNotFound && indexRep<visibleReps.size()) {
		repNotFound = rep != visibleReps[indexRep];
		indexRep++;
	}
	if (repNotFound) {
		//qDebug() << "Rep not in visible reps, do not hide it";
		return;
	} else if (!repNotFound && visibleReps.size()==1) {
		m_wordBoundsInitialized = false;
	}
	GraphicLayer *layer = rep->layer(m_scene, getZFromRep(rep), nullptr);
	if (layer)
	{
		layer->hide();
		disconnect(layer, SIGNAL(boundingRectChanged(QRectF)),this,SLOT(updateWorldExtent(QRectF)));
	}
	//remove controlers link to this rep
	for (DataControler *c : m_externalControler)
		releaseControler(c, rep);

	AbstractInnerView::hideRep(rep);
}

QList<IData *> Abstract2DInnerView::detectWellsIncludedInItem(QGraphicsItem *item)
				{
	QList<IData*> wellList;
	for (int i = 0; i < m_visibleReps.size(); i++)
	{
		if(m_visibleReps[i]->getTypeGraphicRep() == AbstractGraphicRep::Courbe)
		{
			if ( dynamic_cast<iRepGraphicItem *>(m_visibleReps[i]) )
			{
				QGraphicsItem *graphicItem = dynamic_cast<iRepGraphicItem *>(m_visibleReps[i])->graphicsItem();
				if (graphicItem)
				{
					if (graphicItem->collidesWithItem(item))
					{
						wellList << m_visibleReps[i]->data();
					}
				}
			}
		}
	}
	return wellList;
				}

void Abstract2DInnerView::deselectWellsIncludedInItem(QGraphicsItem *item)
{
	QList<AbstractGraphicRep *> wellsRep;
	for (int i = 0; i < m_visibleReps.size(); i++)
	{
		if(m_visibleReps[i]->getTypeGraphicRep() == AbstractGraphicRep::Courbe)
		{
			if ( dynamic_cast<iRepGraphicItem *>(m_visibleReps[i]) )
			{
				QGraphicsItem *graphicItem = dynamic_cast<iRepGraphicItem *>(m_visibleReps[i])->graphicsItem();
				if (graphicItem)
				{
					if ( dynamic_cast <WellHeadRepOnMap *>(m_visibleReps[i]) )
					{
						QRectF bbox = graphicItem->sceneBoundingRect().normalized();
						QPointF svg_center = bbox.center();
						if (item->contains(svg_center))
						{
							wellsRep << m_visibleReps[i];
						}
					}
					else
					{
						if (graphicItem->collidesWithItem(item))
						{
							wellsRep << m_visibleReps[i];
						}
					}
				}
			}
		}
	}
	foreach(AbstractGraphicRep *rep, wellsRep)
	{
		GraphicLayer *layer = rep->layer(m_scene, DATA_ITEM_Z, nullptr);
		layer->hide();
		rep->data()->setAllDisplayPreference(false);
		//remove controlers link to this rep
		for (DataControler *c : m_externalControler)
			releaseControler(c, rep);

		AbstractInnerView::hideRep(rep);
	}

}

void Abstract2DInnerView::cleanupRep(AbstractGraphicRep *rep) {
	// cannot remove controlers link to this rep because item most likely deleted
	AbstractInnerView::cleanupRep(rep);
}

void Abstract2DInnerView::showControler(DataControler *controler,
		AbstractGraphicRep *rep) {
	if (IDataControlerHolder *holder = dynamic_cast<IDataControlerHolder*>(rep)) {
		QGraphicsItem *item = holder->getOverlayItem(controler, nullptr);
		if (item != nullptr) {
			item->setZValue(SCALE_ITEM_Z - 2);
			m_scene->addItem(item);
		}
	}
}

void Abstract2DInnerView::releaseControler(DataControler *controler,
		AbstractGraphicRep *rep) {
	if (IDataControlerHolder *holder = dynamic_cast<IDataControlerHolder*>(rep)) {
		QGraphicsItem *item = holder->releaseOverlayItem(controler);
		if (item != nullptr)
		{
			m_scene->removeItem(item);
			delete item;
		}
	}
}

void Abstract2DInnerView::addExternalControler(DataControler *controler) {
	AbstractInnerView::addExternalControler(controler);
	for (AbstractGraphicRep *rep : m_visibleReps)
		showControler(controler, rep);

}
void Abstract2DInnerView::removeExternalControler(DataControler *controler) {
	AbstractInnerView::removeExternalControler(controler);
	for (AbstractGraphicRep *rep : m_visibleReps)
		releaseControler(controler, rep);
}

QPointF Abstract2DInnerView::ConvertToImage(QPointF point)
{
	for (AbstractGraphicRep *rep : m_visibleReps)
	{
		if (IMouseImageDataProvider *provider =
				dynamic_cast<IMouseImageDataProvider*>(rep))
		{
			IMouseImageDataProvider::MouseInfo info;
			if (provider->mouseData(point.x(), point.y(), info))
			{
				return QPointF(info.i,info.j);
			}
		}
	}
	return QPointF(-1,-1);
}

void Abstract2DInnerView::deleteData (QGraphicsItem *item)
{
	for (AbstractGraphicRep *rep : m_visibleReps) {
		if (dynamic_cast<iGraphicToolDataControl*>(rep))
		{
			dynamic_cast<iGraphicToolDataControl*>(rep)->deleteGraphicItemDataContent(item);
		}
	}
}

void Abstract2DInnerView::startGraphicToolsDialog()
{
	GraphicToolsWidget::showPalette(title());
}

void Abstract2DInnerView::saveGraphicLayer()
{
	SaveGraphicLayerDialog *dialog = new SaveGraphicLayerDialog(this);
	QString graphicsLayersDirPath = GraphicsLayersDirPath();
	QDir dir(graphicsLayersDirPath);
	if (!dir.exists(graphicsLayersDirPath))
		dir.mkpath(".");

	int result = dialog->exec();
	if (result == QDialog::Accepted)
	{
		QString fileName = dialog->fileName();
		if (viewType() == InlineView)
		{
			fileName += ("_" + QString::number(dynamic_cast<SingleSectionView *> (this)->sliceValueWorld()));
			fileName += ".section";
		}
		else
		{
			fileName += ".map";
		}
		dynamic_cast<GraphicSceneEditor *> (m_scene)->save_state(graphicsLayersDirPath+fileName);
	}
}
/*
QString Abstract2DInnerView::GraphicsLayersDirPath()
{
	QStringList surveysNames;
	QList<SeismicSurvey*> surveys;
	QString path = "";

	WorkingSetManager* workingmanager = GeotimeGraphicsView::getWorkingSetManager();

	for (IData* surveyData : workingmanager->folders().seismics->data()) {
		if (SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(surveyData)) {
			surveys.push_back(survey);
			surveysNames.push_back(survey->name());
		}
	}
	if (surveys[0])
	{
		path = surveys[0]->idPath() + "ImportExport/IJK/GraphicLayers/";
	}
	return path;
}*/

void Abstract2DInnerView::loadCultural()
{
	QString graphicsLayersDirPath = GraphicsLayersDirPath();
	GraphicLayerSelectorDialog dialog(graphicsLayersDirPath, this);
	int result = dialog.exec();
	if (result == QDialog::Accepted)
	{
		QString fileName = dialog.getSelectedString();
		dynamic_cast<GraphicSceneEditor *> (m_scene)->restore_state(graphicsLayersDirPath+fileName +((viewType()==InlineView)?".section" :".map") );
	}
}

int Abstract2DInnerView::getZFromRep(AbstractGraphicRep* rep) {
	int zValue = DATA_ITEM_Z;
	if (rep->getTypeGraphicRep()==AbstractGraphicRep::Courbe) {
		zValue = COURBE_ITEM_Z;
	} else if(rep->getTypeGraphicRep()==AbstractGraphicRep::ImageRgt) {
	    zValue = DATA_ITEM_Z + 1;
	}

	return zValue;
}

Abstract2DInnerView::~Abstract2DInnerView() {
	m_verticalAxisScene->clear();
	m_horizontalAxisScene->clear();
	m_scene->clear();

}

void Abstract2DInnerView::setNameItem(QString name)
{
	GraphEditor_Path* bezier1 = (dynamic_cast<GraphicSceneEditor *> (m_scene))->getSelectedBezier();
	if(bezier1!= nullptr) bezier1->setNameId(name);
}
/*
void Abstract2DInnerView::refreshOrtho(QVector3D normal)
{
	qDebug()<<"Abstract2DInnerView refresh ortho : "<<normal;
	emit updateOrthoFrom3D(normal);
	//dynamic_cast<GraphicSceneEditor *> (m_scene)->
}*/

void Abstract2DInnerView::setNurbsSelected(QString name)
{
	emit selectedNurbs(name);
}

void Abstract2DInnerView::setNurbsDeleted(QString name)
{
	emit deletedNurbs(name);
}

void Abstract2DInnerView::directriceDeleted(QString name)
{
//	emit deletedDirectriceNurbs( name);
}

const MtLengthUnit* Abstract2DInnerView::depthLengthUnit() const {
	return m_depthLengthUnit;
}

void Abstract2DInnerView::setDepthLengthUnit(const MtLengthUnit* depthLengthUnit) {
	setDepthLengthUnitProtected(depthLengthUnit);
}

void Abstract2DInnerView::setDepthLengthUnitProtected(const MtLengthUnit* depthLengthUnit) {
	if ((*m_depthLengthUnit)!=(*depthLengthUnit)) {
		StatusBar *b = statusBar();
		b->clearDepth();

		m_depthLengthUnit = depthLengthUnit;
	}
}

void Abstract2DInnerView::setScenesTopCornerWidget(QWidget* widget) {
	if (m_boxTopCornerPreviousWidget) {
		m_boxTopCornerPreviousWidget->deleteLater();
	}
	m_boxTopCornerPreviousWidget = widget;
	m_box->addWidget(widget, 0, 0);
}
