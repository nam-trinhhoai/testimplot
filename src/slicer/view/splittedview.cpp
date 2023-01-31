#include <QMouseEvent>
#include <QWidget>
#include <QList>
#include <QComboBox>
#include <QString>
#include <QToolBar>
#include <QMenu>
#include <QPushButton>
#include <kddockwidgets/MainWindow.h>
#include "splittedview.h"
#include "basemapview.h"
#include "stackbasemapview.h"
#include "abstractgraphicrep.h"
#include "rgblayerrgtrep.h"
#include "stacklayerrgtrep.h"
#include "abstractsectionview.h"
#include "wellheadreponmap.h"
#include "wellborereponmap.h"
#include "wellheadlayeronmap.h"
#include "wellborelayeronmap.h"
#include "GraphicEditorFormsRep.h"
#include "fixedrgblayersfromdatasetandcuberep.h"
#include "abstract2Dinnerview.h"
#include "qgraphicsitem.h"

QString SplittedView::viewModeLabel(eViewMode e)
{
	static QString splitLabel("Split Mode");
	static QString tabLabel("Tab Mode");

	if(e == eTypeTabMode) {
		return tabLabel;
	}

	return splitLabel;
}

SplittedView::SplittedView(ViewType v,QList<AbstractGraphicRep*>repList,eViewMode eMode,AbstractInnerView *parent):m_parent(parent),
KDDockWidgets::MainWindow("SplittedView", KDDockWidgets::MainWindowOption_None, nullptr)
{
	setAttribute(Qt::WA_DeleteOnClose);
	setDockNestingEnabled(true);

	m_eViewMode = eMode;

	QToolBar *  m_toolbar =addToolBar("Main Toolbar");
	m_toolbar->setStyleSheet("background-color:#32414B;");

	m_viewMode = new QPushButton("Tab Mode");
	connect(m_viewMode,SIGNAL(clicked()),this,SLOT(changeviewMode()));

	m_toolbar->addWidget(m_viewMode);

	m_NbColumn = 0;
	for(int i = 0;i<repList.size();i++){
		this->addView(repList,repList[i],v);
	}

	if((m_InnerViews.size() % 2) != 0){
		this->addView(repList,nullptr,v);
	}

	changeviewMode();
}

void SplittedView::changeviewMode(){

	switch(m_eViewMode){
	case eTypeTabMode:
		splitView();
		break;
	case eTypeSplitMode:
		tabView();
		break;
	}
}

void SplittedView::tabView(){

	if(m_eViewMode != eTypeTabMode){

		m_eViewMode = eTypeTabMode;
#if 0
		m_viewMode->setText("Splitted Mode");
#else
		m_viewMode->setText(SplittedView::viewModeLabel(eTypeSplitMode));
#endif
		int index = 0;
		QList<AbstractInnerView *> keys= m_InnerViews.keys();
		for(int index = 0; index < keys.size(); index++){
			if(index == 0){
				addDockWidget(keys[index],KDDockWidgets::Location_OnRight, nullptr);
			} else {
				keys[index-1]->addDockWidgetAsTab(keys[index]);
			}
		}
	}
}

void SplittedView::splitView(){

	if(m_eViewMode != eTypeSplitMode){
		m_eViewMode = eTypeSplitMode;
#if 0
		m_viewMode->setText("Tab Mode");
#else
		m_viewMode->setText(SplittedView::viewModeLabel(eTypeTabMode));
#endif
		m_NbColumn = 0;

		QList<AbstractInnerView *> keys= m_InnerViews.keys();
		for(int index = 0; index < keys.size(); index++){
			if(m_NbColumn < 2 && (index != 0)){
				m_NbColumn++;
				addDockWidget(keys[index], KDDockWidgets::Location_OnRight, keys[index -1]);
			} else {
				m_NbColumn = 1;
				addDockWidget(keys[index], KDDockWidgets::Location_OnBottom,nullptr);
			}
		}
	}
}

void SplittedView::geometryChanged(AbstractInnerView *newView,const QRect & geom){
	newView->window()->setGeometry(geom);
}

void SplittedView::unregisterView(AbstractInnerView * pInnerView){
	QMapIterator<AbstractInnerView*, QList<AbstractGraphicRep*>> iter(m_InnerViews);
	int index = 0;
	while (iter.hasNext()) {
		iter.next();
		AbstractInnerView *key = iter.key();
		QList<AbstractGraphicRep*> value = iter.value();
		if(key == pInnerView){
			for(int i = 0; i < m_Rep.size();i++){
				if(pInnerView->title() == m_Rep[i]->name())    {
					delete m_Rep[i];
					m_Rep.removeAt(i);
					break;
				}
			}

			for(int i=0;i<value.size();i++){
				delete value[i];
			}
			m_InnerViews.remove(key);

			//delete
			key->deleteLater();
		}
		index++;
	}
}

AbstractInnerView* SplittedView::createInnerView(ViewType v,AbstractGraphicRep *pRep){
	AbstractInnerView* pInnerView = nullptr;
	QString str = "Empty View";
	if(pRep != nullptr){
		str = pRep->name();
	}
	pInnerView = new StackBaseMapView(false,str,eModeSplitView,this);
	return pInnerView;
}

QList<AbstractInnerView*> SplittedView::getInnerViews(){
     return m_InnerViews.keys();
}

void SplittedView::addView(QList<AbstractGraphicRep*>repList,AbstractGraphicRep* pRep,ViewType v){
	bool isRepSelected = false;

	AbstractInnerView* pInnerView = nullptr;
	AbstractGraphicRep *pLayer =  nullptr;

	if(dynamic_cast<RGBLayerRGTRep*>(pRep) != nullptr){
		pInnerView = createInnerView(v,pRep);
		pLayer= new RGBLayerRGTRep(dynamic_cast<RGBLayerRGTRep*>(pRep)->rgbLayerSlice(),pInnerView);
		isRepSelected = true;
	} else if(dynamic_cast<StackLayerRGTRep*>(pRep) != nullptr){
		pInnerView = createInnerView(v,pRep);
		pLayer = new StackLayerRGTRep(dynamic_cast<StackLayerRGTRep*>(pRep)->layerSlice(),pInnerView);
		isRepSelected = true;
	} else if(dynamic_cast<FixedRGBLayersFromDatasetAndCubeRep*>(pRep) != nullptr){
		pInnerView = createInnerView(v,pRep);
		pLayer = new FixedRGBLayersFromDatasetAndCubeRep(dynamic_cast<FixedRGBLayersFromDatasetAndCubeRep*>(pRep)->fixedRGBLayersFromDataset(),pInnerView);
		isRepSelected = true;
	} else {
		if(pRep == nullptr){
			pInnerView = createInnerView(UndefinedView,pRep);
			//pLayer= new AbstractGraphicRep(pInnerView);
			isRepSelected = true;
		}
	}
	m_dynamicItems.clear();
	if(pRep != nullptr){
		Abstract2DInnerView* p2DParent = dynamic_cast<Abstract2DInnerView*>(m_parent) ;
		if(p2DParent != nullptr){
			GraphicSceneEditor *pgGraphicScene = dynamic_cast<GraphicSceneEditor*>(p2DParent->scene());
			if(pgGraphicScene != nullptr){
				m_dynamicItems = pgGraphicScene->CloneSceneItem();
			}
		}
		Abstract2DInnerView* p2DInnerview = dynamic_cast<Abstract2DInnerView*>(pInnerView);
		if(p2DInnerview != nullptr){
			GraphicSceneEditor *pgGraphicScene = dynamic_cast<GraphicSceneEditor*>(p2DInnerview->scene());
			if(pgGraphicScene != nullptr){
				foreach(QGraphicsItem *p,m_dynamicItems){
					pgGraphicScene->addItem(p);
					pgGraphicScene->saveItem(p);
				}
			}
		}
	}
	if (isRepSelected){
		QList<AbstractGraphicRep*> listCourbeRep;
		if(pLayer != nullptr){
			m_Rep.push_back(pLayer);

			for(int j = 0;j < repList.size();j++){
				if(dynamic_cast<WellHeadRepOnMap*>(repList[j]) != nullptr){
					WellHeadRepOnMap * layer = new WellHeadRepOnMap(dynamic_cast<WellHeadRepOnMap*>(repList[j])->wellHead(),pInnerView);
					listCourbeRep.push_back(layer);
				}
				if(dynamic_cast<WellBoreRepOnMap*>(repList[j]) != nullptr){
					WellBoreRepOnMap * layer = new WellBoreRepOnMap(dynamic_cast<WellBoreRepOnMap*>(repList[j])->wellBore(),pInnerView);
					listCourbeRep.push_back(layer);
				}
			}
		}

		m_InnerViews[pInnerView] = listCourbeRep;

		connect(pInnerView, SIGNAL(viewMouseMoved(MouseTrackingEvent *)), this,SLOT(innerViewMouseMoved(MouseTrackingEvent *)));
		connect(pInnerView, SIGNAL(viewAreaChanged(const QPolygonF & )), this,SLOT(viewPortChanged(const QPolygonF &)));
		connect(pInnerView, SIGNAL(isClosing(AbstractInnerView * )), this,SLOT(unregisterView(AbstractInnerView *)));
		//connect(pInnerView, &AbstractInnerView::viewEnter,pInnerView,&AbstractInnerView::activateWindow);
		connect(pInnerView, &AbstractInnerView::viewEnter, pInnerView,QOverload<>::of(&AbstractInnerView::setFocus));
		connect(pInnerView, SIGNAL(askGeometryChanged(AbstractInnerView * ,const QRect & )), this,SLOT(geometryChanged(AbstractInnerView *,const QRect & )));
	}
}

void SplittedView::viewPortChanged(const QPolygonF &poly) {
	QObject* senderView = sender();
	bool isSenderBaseMap = dynamic_cast<BaseMapView*>(senderView)!=nullptr || dynamic_cast<StackBaseMapView*>(senderView)!=nullptr;

	QMapIterator<AbstractInnerView*, QList<AbstractGraphicRep*>> iter(m_InnerViews);
	while (iter.hasNext()) {
		iter.next();
		AbstractInnerView *v = iter.key();
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
		}
	}
}

void SplittedView::innerViewMouseMoved(MouseTrackingEvent *event) {

	QMapIterator<AbstractInnerView*, QList<AbstractGraphicRep*>> iter(m_InnerViews);
	while (iter.hasNext()) {
		iter.next();
		AbstractInnerView *v = iter.key();
		if (v != sender())
			v->externalMouseMoved(event);
	}
	emit viewMouseMoved(event);
}

void SplittedView::showRep(){

	if(m_InnerViews.size() != 0){
		QMapIterator<AbstractInnerView*, QList<AbstractGraphicRep*>> iter(m_InnerViews);
		Abstract2DInnerView *pInnerView = nullptr;

		AbstractInnerView *pInner = nullptr;
		while (iter.hasNext()) {
			iter.next();
			pInner = iter.key();
			QList<AbstractGraphicRep*> courbe = iter.value();
			AbstractGraphicRep *pLayer = nullptr;
			for(int i = 0; i < m_Rep.size();i++){
				if(pInner->title() == m_Rep[i]->name()){
					//pLayer = m_Rep[i];
					pInner->showRep(m_Rep[i]);
					break;
				}
			}

			//            if(dynamic_cast<BaseMapView*>(pInner) != nullptr){
			//                pInnerView = dynamic_cast<StackBaseMapView*>(pInner);
			//            }

			if(dynamic_cast<StackBaseMapView*>(pInner) != nullptr){
				pInnerView = dynamic_cast<StackBaseMapView*>(pInner);
			}

			for(int j =0;j<  courbe.size();j++){
				pInner->showRep(courbe[j]);
			}
		}
	}
}

SplittedView::~SplittedView(){
	for(int i = 0;i < m_Rep.size();i++){
		delete m_Rep[i];
	}

	QMapIterator<AbstractInnerView*, QList<AbstractGraphicRep*>> iter(m_InnerViews);
	int index = 0;
	while (iter.hasNext()) {
		iter.next();
		AbstractInnerView *pInner = iter.key();
		QList<AbstractGraphicRep*> courbe = iter.value();
		for(int i = 0;i < courbe.size();i++){
			delete courbe[i];
		}
		GraphicToolsWidget::removeInnerView(dynamic_cast<Abstract2DInnerView*>(pInner));
		delete pInner;
	}
}
