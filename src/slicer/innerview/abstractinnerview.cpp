#include "abstractinnerview.h"
#include "idatacontrolerholder.h"
#include "idatacontrolerprovider.h"
#include "abstractgraphicrep.h"
#include "splittedview.h"
#include "datasetrep.h"
#include "slicerep.h"
#include "randomrep.h"
#include <QMdiArea>
#include <QCloseEvent>
#include <QProxyStyle>
#include <QPalette>
#include <QStyleOptionComplex>
#include <QColor>
#include <QDebug>
#include <QPushButton>
#include <QSizeGrip>
#include <QActionGroup>
#include <QMenu>
#include <QLabel>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QCheckBox>
#include "dockwidgetsizegrid.h"
#include "geotimegraphicsview.h"
#include "workingsetmanager.h"
#include "folderdata.h"
#include "seismicsurvey.h"
#include "isochronprovider.h"

AbstractInnerView::AbstractInnerView(bool restictToMonoTypeSplit,
		QString uniqueName,eModeView typeView) :
		KDDockWidgets::DockWidget(uniqueName  ), m_modeView(typeView) {
	m_restictToMonoTypeSplitChild = restictToMonoTypeSplit;
	setAttribute(Qt::WA_DeleteOnClose);
	setWindowFlag(Qt::SubWindow, true);

	m_viewType = ViewType::UndefinedView;
	m_currentViewIndex = 0;
}


QWidget* AbstractInnerView::generateSizeGrip() {
	QWidget *status = new QWidget(this);
	QHBoxLayout *hlayout = new QHBoxLayout(status);
	hlayout->setContentsMargins(0, 0, 0, 0);
	hlayout->addWidget(new QLabel(" "));
	DockWidgetSizeGrid *sizegrip = new DockWidgetSizeGrid(this);
	connect(sizegrip, SIGNAL(geometryChanged(const QRect &)), this,
			SLOT(geometryChanged(const QRect &)));
	hlayout->addWidget(sizegrip);

	return status;
}
void AbstractInnerView::geometryChanged(const QRect &geom) {
	emit askGeometryChanged(this, geom);
}
void AbstractInnerView::emitSplit(int value, Qt::Orientation orientation) {
	if (value == ViewType::BasemapView) {
		askToSplit(this, ViewType::BasemapView, false, orientation);
	} else if (value == ViewType::StackBasemapView) {
		askToSplit(this, ViewType::StackBasemapView, false, orientation);
	} else if (value == ViewType::InlineView) {
		askToSplit(this, ViewType::InlineView, false, orientation);
	} else if (value == ViewType::XLineView) {
		askToSplit(this, ViewType::XLineView, false, orientation);
	} else if (value == ViewType::RandomView) {
		askToSplit(this, ViewType::RandomView, false, orientation);
	} else if (value == ViewType::View3D) {
		askToSplit(this, ViewType::View3D, false, orientation);
	}
}

void AbstractInnerView::showPalette() {
	for(int i=0;i<m_visibleReps.size();i++){
		if(m_visibleReps[i]->propertyPanel() != nullptr){
			QWidget *parmPanel = new QWidget(nullptr);
			QWidget *propertyPanel = m_visibleReps[i]->propertyPanel() ;
			QVBoxLayout *parameterLayout = new QVBoxLayout(parmPanel);
			parameterLayout->addWidget(propertyPanel);
			parameterLayout->addSpacerItem(new QSpacerItem(0, 0, QSizePolicy::Minimum,QSizePolicy::Expanding));
			parmPanel->show();
			break;
		}
	}
}

void AbstractInnerView::splitDockWidgetMulti() {
	int nbView = 0;

	for(int i=0;i<m_visibleReps.size();i++){
		if((m_visibleReps[i]->getTypeGraphicRep() == AbstractGraphicRep::Image)
				&& ((m_viewType == BasemapView) || (m_viewType == StackBasemapView))){
			nbView++;
		}
	}

	if(nbView > 1){
		SplittedView * view = new SplittedView(m_viewType,m_visibleReps,eTypeTabMode,this);
		view->setWindowTitle("Splitted View");
		view->setVisible(true);
		view->showRep();
	}
}

QPair<QMenu*, QActionGroup*> AbstractInnerView::generateViewMenu() {
	QMenu *menu = new QMenu(this);
	QActionGroup *actions1 = new QActionGroup(menu);
	actions1->addAction(
			menu->addAction(QIcon(":/slicer/icons/map_gray.png"),
					tr("BaseMap")))->setData(ViewType::BasemapView);
	actions1->addAction(
			menu->addAction(QIcon(":/slicer/icons/inline_gray.png"),
					tr("Inline")))->setData(ViewType::InlineView);
	actions1->addAction(
			menu->addAction(QIcon(":/slicer/icons/xline_gray.png"),
					tr("Xline")))->setData(ViewType::XLineView);
	actions1->addAction(
			menu->addAction(QIcon(":/slicer/icons/3d_gray.png"), tr("3D View")))->setData(
			ViewType::View3D);
	actions1->addAction(
			menu->addAction(QIcon(":/slicer/icons/random_gray.png"), tr("Random View")))->setData(
			ViewType::RandomView);

	return QPair<QMenu*, QActionGroup*>(menu, actions1);
}

void AbstractInnerView::enterEvent( QEnterEvent* event) {
	//qDebug()<<" enter event "<<m_viewType;
	emit viewEnter(this);
	KDDockWidgets::DockWidget::enterEvent(event);
}

QPushButton* AbstractInnerView::createTitleBarButton(const QString &iconPath,
		const QString &tooltip) const {
	QPushButton *split = new QPushButton(QIcon(iconPath), "");
	split->setIconSize(QSize(8, 8));
	split->setToolTip(tooltip);
	split->setDefault(false);
	split->setAutoDefault(false);
	split->setStyleSheet("min-width: 8px;");
	return split;
}

void AbstractInnerView::setViewIndex(int val) {
	m_currentViewIndex = val;
	updateTile(m_suffixTitle);
}

QString AbstractInnerView::getBaseTitle() const {
	QString baseTitle;
	if (m_defaultTitle.isNull() || m_defaultTitle.isEmpty()) {
		QString viewName;
		if (m_viewType == ViewType::BasemapView) {
			viewName = "Basemap";
		} else if (m_viewType == ViewType::StackBasemapView) {
			viewName = "Basemap";
		} else if (m_viewType == ViewType::InlineView) {
			viewName = "Inline";
		} else if (m_viewType == ViewType::XLineView) {
			viewName = "XLine";
		} else if (m_viewType == ViewType::RandomView) {
			viewName = "Random";
		} else if (m_viewType == ViewType::View3D) {
			viewName = "View3D";
		} else {
			viewName = "View";
		}
		viewName += " " + QString::number(m_currentViewIndex);
		baseTitle = viewName;
	} else {
		baseTitle = m_defaultTitle;
	}
	return baseTitle;
}

QString AbstractInnerView::suffixTitle() const {
	return m_suffixTitle;
}

void AbstractInnerView::updateTile(const QString &name) {
	QString viewName;//(
	//		(std::string("View ") + std::to_string(m_currentViewIndex)).c_str());
	QString title;

	if (m_defaultTitle.isNull() || m_defaultTitle.isEmpty()) {
		if (m_viewType == ViewType::BasemapView) {
			viewName = "Basemap";
		} else if (m_viewType == ViewType::StackBasemapView) {
			viewName = "Basemap";
		} else if (m_viewType == ViewType::InlineView) {
			viewName = "Inline";
		} else if (m_viewType == ViewType::XLineView) {
			viewName = "XLine";
		} else if (m_viewType == ViewType::RandomView) {
			viewName = "Random";
		} else if (m_viewType == ViewType::View3D) {
			viewName = "View3D";
		} else {
			viewName = "View";
		}
		viewName += " " + QString::number(m_currentViewIndex);
		title = viewName;
		if (!name.isEmpty()) {
			title = viewName + ": " + name;
		}
	} else {
		title = m_defaultTitle;
	}

	setWindowTitle(title);
	setTitle("  "+title);
	m_suffixTitle = name;
}


QString AbstractInnerView::GraphicsLayersDirPath()
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
}

void AbstractInnerView::setDefaultTitle(QString title) { // use old title creation process if title is null or empty
	m_defaultTitle = title;
	updateTile(m_suffixTitle);
}

AbstractGraphicRep* AbstractInnerView::lastRep() {
	if (m_visibleReps.empty())
		return nullptr;
	else
		return m_visibleReps.last();
}

void AbstractInnerView::registerPickingTask(PickingTask *task) {
	m_pickingTask.push_back(task);
}
void AbstractInnerView::unregisterPickingTask(PickingTask *task) {
	m_pickingTask.removeOne(task);
}

ViewType AbstractInnerView::viewType() const {
	return m_viewType;
}

void AbstractInnerView::showRep(AbstractGraphicRep *rep) {
	m_visibleReps.push_back(rep);

	//if the current rep is a controler provider notify it
	if (IDataControlerProvider *provider =
			dynamic_cast<IDataControlerProvider*>(rep)) {
		DataControler* controler = provider->dataControler();
		m_repToControlerMap[rep] = controler;
		emit controlerActivated(controler);
	}
	QMetaObject::Connection conn = connect(rep, &AbstractGraphicRep::destroyed, [this, rep]() {
		cleanupRep(rep);
	});

	if((dynamic_cast<DatasetRep*>(rep) != nullptr)
			|| (dynamic_cast<SliceRep*>(rep) != nullptr)
			|| ((dynamic_cast<RandomRep*>(rep) != nullptr))// && (dynamic_cast<RandomRep*>(rep)->isUpdatedFlag() == false))
	) {
		emit repAdded(rep);
	}
	m_destructionConnectionMap[rep] = conn;
}

void AbstractInnerView::hideRep(AbstractGraphicRep *rep) {
	m_visibleReps.removeOne(rep);

	//if this rep is a provider notify it!
	if (IDataControlerProvider *provider =
			dynamic_cast<IDataControlerProvider*>(rep))
		emit controlerDesactivated(provider->dataControler());

	disconnect(m_destructionConnectionMap[rep]);
	m_destructionConnectionMap.remove(rep);
	m_repToControlerMap.remove(rep); // remove DataControler if present
}

void AbstractInnerView::cleanupRep(AbstractGraphicRep *rep) {
	m_visibleReps.removeOne(rep);

	//if this rep is a provider notify it!
	if (m_repToControlerMap.contains(rep)) {
		emit controlerDesactivated(m_repToControlerMap[rep]);
		m_repToControlerMap.remove(rep);
	}

	disconnect(m_destructionConnectionMap[rep]);
	m_destructionConnectionMap.remove(rep);
}

void AbstractInnerView::addExternalControler(DataControler *controler) {
	m_externalControler.push_back(controler);

}
void AbstractInnerView::removeExternalControler(DataControler *controler) {
	m_externalControler.removeOne(controler);
}

QList<DataControler*> AbstractInnerView::getControlers() const {
	QList<DataControler*> result;
	for (AbstractGraphicRep *r : m_visibleReps) {
		if (IDataControlerProvider *provider =
				dynamic_cast<IDataControlerProvider*>(r))
			result.push_back(provider->dataControler());
	}
	return result;
}

void AbstractInnerView::closeEvent(QCloseEvent *event) {
	emit isClosing(this);
	event->accept();

	// Did not manage to destroy the widget even with setAttribute(Qt::WA_DeleteOnClose);
	// There may still be memory leaks issue because in the minimal exemple,
	// the setWidget function does not seem to take ownership of the widget.
	// with more investigation, deleteLater() could be removed.
	deleteLater();
}

AbstractInnerView::~AbstractInnerView() {
	for (QMetaObject::Connection conn : m_destructionConnectionMap) {
		disconnect(conn);
	}
}

const QList<AbstractGraphicRep*>& AbstractInnerView::getVisibleReps() const {
	return m_visibleReps;
}

IsoSurfaceBuffer AbstractInnerView::getHorizonBuffer()
{

	for(int i= m_visibleReps.count()-1;i>=0;i--)
	{
		if(m_visibleReps[i] != nullptr)
		{
			IsoChronProvider* isochron = dynamic_cast<IsoChronProvider*>(m_visibleReps[i]->data());
			if( isochron != nullptr)
			{
				return  isochron->getIsoBuffer();
			}
		}
	}
//	qDebug()<<"Errror getHorizonBuffer isoBuffer introuvable";

	IsoSurfaceBuffer buffer;
	return buffer;
}

QPushButton* AbstractInnerView::getPaletteButton() {
	QPushButton* paletteView = nullptr;
	if(m_modeView == eModeSplitView){
		paletteView = createTitleBarButton(":/slicer/icons/palette.png", "Palette");
		connect(paletteView, SIGNAL(clicked()), this,SLOT(showPalette()));
	}
	return paletteView;
}

QPushButton* AbstractInnerView::getSplitButton() {
	QPushButton *splitMultiView = createTitleBarButton(":/slicer/icons/mv_split.png", "Split multi view");
	connect(splitMultiView, SIGNAL(clicked()), this,SLOT(splitDockWidgetMulti()));
	return splitMultiView;
}


























