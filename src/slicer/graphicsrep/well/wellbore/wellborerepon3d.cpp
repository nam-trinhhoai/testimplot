#include "wellborerepon3d.h"
#include "wellbore.h"
#include "wellborelayer3d.h"
#include "wellboreproppanelon3d.h"
#include "abstractinnerview.h"
#include "wellpickrep.h"
#include "wellpick.h"

#include "workingsetmanager.h"
#include <QMenu>
#include <QAction>
#include "viewqt3d.h"

WellBoreRepOn3D::WellBoreRepOn3D(WellBore *wellBore, AbstractInnerView *parent) :
	AbstractGraphicRep(parent) {
	m_data = wellBore;
	m_layer3D = nullptr;
	m_propPanel = nullptr;
	m_sampleUnit = SampleUnit::NONE;

	connect(m_data,&WellBore::deletedMenu,this,&WellBoreRepOn3D::deleteWellBoreRepOn3D); // MZR 18082021
	connect(m_data, &WellBore::boreUpdated, this, &WellBoreRepOn3D::reExtractDeviation);
}

WellBoreRepOn3D::~WellBoreRepOn3D() {
	if (m_layer3D != nullptr)
		delete m_layer3D;
	if (m_propPanel!=nullptr)
		delete m_propPanel;
}

WellBore* WellBoreRepOn3D::wellBore() const {
	return m_data;
}

IData* WellBoreRepOn3D::data() const {
	return m_data;
}

QString WellBoreRepOn3D::name() const {
	return m_data->name();
}

QWidget* WellBoreRepOn3D::propertyPanel() {
	if (m_propPanel == nullptr) {
		WellBorePropPanelOn3D* propPanel = new WellBorePropPanelOn3D(this, m_defaultWidth,
				m_minimalWidth, m_maximalWidth, m_logMin, m_logMax, m_defaultColor, m_parent);
		m_propPanel = propPanel;
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
		connect(propPanel, &WellBorePropPanelOn3D::stateUpdated,
				this, &WellBoreRepOn3D::updatedParameterState);
	}
	return m_propPanel;
}

GraphicLayer * WellBoreRepOn3D::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)
{
	return nullptr;
}

Graphic3DLayer * WellBoreRepOn3D::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) {
	if (m_layer3D == nullptr) {
		ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(m_parent);


		m_layer3D = new WellBoreLayer3D(this,parent, root, camera);
		connect(m_layer3D, &WellBoreLayer3D::layerShownChanged, this, &WellBoreRepOn3D::updatePicks);


		if(view3D != nullptr)
		{
			//connect(view3D, SIGNAL(signalSimplifyWell(int)), m_layer3D, SLOT(setDistanceSimplification(int)) ) ;
			connect(view3D, SIGNAL(signalSimplifyWell(double)), m_layer3D, SLOT(setDistanceSimplification(double)) ) ;
			connect(view3D, SIGNAL(signalWireframeWell(bool)), m_layer3D, SLOT(setWireframe(bool)) ) ;
			connect(view3D, SIGNAL(signalShowNormalsWell(bool)), m_layer3D, SLOT(setShowNormals(bool)) ) ;
			connect(view3D, SIGNAL(signalSimplifyLogs(int)), m_layer3D, SLOT(setIncrementLogs(int)) ) ;
			connect(view3D, SIGNAL(signalThicknessLog(int)), m_layer3D, SLOT(setThicknessLog(int)) ) ;
			connect(view3D, SIGNAL(signalColorLog(QColor)), m_layer3D, SLOT(setColorLog(QColor)) ) ;
			connect(view3D, SIGNAL(signalColorWell(QColor)), m_layer3D, SLOT(setColorWell(QColor)) ) ;
			connect(view3D, SIGNAL(signalColorSelectedWell(QColor)), m_layer3D, SLOT(setColorSelectedWell(QColor)) ) ;
			connect(view3D, SIGNAL(signalDiameterWell(int)), m_layer3D, SLOT(setDiameterWell(int)) ) ;
			view3D->sendSimplifyWell();
			view3D->sendWireframeWell();
			view3D->sendShowNormalsWell();
			view3D->sendThicknessLog();
			view3D->sendColorLog();
			view3D->sendDiameterWell();
			view3D->sendColorWell();
			view3D->sendColorSelectedWell();

			if(m_propPanel!= nullptr)
			{
				// comment connect pointing to non existing slot, the rep/layer/proppanel need to be checked
				// There is too many issues with the property panel
				//connect(view3D, SIGNAL(signalColorLog(QColor)), m_propPanel, SLOT(setColorLog(QColor)) ) ;
				m_propPanel->setDefaultColor(view3D->getLogColor());
			}
		}

	}

	return m_layer3D;
}

SampleUnit WellBoreRepOn3D::sampleUnit() const {
	return m_sampleUnit;
}

bool WellBoreRepOn3D::setSampleUnit(SampleUnit type) {
	if (type==SampleUnit::TIME && !m_data->isWellCompatibleForTime(true)) {
		m_sampleUnit = SampleUnit::NONE;
		return false;
	} else {
		m_sampleUnit = type;
		return true;
	}
}

QList<SampleUnit> WellBoreRepOn3D::getAvailableSampleUnits() const {
	QList<SampleUnit> list;
	if (m_data->isTfpDefined()) {
		list.push_back(SampleUnit::TIME);
	}
	list.push_back(SampleUnit::DEPTH);
	return list;
}

QString WellBoreRepOn3D::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	if (list.contains(sampleUnit)) {
		return "Failure to load supported unit";
	} else{
		return "Unknown unit";
	}
}

void WellBoreRepOn3D::updatePicks() {
	bool isShown = m_layer3D!=nullptr && m_layer3D->isShown();

	const QList<AbstractGraphicRep*>& reps = m_parent->getVisibleReps();
	std::size_t index = 0;
	while (index<reps.size()) {
		WellPickRep* pickRep = dynamic_cast<WellPickRep*>(reps[index]);
		if (pickRep!=nullptr && dynamic_cast<WellPick*>(pickRep->data())->wellBore()==m_data) {
			pickRep->wellBoreLayerChanged(isShown, this);
		}
		index++;
	}
}

bool WellBoreRepOn3D::isLayerShown() const {
	return m_layer3D!=nullptr && m_layer3D->isShown();
}

void WellBoreRepOn3D::updatedParameterState() {
	if (m_propPanel!=nullptr) {
		m_defaultWidth = m_propPanel->defaultWidth();
		m_minimalWidth = m_propPanel->minimalWidth();
		m_maximalWidth = m_propPanel->maximalWidth();
		m_logMin = m_propPanel->logMin();
		m_logMax = m_propPanel->logMax();
		m_defaultColor = m_propPanel->defaultColor();

		if(m_layer3D != nullptr)
		{
			m_layer3D->setDefaultWidth(m_defaultWidth);
			m_layer3D->setMinimalWidth(m_minimalWidth);
			m_layer3D->setMaximalWidth(m_maximalWidth);
			m_layer3D->setLogMin(m_logMin);
			m_layer3D->setLogMax(m_logMax);
			m_layer3D->setDefaultColor(m_defaultColor);

			if (m_layer3D->isShown()) {
				m_layer3D->updateLog();
				//m_layer3D->hide();
				//m_layer3D->show();
			}
		}
	}
}

// MZR 18082021
void WellBoreRepOn3D::buildContextMenu(QMenu *menu){
	QAction *deleteAction = new QAction(tr("Delete Wells 2"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteWellBoreRepOn3D()));
}

void WellBoreRepOn3D::deleteWellBoreRepOn3D(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	disconnect(m_data, nullptr, this, nullptr);
	m_data->deleteRep();

	if(m_layer3D != nullptr){
		m_layer3D->hide();
	}
	WorkingSetManager *manager = m_data->workingSetManager();
	manager->deleteWellHead(m_data->wellHead());

	this->deleteLater();
}

AbstractGraphicRep::TypeRep WellBoreRepOn3D::getTypeGraphicRep() {
    return AbstractGraphicRep::Image3D;
}

void WellBoreRepOn3D::reExtractDeviation() {
	if (m_layer3D) {
		m_layer3D->refresh();
	}
}
