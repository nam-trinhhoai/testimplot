#include "stratisliceattributereponslice.h"

#include "stratisliceattributeproppanelonslice.h"
#include "abstractstratisliceattribute.h"
#include "abstractinnerview.h"
#include "stratisliceattributeslicelayer.h"
#include "cubeseismicaddon.h"
#include "stratislice.h"
#include "seismic3dabstractdataset.h"

StratiSliceAttributeRepOnSlice::StratiSliceAttributeRepOnSlice(AbstractStratiSliceAttribute *stratislice,const IGeorefImage * const transfoProvider, SliceDirection dir,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISliceableRep(),m_transfoProvider(transfoProvider){
	m_stratislice = stratislice;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_dir=dir;
	m_name = m_stratislice->name();
	m_currentSlice=0;
}


StratiSliceAttributeRepOnSlice::~StratiSliceAttributeRepOnSlice() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

AbstractStratiSliceAttribute* StratiSliceAttributeRepOnSlice::stratiSliceAttribute() const {
	return m_stratislice;
}

void StratiSliceAttributeRepOnSlice::setSliceIJPosition(int imageVal)
{
	m_currentSlice=imageVal;
	if(m_layer!=nullptr)
		m_layer->setSliceIJPosition(imageVal);
}

IData* StratiSliceAttributeRepOnSlice::data() const {
	return m_stratislice;
}


QWidget* StratiSliceAttributeRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_stratislice->initialize();
		m_propPanel = new StratiSliceAttributePropPanelOnSlice(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* StratiSliceAttributeRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_stratislice->initialize();
		m_layer = new StratiSliceAttributeSliceLayer(this,m_dir,m_transfoProvider,m_currentSlice, scene, defaultZDepth, parent);
	}
	return m_layer;
}

bool StratiSliceAttributeRepOnSlice::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> StratiSliceAttributeRepOnSlice::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_stratislice->stratiSlice()->seismic()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString StratiSliceAttributeRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep StratiSliceAttributeRepOnSlice::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}
