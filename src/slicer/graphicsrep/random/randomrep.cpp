#include "randomrep.h"
#include "cudaimagepaletteholder.h"
#include "randomproppanel.h"
#include "randomlayer.h"
#include "qgllineitem.h"
#include "qglisolineitem.h"
#include "seismic3dabstractdataset.h"
#include "abstractinnerview.h"
#include "randomlineview.h"
#include "affine2dtransformation.h"
#include "cubeseismicaddon.h"
#include "seismicsurvey.h"
#include <iostream>

#include <QAction>
#include <QMenu>

RandomRep::RandomRep(Seismic3DAbstractDataset *data,
		const LookupTable& attributeLookupTable, AbstractInnerView *parent) :
	  AbstractGraphicRep(parent), IMouseImageDataProvider() {
	m_data = data;
	m_channel = 0;

	m_image = nullptr;
	m_defaultLookupTable = attributeLookupTable;
	m_transformation = nullptr;

	m_propPanel = nullptr;
	m_layer = nullptr;
	m_controler = nullptr;
	m_showColorScale = false;

	m_name=m_data->name();
	//m_UpdatedRep = false;
	// MZR 19082021
	m_data->addRep(this);
	connect(m_data,&Seismic3DAbstractDataset::deletedMenu,this,&RandomRep::deleteRandomRep);

	connect(m_data, SIGNAL(rangeLockChanged()), this, SLOT(rangeLockChanged()));
}

void RandomRep::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
}

IData* RandomRep::data() const {
	return m_data;
}


Seismic3DAbstractDataset* RandomRep::getdataset() const
{
	return m_data;
}

void RandomRep::showColorScale(bool val) {
	m_showColorScale = val;
	m_layer->showColorScale(m_showColorScale);
}
bool RandomRep::colorScale() const {
	return m_showColorScale;
}

QWidget* RandomRep::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new RandomPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* RandomRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		bool result = createImagePaletteHolder();
		if (result) {
			m_layer = new RandomLayer(this, scene, defaultZDepth, parent);
			m_layer->showColorScale(m_showColorScale);
			connect(m_layer, &RandomLayer::hidden, this, &RandomRep::relayHidden);
		}
	}
	return m_layer;
}


void RandomRep::setPolyLine(QPolygonF poly)
{
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());
	if (randomView)
	{
		m_discreatePolygon = randomView->discreatePolyLine();
		//if (m_discreatePolygon.size()>0  ) {
			if(m_image == nullptr)
			{
				createImagePaletteHolder();
			}
			else
			{
				if(m_discreatePolygon.size() != m_image->width())
				{
					//QVector2D range = m_image->range();
					float opacity = m_image->opacity();
					LookupTable palette = m_image->lookupTable();
					cleanImage();
					createImagePaletteHolder();
					m_image->setLookupTable(palette);
				    m_image->setOpacity(opacity);
				    if (m_data->isRangeLocked()) {
				    	m_image->setRange(m_data->lockedRange());
				    }

				}
				else
				{
					loadRandom();

				}

			}

		//}
		/*else
		{
			cleanImage();
		}*/

		if (m_layer != nullptr) {
			m_layer->updateImage();
		}
	}

}

void RandomRep::cleanImage()
{
	if(m_image != nullptr)
	{
		m_cache.reset(nullptr);
		m_image->deleteLater();
		m_image =nullptr;
	}
}

void RandomRep::deleteLayer() {
    if (m_layer != nullptr)
         delete m_layer;

    if(m_image != nullptr){
        delete m_image;
    }

    if (m_propPanel != nullptr)
            delete m_propPanel;

    m_layer = nullptr;
    m_image = nullptr;
    m_propPanel = nullptr;
}

void RandomRep::refreshLayer() {
	if (m_layer == nullptr)
		return;

	m_layer->refresh();
}

RandomRep::~RandomRep() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;

	m_data->deleteRep(this);
}

void RandomRep::loadRandom() {
//	m_currentSlice = z;
//	m_data->loadInlineXLine(m_image, m_dir, z);

	if (m_image!=nullptr) {
		m_data->loadRandomLine(m_image, m_discreatePolygon, m_channel, m_cache.get()); //(void*) m_cache.data());
	}
}

bool RandomRep::mouseData(double x, double y, MouseInfo &info) {
	double value;
	bool valid = false;
	if (m_image!=nullptr) {
		valid = IGeorefImage::value(m_image, x, y, info.i, info.j, value);
		info.valuesDesc.push_back("Image");
		info.values.push_back( value);
		info.depthValue = true;
		info.depth=y;
		info.depthUnit = m_data->cubeSeismicAddon().getSampleUnit();
	}
	return valid;
}

bool RandomRep::createImagePaletteHolder() {
	bool isValid = false;
	RandomLineView* randomView = dynamic_cast<RandomLineView*>(view());
	if (randomView) {
		m_discreatePolygon = randomView->discreatePolyLine();
		//if (m_discreatePolygon.size()>0) {
			const AffineTransformation* sampleTransform = m_data->sampleTransformation();
			std::array<double, 6> transform;

			transform[0]=0;
			transform[1]=1;
			transform[2]=0;

			transform[3]=sampleTransform->b();
			transform[4]=0;
			transform[5]=sampleTransform->a();

			m_transformation = new Affine2DTransformation(m_discreatePolygon.size(), m_data->height(), transform, this);
			m_image = new CUDAImagePaletteHolder(
					m_discreatePolygon.size(), m_data->height(),
					m_data->sampleType(),
					m_transformation, randomView);
			m_image->setLookupTable(m_defaultLookupTable);
			if(m_name.toLower().contains("rgt")){
			    m_image->setOpacity(0.5f);
			}

			if (m_data->isRangeLocked()) {
				m_image->setRange(m_data->lockedRange());
			}

			//m_cache.resize(m_image->width() * m_image->height() * m_data->dimV() * m_data->sampleType().byte_size());
			m_cache.reset(m_data->createRandomCache(m_discreatePolygon));

			connect(m_image, SIGNAL(dataChanged()), this, SLOT(dataChanged()));
			isValid = true;
	//	}
	}
	if (isValid) {
		loadRandom();
	}
	return isValid;
}

bool RandomRep::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> RandomRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString RandomRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

int RandomRep::channel() const {
	return m_channel;
}

void RandomRep::setChannel(int channel) {
	m_channel = channel;
	loadRandom();
	emit channelChanged(channel);
}

const std::vector<char>& RandomRep::lockCache() {
	m_cacheMutex.lock();
	return m_cache->buffer();
}

void RandomRep::unlockCache() {
	m_cacheMutex.unlock();
}

// MZR 19082021
void RandomRep::buildContextMenu(QMenu *menu){
	QAction *deleteAction = new QAction(tr("Delete seismic"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteRandomRep()));
}

void RandomRep::deleteRandomRep(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	WorkingSetManager *manager = const_cast<WorkingSetManager*>(m_data->workingSetManager());
	SeismicSurvey* survey = const_cast<SeismicSurvey*>(m_data->survey());
	QList<Seismic3DAbstractDataset*> list = survey->datasets();
	for (int i = 0; i < list.size(); ++i) {
		if (list.at(i)->name()== m_name){
			Seismic3DAbstractDataset* dataSet = list.at(i);
			dataSet->deleteRep(this);
			disconnect(m_data,nullptr,this,nullptr);
			dataSet->deleteRep();
			

			if(dataSet->getRepListSize() == 0)
				survey->removeDataset(dataSet);

			break;
		}
	}

	this->deleteLater();
}

void RandomRep::rangeLockChanged() {
	if (m_data->isRangeLocked() && m_image) {
		m_image->setRange(m_data->lockedRange());
	}
}

AbstractGraphicRep::TypeRep RandomRep::getTypeGraphicRep() {
	if (m_data->type()==Seismic3DAbstractDataset::CUBE_TYPE::RGT) {
		return AbstractGraphicRep::ImageRgt;
	} else {
		 return AbstractGraphicRep::Image;
	}
}

void RandomRep::relayHidden() {
   emit layerHidden();
}
