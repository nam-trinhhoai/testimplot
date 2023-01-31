#include "datasetrep.h"
#include <iostream>
#include <QMenu>
#include <QAction>
#include <QMessageBox>
#include <QFileDialog>
#include "slicelayer.h"
#include "seismic3dabstractdataset.h"
#include "seismicdataset3Dlayer.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "cudaimagepaletteholder.h"
#include "dataset3Dslicerep.h"
#include "abstractinnerview.h"
#include "fixedrgblayersfromdataset.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "workingsetmanager.h"
#include "videolayer.h"
#include "workingsetmanager.h"
#include "seismicsurvey.h"
#include "datacontroler.h"

DatasetRep::DatasetRep(Seismic3DAbstractDataset *data, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = data;
	m_layer = nullptr;
	m_name = m_data->name();

	connect(m_data,&Seismic3DAbstractDataset::deletedMenu,this,&DatasetRep::deleteDatasetRep);// MZR 19082021
	m_data->addRep(this);
}

IData* DatasetRep::data() const {
	return m_data;
}

void DatasetRep::buildContextMenu(QMenu *menu) {
	QAction *inlineAction = new QAction(tr("Add Inline"), this);
	menu->addAction(inlineAction);
	connect(inlineAction, SIGNAL(triggered()), this, SLOT(createInline()));

	QAction *xlineAction = new QAction(tr("Add Xline"), this);
	menu->addAction(xlineAction);
	connect(xlineAction, SIGNAL(triggered()), this, SLOT(createXline()));

	QAction *surfaceAction = new QAction(tr("Add Animated Surfaces"), this);
    menu->addAction(surfaceAction);
    connect(surfaceAction, SIGNAL(triggered()), this, SLOT(createAnimatedSurfaces()));

	QAction *surfaceFromCubeAction = new QAction(tr("Add Animated Surfaces from cube"), this);
    menu->addAction(surfaceFromCubeAction);
    connect(surfaceFromCubeAction, SIGNAL(triggered()), this, SLOT(createAnimatedSurfacesFromCube()));

	QAction *videoAction = new QAction(tr("Add video layer"), this);
	menu->addAction(videoAction);
	connect(videoAction, SIGNAL(triggered()), this, SLOT(createVideoLayer()));

	QAction *deleteAction = new QAction(tr("Delete seismic"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteDatasetRep()));
}

void DatasetRep::createInline() {
	std::array<double, 6> transfo = m_data->ijToInlineXlineTransfo()->direct();

	CUDAImagePaletteHolder *slice = new CUDAImagePaletteHolder(m_data->width(),
			m_data->height(), m_data->sampleType(),
			m_data->ijToInlineXlineTransfoForInline(), m_parent);
	slice->setLookupTable(m_data->defaultLookupTable());

	QPair<QVector2D, AffineTransformation> rangeAndStep(
			QVector2D(transfo[3],
					transfo[3] + transfo[5] * (m_data->depth() - 1)),
			AffineTransformation(transfo[5], transfo[3]));
	Dataset3DSliceRep *rep = new Dataset3DSliceRep(m_data, slice, rangeAndStep,
			SliceDirection::Inline, m_parent);
	insertChildRep(rep);

	connect(this,&DatasetRep::delete3DRep,rep,&Dataset3DSliceRep::delete3DRep);
}

void DatasetRep::createXline() {
	std::array<double, 6> transfo = m_data->ijToInlineXlineTransfo()->direct();
	CUDAImagePaletteHolder *slice = new CUDAImagePaletteHolder(m_data->depth(),
			m_data->height(), m_data->sampleType(),
			m_data->ijToInlineXlineTransfoForXline(), this);
	slice->setLookupTable(m_data->defaultLookupTable());
	QPair<QVector2D, AffineTransformation> rangeAndStep(
			QVector2D(transfo[0],
					transfo[0] + transfo[1] * (m_data->width() - 1)),
			AffineTransformation(transfo[1], transfo[0]));
	Dataset3DSliceRep *rep = new Dataset3DSliceRep(m_data, slice, rangeAndStep,
			SliceDirection::XLine, m_parent);
	insertChildRep(rep);

	connect(this,&DatasetRep::delete3DRep,rep,&Dataset3DSliceRep::delete3DRep);
}

QWidget* DatasetRep::propertyPanel() {
	return nullptr;
}
GraphicLayer* DatasetRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {

	return nullptr;
}

Graphic3DLayer* DatasetRep::layer3D(QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) {

	if (m_layer == nullptr)
		m_layer = new SeismicDataset3DLayer(this, parent, root, camera);

	return m_layer;
}

DatasetRep::~DatasetRep() {
	if (m_layer!=nullptr) {
		delete m_layer;
	}
	m_data->deleteRep(this);
}

void DatasetRep::createAnimatedSurfaces() {
	WorkingSetManager* manager = m_data->workingSetManager();
	FixedRGBLayersFromDataset* animateData = FixedRGBLayersFromDataset::createDataFromDatasetWithUI(
			QString("Animate ")+m_data->name(), manager, 
			m_data, m_data);
	if (animateData!=nullptr) {
		manager->addFixedRGBLayersFromDataset(animateData);
	} else {
		QMessageBox::information(view(), tr("Load Animated Surfaces"), tr("Could not create animated surfaces"));
	}
}

void DatasetRep::createAnimatedSurfacesFromCube() {
	WorkingSetManager* manager = m_data->workingSetManager();
	FixedRGBLayersFromDatasetAndCube* animateData = FixedRGBLayersFromDatasetAndCube::createDataFromDatasetWithUI(
			QString("Animate ")+m_data->name(), manager,
			m_data->survey(), m_data);
	if (animateData!=nullptr) {
		manager->addFixedRGBLayersFromDatasetAndCube(animateData);
	} else {
		QMessageBox::information(view(), tr("Load Animated Surfaces"), tr("Could not create animated surfaces"));
	}
}

void DatasetRep::createVideoLayer() {
	QString mediaPath = QFileDialog::getOpenFileName(view());
	VideoLayer* layer =  new VideoLayer(m_data->workingSetManager(), mediaPath, m_data);
	m_data->workingSetManager()->addVideoLayer(layer);
}

// MZR 16072021
void DatasetRep::deleteDatasetRep() {
	m_parent->hideRep(this);
	emit deletedRep(this);

	WorkingSetManager *manager = const_cast<WorkingSetManager*>(m_data->workingSetManager());
	SeismicSurvey* survey = const_cast<SeismicSurvey*>(m_data->survey());
	QList<Seismic3DAbstractDataset*> list = survey->datasets();
	for (int i = 0; i < list.size(); ++i) {
		if (list.at(i)->name()== m_name){
			emit delete3DRep();
			Seismic3DAbstractDataset* dataSet = list.at(i);
			dataSet->deleteRep(this);
			//if(dataSet->getTreeDeletionProcess() == false){
				disconnect(m_data,nullptr,this,nullptr);
				dataSet->deleteRep();
			//}
			if(dataSet->getRepListSize() == 0)
			    survey->removeDataset(dataSet);

			break;
		}
	}

	this->deleteLater();
}

bool DatasetRep::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> DatasetRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_data->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString DatasetRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}


void DatasetRep::setDataControler(DataControler *controler) {
	m_controler = controler;
}
DataControler* DatasetRep::dataControler() const {
	return m_controler;
}

AbstractGraphicRep::TypeRep DatasetRep::getTypeGraphicRep() {
    return Image3D;
}
