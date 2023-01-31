#include "basemapview.h"

#include <QGraphicsView>
#include <QMenu>

#include <iostream>

#include "../dialog/exportmultilayerblocdialog.h"
#include "basemapqglgraphicsview.h"
#include "qglgriditem.h"
#include "qglscalebaritem.h"
#include "qglgridaxisitem.h"

#include "rulerpicking.h"
//#include "GeRectangle.h"
//#include "GeEllipse.h"
//#include "GePolygon.h"
//#include "GeRectanglePicking.h"
//#include "GeObjectId.h"
//#include "GeGlobalParameters.h"

#include "rgblayerslice.h"
#include "LayerSlice.h"
#include "abstractgraphicrep.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "fixedrgblayersfromdatasetandcuberep.h"
#include "sismagedbmanager.h"
#include "seismic3ddataset.h"
#include "layerings.h"
#include "cultural.h"
#include "culturals.h"
#include "isochron.h"
#include "stringselectordialog.h"
#include "exportlayerdialog.h"
#include "GraphicsPointerExt.h"
#include "culturalcategory.h"
#include "KohonenOnMultispectralImages.h"
#include "folderdata.h"
#include "workingsetmanager.h"
#include "selectorcreatelayerdialog.h"
#include "rgblayerfromdataset.h"
#include "fixedlayerfromdataset.h"


int BaseMapView::GRID_ITEM_Z = 0;

BaseMapView::BaseMapView(bool restictToMonoTypeSplit,QString uniqueName,eModeView typeView, AbstractGraphicsView* geoTimeView) :
		Abstract2DInnerView(restictToMonoTypeSplit,new BaseMapQGLGraphicsView(),new BaseMapQGLGraphicsView(),new BaseMapQGLGraphicsView(), uniqueName,typeView, geoTimeView) {
	m_viewType = ViewType::BasemapView;
	m_baseGridItem = new QGLGridItem(m_worldBounds);
	m_baseGridItem->setZValue(GRID_ITEM_Z);
	m_scene->addItem(m_baseGridItem);

	m_verticalAxis=new QGLGridAxisItem(m_worldBounds,VERTICAL_AXIS_SIZE,QGLGridAxisItem::Direction::VERTICAL);
	m_verticalAxisScene->addItem(m_verticalAxis);
	m_verticalAxisScene->setSceneRect(m_verticalAxis->boundingRect());

	m_horizontalAxis=new QGLGridAxisItem(m_worldBounds,HORIZONTAL_AXIS_SIZE,QGLGridAxisItem::Direction::HORIZONTAL);
	m_horizontalAxisScene->addItem(m_horizontalAxis);
	m_horizontalAxisScene->setSceneRect(m_horizontalAxis->boundingRect());

}

void BaseMapView::showRep(AbstractGraphicRep *rep) {
	Abstract2DInnerView::showRep(rep);
	if(m_visibleReps.size()==1)
		resetZoom();
}

void BaseMapView::updateAxisExtent(const QRectF &worldExtent)
{
	QRectF hWorldExtent = worldExtent;
	hWorldExtent.setY(0);
	hWorldExtent.setHeight(HORIZONTAL_AXIS_SIZE);
	m_horizontalAxis->updateWorldExtent(hWorldExtent);
	m_horizontalAxisScene->setSceneRect(hWorldExtent);

	QRectF vWorldExtent = worldExtent;
	vWorldExtent.setX(0);
	vWorldExtent.setWidth(VERTICAL_AXIS_SIZE);
	m_verticalAxis->updateWorldExtent(vWorldExtent);
	m_verticalAxisScene->setSceneRect(vWorldExtent);
}

bool BaseMapView::updateWorldExtent(const QRectF &worldExtent) {
	bool changed = Abstract2DInnerView::updateWorldExtent(worldExtent);
	if (changed)
	{
		updateAxisExtent(m_worldBounds);
		m_baseGridItem->updateWorldExtent(m_worldBounds);
	}
	return changed;
}

bool BaseMapView::absoluteWorldToViewWorld(MouseTrackingEvent &event)
{
	return true;
}
bool BaseMapView::viewWorldToAbsoluteWorld(MouseTrackingEvent &event)
{
	return true;
}

void BaseMapView::contextualMenuFromGraphics(double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& menu) {
//	QMenu menu("Seismic", m_view);
	m_contextualWorldX = worldX;
	m_contextualWorldY = worldY;



	QAction *displayGraphicTool = menu.addAction(("Display Graphic Tools"));
	connect(displayGraphicTool, SIGNAL(triggered(bool)), this,
			SLOT(startGraphicToolsDialog()));

	QAction *saveGraphicLayer = menu.addAction(("Save Graphic Layer"));
	connect(saveGraphicLayer, SIGNAL(triggered(bool)), this,
			SLOT(saveGraphicLayer()));

	QAction *loadCultural = menu.addAction(("Load/Manage Graphic Layer"));
	connect(loadCultural, SIGNAL(triggered(bool)), this,
			SLOT(loadCultural()));

	QAction *rulerAction = menu.addAction(("Demonstrate Ruler"));
	rulerAction->setCheckable(m_isRulerOn);
	connect(rulerAction, SIGNAL(triggered(bool)), this,
			SLOT(startRuler(bool)));

//	QAction *rectanglePickingAction = menu.addAction(("Demonstrate Rectangle Picking"));
//	rectanglePickingAction->setCheckable(m_isRectanglePickingOn);
//	connect(rectanglePickingAction, SIGNAL(triggered(bool)), this,
//			SLOT(startRectanglePicking(bool)));
//
//	QAction *ellipsePickingAction = menu.addAction(("Demonstrate Ellipse Picking"));
//	ellipsePickingAction->setCheckable(m_isEllipsePickingOn);
//	connect(ellipsePickingAction, SIGNAL(triggered(bool)), this,
//			SLOT(startEllipsePicking(bool)));
//
//	QAction *polygonPickingAction = menu.addAction(("Demonstrate Polygon Picking"));
//	polygonPickingAction->setCheckable(m_isPolygonPickingOn);
//	connect(polygonPickingAction, SIGNAL(triggered(bool)), this,
//			SLOT(startPolygonPicking(bool)));

	std::size_t index = 0;
	while (index<m_visibleReps.size() && (dynamic_cast<RGBLayerSlice*>(m_visibleReps[index]->data())==nullptr &&
			dynamic_cast<LayerSlice*>(m_visibleReps[index]->data())==nullptr &&
			dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(m_visibleReps[index]->data())==nullptr)) {
		index++;
	}
	if (index<m_visibleReps.size()) {
		QAction *actionExport2Sismage = menu.addAction(("Export to Sismage"));
		QObject::connect(actionExport2Sismage, &QAction::triggered, this, &BaseMapView::export2Sismage);

		QAction *actionComputeTmap = menu.addAction(("Compute Tmap on MultiSpectral image"));
		QObject::connect(actionComputeTmap, &QAction::triggered, this, &BaseMapView::computeTmap);
	}

	index = 0;
	while (index<m_visibleReps.size() &&
			(dynamic_cast<FixedRGBLayersFromDatasetAndCubeRep*>(m_visibleReps[index])==nullptr)) {
		index++;
	}
	if (index<m_visibleReps.size()) {
		QAction *actionExport2Sismage = menu.addAction(("Export Multi Layer to Sismage"));
		QObject::connect(actionExport2Sismage, &QAction::triggered, this, &BaseMapView::exportMultiLayer2Sismage);
	}


//	QPoint mapPos = m_view->mapFromScene(QPointF( worldX, worldY));
//	QPoint globalPos = m_view->mapToGlobal(mapPos);
//	menu.exec(globalPos);
}

/**
 * Start demo ruler, checked not used
 */




void BaseMapView::startRuler(bool checked) {
	if (!m_isRulerOn) {
		m_isRulerOn = true;
		m_rulerPicking = new RulerPicking(this, 1);
		m_rulerPicking->initCanvas( m_scene);
		this->registerPickingTask(m_rulerPicking);
	} else {
		m_isRulerOn = false;
		m_rulerPicking->releaseCanvas( m_scene);
		this->unregisterPickingTask(m_rulerPicking);
		delete m_rulerPicking;
		m_rulerPicking = nullptr;
	}
}

/**
 * Start demo rectangle picking, checked not used
 */
//void BaseMapView::startRectanglePicking(bool checked) {
//	if (!m_isRectanglePickingOn) {
//		m_isRectanglePickingOn = true;
//
//
//		GeGlobalParameters globalParameters;
//		m_rectanglePicking = new GeRectanglePicking(globalParameters,
//				this);
//		m_rectanglePicking->initCanvas(m_scene);
//		this->registerPickingTask(m_rectanglePicking);
//
////		QColor color(Qt::green);
////		GeObjectId objectId(1, 1);
////
////		m_scene->addItem(m_geRectangle);
//	} else {
//		m_isRectanglePickingOn = false;
//		this->unregisterPickingTask(m_rectanglePicking);
////		if ( m_geRectangle != nullptr ) {
////			delete m_geRectangle;
////			m_geRectangle = nullptr;
////		}
//	}
//}

/**
 * Start demo ellipse picking, checked not used
 */
//void BaseMapView::startEllipsePicking(bool checked) {
//	if (!m_isEllipsePickingOn) {
//		m_isEllipsePickingOn = true;
//		QRectF rect(m_contextualWorldX + 50, m_contextualWorldY + 50,
//				2000, 1500);
//		QColor color(Qt::green);
//		GeObjectId objectId(1, 1);
//		GeGlobalParameters globalParameters;
//		m_geEllipse = new GeEllipse(
//			globalParameters,
//			objectId, color, true, rect, this);
//		m_scene->addItem(m_geEllipse);
//	} else {
//		m_isEllipsePickingOn = false;
//		if ( m_geEllipse != nullptr ) {
//			delete m_geEllipse;
//			m_geEllipse = nullptr;
//		}
//	}
//}

/**
 * Start demo polygon picking, checked not used
 */
//void BaseMapView::startPolygonPicking(bool checked) {
//	if (!m_isPolygonPickingOn) {
//		m_isPolygonPickingOn = true;
//		QColor color(Qt::green);
////		GeObjectId objectId(1, 1);
////		GeGlobalParameters globalParameters;
////		m_gePolygon = new GePolygon(
////			globalParameters,
////			objectId, color, true, this);
////
////		QPolygon poly;
////		poly << QPoint(m_contextualWorldX + 50, m_contextualWorldY + 50) <<
////				QPoint(m_contextualWorldX + 50, m_contextualWorldY - 50) <<
////				QPoint(m_contextualWorldX - 50, m_contextualWorldY - 50) <<
////				QPoint(m_contextualWorldX - 50, m_contextualWorldY + 50);
////		m_gePolygon->setPolygon(poly);
////		m_scene->addItem(m_gePolygon);
//		GeGlobalParameters globalParameters;
//		GraphicsPointerExt* ext = new GraphicsPointerExt(globalParameters, this); // TEST TO REMOVE
//		registerPickingTask(ext);
//		ext->setCurrentAction(GraphicsPointerExt::PolygonType);
//		connect(ext, &GraphicsPointerExt::endEditionItem, [this, ext](QGraphicsItem* item) {
//			unregisterPickingTask(ext);
//			delete ext;
//			GePolygon* polygonItem = qgraphicsitem_cast<GePolygon*>(item);
//			if (polygonItem!=nullptr) {
//				QPolygonF poly = polygonItem->polygon();
//			}
//			scene()->removeItem(item);
//			delete item;
//		});
//	} else {
//		m_isPolygonPickingOn = false;
//		if ( m_gePolygon != nullptr ) {
//			delete m_gePolygon;
//			m_gePolygon = nullptr;
//		}
//	}
//}

void BaseMapView::export2Sismage() {
	long index = m_visibleReps.size()-1;
	while (index>=0 && (
			dynamic_cast<RGBLayerSlice*>(m_visibleReps[index]->data())==nullptr &&
			dynamic_cast<LayerSlice*>(m_visibleReps[index]->data())==nullptr &&
			dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(m_visibleReps[index]->data())==nullptr)) {
		index--;
	}
    if (index>=0) {
    	LayerSlice* datasetGrayCVisual = dynamic_cast<LayerSlice*>(m_visibleReps[index]->data());
		RGBLayerSlice* datasetRgbCVisual = dynamic_cast<RGBLayerSlice*>(m_visibleReps[index]->data());
		if ( datasetGrayCVisual != nullptr) {
			// ---- Export to Layer
			Seismic3DDataset * datasetS = datasetGrayCVisual->seismic();

			std::string layerinsPath =
							SismageDBManager::datasetPath2LayerPath(datasetS->path().c_str());

			if( layerinsPath.empty() ) return;

			Layerings layerings(layerinsPath, "NextVision");

			std::vector<std::string> namesVector = layerings.getNames();
			QStringList namesList;
			for (int i = 0; i < namesVector.size(); i++) {
					QString nameQS(namesVector.at(i).c_str());
					namesList.append(nameQS);
			}

			StringSelectorDialog ssd(&namesList, QString("Layering selection"));
			if ( ssd.exec() != QDialog::Accepted)
				return;
			int selectedIndex = ssd.getSelectedIndex();
			QString selectedString = ssd.getSelectedString();
			std::cout << "SELECT STRING " << selectedIndex << " " << selectedString.toStdString() << std::endl;
			if (selectedIndex == -1)
				return;
			Layering* layering = layerings.getLayering(namesVector.at(selectedIndex));

			// Il faut retrouver la data qui contient le process
			layering->saveInto( datasetGrayCVisual );

			return;
		} else if ( datasetRgbCVisual != nullptr) {
			//RGBLayerSlice* datasetRgbCVisual = dynamic_cast<RGBLayerSlice*>(m_visibleReps[index]->data());
			// ---- Export to Cultural
			Seismic3DDataset * datasetS = datasetRgbCVisual->layerSlice()->seismic();

			std::string culturalsPath =
							SismageDBManager::datasetPath2CulturalPath(datasetS->path().c_str());

			if( culturalsPath.empty() ) return;

			int dimH = datasetS->width();
			int dimW = datasetS->depth();
			Culturals culturals(culturalsPath);

			std::vector<std::string> namesVector = culturals.getNames(dimW, dimH);
			QStringList namesList;
			for (int i = 0; i < namesVector.size(); i++) {
					QString nameQS(namesVector.at(i).c_str());
					namesList.append(nameQS);
			}

			ExportLayerDialog ssd(namesList, QString("GeoRef Culturals selection"));
			if ( ssd.exec() != QDialog::Accepted || (!ssd.isNewName() && ssd.getSelectedIndex()==-1) || (ssd.isNewName() &&
					(ssd.newName().isNull() || ssd.newName().isEmpty() )))
					return;

			bool isNewNameInList = false;
			int selectedIndex = ssd.getSelectedIndex();
			QString selectedString = ssd.getSelectedString();
			if (ssd.isNewName()) {
				std::size_t idx=0;
				while (idx<namesList.count() && !isNewNameInList) {
					isNewNameInList = namesList[idx].compare(ssd.newName())==0;
					if (isNewNameInList) {
						selectedIndex = idx;
						selectedString = ssd.newName();
					}
					idx++;
				}
			}

			if (!ssd.isNewName() || (isNewNameInList && ssd.isNewName())) {
				std::cout << "SELECT STRING " << selectedIndex << " " << selectedString.toStdString() << std::endl;
				if (selectedIndex == -1)
						return;
				Cultural* cultural = culturals.getCultural(namesVector.at(selectedIndex));

				// Il faut retrouver la data qui contient le process
				cultural->saveInto( datasetRgbCVisual );
			} else {
				CulturalCategory  nvCulturalCategory(culturalsPath, "NextVision");
				Cultural cultural(culturalsPath, ssd.newName().toStdString(), nvCulturalCategory);

				// Il faut retrouver la data qui contient le process
				cultural.saveInto( datasetRgbCVisual, true);
			}
			return;
		} else {
			FixedRGBLayersFromDatasetAndCube* layer = dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(m_visibleReps[index]->data());
			// ---- Export to Cultural
			std::string culturalsPath = SismageDBManager::surveyPath2CulturalPath(
					layer->surveyPath().toStdString());

			if( culturalsPath.empty() ) return;

			int dimH = layer->width();
			int dimW = layer->depth();
			Culturals culturals(culturalsPath);

			std::vector<std::string> namesVector = culturals.getNames(dimW, dimH);
			QStringList namesList;
			for (int i = 0; i < namesVector.size(); i++) {
					QString nameQS(namesVector.at(i).c_str());
					namesList.append(nameQS);
			}

			ExportLayerDialog ssd(namesList, QString("GeoRef Culturals selection"));
			if ( ssd.exec() != QDialog::Accepted || (!ssd.isNewName() && ssd.getSelectedIndex()==-1) || (ssd.isNewName() &&
					(ssd.newName().isNull() || ssd.newName().isEmpty() )))
					return;

			bool isNewNameInList = false;
			int selectedIndex = ssd.getSelectedIndex();
			QString selectedString = ssd.getSelectedString();
			if (ssd.isNewName()) {
				std::size_t idx=0;
				while (idx<namesList.count() && !isNewNameInList) {
					isNewNameInList = namesList[idx].compare(ssd.newName())==0;
					if (isNewNameInList) {
						selectedIndex = idx;
						selectedString = ssd.newName();
					}
					idx++;
				}
			}

			CUDAImagePaletteHolder *red = new CUDAImagePaletteHolder(
					layer->getNbTraces(), layer->getNbProfiles(),
							ImageFormats::QSampleType::INT16/*,
							m_data->ijToInlineXlineTransfoForXline(), parent*/);
			CUDAImagePaletteHolder *green = new CUDAImagePaletteHolder(
					layer->getNbTraces(), layer->getNbProfiles(),
							ImageFormats::QSampleType::INT16);
			CUDAImagePaletteHolder *blue = new CUDAImagePaletteHolder(
					layer->getNbTraces(), layer->getNbProfiles(),
							ImageFormats::QSampleType::INT16);

			CUDAImagePaletteHolder *iso = new CUDAImagePaletteHolder(
					layer->getNbTraces(), layer->getNbProfiles(),
							ImageFormats::QSampleType::INT16);

			layer->getImageForIndex(layer->currentImageIndex(), red, green, blue, iso);

			int minimumValue = 0;
			if (layer->isMinimumValueActive()) {
				minimumValue = std::floor(layer->minimumValue() * 255);
			}
			if (!ssd.isNewName() || (isNewNameInList && ssd.isNewName())) {
				std::cout << "SELECT STRING " << selectedIndex << " " << selectedString.toStdString() << std::endl;
				if (selectedIndex == -1)
						return;
				Cultural* cultural = culturals.getCultural(namesVector.at(selectedIndex));

				// Il faut retrouver la data qui contient le process
				cultural->saveInto(red, green, blue, minimumValue, layer->ijToXYTransfo());
			} else {
				CulturalCategory  nvCulturalCategory(culturalsPath, "NextVision");
				Cultural cultural(culturalsPath, ssd.newName().toStdString(), nvCulturalCategory);

				// Il faut retrouver la data qui contient le process
				cultural.saveInto(red, green, blue, minimumValue, layer->ijToXYTransfo(), true);
			}
			return;
		}
	}
}

//
void BaseMapView::exportMultiLayer2Sismage() {
	long index = m_visibleReps.size()-1;
	while (index>=0 && (
			dynamic_cast<FixedRGBLayersFromDatasetAndCubeRep*>(m_visibleReps[index])==nullptr )) {
		index--;
	}
    if (index<0) return;

    FixedRGBLayersFromDatasetAndCubeRep* datasetVisual = dynamic_cast<FixedRGBLayersFromDatasetAndCubeRep*>
			(m_visibleReps[index]);
	if ( datasetVisual == nullptr) return;

	FixedRGBLayersFromDatasetAndCube * cube =
			dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(
					datasetVisual->fixedRGBLayersFromDataset());

	QString dialogName ("Export Multi-Layer " + cube->name());
	ExportMultiLayerBlocDialog* dialog = new ExportMultiLayerBlocDialog(dialogName, cube, NULL);
	dialog->resize(500, 250);
	dialog->exec();

}

void BaseMapView::computeTmap() {
	long index = m_visibleReps.size()-1;
	while (index>=0 && (
			dynamic_cast<RGBLayerSlice*>(m_visibleReps[index]->data())==nullptr &&
			dynamic_cast<LayerSlice*>(m_visibleReps[index]->data())==nullptr)) {
		index--;
	}
    if (index>=0) {
    	LayerSlice* data = nullptr;
    	if (RGBLayerSlice* rgblayerslice = dynamic_cast<RGBLayerSlice*>(m_visibleReps[index]->data())) {
    		data = rgblayerslice->layerSlice();
    	} else if (LayerSlice* layerslice = dynamic_cast<LayerSlice*>(m_visibleReps[index]->data())) {
    		data = layerslice;
    	}
		QString tmapLabel;
		RgbLayerFromDataset* savePropertiesRgb = nullptr;
		FixedLayerFromDataset* savePropertiesGray = nullptr;
		int tmapSize = 256;
		int exampleStep = 10;
		bool isTmap = true;
		if (data!=nullptr && data->isModuleComputed()) {
			QList<RgbLayerFromDataset*> currentLayersRgb;
			QList<FixedLayerFromDataset*> currentLayersGray;
			for (IData* idata : data->workingSetManager()->folders().horizonsFree->data()) {
				if (RgbLayerFromDataset* layer = dynamic_cast<RgbLayerFromDataset*>(idata)){
					if (layer->width()==data->width() && layer->depth()==data->depth()) {
						currentLayersRgb.push_back(layer);
					}
				} else if (FixedLayerFromDataset* layer = dynamic_cast<FixedLayerFromDataset*>(idata)){
					if (layer->width()==data->width() && layer->depth()==data->depth()) {
						currentLayersGray.push_back(layer);
					}
				}
			}
			SelectOrCreateLayerDialog dialog(currentLayersRgb, currentLayersGray, "pca", "tmap");
			dialog.setParamTmapSize(tmapSize);
			dialog.setMaxExampleStep(std::max(data->width(), data->depth()));
			dialog.setParamExampleStep(exampleStep);
			int code = dialog.exec();

			if (code==QDialog::Accepted) {
				isTmap = dialog.isTmapChoosen();
				tmapLabel = dialog.label();
				tmapSize = dialog.paramTmapSize();
				exampleStep = dialog.paramExampleStep();
				if (dialog.isPcaChoosen()) {
					if (dialog.isLayerNew() || dialog.layerIndex()>=currentLayersRgb.size() || dialog.layerIndex()<0) {
						savePropertiesRgb = new RgbLayerFromDataset(dialog.layer(), data->workingSetManager(), data->seismic());
						data->workingSetManager()->addRgbLayerFromDataset(savePropertiesRgb);
					} else {
						savePropertiesRgb = currentLayersRgb[dialog.layerIndex()];
					}
				} else {
					if (dialog.isLayerNew() || dialog.layerIndex()>=currentLayersGray.size() || dialog.layerIndex()<0) {
						savePropertiesGray = new FixedLayerFromDataset(dialog.layer(), data->workingSetManager(), data->seismic());
						data->workingSetManager()->addFixedLayerFromDataset(savePropertiesGray);
					} else{
						savePropertiesGray = currentLayersGray[dialog.layerIndex()];
					}
				}
			}

		}
		// recheck in case module has benn modified
    	if (data!=nullptr && data->isModuleComputed() && ((savePropertiesRgb!=nullptr && !isTmap) ||
    			(isTmap && savePropertiesGray!=nullptr))) {

    		std::vector<const float*> stack;
    		stack.resize((data->getNbOutputSlices()-2)/2);
    		for (std::size_t idx=0; idx<(data->getNbOutputSlices()-2)/2; idx++) {
    			stack[idx] = data->getModuleData(idx+2);
    		}
    		KohonenCudaPlanarImage2D image(stack, data->width(), data->depth(), -9999);
    		KohonenOnMultispectralImages kohonen(&image, isTmap, false);
    		if (isTmap) {
    			kohonen.setOutputHorizonProperties(savePropertiesGray, tmapLabel);
    		} else {
    			kohonen.setOutputHorizonProperties(savePropertiesRgb, tmapLabel);
    		}
    		kohonen.compute(tmapSize, exampleStep);
    	} else if (data!=nullptr && !data->isModuleComputed()) {
    		qDebug() << "BaseMapView::computeTmap data module not computed";
    	}
    }
}

double BaseMapView::getWellMapWidth() const {
	return m_wellMapWidth;
}

void BaseMapView::setWellMapWidth(double value) {
	if(m_wellMapWidth != value) {
		m_wellMapWidth = value;
		emit signalWellMapWidth(m_wellMapWidth);
	}
}

BaseMapView::~BaseMapView() {
	if ( m_rulerPicking != nullptr)
		delete m_rulerPicking;
}

