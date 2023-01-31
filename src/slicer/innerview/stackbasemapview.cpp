#include "stackbasemapview.h"

#include <QGraphicsView>
#include <QSlider>
#include <QSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QListIterator>
#include <iostream>

#include "basemapqglgraphicsview.h"
#include "isliceablerep.h"
#include "abstractgraphicrep.h"
#include "stacklayerrgtrep.h"
#include "qglgriditem.h"
#include "qglscalebaritem.h"
#include "qglgridaxisitem.h"

#if 1
#include "splittedview.h"
#endif

//#include "GeRectangle.h"
//#include "GeEllipse.h"
//#include "GePolygon.h"
//#include "GeObjectId.h"
//#include "GeGlobalParameters.h"
#include "LayerSlice.h"
#include "rgblayerslice.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "abstractgraphicrep.h"
#include "fixedrgblayersfromdatasetandcuberep.h"
#include "sismagedbmanager.h"
#include "seismic3ddataset.h"
#include "layerings.h"
#include "cultural.h"
#include "culturals.h"
#include "culturalcategory.h"
#include "stringselectordialog.h"
#include "exportmultilayerblocdialog.h"
#include "exportlayerdialog.h"
#include "fixedlayersfromdatasetandcube.h"
#include "seismicsurvey.h"
#include "folderdata.h"

#include "nurbswidget.h"

int StackBaseMapView::GRID_ITEM_Z = 0;

StackBaseMapView::StackBaseMapView(bool restictToMonoTypeSplit,QString uniqueName,eModeView typeView, KDDockWidgets::MainWindow* geoTimeView):
								Abstract2DInnerView(restictToMonoTypeSplit,new BaseMapQGLGraphicsView(),new BaseMapQGLGraphicsView(),new BaseMapQGLGraphicsView(), uniqueName,typeView, geoTimeView) {
	m_viewType = ViewType::StackBasemapView;
	m_baseGridItem = new QGLGridItem(m_worldBounds);
	m_baseGridItem->setZValue(GRID_ITEM_Z);
	m_scene->addItem(m_baseGridItem);

	m_verticalAxis=new QGLGridAxisItem(m_worldBounds,VERTICAL_AXIS_SIZE,QGLGridAxisItem::Direction::VERTICAL);
	m_verticalAxisScene->addItem(m_verticalAxis);
	m_verticalAxisScene->setSceneRect(m_verticalAxis->boundingRect());

	m_horizontalAxis=new QGLGridAxisItem(m_worldBounds,HORIZONTAL_AXIS_SIZE,QGLGridAxisItem::Direction::HORIZONTAL);
	m_horizontalAxisScene->addItem(m_horizontalAxis);
	m_horizontalAxisScene->setSceneRect(m_horizontalAxis->boundingRect());
	
	m_currentStack = 0;

	QWidget *stackBox = createStackBox();
	stackBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	m_mainLayout->insertWidget(0,stackBox);

#if 0
	QWidget *selectViewModeArea = createViewModeSelector();
	selectViewModeArea->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_mainLayout->insertWidget(0,selectViewModeArea);
#endif
}

void StackBaseMapView::showRep(AbstractGraphicRep *rep) {
	StackLayerRGTRep* stack = dynamic_cast<StackLayerRGTRep*>(rep);
	if(stack!=nullptr){
		int currentMin = m_stackSlider->minimum();
		int currentMax = m_stackSlider->maximum();
		QVector2D range = stack->stackRange();
		m_repToRange[rep] = range;
		recomputeRange();
		stack->setSliceIJPosition(m_stackSlider->value());


		connect(stack, &StackLayerRGTRep::stackRangeChanged, this, [this, stack](QVector2D newRange) {
			m_repToRange[stack] = newRange;
			recomputeRange();
		});
	}

	Abstract2DInnerView::showRep(rep);

	if (stack == nullptr){
		if(m_visibleReps.size()==1)
			m_stackLabel->setText("");
	} else {
		QString str = stack->getLabelFromPosition(m_stackSlider->value());
		if(str == ""){
			m_stackSlider->setValue(0);
			str = stack->getLabelFromPosition(m_stackSlider->value());
		}
		m_stackLabel->setText(str);
	}


	NurbsWidget::setView2D(this);
	if(m_visibleReps.size()==1)
	{
		resetZoom();

	}
}

void StackBaseMapView::hideRep(AbstractGraphicRep *rep) {
	Abstract2DInnerView::hideRep(rep);
	recomputeRange();
}

void StackBaseMapView::cleanupRep(AbstractGraphicRep *rep) {
	Abstract2DInnerView::cleanupRep(rep);
	recomputeRange();
}

void StackBaseMapView::updateAxisExtent(const QRectF &worldExtent)
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

bool StackBaseMapView::updateWorldExtent(const QRectF &worldExtent) {
	bool changed = Abstract2DInnerView::updateWorldExtent(worldExtent);
	if (changed)
	{
		updateAxisExtent(m_worldBounds);
		m_baseGridItem->updateWorldExtent(m_worldBounds);
	}
	return changed;
}

bool StackBaseMapView::absoluteWorldToViewWorld(MouseTrackingEvent &event)
{
	return true;
}
bool StackBaseMapView::viewWorldToAbsoluteWorld(MouseTrackingEvent &event)
{
	return true;
}

void StackBaseMapView::contextualMenuFromGraphics(double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& menu) {
	//	QMenu menu("Seismic", m_view);
	m_contextualWorldX = worldX;
	m_contextualWorldY = worldY;

	/*QAction *createnurbsAction = menu.addAction(("Create Nurbs"));
			connect(createnurbsAction, SIGNAL(triggered(bool)), this,
					SLOT(startCreateNurbs(bool)));*/

	QAction *displayGraphicTool = menu.addAction(("Display Graphic Tools"));
	connect(displayGraphicTool, SIGNAL(triggered(bool)), this,
			SLOT(startGraphicToolsDialog()));

	QAction *saveGraphicLayer = menu.addAction(("Save Graphic Layer"));
	connect(saveGraphicLayer, SIGNAL(triggered(bool)), this,
			SLOT(saveGraphicLayer()));

	QAction *loadCultural = menu.addAction(("Load/Manage Graphic Layer"));
	connect(loadCultural, SIGNAL(triggered(bool)), this,
			SLOT(loadCultural()));

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
			dynamic_cast<LayerSlice*>(m_visibleReps[index]->data())==nullptr) &&
			dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(m_visibleReps[index]->data())==nullptr &&
			dynamic_cast<FixedLayersFromDatasetAndCube*>(m_visibleReps[index]->data())==nullptr) {
		index++;
	}
	if (index<m_visibleReps.size()) {
		QAction *actionExport2Sismage = menu.addAction(("Export to Sismage"));
		QObject::connect(actionExport2Sismage, &QAction::triggered, this, &StackBaseMapView::export2Sismage);
	}

	index = 0;
	while (index<m_visibleReps.size() &&
			(dynamic_cast<FixedRGBLayersFromDatasetAndCubeRep*>(m_visibleReps[index])==nullptr)) {
		index++;
	}
	if (index<m_visibleReps.size()) {
		QAction *actionExport2Sismage = menu.addAction(("Export Multi Layer to Sismage"));
		QObject::connect(actionExport2Sismage, &QAction::triggered, this, &StackBaseMapView::exportMultiLayer2Sismage);
	}

	//	QPoint mapPos = m_view->mapFromScene(QPointF( worldX, worldY));
	//	QPoint globalPos = m_view->mapToGlobal(mapPos);
	//	menu.exec(globalPos);
}

void StackBaseMapView::startCreateNurbs(bool checked)
{
	//GraphicToolsWidget::showPalette(title());
	//NurbsWidget::setView2D(this);
	NurbsWidget::showWidget();
	//NurbsWidget* nurbsWid = new NurbsWidget(nullptr);
	//nurbsWid->show();
}
/**
 * Start demo rectangle picking, checked not used
 */
//void StackBaseMapView::startRectanglePicking(bool checked) {
//	if (!m_isRectanglePickingOn) {
//		m_isRectanglePickingOn = true;
//		QRectF rect(m_contextualWorldX + 50, m_contextualWorldY + 50,
//				2000, 1500);
//		QColor color(Qt::green);
//		GeObjectId objectId(1, 1);
//		GeGlobalParameters globalParameters;
//		m_geRectangle = new GeRectangle(
//				globalParameters,
//				"MyRectangle1",	objectId, color, true, rect, this);
//		m_scene->addItem(m_geRectangle);
//	} else {
//		m_isRectanglePickingOn = false;
//		if ( m_geRectangle != nullptr ) {
//			delete m_geRectangle;
//			m_geRectangle = nullptr;
//		}
//	}
//}

/**
 * Start demo ellipse picking, checked not used
 */
//void StackBaseMapView::startEllipsePicking(bool checked) {
//	if (!m_isEllipsePickingOn) {
//		m_isEllipsePickingOn = true;
//		QRectF rect(m_contextualWorldX + 50, m_contextualWorldY + 50,
//				2000, 1500);
//		QColor color(Qt::green);
//		GeObjectId objectId(1, 1);
//		GeGlobalParameters globalParameters;
//		m_geEllipse = new GeEllipse(
//				globalParameters,
//				objectId, color, true, rect, this);
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
//void StackBaseMapView::startPolygonPicking(bool checked) {
//	if (!m_isPolygonPickingOn) {
//		m_isPolygonPickingOn = true;
//		QColor color(Qt::green);
//		GeObjectId objectId(1, 1);
//		GeGlobalParameters globalParameters;
//		m_gePolygon = new GePolygon(
//				globalParameters,
//				objectId, color, true, this);
//		m_scene->addItem(m_gePolygon);
//	} else {
//		m_isPolygonPickingOn = false;
//		if ( m_gePolygon != nullptr ) {
//			delete m_gePolygon;
//			m_gePolygon = nullptr;
//		}
//	}
//}

void StackBaseMapView::updateStackIndex(int stackIndex) {
	m_currentStack = stackIndex;

	for (AbstractGraphicRep *r : m_visibleReps) {
		if (ISliceableRep *slice = dynamic_cast<ISliceableRep*>(r)) {
			slice->setSliceIJPosition(stackIndex);
		}
	}
	defineStackVal(stackIndex);
}

StackBaseMapView::~StackBaseMapView() {
}

void StackBaseMapView::defineStackMinMax(const QVector2D &imageMinMax,
		int step) {
	QSignalBlocker b1(m_stackSlider);
	m_stackSlider->setMinimum((int) imageMinMax.x());
	m_stackSlider->setMaximum((int) imageMinMax.y());
	m_stackSlider->setSingleStep(step);
	int pageStep = (int) ((imageMinMax.y() - imageMinMax.x()) * 5.0 / 100);
	m_stackSlider->setPageStep(pageStep);
	m_stackSlider->setTickInterval(step);

	QSignalBlocker b2(m_stackSpin);
	m_stackSpin->setMinimum((int) imageMinMax.x());
	m_stackSpin->setMaximum((int) imageMinMax.y());
	m_stackSpin->setSingleStep(step);
}

void StackBaseMapView::defineStackVal(int image) {
	QSignalBlocker b1(m_stackSlider);
	m_stackSlider->setValue(image);

	QSignalBlocker b2(m_stackSpin);
	m_stackSpin->setValue(image);

	StackLayerRGTRep* rep = nullptr;
	for (int i = m_visibleReps.size(); i >0 ; --i) {
		rep = dynamic_cast<StackLayerRGTRep*>(m_visibleReps.at(i-1));
		if (rep != nullptr) {
			if(rep->getLabelFromPosition(image) != ""){
				m_stackLabel->setText(rep->getLabelFromPosition(image));
			}
			break;
		}
	}

	if (rep == nullptr) {
		m_stackLabel->setText("");
	}
}

QWidget* StackBaseMapView::createStackBox() {
	QWidget *controler = new QWidget(this);
	m_stackSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_stackSlider->setSingleStep(1);
	m_stackSlider->setTracking(true);

	m_stackSlider->setTickInterval(10);
	m_stackSlider->setMinimum(0);
	m_stackSlider->setMaximum(1);
	m_stackSlider->setValue(0);

	m_stackSpin = new QSpinBox();
	m_stackSpin->setMinimum(0);
	m_stackSpin->setMaximum(1);
	m_stackSpin->setSingleStep(1);
	m_stackSpin->setValue(0);
	m_stackLabel = new QLabel;

	m_stackSpin->setWrapping(false);

	connect(m_stackSpin, SIGNAL(valueChanged(int)), this,SLOT(updateStackIndex(int )));
	connect(m_stackSlider, SIGNAL(valueChanged(int)), this,SLOT(updateStackIndex(int)));

	QHBoxLayout *hBox = new QHBoxLayout(controler);

	hBox->addWidget(m_stackSpin);
	hBox->addWidget(m_stackSlider);
	hBox->addWidget(m_stackLabel);

	QPushButton* paletteButton = getPaletteButton();
	if (paletteButton) {
		hBox->addWidget(paletteButton,0,Qt::AlignRight);
	}
	hBox->addWidget(getSplitButton(),0,Qt::AlignRight);
	return controler;
}

QWidget* StackBaseMapView::createViewModeSelector() {
	QComboBox* displayViewCombo = new QComboBox;
	displayViewCombo->addItem("Stack Mode", QVariant(0));
	displayViewCombo->addItem(SplittedView::viewModeLabel(eTypeSplitMode), QVariant(1));
	displayViewCombo->addItem(SplittedView::viewModeLabel(eTypeTabMode)  , QVariant(2));
	displayViewCombo->setStyleSheet("QComboBox::item{height: 20px}");
	return displayViewCombo;
}

void StackBaseMapView::recomputeRange() {
	std::map<AbstractGraphicRep*, QVector2D>::iterator it = m_repToRange.begin();
	int minimum = std::numeric_limits<int>::max();
	int maximum = std::numeric_limits<int>::min();
	while(it!=m_repToRange.end()) {
		if (minimum>it->second.x()) {
			minimum = it->second.x();
		}
		if (maximum<it->second.y()) {
			maximum = it->second.y();
		}
		it++;
	}
	if (minimum>maximum) {
		minimum = 0;
		maximum = 0;
	}
	QVector2D outVec(minimum, maximum);
	defineStackMinMax(outVec, 1);
}

void StackBaseMapView::export2Sismage() {
	long index = m_visibleReps.size()-1;
	while (index>=0 && (dynamic_cast<RGBLayerSlice*>(m_visibleReps[index]->data())==nullptr &&
			dynamic_cast<LayerSlice*>(m_visibleReps[index]->data())==nullptr) &&
			dynamic_cast<FixedLayersFromDatasetAndCube*>(m_visibleReps[index]->data())==nullptr &&
			dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(m_visibleReps[index]->data())==nullptr) {
		index--;
	}
	if (index>=0) {
		LayerSlice* datasetGrayCVisual = dynamic_cast<LayerSlice*>(m_visibleReps[index]->data());
		RGBLayerSlice* datasetRgbCVisual = dynamic_cast<RGBLayerSlice*>(m_visibleReps[index]->data());
		FixedLayersFromDatasetAndCube* fixedLayer = dynamic_cast<FixedLayersFromDatasetAndCube*>(m_visibleReps[index]->data());
		FixedRGBLayersFromDatasetAndCube* fixedRGBLayer = dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(m_visibleReps[index]->data());
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
		} else if (fixedLayer!=nullptr) {
			SeismicSurvey* survey = nullptr;
			if (fixedLayer!=nullptr) {
				QList<IData*> datas = fixedLayer->workingSetManager()->folders().seismics->data();
				long i = 0;
				while (survey==nullptr && i<datas.size()) {
					survey = dynamic_cast<SeismicSurvey*>(datas[i]);
					i++;
				}
			}
			if (fixedLayer!=nullptr && survey!=nullptr) {
				// ---- Export to Cultural
				QDir dir(survey->idPath());
				std::string culturalsPath =
						SismageDBManager::surveyPath2CulturalPath(dir.absolutePath().toStdString());

				if( culturalsPath.empty() ) return;

				int dimH = fixedLayer->width();
				int dimW = fixedLayer->depth();
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
					cultural->saveInto(fixedLayer->image(), fixedLayer->ijToXYTransfo());
				} else {
					CulturalCategory  nvCulturalCategory(culturalsPath, "NextVision");
					Cultural cultural(culturalsPath, ssd.newName().toStdString(), nvCulturalCategory);

					// Il faut retrouver la data qui contient le process
					cultural.saveInto(fixedLayer->image(), fixedLayer->ijToXYTransfo(), true);
				}
			}
		} else if (fixedRGBLayer!=nullptr) {
			SeismicSurvey* survey = nullptr;
			QList<IData*> datas = fixedRGBLayer->workingSetManager()->folders().seismics->data();
			long i = 0;
			while (survey==nullptr && i<datas.size()) {
				survey = dynamic_cast<SeismicSurvey*>(datas[i]);
				i++;
			}

			if (survey!=nullptr) {
				// ---- Export to Cultural
				QDir dir(survey->idPath());
				std::string culturalsPath =
						SismageDBManager::surveyPath2CulturalPath(dir.absolutePath().toStdString());

				if( culturalsPath.empty() ) return;

				int dimH = fixedRGBLayer->width();
				int dimW = fixedRGBLayer->depth();
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

				ImageFormats::QSampleType imageType = ImageFormats::QSampleType::INT16;
				ImageFormats::QSampleType isoType = ImageFormats::QSampleType::INT16;
				if (fixedRGBLayer->image()) {
					imageType = fixedRGBLayer->image()->sampleType();
				}
				if (fixedRGBLayer->isoSurfaceHolder()) {
					imageType = fixedRGBLayer->isoSurfaceHolder()->sampleType();
				}
				CUDAImagePaletteHolder *red = new CUDAImagePaletteHolder(
						fixedRGBLayer->getNbTraces(), fixedRGBLayer->getNbProfiles(), imageType);
				CUDAImagePaletteHolder *green = new CUDAImagePaletteHolder(
						fixedRGBLayer->getNbTraces(), fixedRGBLayer->getNbProfiles(), imageType);
				CUDAImagePaletteHolder *blue = new CUDAImagePaletteHolder(
						fixedRGBLayer->getNbTraces(), fixedRGBLayer->getNbProfiles(), imageType);

				CUDAImagePaletteHolder *iso = new CUDAImagePaletteHolder(
						fixedRGBLayer->getNbTraces(), fixedRGBLayer->getNbProfiles(), isoType);

				fixedRGBLayer->getImageForIndex(fixedRGBLayer->currentImageIndex(), red, green, blue, iso);

				int minimumValue = 0;
				if (fixedRGBLayer->isMinimumValueActive()) {
					minimumValue = std::floor(fixedRGBLayer->minimumValue() * 255);
				}
				if (!ssd.isNewName() || (isNewNameInList && ssd.isNewName())) {
					std::cout << "SELECT STRING " << selectedIndex << " " << selectedString.toStdString() << std::endl;
					if (selectedIndex == -1)
							return;
					Cultural* cultural = culturals.getCultural(namesVector.at(selectedIndex));

					// Il faut retrouver la data qui contient le process
					cultural->saveInto(red, green, blue, minimumValue, fixedRGBLayer->ijToXYTransfo());
				} else {
					CulturalCategory  nvCulturalCategory(culturalsPath, "NextVision");
					Cultural cultural(culturalsPath, ssd.newName().toStdString(), nvCulturalCategory);

					// Il faut retrouver la data qui contient le process
					cultural.saveInto(red, green, blue, minimumValue, fixedRGBLayer->ijToXYTransfo(), true);
				}
				delete red;
				delete green;
				delete blue;
				delete iso;
			}
		}
	}
}

void StackBaseMapView::exportMultiLayer2Sismage() {
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

double StackBaseMapView::getWellMapWidth() const {
	return m_wellMapWidth;
}

void StackBaseMapView::setWellMapWidth(double value) {
	if(m_wellMapWidth != value) {
		m_wellMapWidth = value;
		emit signalWellMapWidth(m_wellMapWidth);
	}
}
