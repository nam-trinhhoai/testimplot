// #define __DEBUG_RGTPICKING__

#include <string>
#include <iostream>

#include <qboxlayout.h>
#include <QLineEdit>
#include <QWidget>
#include <QMessageBox>

#include <qlabel.h>
#include <qpushbutton.h>
#include <qgroupbox.h>
#include <QFormLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QMainWindow>
#include <QDockWidget>
#include <QScrollArea>
#include <QInputDialog>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QListWidget>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QFormLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QImage>
#include <QIcon>
#include <QApplication>
#include <QGraphicsScene>
#include <QStack>

#include "Xt.h"
#include "DataSelectorDialog.h"
#include "stringselectordialog.h"
#include "dialog/validator/SimpleDoubleValidator.h"
#include "dialog/validator/OutlinedQLineEdit.h"
#include "abstractsectionview.h"
#include "basemapview.h"
#include "stackbasemapview.h"
#include "basemapgraphicsview.h"
#include "randomlineview.h"
#include "slicer/data/layerslice/LayerSlice.h"
#include "slicer/data/layerslice/rgblayerslice.h"
#if 0
#include "slicer/view/splittedview.h"
#endif
#include "wellpick.h"
#include "monotypegraphicsview.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "seismicsurvey.h"
#include "slicerep.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "itemlistselectorwithcolor.h"
#include "workingsetmanager.h"

#include "sliceutils.h"
#include "basemapview.h"
#include "basemapgraphicsview.h"
#include "multiseedhorizon.h"
#include "multiseedslicerep.h"
#include "multiseedrandomrep.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "geotimegraphicsview.h"
#include "qgraphicsreptreewidgetitem.h"
#include "qinnerviewtreewidgetitem.h"
#include "viewqt3d.h"
#include "fixedlayerfromdataset.h"
#include "fixedlayerfromdatasetrep.h"
#include "SeismicPropagator.h"
#include "LayerRGTInterpolator.h"
#include "PatchCompositionProcess.h"
#include "ijkhorizon.h"
#include "savehorizondialog.h"
#include "selectseedfrommarkers.h"
#include "folderdata.h"
#include "horizonfolderdata.h"
#include "wellbore.h"
#include "GeotimeMarkerQCDialog.h"
#include "GraphicToolsWidget.h"

#include "cultural.h"
#include "randomrep.h"
#include "basemapsurface.h"
#include "CUDAImageMask.h"
#include "mtlengthunit.h"
#include <SeismicManager.h>
#include <QFileUtils.h>
#include "freehorizon.h"
// #include <freehorizonUtil.h>
#include <freeHorizonManager.h>
#include <CubeIO.h>

//#include "viewers/view2d/ViewHorizonExtension.h"
//#include "viewers/view2d/MultiSeedHorizonExtension.h"
//#include "viewers/view2d/visual/DatasetSyncViewer2dVisual.h"
//#include "viewers/geotimeview/GeotimeView.h"
//#include "LayerSpectrumProcess.h"

//#include "itkImageFileReader.h"
//#include "itkImageFileWriter.h"

//#include "CubeIO.h"
//#include "CubeIOException.h"
//#include "Cube.h"

#include <limits>
#include <png.h>
#include <ihm.h>
#include <geotimepath.h>
#include <rgtGraphLabelRead.h>
#include "DataSelectorDialog.h"

#define DEBUG0 fprintf(stderr, "%s %d\n", __FILE__, __LINE__);

class WindowWidgetPoper2: public QMainWindow,
public WidgetPoperTrait {
public:
	void popWidget(QWidget* widget) override {
		QDockWidget *dock = new QDockWidget(widget->windowTitle(), this);

		QScrollArea* scrollWidget = new QScrollArea(dock);
		scrollWidget->setWidget(widget);
		widget->setParent(scrollWidget);
		scrollWidget->setWidgetResizable(true);

		dock->setAllowedAreas(Qt::RightDockWidgetArea);
		dock->setWidget(scrollWidget);
		dock->setAttribute(Qt::WA_DeleteOnClose);
		addDockWidget(Qt::RightDockWidgetArea, dock);
	}
};

//const char * LayerSpectrumDialog::EXT_STR = "_Converted";

LayerSpectrumDialog::LayerSpectrumDialog(
		Seismic3DAbstractDataset *datasetS, int channelS, Seismic3DAbstractDataset *datasetT,
		int channelT, GeotimeGraphicsView* viewer,Abstract2DInnerView* emitingView,
		QWidget *parent): QWidget(),
				m_datasetS( datasetS), m_datasetT( datasetT),
				m_originViewer (viewer),
				//				m_pickingTaskSection(Qt::LeftButton, this),
				//m_pickingTaskMap(Qt::LeftButton, this),
				m_geotimeViewTrackerExtension({Qt::Key_U, Qt::Key_C, Qt::Key_P, Qt::Key_W}, this)/*,
				m_originMainViewer(viewerMain)*/ {

	this->setAttribute(Qt::WA_DeleteOnClose);
	m_channelS = channelS;
	m_channelT = channelT;
	m_method = eComputeMethd_Spectrum;

	createLayerSlice();

	setupHorizonExtension();

	WorkingSetManager *wm = m_datasetS->workingSetManager();
	QList<IData*> datas = wm->folders().horizonsFree->data();
	//	std::vector<QString>* horizonNames = wm->get_horizonNames();
	//	std::vector<QString>* horizonPaths = wm->get_horizonPaths();
	//	std::vector<QString>* horizonExtractionDataPaths = wm->get_horizonExtractionDataPaths();

	m_horizonNames.clear();
	m_horizonPaths.clear();
	m_horizonDatas.clear();
	for (int i=0; i<datas.size(); i++) {
		IJKHorizon* horizon = dynamic_cast<IJKHorizon*>(datas[i]);
		FreeHorizon* freeHorizon = dynamic_cast<FreeHorizon*>(datas[i]);
		if (horizon!=nullptr) {
			bool isCubeCompatible = filterHorizon(horizon);

			if (isCubeCompatible) {
				long id = m_horizonNextId++;
				m_horizonNames[id] = horizon->name();//(*horizonNames)[i]);
				m_horizonPaths[id] = horizon->path();//(*horizonPaths)[i]);
				m_horizonDatas[id] = nullptr;
			}
		} else if (freeHorizon!=nullptr) {
			bool isCubeCompatible = filterHorizon(freeHorizon);

			if (isCubeCompatible) {
				long id = m_horizonNextId++;
				m_horizonNames[id] = freeHorizon->name();//(*horizonNames)[i]);
				m_horizonPaths[id] = "";//(*horizonPaths)[i]);
				m_horizonDatas[id] = freeHorizon;
			}
		}
	}

	fprintf(stderr, "size %ld\n", m_horizonNames.size());
	int i=0;
	for (auto it=m_horizonNames.cbegin(); it!=m_horizonNames.cend(); it++) {
		fprintf(stderr, "------> %d %s\n", i, it->second.toStdString().c_str());
		i++;
	}

	QVBoxLayout * mainLayout=new QVBoxLayout(this);
	mainLayout->addWidget(initWidget());

	m_emitingView = emitingView;
	if(emitingView != nullptr){
		connect(emitingView,&AbstractInnerView::repAdded,this,&LayerSpectrumDialog::updateRgtList);
	}
	//	m_originViewer->registerPickingTask(&m_pickingTaskSection);
	//	connect(&m_pickingTaskSection, &PointPickingTask::pointPicked, [this](double worldX,double worldY) {
	//		if (m_seedEdit) {
	////			if (m_seedEdit && m_isMultiSeedActive) {
	////			//					m_redoButtonList.clear();
	////			//					m_horizonExtenstion->addPointAndSelect(imagePoint);
	////			//				} else if (m_seedEdit && !m_isMultiSeedActive) {
	////			//					if (m_seeds.size()==0) {
	////			//						m_horizonExtenstion->addPointAndSelect(imagePoint);
	////			//					} else {
	////			//						m_horizonExtenstion->moveSelectedPoint(imagePoint);
	////			//					}
	////			//				}
	//			double imageX, imageY;
	//			if (m_horizonRep->direction()==SliceDirection::Inline) {
	//				dynamic_cast<Seismic3DDataset*>(m_datasetS)->ijToInlineXlineTransfoForInline()->worldToImage(worldX, worldY, imageX, imageY);
	//			} else {
	//				dynamic_cast<Seismic3DDataset*>(m_datasetS)->ijToInlineXlineTransfoForXline()->worldToImage(worldX, worldY, imageX, imageY);
	//			}
	//
	//			MultiSeedSliceRep* horizonRep = m_horizonRep;
	//			std::size_t index=0;
	//
	//			if (imageX>=m_datasetS->image()->width() || imageX<0 ||
	//				imageY>=m_datasetS->image()->height() || imageY<0) {
	//				return;
	//			}
	//
	//			/*const QList<AbstractGraphicRep*> visibleReps = m_originViewer->getVisibleReps();
	//			while(index<visibleReps.size() && horizonRep==nullptr) {
	//				horizonRep = dynamic_cast<MultiSeedSliceRep*>(visibleReps[index]);
	//				index++;
	//			}*/
	//			if (horizonRep!=nullptr && m_isMultiSeedActive) {
	//				m_redoButtonList.clear();
	//				horizonRep->addPointAndSelect(imageX, imageY);
	//			} else if (horizonRep!=nullptr && !m_isMultiSeedActive) {
	//				QPoint imagePoint(imageX, imageY);
	//				if (m_seeds.size()==0) {
	//					horizonRep->addPointAndSelect(imagePoint);
	//				} else {
	//					horizonRep->moveSelectedPoint(imagePoint);
	//				}
	//			}
	//		}
	//	});
	m_originViewer->installEventFilter(&m_geotimeViewTrackerExtension);

	//	connect(&m_geotimeViewTrackerExtension, &LayerSpectrumTrackerExtension::hoverMovedFromSynchroSignal, [this]
	//			(QPoint imagePoint, PointF2DWithUnit scenePoint, view2d::SyncViewer2d* originViewer) {
	//		if (this->m_syncView!=nullptr && originViewer==m_originViewer) {
	//			std::vector<view2d::SyncViewer2dVisual*> visuals = m_syncView->getVisuals();
	//			if (visuals.size()>0) {
	//				if (m_originMainViewer->getDirection()==view2d::View2dDirection::Z) {
	//					QPoint pt(imagePoint.y(), m_originMainViewer->getCurrentSlice());
	//					PointF2DWithUnit ptScene = visuals[0]->visualToScene(pt);
	//					m_syncView->hoverMovedDownTop(pt, ptScene, originViewer);
	//				} else if (m_originMainViewer->getDirection()==view2d::View2dDirection::Y) {
	//					QPoint pt(m_originMainViewer->getCurrentSlice(), imagePoint.y());
	//					PointF2DWithUnit ptScene = visuals[0]->visualToScene(pt);
	//					m_syncView->hoverMovedDownTop(pt, ptScene, originViewer);
	//				}
	//			}
	//		}
	//	});

	//	connect(&m_geotimeViewTrackerExtension, &LayerSpectrumTrackerExtension::leftButtonPressSignal, [this]
	//			(QPoint imagePoint, PointF2DWithUnit scenePoint, view2d::SyncViewer2d* originViewer) {
	//		if (originViewer==m_originViewer) {
	//			if (m_horizonExtenstion!=nullptr) {
	//				if (m_seedEdit && m_isMultiSeedActive) {
	//					m_redoButtonList.clear();
	//					m_horizonExtenstion->addPointAndSelect(imagePoint);
	//				} else if (m_seedEdit && !m_isMultiSeedActive) {
	//					if (m_seeds.size()==0) {
	//						m_horizonExtenstion->addPointAndSelect(imagePoint);
	//					} else {
	//						m_horizonExtenstion->moveSelectedPoint(imagePoint);
	//					}
	//				}
	//			}
	//		}
	//	});
	connect(&m_geotimeViewTrackerExtension, &ContinuousPressEventFilter::keyPressSignal, [this](int key) {
		qDebug() << "LayerSpectrumDialog : lambda from CPEF";
		if (key==Qt::Key_U) {
			qDebug() << "LayerSpectrumDialog : lambda from CPEF : got U";
			undoHorizonModification();
		} else if (key==Qt::Key_C && m_horizon!=nullptr) {
			qDebug() << "LayerSpectrumDialog : lambda from CPEF : got C";
			//m_horizon->setBehaviorMode(FIXED);
			//disconnect(m_horizonExtenstion, &view2d::ViewHorizonExtension::currentTauChanged, this, QOverload<int,bool>::of(&LayerSpectrumDialog::setGeologicalTime));
			updateData();
			//			trt_compute();
		} else if (key==Qt::Key_P && m_horizon!=nullptr) {
			propagateHorizon();
		}
		else if (key == Qt::Key_W)
		{
			patchConstrain();
		}
	});
	//
	//	connect(&m_geotimeViewTrackerExtension, &LayerSpectrumTrackerExtension::directionChanged, [this](view2d::View2dDirection dir) {
	//		if (m_horizonExtenstion!=nullptr) {
	//			view2d::DatasetSyncViewer2dVisual* rgtVisual = nullptr;
	//			std::vector<view2d::SyncViewer2dVisual*> visuals = m_originViewer->getVisuals();
	//			int i=0;
	//			while(i<visuals.size()) {
	//				view2d::DatasetSyncViewer2dVisual* visual = dynamic_cast<view2d::DatasetSyncViewer2dVisual*>(visuals[i]);
	//				if (visual!=nullptr && visual->getDataset()==m_datasetT) {
	//					rgtVisual = visual;
	//				}
	//				i++;
	//			}
	//			if (rgtVisual!=nullptr) {
	//				if (dir==view2d::View2dDirection::Z) {
	//					m_horizonExtenstion->setPixelSampleRateY(rgtVisual->getDatasetGroup()->getDBStepY());
	//				} else if (dir==view2d::View2dDirection::Y) {
	//					m_horizonExtenstion->setPixelSampleRateY(rgtVisual->getDatasetGroup()->getDBStepZ());
	//				}
	//				m_horizonExtenstion->updateGraphicRepresentation();
	//			}
	//		}
	//	});

	//m_originViewer->getScene()->views().first()->installEventFilter(&m_geotimeViewTrackerExtension);

	connect(wm->folders().horizonsFree, &FolderData::dataAdded, this, &LayerSpectrumDialog::newLayerAdded);
}

void LayerSpectrumDialog::updateRgtList(AbstractGraphicRep *rep){
	IData* ds = rep->data();
	QString name = ds->name();
	Seismic3DDataset* dataset = nullptr;

	if ((dataset = dynamic_cast<Seismic3DDataset*>(ds)) != nullptr) {

		SliceRep* sliceRep = nullptr;
		RandomRep* randomRep = nullptr;
		int channel = 0;
		if( ((sliceRep = dynamic_cast<SliceRep*>(rep))!= nullptr)
				|| ((randomRep = dynamic_cast<RandomRep*>(rep)) != nullptr)) {
			if (sliceRep) {
				channel = sliceRep->channel();
			} else if (randomRep) {
				channel = randomRep->channel();
			}
		}

		if((sliceRep != nullptr) || (randomRep != nullptr)){

			if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::RGT && dataset->dimV()==1){

				if(m_rgtCombo != nullptr){
					m_rgtCombo->clear();
				}

				for(int i = 0;i< m_allDatasets.size();i++){
					if(m_allDatasets[i] == m_datasetT){
						m_allDatasets.erase(i);
						break;
					}
				}

				std::size_t id = nextDatasetIndex();
				m_allDatasets[id] = dataset;

				bool oldCompatibleWithNew = m_datasetT->isCompatible(dataset);

				m_datasetT = dataset;
				m_channelT = channel;
				if (m_horizon!=nullptr) {
					m_horizon->changeRgtVolume(dynamic_cast<Seismic3DDataset*>(m_datasetT), m_channelT);
				}

				QListWidgetItem* item = new QListWidgetItem;
				item->setText(m_datasetT->name());
				item->setData(Qt::UserRole, (qulonglong) id);
				if(m_rgtCombo != nullptr){
					m_rgtCombo->addItem(item);
				}

				//createLayerSlice(eUpdateDatasetT);
				clearLayerSlices();

				m_VectorData.clear();

				if (!oldCompatibleWithNew) {
					//  clear all then re add the new rgt, this is faster than removing all non rgt
					m_allDatasets.clear();
					m_allDatasets[id] = dataset;
					m_sismiqueCombo->clear();
					m_patchList->clear();
					m_constrainMissingWarned = false;
					m_datasetS = nullptr;

					fullClearHorizon();

					SeismicSurvey* survey = m_datasetT->survey();
					for (Seismic3DAbstractDataset* dataset : survey->datasets()) {
						addDataset(dataset);
					}

					reloadHorizonList();

					std::vector<Seismic3DAbstractDataset*> selectedDatasets = getSelectedDatasetsInView(m_originViewer);

					for (Seismic3DAbstractDataset* dataset : selectedDatasets) {
						if (!m_datasetT->isCompatible(dataset)) {
							continue;
						}

						disconnect(m_sismiqueCombo, &QListWidget::itemSelectionChanged,this,&LayerSpectrumDialog::updateDataSet);
						bool foundItem = false;
						int i=0;
						while (!foundItem && i<m_sismiqueCombo->count()) {
							QListWidgetItem* item = m_sismiqueCombo->item(i);
							foundItem = item->text().compare(dataset->name())==0;
							if (foundItem) {
								item->setSelected(true);
								bool idOk;
								int itemId = item->data(Qt::UserRole).toInt(&idOk);
								if (idOk) {
									m_datasetS = m_allDatasets[itemId];
								}
							} else {
								i++;
							}
						}
						connect(m_sismiqueCombo, &QListWidget::itemSelectionChanged,this,&LayerSpectrumDialog::updateDataSet);

						if (foundItem && m_datasetS!=nullptr) {
							fullClearHorizon();
							setupHorizonExtension();
						}
					}
					clearLayerSlices();
					m_VectorData.clear();
				}

				bool foundFirstData = false;
				QList<QListWidgetItem*> selection = m_sismiqueCombo->selectedItems();
				for (QListWidgetItem* item : selection) {
					bool ok;
					std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);
					if (ok) {
						Seismic3DDataset* dataset = m_allDatasets[id];

						if(dataset->type() == Seismic3DAbstractDataset::Seismic){
							m_datasetS = dataset;
							createLayerSlice(eUpdateDatasetS);

							QVector<IData*>  vectorData =  m_mapData[dataset];

							if (m_method == eComputeMethd_Anisotropy) {
								if (!foundFirstData) {
									foundFirstData = true;
									m_data = dynamic_cast<LayerSlice*>(vectorData[m_method]);
									m_VectorData.push_back(m_data);
									if(m_data->ProcessDeletion() == true){
										m_data->setProcessDeletion(false);
									}
									m_data->getDatasetS()->workingSetManager()->addLayerSlice(m_data);
								}
							} else {
								m_data = dynamic_cast<LayerSlice*>(vectorData[m_method]);
								m_VectorData.push_back(m_data);

								if(m_method == eComputeMethd_Spectrum){
									m_dataRGB = dynamic_cast<RGBLayerSlice*>(vectorData[eComputeMethd_RGB]);
									m_data->getDatasetS()->workingSetManager()->addRGBLayerSlice(m_dataRGB);
								}
								if(m_data->ProcessDeletion() == true){
									m_data->setProcessDeletion(false);
								}
								m_data->getDatasetS()->workingSetManager()->addLayerSlice(m_data);
							}
						}
					}
				}
			}
		}
	}
}

void LayerSpectrumDialog::createLayer()
{
	WorkingSetManager *manager = m_datasetS->workingSetManager();
	QVector<IData*> vector;

	for(int iIndex = eComputeMethd_Morlet; iIndex < eComputeMethd_RGB; iIndex++)
	{
		LayerSlice *pData = new LayerSlice(manager,	dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS, dynamic_cast<Seismic3DDataset*>(m_datasetT), m_channelT,iIndex);

		if(pData != nullptr){
			pData->setPolarity(m_polarity>=0);
			pData->setDistancePower(m_distancePower);
			pData->setUseSnap(m_useSnap);
			pData->setUseMedian(m_useMedian);
			pData->setLWXMedianFilter(m_halfLwxMedianFilter*2+1);
			pData->setHatPower(m_hatPower);
			pData->setFreqMin(m_freqMin);
			pData->setFreqMax(m_freqMax);
			pData->setFreqStep(m_freqStep);
			pData->setWindowSize(m_spectrumWindowSize);
			pData->setW(m_w);
			pData->setShift(m_shift);

			vector.push_back(pData);
		}
	}

	m_data = dynamic_cast<LayerSlice*>(vector[m_method]);
	// let other functions manage which data need to be displayed for Anisotropy
	if (m_method != eComputeMethd_Anisotropy) {
		manager->addLayerSlice(m_data);
	}
	//if(eComputeMethd_Spectrum == m_method){
	m_dataRGB = new RGBLayerSlice(manager, dynamic_cast<LayerSlice*>(vector[eComputeMethd_Spectrum]), dynamic_cast<LayerSlice*>(vector[eComputeMethd_Spectrum]));
	vector.push_back(this->m_dataRGB);
	manager->addRGBLayerSlice(m_dataRGB);
	//}
	m_mapData[m_datasetS] = vector;
}

void LayerSpectrumDialog::removeLayer(Seismic3DAbstractDataset *pSeismic){

	WorkingSetManager *manager = m_datasetS->workingSetManager();
	if (m_mapData.contains(pSeismic) == true){
		QVector<IData*> vectorData = m_mapData[pSeismic];
		m_dataRGB = dynamic_cast<RGBLayerSlice*>(vectorData[vectorData.size()-1]);
		for (int  i = 0; i < vectorData.size(); i++) {
			if(i != eComputeMethd_RGB){
				if ( vectorData[i]!=nullptr) {
					LayerSlice* pData = dynamic_cast<LayerSlice*>(vectorData[i]);
					if(pData != nullptr)
						manager->removeLayerSlice(pData); // remove delete IData
				}
			}else {
				RGBLayerSlice* pData = dynamic_cast<RGBLayerSlice*>(vectorData[i]);
				if(pData != nullptr)
					manager->removeRGBLayerSlice(pData);
			}
		}
		m_mapData.remove(pSeismic);
	}
}

void LayerSpectrumDialog::createLayerSlice(eUpdateDataSet eUpdate){

	m_isComputationValid = false;
	m_isComputationRunning = false;
	m_isWindowChanged = true;

	if(eUpdate == eUpdateDatasetT)
	{
		clearLayerSlices();

		createLayer();
	}
	else
	{
		// Begin MZR 09072021
		if (m_mapData.contains(m_datasetS) == false){
			createLayer();
		} else {
			WorkingSetManager *manager = m_datasetS->workingSetManager();
			QVector<IData*> vector = m_mapData[this->m_datasetS];
			m_data = dynamic_cast<LayerSlice*>(vector[m_method]);
			if (m_method != eComputeMethd_Anisotropy) {
				manager->addLayerSlice(m_data);
			}
			if(eComputeMethd_Spectrum == m_method){
				m_dataRGB= dynamic_cast<RGBLayerSlice*>(vector[eComputeMethd_RGB]);
				manager->addRGBLayerSlice(m_dataRGB);
			}
		}
		// End MZR 09072021
	}
}

// created in a hurry, reseting the below variables without protection will lead to crashes
void LayerSpectrumDialog::clearLayerSlices() {
	m_isComputationValid = false;
	m_isComputationRunning = false;
	m_isWindowChanged = true;

	SeismicSurvey *baseSurvey=m_datasetS->survey();
	WorkingSetManager *manager = m_datasetS->workingSetManager();

	for (Seismic3DAbstractDataset *seismic : baseSurvey->datasets() ) {

		if(seismic->type() == Seismic3DAbstractDataset::Seismic){
			if (m_mapData.contains(seismic) == true){
				QVector<IData*>  vectorData = m_mapData[seismic];
				m_dataRGB = dynamic_cast<RGBLayerSlice*>(vectorData[vectorData.size()-1]);
				for (int  i = 0; i < vectorData.size(); i++) {
					if(i != eComputeMethd_RGB){
						if ( vectorData[i]!=nullptr) {
							LayerSlice* pData = dynamic_cast<LayerSlice*>(vectorData[i]);
							if(pData != nullptr)
								manager->removeLayerSlice(pData); // remove delete IData
						}
					}else {
						RGBLayerSlice* pData = dynamic_cast<RGBLayerSlice*>(vectorData[i]);
						if(pData != nullptr)
							manager->removeRGBLayerSlice(pData);
					}
				}
				m_mapData.remove(seismic);
			}
		}
	}
}

void LayerSpectrumDialog::setGeoTimeView(GeotimeGraphicsView* pGeoTimeView){
	m_originViewer = pGeoTimeView;
	m_datasetS = nullptr;
}

LayerSpectrumDialog::~LayerSpectrumDialog() {
	// data cannot be deleted yet
	//	if (m_horizonExtenstion!=nullptr) {
	//		this->m_originMainViewer->getViewExtensions().removeExtension(m_horizonExtenstion);
	//	}
	//	m_originViewer->getScene()->views().first()->removeEventFilter(&m_geotimeViewTrackerExtension);
	if(m_originViewer != nullptr){
		m_originViewer->removeEventFilter(&m_geotimeViewTrackerExtension);
	}

	if(m_datasetS != nullptr){
		WorkingSetManager* manager = m_datasetS->workingSetManager();
		//manager->removeRGBLayerSlice(m_dataRGB); // remove delete IData

		SeismicSurvey *baseSurvey=m_datasetS->survey();
		for (Seismic3DAbstractDataset *seismic : baseSurvey->datasets() ) {
			if(seismic->type() == Seismic3DAbstractDataset::Seismic){
				qDebug() << seismic->name();
				if (m_mapData.contains(seismic) == true){
					QVector<IData*>  vectorData = m_mapData[seismic];
					m_dataRGB = dynamic_cast<RGBLayerSlice*>(vectorData[vectorData.size()-1]);
					for (int  i = 0; i < vectorData.size(); i++) {//IData* layerData : vectorData) {
						if(m_dataRGB != vectorData[i]){
							//qDebug() << vectorData[i]->name();
							if ( vectorData[i]!=nullptr) {
								LayerSlice* pData = dynamic_cast<LayerSlice*>(vectorData[i]);
								if(pData != nullptr)
									manager->removeLayerSlice(pData); // remove delete IData
							}
						}
						else
						{
							RGBLayerSlice* pData = dynamic_cast<RGBLayerSlice*>(vectorData[i]);
							if(pData != nullptr)
								manager->removeRGBLayerSlice(pData);
						}
					}
				}
			}
		}

		//manager->removeLayerSlice(m_data); // remove delete IData
		//	m_data->deleteLater();
		if (m_horizon!=nullptr) {
			manager->removeMultiSeedHorizon(m_horizon); // remove delete IData
			//		m_horizon->deleteLater();
		}
		if (m_constrainData!=nullptr) {
			FixedLayerFromDataset* layer = m_constrainData.release();
			manager->removeFixedLayerFromDataset(layer); // remove delete IData
		}
		for(QMetaObject::Connection con : m_conn)
			QObject::disconnect(con);
	}
	if ( rgtGraphLabelRead ) delete rgtGraphLabelRead;
}

void LayerSpectrumDialog::setSeeds(const std::vector<RgtSeed>& seeds) {
	if (m_seeds!=seeds) {
		if (m_horizon!=nullptr) {
			m_horizon->setSeeds(seeds);
			m_isComputationValid = false;
		}
		m_seeds = seeds;
	}
}

void LayerSpectrumDialog::setPolarity(int pol) {
	if (pol!=m_polarity) {
		m_polarity = pol;
		if (m_horizon!=nullptr) {
			m_horizon->setPolarity(m_polarity);
		}
		if (m_polarity==1) {
			m_polarityComboBox->setCurrentIndex(0);
		} else if(m_polarity==-1) {
			m_polarityComboBox->setCurrentIndex(1);
		} else if (m_polarity==2) {
			m_polarityComboBox->setCurrentIndex(2);
		} else if(m_polarity==-2) {
			m_polarityComboBox->setCurrentIndex(3);
		}
	}
}

/*void LayerSpectrumDialog::setGeologicalTime(int tau, bool polarity) {
	if (tau!=this->m_tau) {
		this->m_tau = tau;
		if (m_tauSpinBox) {
			m_tauSpinBox->setValue(tau);
		}

		m_isComputationValid = false;
		//data->setGeologicalTime(tau);
	}
	if ((polarity && m_polarity!=1) || (!polarity && m_polarity!=-1)) {
		m_polarity = (polarity)? 1 : -1;
		m_isComputationValid = false;
	}
}*/

void LayerSpectrumDialog::setPseudoGeologicalTime(int tau) {
	if (tau!=this->m_pseudoTau) {
		this->m_pseudoTau = tau;
		if (m_tauSpinBox && m_tauSpinBox->value()!=m_pseudoTau) {
			m_tauSpinBox->setValue(m_pseudoTau);
		}

		if (m_horizon!=nullptr) {
			m_horizon->setPseudoTau(tau);
			setPolarity(m_horizon->getPolarity());
		}

		m_isComputationValid = false;
	}
}

//void LayerSpectrumDialog::connectViewer(view2d::SyncViewer2d* viewer) {
////	AnnotationSyncViewer2dEditor* baseEditor = new AnnotationSyncViewer2dEditor(data);
////	baseEditor->addInto(viewer);
////	ExtendedViewer extendedViewer;
////	extendedViewer.viewer = viewer;
////	extendedViewer.editor = baseEditor;
////	extendedViewers.push_back(extendedViewer);
//}
//
//void LayerSpectrumDialog::disconnectViewer(view2d::SyncViewer2d* viewer) {
////	auto it = extendedViewers.begin();
////	while (it != extendedViewer.end() && it->viewer != viewer); {
////		it++;
////	}
////	if (it != extendedViewer.end()) {
////		ExtendedViewer extendedViewer = *it;
////		extendedViewer.editor->removeFrom(viewer);
////		extendedViewer.erase(it);
////		extendedViewer.editor->closeEditor();
////	}
//}
template<typename DataType> std::size_t LayerSpectrumDialog::getViewIndex(QString titleName)
{
	QVector<AbstractInnerView*> innerViews = m_originViewer->getInnerViews();
	std::size_t index = 0;

	while(index<innerViews.size()){
		if((dynamic_cast<DataType*>(innerViews[index]) != nullptr)
				&& (dynamic_cast<DataType*>(innerViews[index])->windowTitle().compare(titleName) == 0)){
			break;
		}
		index++;
	}

	return index;
}

template<typename DataType> std::size_t LayerSpectrumDialog::getViewIndexOrCreate(QString titleName)
{
	QVector<AbstractInnerView*> innerViews = m_originViewer->getInnerViews();
	std::size_t index = 0;

	while(index<innerViews.size()){
		if((dynamic_cast<DataType*>(innerViews[index]) != nullptr)
				&& (dynamic_cast<DataType*>(innerViews[index])->windowTitle().compare(titleName) == 0)){
			break;
		}
		index++;
	}

	if (index==innerViews.size()) {
		AbstractInnerView* innerView = m_originViewer->createInnerView(ViewType::StackBasemapView);
		m_originViewer->setInnerViewDefaultName(innerView, titleName);

		QVector<AbstractInnerView*> innerViewsBis = m_originViewer->getInnerViews();
		index = 0;
		while (index<innerViewsBis.size() && innerViewsBis[index]!=innerView) {
			index ++;
		}
	}

	return index;
}

template<typename DataType> void LayerSpectrumDialog::SetDataItem(IData *pData,std::size_t index ){
	QVector<AbstractInnerView*> innerViews = m_originViewer->getInnerViews();
	QInnerViewTreeWidgetItem* rootItem = m_originViewer->getItemFromView(innerViews[index]);
	QStack<QTreeWidgetItem*> stack;
	stack.push(rootItem);

	QGraphicsRepTreeWidgetItem* itemData = nullptr;

	while (stack.size()>0 && itemData==nullptr) {
		QTreeWidgetItem* item = stack.pop();

		std::size_t N = item->childCount();
		for (std::size_t index=0; index<N; index++) {
			stack.push(item->child(index));
		}
		QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
		if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
			const DataType* data = dynamic_cast<const DataType*>(_item->getRep()->data());
			if (data!=nullptr && data==pData) {
				itemData = _item;
			}
		}
	}

	if (itemData!=nullptr) {
		itemData->setCheckState(0, Qt::Checked);
	}
}

void LayerSpectrumDialog::crunch() {
	//	if (m_datasetView!=nullptr) {
	//		return;
	//m_window->hide();
	//m_window->deleteLater();
	//m_datasetView->deleteLater();
	//m_syncView->deleteLater();
	//	}

	//	m_window = new WindowWidgetPoper2;

	//	BaseMapGraphicsView *win = new BaseMapGraphicsView(
	//		m_datasetS->workingSetManager(), this);
	//	win->setWindowTitle("Geotime BaseMap View");
	//	win->setVisible(true);
	//	win->resize(800, 500);
	//
	//	m_datasetView = win;

	//registerWindow(win);


	//	m_syncView = new view2d::SyncMultiView2d(
	//			&view2d::SyncViewer2d::GLOBAL_REGISTER,
	//			m_window, &view2d::SyncMultiView2d::GLOBAL_EDITOR_REGISTER);

	//syncView->setCurrentSlice(m_slice);
	//	m_datasetView = new DatasetSplittedContainer(m_syncView);
	//	m_syncView->setDatasetSplittedContainer(m_datasetView);

	//	m_window->setWindowTitle("Sync 2D view");
	//	m_window->setCentralWidget(m_datasetView);
	//	m_window->setMenuBar(m_datasetView->menuBar());
	//	m_window->setMinimumSize(1000,700);
	//	m_window->addToolBar(Qt::RightToolBarArea,m_datasetView->toolBar());
	//	m_window->show();
	//	m_syncView->setDirection(view2d::View2dDirection::X);
	//
	//	QString rgbImageProp = "RGB_IMAGE";
	//	m_parameters.setValueFrom(rgbImageProp, false);
	//	bool isRGBImage = m_parameters.getValueAs(rgbImageProp, true);
	//
	//	m_data->setName("Spectrum Gray");
	//	m_datasetView->insertData(m_data, m_parameters);
	//	m_syncView->addExtension(this);
	//
	//	m_data->setName("Spectrum RGB");
	//	m_parameters.setValueFrom("RGB_IMAGE", true);
	//	m_datasetView->insertData(m_data, m_parameters);
	//
	//	m_syncView->setDirection(view2d::View2dDirection::X);
	//	m_isWindowChanged = false;
	m_dataRGB->resetFrequencies();
	GeotimeGraphicsView* geotimeView = m_originViewer;
	if (geotimeView!=nullptr) {
		StackBaseMapView* basemap = nullptr;
		StackBaseMapView* stackBasemap = nullptr;
		//	QVector<AbstractInnerView*> innerViews = geotimeView->getInnerViews();
		std::size_t index = 0;

		switch(m_method){
		case eComputeMethd_Spectrum:index = getViewIndex<StackBaseMapView>("Spectrum");	break;
		case eComputeMethd_Gcc:index = getViewIndex<StackBaseMapView>("GCC");break;
		case eComputeMethd_TMAP:index = getViewIndex<StackBaseMapView>("TMAP");break;
		case eComputeMethd_Mean:index = getViewIndex<StackBaseMapView>("Mean");break;
		case eComputeMethd_Anisotropy:index = getViewIndexOrCreate<StackBaseMapView>("Anisotropy");break;
		}
		//if (index<innerViews.size()) {
		//		stackBasemap = dynamic_cast<StackBaseMapView*>(innerViews[index]);
		//		QInnerViewTreeWidgetItem* rootItem = geotimeView->getItemFromView(innerViews[index]);

		// activate gray on stack base map
		if (index<m_originViewer->getInnerViews().size()) {
			stackBasemap = dynamic_cast<StackBaseMapView*>(m_originViewer->getInnerViews()[index]);

			QList<QVector<IData*>>  list = m_mapData.values();
			for(QVector<IData*> v :list){
				LayerSlice* pData =  dynamic_cast<LayerSlice*>(v[m_method]);
				SetDataItem<LayerSlice>(pData,index);
			}
		}
		// activate rgb on basemap
		index = getViewIndex<StackBaseMapView>("RGB");
		if (index<m_originViewer->getInnerViews().size()) {
			basemap = dynamic_cast<StackBaseMapView*>(m_originViewer->getInnerViews()[index]);

			QList<QVector<IData*>>  list = m_mapData.values();
			for(QVector<IData*> v :list){
				if(m_method == eComputeMethd_Spectrum){
					RGBLayerSlice* pData =  dynamic_cast<RGBLayerSlice*>(v[eComputeMethd_RGB]);
					SetDataItem<RGBLayerSlice>(pData,index);
				}
			}
		}
		// activate rgb on 3d view
		index = 0;
		while(index<m_originViewer->getInnerViews().size() && dynamic_cast<ViewQt3D*>(m_originViewer->getInnerViews()[index])==nullptr) {
			index++;
		}
		if (index<m_originViewer->getInnerViews().size()) {
			QList<QVector<IData*>>  list = m_mapData.values();
			for(QVector<IData*> v :list){
				if(m_method == eComputeMethd_Spectrum){
					RGBLayerSlice* pData =  dynamic_cast<RGBLayerSlice*>(v[eComputeMethd_RGB]);
					SetDataItem<RGBLayerSlice>(pData,index);
				}
			}
		}

		//		if (stackBasemap!=nullptr) {
		//			stackBasemap->registerPickingTask(&m_pickingTaskMap);
		//		}
		//		if (basemap!=nullptr) {
		//			basemap->registerPickingTask(&m_pickingTaskMap);
		//		}
		//		connect(&m_pickingTaskMap, &PointPickingTask::pointPicked, [this](double cartoX,double cartoY) {
		//			double worldX, worldY;
		//			dynamic_cast<Seismic3DDataset*>(m_datasetS)->survey()->inlineXlineToXYTransfo()->worldToImage(cartoX, cartoY, worldX, worldY);
		//			if (m_datasetS->direction()==SliceDirection::Inline) {
		//				m_datasetS->setSliceWorldPosition(worldY);
		//			} else {
		//				m_datasetS->setSliceWorldPosition(worldX);
		//			}
		//		});
	}
	m_isWindowChanged = false;
}

/**
 *
 */
//void LayerSpectrumDialog::hoverMovedFromSynchro(QPoint imagePoint, PointF2DWithUnit scenePoint,
//		view2d::SyncViewer2d* originViewer) {
////	qDebug() << "Mouse Hover Moved: " << imagePoint.x() << "/" << imagePoint.y() <<
////			"Scene Point: " << scenePoint.getX() << "/" << scenePoint.getY();
//	if (originViewer==m_originViewer) {
//		return;
//	}
//
//	view2d::DatasetSyncViewer2dVisual* rgtVisual = nullptr;
//	std::vector<view2d::SyncViewer2dVisual*> visuals = m_originViewer->getVisuals();
//	int i=0;
//	while(rgtVisual==nullptr && i<visuals.size()) {
//		view2d::DatasetSyncViewer2dVisual* visual = dynamic_cast<view2d::DatasetSyncViewer2dVisual*>(visuals[i]);
//		if (visual!=nullptr && visual->getDataset()==m_datasetT) {
//			rgtVisual = visual;
//		}
//	}
//	if (rgtVisual!=nullptr) {
//		// TODO tracking for layer
//		long x_rgt = 0;
//		QPoint pt(x_rgt, 0);
//		if (m_originViewer->getDirection()==view2d::View2dDirection::Z) {
//			pt.setY(imagePoint.x());
//		} else if (m_originViewer->getDirection()==view2d::View2dDirection::Y) {
//			pt.setY(imagePoint.y());
//		}
//		if (m_horizonExtenstion!=nullptr) {
//			const QPolygon& horizon = m_horizonExtenstion->getMainHorizon();
//			std::size_t index = 0;
//			while (index<horizon.size() && pt.y()!=horizon[index].y()) {
//				index++;
//			}
//			if (index<horizon.size()) {
//				pt.setX(horizon[index].x());
//			}
//		}
//		/*while (x_rgt<rgtVisual->getImageSize().width()-1 && rgtVisual->valueAtImage(pt)<m_tau) {
//			x_rgt ++;
//			pt.setX(x_rgt);
//		}*/
//
//		PointF2DWithUnit ptSceneUnit = rgtVisual->visualToScene(pt.x(), pt.y());
//		m_originViewer->synchroHoverMovedTopDown(pt, ptSceneUnit);
//	}
//}

//void LayerSpectrumDialog::leftButtonPressFromSynchro(QPoint imagePoint, PointF2DWithUnit scenePoint,
//		view2d::SyncViewer2d* originViewer) {
//	if (originViewer==m_originViewer) {
//		return;
//	}
//
//	qDebug() << "Left Button pressed on origin viewer, change slice to " << imagePoint.y();
//	if (m_originViewer->getDirection()==view2d::View2dDirection::Z) {
//		m_originMainViewer->setCurrentSlice(imagePoint.y());
//	} else if (m_originViewer->getDirection()==view2d::View2dDirection::Y) {
//		m_originMainViewer->setCurrentSlice(imagePoint.x());
//	}
//}

void LayerSpectrumDialog::setSynchroSlice(int slice) {
	qDebug() << "Change slice from Synchro Menu to " << slice;
	//m_originMainViewer->setCurrentSlice(slice);
}

/**
 *
 */
//model::DatasetGroup* LayerSpectrumDialog::getOrCreateDatasetSpectrum(
//		model::Dataset* inputDataset,
//		std::string newGroupName, std::string newDatasetName, int nbSlices) {
//	murat::model::DatasetGroup *inputGroup = inputDataset->getDatasetGroup();
//	murat::model::BorePart *bore = inputDataset->getDatasetGroup()->getBorePart();
//	murat::model::DatasetGroup *group = bore->getExistingDatasetGroup(newGroupName);
//	int dimI, dimJ, dimK;
//	double originX = 0,  originY = 0,  originZ = 0;
//	double stepX, stepY, stepZ;
//	int dimIOut, dimJOut, dimKOut;
//	double originXOut = 0,  originYOut = 0,  originZOut = 0;
//	double stepXOut, stepYOut, stepZOut;
//	if (group == nullptr) {
//		inputGroup->getDimensions(dimI,  dimJ,  dimK, originX,  originY,  originZ, stepX, stepY,  stepZ);
//		dimIOut = dimJ; // Samples
//		dimJOut = dimK; // traces
//		dimKOut = nbSlices; // Profiles
//		originXOut = 0,  originYOut = 0,  originZOut = 0;
//		stepXOut = stepX;
//		stepYOut = stepY;
//		stepZOut = stepZ;
//		group = bore->createDatasetGroup(QString::fromStdString(newGroupName),
//				dimIOut, dimJOut, dimKOut, originXOut, originYOut,
//				originZOut, stepXOut, stepYOut, stepZOut,
//				inputGroup->getOriginUnit(), inputGroup->getStepUnit());
//	} else {
////		group->getDimensions(dimI,  dimJ,  dimK, originX,  originY,  originZ, stepX, stepY,  stepZ);
////		if (dimI != box->dx / box->stepX || dimJ != box->dy / box->stepY ||dimK != box->dz / box->stepZ ) {
////			QMessageBox::warning(this->parentWidget(),"Information error",
////					"Sorry 3D box and existing Dataset Group should have same dimensions.");
////			return nullptr;
////		}
//	}
//	return group;
////	murat::SampleType sampleType = murat::SampleType::INT16;
////	// Output Dataset
////	std::string outputName = "LayerSpectrum";
////	murat::CubeFormats cubeFormat = murat::CubeFormats::murat;
////	double energy = inputDataset->getEnergy();
////
////	return TarumGenericProcessDialog::getOutputDataset(group, outputName, sampleType,
////			cubeFormat, 0, inputDataset->getEnergyUnit(), this);
//}

void LayerSpectrumDialog::UpdateDatasets(Seismic3DAbstractDataset *pDataSet)
{
	SliceRep* sliceRep = nullptr;
	RandomRep* randomRep = nullptr;
	int channel;

	for(AbstractGraphicRep* rep : pDataSet->getRepList()){
		if (!(sliceRep = dynamic_cast<SliceRep*>(rep)) && ! (randomRep = dynamic_cast<RandomRep*>(rep))) {
			continue;
		} else {
			if (sliceRep) {
				channel = sliceRep->channel();
			} else if (randomRep) {
				channel = randomRep->channel();
			}
		}
	}

	m_datasetS = pDataSet;
	m_channelS = channel;
	createLayerSlice();
}

void LayerSpectrumDialog::updateDataSet(){
	this->m_isComputationValid = false;
	this->m_isWindowChanged = true;

	m_meanDatasets.clear();
	m_VectorData.clear();

	QList<QListWidgetItem*> selection = m_sismiqueCombo->selectedItems();
	if (m_method == eComputeMethd_Anisotropy) {
		// update mean part
		for (QListWidgetItem* item : selection) {
			bool ok;
			std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);
			if (ok) {
				Seismic3DDataset* dataset = m_allDatasets[id];
				UpdateDatasets(dataset);
				m_meanDatasets.push_back(std::pair<Seismic3DDataset*, int>(dataset, 0));
			}
		}

		// update data part
		/*
		 *  may not be good enough
		 *  perhaps avoiding m_VectorData.clear() would be better
		 *  but will need a scan to detect if the datas in m_VectorData are from the right method and are still valid
		 */
		std::pair<bool, LayerSlice*> foundData = searchFirstValidDataFromMethod(m_method);
		if (foundData.second!=nullptr) {
			if (!foundData.first) {
				foundData.second->getDatasetS()->workingSetManager()->addLayerSlice(foundData.second);

				// may lack a foundData.second->setProcessDeletion(false);
			}
			m_VectorData.push_back(foundData.second);
		}
	} else {
		for (QListWidgetItem* item : selection) {
			bool ok;
			std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);
			if (ok) {
				Seismic3DDataset* dataset = m_allDatasets[id];
				UpdateDatasets(dataset);
				m_meanDatasets.push_back(std::pair<Seismic3DDataset*, int>(dataset, 0));
				QVector<IData*>  vectorData =  m_mapData[dataset];
				m_VectorData.push_back(vectorData[m_method]);
			}
		}
	}

	selection = m_rgtCombo->selectedItems();
	for (QListWidgetItem* item : selection) {
		bool ok;
		std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);
		if (ok) {
			Seismic3DDataset* dataset = m_allDatasets[id];
			m_meanDatasets.push_back(std::pair<Seismic3DDataset*, int>(dataset, 0));
		}
	}

	updateAngleTree();
}

void LayerSpectrumDialog::updateAngleTree() {
	if (m_anisotropyTree==nullptr) {
		return;
	}

	std::map<std::size_t, bool> usedMaps;

	for (int i=0; i<m_anisotropyTree->topLevelItemCount(); i++) {
		bool ok;
		std::size_t id = m_anisotropyTree->topLevelItem(i)->data(0, Qt::UserRole).toULongLong(&ok);
		if (ok) {
			usedMaps[id] = false;
		}
	}

	QList<QListWidgetItem*> selection = m_sismiqueCombo->selectedItems();
	for (QListWidgetItem* item : selection) {
		bool ok;
		std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);
		if (ok) {
			std::map<std::size_t, bool>::iterator it = usedMaps.find(id);
			if (it==usedMaps.end()) {
				QTreeWidgetItem* treeItem = new QTreeWidgetItem;
				treeItem->setText(0, item->text());
				treeItem->setData(0, Qt::UserRole, (qulonglong) id);
				treeItem->setData(1, Qt::EditRole, 0.0f);
				treeItem->setData(1, Qt::UserRole, (qulonglong) id);
				m_anisotropyTree->addTopLevelItem(treeItem);
				usedMaps[id] = true;

				m_anisotropyTree->openPersistentEditor(treeItem, 1);
				//m_anisotropyTree->setItemWidget(treeItem, 1, new QDoubleSpinBox);
			} else {
				it->second = true;
			}
		}
	}

	for (int i=0; i<m_anisotropyTree->topLevelItemCount(); i++) {
		bool ok;
		std::size_t id = m_anisotropyTree->topLevelItem(i)->data(0, Qt::UserRole).toULongLong(&ok);
		if (ok) {
			std::map<std::size_t, bool>::iterator it = usedMaps.find(id);
			if (it==usedMaps.end() || !it->second) {
				m_anisotropyTree->closePersistentEditor(m_anisotropyTree->topLevelItem(i), 1);
				QTreeWidgetItem* item = m_anisotropyTree->takeTopLevelItem(i);
				delete item;
			}
		}
	}
}

void LayerSpectrumDialog::anisotropyDataChanged(const QModelIndex& topLeft,
		const QModelIndex& bottomRight, const QVector<int>& roles) {
	for (int i : roles) {
		if (i == Qt::EditRole) {
			QVariant var = topLeft.data(Qt::UserRole);
			bool okId, okEdit;
			std::size_t id = var.toULongLong(&okId);

			var = topLeft.data(Qt::EditRole);
			float val = var.toULongLong(&okEdit);
			if (okId && okEdit) {
				m_isComputationValid = false;
				return;
			}
		}
	}
}

//void LayerSpectrumDialog::updateDataSetS(){
//	this->m_isComputationValid = false;
//	this->m_isWindowChanged = true;
//	QList<QListWidgetItem*> selection = m_sismiqueCombo->selectedItems();
//    m_meanDatasets.clear();
//	for (QListWidgetItem* item : selection) {
//		bool ok;
//		std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);
//		if (ok) {
//			Seismic3DDataset* dataset = m_allDatasets[id];
//			UpdateDatasets(dataset);
//			m_meanDatasets.push_back(std::pair<Seismic3DDataset*, int>(dataset, 0));
//		}
//	}
//}

QWidget* LayerSpectrumDialog::initWidget() {
	QWidget* topLevelWidget = new QWidget;
	QVBoxLayout* topLevelLayout = new QVBoxLayout;
	topLevelWidget->setLayout(topLevelLayout);

	QWidget* widget = new QWidget;
	QVBoxLayout* mainLayout = new QVBoxLayout;
	widget->setLayout(mainLayout);

#if 0
	m_displayViewCombo = new QComboBox;
	m_displayViewCombo->addItem("Stack Mode", QVariant(0));
	m_displayViewCombo->addItem(SplittedView::viewModeLabel(eTypeSplitMode), QVariant(1));
	m_displayViewCombo->addItem(SplittedView::viewModeLabel(eTypeTabMode)  , QVariant(2));
	m_displayViewCombo->setStyleSheet("QComboBox::item{height: 20px}");
	mainLayout->addWidget(m_displayViewCombo);
#endif
	// COMBO METHOD
	m_methodCombo = new QComboBox();
	m_methodCombo->addItem("Morlet");
	m_methodCombo->addItem("Spectrum");
	m_methodCombo->addItem("GCC/Coherence");
	m_methodCombo->addItem("TMap");
	m_methodCombo->addItem("Layer Attribute (Mean)");
	m_methodCombo->addItem("Anisotropy");
	// m_methodCombo->addItem("GCC mean");

	/*
	// JD
	m_methodCombo->addItem("Coherency");
	m_methodCombo->addItem("Root mean square");
	m_methodCombo->addItem("Min");
	m_methodCombo->addItem("Max");
	m_methodCombo->addItem("Absolute average");
	m_methodCombo->addItem("Most represented");
	m_methodCombo->addItem("Absolute add up");
	m_methodCombo->addItem("Absolute max");
	m_methodCombo->addItem("Add up");
	m_methodCombo->addItem("Average");
	m_methodCombo->addItem("Chaos");
	m_methodCombo->addItem("Depth/time of absolute max");
	m_methodCombo->addItem("Depth/time of max");
	m_methodCombo->addItem("Depth/time of min");
	m_methodCombo->addItem("Nearest middle horizon");
	m_methodCombo->addItem("Negative peak average");
	m_methodCombo->addItem("Negative peak count");
	m_methodCombo->addItem("Number of sample");
	m_methodCombo->addItem("Positive peak average");
	m_methodCombo->addItem("Positive peak count");
	m_methodCombo->addItem("Signed absolute max");
	m_methodCombo->addItem("Spline middle horizon");
	m_methodCombo->addItem("true thickness");
	 */

	m_methodCombo->setStyleSheet("QComboBox::item{height: 20px}");
	mainLayout->addWidget(m_methodCombo);

#if 1
	QGridLayout *gridLayout = new QGridLayout(this);
#else
	QGridLayout *gridLayout = new QGridLayout;
#endif
	gridLayout->setContentsMargins(0, 0, 0, 0);

#if 0
	QLabel *labelDisplayView = new QLabel("View in");
	gridLayout->addWidget(labelDisplayView, 0, 0, 1, 1);
	QComboBox* tmp_displayViewCombo = new QComboBox;
	tmp_displayViewCombo->addItem("Stack Mode", QVariant(0));
	tmp_displayViewCombo->addItem(SplittedView::viewModeLabel(eTypeSplitMode), QVariant(1));
	tmp_displayViewCombo->addItem(SplittedView::viewModeLabel(eTypeTabMode)  , QVariant(2));
	gridLayout->addWidget(tmp_displayViewCombo, 0, 1, 1, 1);
	tmp_displayViewCombo->setMaximumHeight(20);
#endif

	QLabel *labelRgt = new QLabel("Rgt");
	gridLayout->addWidget(labelRgt, 0, 0, 1, 1);
	//m_rgtCombo = new QComboBox(nullptr);
	m_rgtCombo = new QListWidget(nullptr);
	m_rgtCombo->setSelectionMode(QAbstractItemView::MultiSelection);
	gridLayout->addWidget(m_rgtCombo, 0, 1, 1, 1);
	m_rgtCombo->setMaximumHeight(40);

	QLabel *labelSismique = new QLabel("Seismics");
	gridLayout->addWidget(labelSismique, 1, 0, 1, 1);
	//m_sismiqueCombo = new QComboBox(nullptr);
	m_sismiqueCombo = new QListWidget(nullptr);
	gridLayout->addWidget(m_sismiqueCombo, 1, 1, 1, 1);

	QLabel* labelPatch = new QLabel("Patch");
	gridLayout->addWidget(labelPatch, 2, 0, 1, 1);
	m_patchList = new QListWidget(nullptr);
	gridLayout->addWidget(m_patchList, 2, 1, 1, 1);

#if 1
	gridLayout->setRowStretch(0, 0);
	gridLayout->setRowStretch(1, 1);
	gridLayout->setRowStretch(2, 1);
	gridLayout->setRowMinimumHeight(0, 40);
	gridLayout->setRowMinimumHeight(1, 100);
	gridLayout->setRowMinimumHeight(2, 60);
#else
	gridLayout->setRowStretch(0, 0);
	gridLayout->setRowStretch(1, 0);
	gridLayout->setRowStretch(2, 1);
	gridLayout->setRowStretch(3, 1);
	gridLayout->setRowMinimumHeight(1, 20);
	gridLayout->setRowMinimumHeight(1, 40);
	gridLayout->setRowMinimumHeight(2, 100);
	gridLayout->setRowMinimumHeight(3, 60);
#endif

	m_sismiqueCombo->setSelectionMode(QAbstractItemView::MultiSelection);
	connect (m_sismiqueCombo, &QListWidget::itemSelectionChanged,this,&LayerSpectrumDialog::updateDataSet);
	connect (m_rgtCombo, &QListWidget::itemSelectionChanged,this,&LayerSpectrumDialog::updateDataSet);
	connect (m_patchList, &QListWidget::itemSelectionChanged,this,&LayerSpectrumDialog::updatePatchDataSet);

	const QList<Seismic3DAbstractDataset*>& listDatasets = m_datasetS->survey()->datasets();
	for (std::size_t i=0; i<listDatasets.count(); i++) {
		Seismic3DDataset* cpuDataset = dynamic_cast<Seismic3DDataset*>(listDatasets[i]);

		// this function will auto select the dataset in m_datasetS
		addDataset(cpuDataset);

		if ( (cpuDataset!=nullptr) && (cpuDataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::RGT)
				&& (cpuDataset->dimV()==1) && (m_datasetT == cpuDataset)){
			std::size_t id = nextDatasetIndex();
			m_allDatasets[id] = cpuDataset;
			QListWidgetItem* item = new QListWidgetItem;
			item->setText(m_datasetT->name());
			item->setData(Qt::UserRole, (qulonglong) id);
			m_rgtCombo->addItem(item);
		}
	}

	mainLayout->addLayout(gridLayout);

	connect (m_methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &LayerSpectrumDialog::methodChanged);

	m_seedModeButton = new QPushButton;
	m_seedModeButton->setCheckable(true);
	mainLayout->addWidget(m_seedModeButton);

	QWidget* multiSeedHolder = new QWidget;
	QHBoxLayout* multiSeedForm = new QHBoxLayout;
	multiSeedHolder->setLayout(multiSeedForm);
	mainLayout->addWidget(multiSeedHolder);

	m_seedEditButton = new QPushButton;
	m_seedEditButton->setText("Stop Edit");
	m_seedEditButton->setCheckable(true);
	m_seedEditButton->setChecked(true);
	multiSeedForm->addWidget(m_seedEditButton);

	m_undoButton = new QPushButton;
	m_undoButton->setText("Undo");
	multiSeedForm->addWidget(m_undoButton);

	m_redoButton = new QPushButton;
	m_redoButton->setText("Redo");
	multiSeedForm->addWidget(m_redoButton);

	m_loadHorizonButton = new QPushButton;
	m_loadHorizonButton->setText("Load");
	multiSeedForm->addWidget(m_loadHorizonButton);

	m_saveButton = new QPushButton;
	m_saveButton->setText("Save");
	multiSeedForm->addWidget(m_saveButton);

	m_releaseButton = new QPushButton;
	m_releaseButton->setText("Release");
	multiSeedForm->addWidget(m_releaseButton);

	QWidget* propagationOptionHolder = new QWidget;
	QVBoxLayout* propagationPatchOptionLayout = new QVBoxLayout;
	QHBoxLayout* propagationOptionLayout = new QHBoxLayout;
	QHBoxLayout* patchOptionLayout = new QHBoxLayout;
	propagationPatchOptionLayout->addLayout(propagationOptionLayout);
	propagationPatchOptionLayout->addLayout(patchOptionLayout);
	propagationOptionHolder->setLayout(propagationPatchOptionLayout);
	mainLayout->addWidget(propagationOptionHolder);

	m_propagateButton = new QPushButton("Propagate");
	propagationOptionLayout->addWidget(m_propagateButton);

	m_undoPropagationButton = new QPushButton("Undo Propagation");
	propagationOptionLayout->addWidget(m_undoPropagationButton);
	// m_undoPropagationButton->hide();

	m_toggleInterpolation = new QPushButton("Interpolate");
	m_toggleInterpolation->setCheckable(true);
	m_toggleInterpolation->setChecked(false);
	propagationOptionLayout->addWidget(m_toggleInterpolation);

	QHBoxLayout* DataEditLayout = new QHBoxLayout;
	m_cloneAndKeepButton = new QPushButton("Clone and Keep");
	m_eraseDataButton = new QPushButton("Erase Data");
	DataEditLayout->addWidget(m_cloneAndKeepButton);
	DataEditLayout->addWidget(m_eraseDataButton);
	mainLayout->addLayout(DataEditLayout);

	QPushButton* patchButton = new QPushButton("Patch");
	QPushButton* patchNeighbourButton = new QPushButton("Neighbour patches");
	QPushButton* m_undoPatchButton = new QPushButton("Undo Patch");
	QPushButton* m_erasePatchButton = new QPushButton("Erase Patch");
	QLabel *patchThresholdLabel = new QLabel("Threshold:");
	m_patchThreshold = new QLineEdit("10");
	patchOptionLayout->addWidget(patchButton);
	patchOptionLayout->addWidget(m_undoPatchButton);
	patchOptionLayout->addWidget(m_erasePatchButton);
	patchOptionLayout->addWidget(patchNeighbourButton);
	patchOptionLayout->addWidget(patchThresholdLabel);
	patchOptionLayout->addWidget(m_patchThreshold);



	// COMMON
	QWidget* m_commonFormHolder = new QWidget;
	m_commonForm = new QFormLayout;
	m_commonFormHolder->setLayout(m_commonForm);
	mainLayout->addWidget(m_commonFormHolder);

	m_tauHolder = new QWidget;
	QHBoxLayout* tauLayout = new QHBoxLayout;
	m_tauHolder->setLayout(tauLayout);

	QPushButton* minusButton = new QPushButton;
	QImage minusImage(30, 30, QImage::Format_ARGB32);
	for (int j=0; j<minusImage.height(); j++) {
		unsigned char* buf = minusImage.scanLine(j);
		unsigned char val = 0;
		if (j>=10 && j<20) {
			val = 255;
		}
		for (int i=0; i<minusImage.width(); i++) {
			for (int c=0; c<4; c++) {
				buf[4*i+c] = val;
			}
		}
	}
	QPixmap minusPixmap = QPixmap::fromImage(minusImage);
	QIcon minusIcon(minusPixmap);
	minusButton->setIcon(minusIcon);
	tauLayout->addWidget(minusButton);

	m_tauSpinBox = new LayerSpectrumDialogSpinBox;
	m_tauSpinBox->setMinimum(0);
	m_tauSpinBox->setMaximum(std::numeric_limits<short>::max());
	m_tauSpinBox->setValue(m_pseudoTau);
	m_tauSpinBox->setSingleStep(m_tauStep);
	m_tauSpinBox->setButtonSymbols(QAbstractSpinBox::PlusMinus);
	tauLayout->addWidget(m_tauSpinBox);

	QPushButton* plusButton = new QPushButton;
	QImage plusImage(30, 30, QImage::Format_ARGB32);
	for (int j=0; j<plusImage.height(); j++) {
		unsigned char* buf = plusImage.scanLine(j);
		unsigned char val = 0;
		if (j>=10 && j<20) {
			val = 255;
		}
		for (int i=0; i<plusImage.width(); i++) {
			unsigned char _val = val;
			if (i>=10 && i<20) {
				_val = 255;
			}
			for (int c=0; c<4; c++) {
				buf[4*i+c] = _val;
			}
		}
	}
	QPixmap plusPixmap = QPixmap::fromImage(plusImage);
	QIcon plusIcon(plusPixmap);
	plusButton->setIcon(plusIcon);
	tauLayout->addWidget(plusButton);

	m_tauHolderLabel = new QLabel("Tau");
	m_commonForm->addRow(m_tauHolderLabel, m_tauHolder);

	m_tauStepSpinBox = new QSpinBox;
	m_tauStepSpinBox->setMinimum(1);
	m_tauStepSpinBox->setMaximum(std::numeric_limits<short>::max());
	m_tauStepSpinBox->setValue(m_tauStep);
	m_tauStepLabel = new QLabel("Step tau");
	m_commonForm->addRow(m_tauStepLabel, m_tauStepSpinBox);

	m_referenceList = new ItemListSelectorWithColor;
	initReferenceComboBox();
	m_commonForm->addRow("Reference", m_referenceList);

//	QPushButton* seedFromMarkerButton = new QPushButton("Load");
//	m_commonForm->addRow("Marker to Seed", seedFromMarkerButton);

	// Geotime QC With picks
	QPushButton* geotimeQCWithPicksButton = new QPushButton("Geotime QC with Picks");
	m_commonForm->addRow("Geotime QC", geotimeQCWithPicksButton);

	m_distancePowerSpinBox = new QSpinBox;
	m_distancePowerSpinBox->setMinimum(1);
	m_distancePowerSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_distancePowerSpinBox->setValue(m_distancePower);
	m_distancePowerLabel = new QLabel("distance power");
	m_commonForm->addRow(m_distancePowerLabel, m_distancePowerSpinBox);

	m_useSnapCheckBox = new QCheckBox;
	m_useSnapCheckBox->setCheckState(m_useSnap ? Qt::Checked : Qt::Unchecked);
	m_useSnapLabel = new QLabel("Use snap");
	m_commonForm->addRow(m_useSnapLabel, m_useSnapCheckBox);

	m_polarityComboBox = new QComboBox;
	m_polarityComboBox->addItem("+", QVariant(1));
	m_polarityComboBox->addItem("-", QVariant(-1));
	m_polarityComboBox->addItem("zc inc", QVariant(2));
	m_polarityComboBox->addItem("zc dec", QVariant(-2));
	m_polarityLabel = new QLabel("Polarity");
	m_commonForm->addRow(m_polarityLabel, m_polarityComboBox);

	m_snapWindowSpinBox = new QSpinBox;
	m_snapWindowSpinBox->setMinimum(1);
	m_snapWindowSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_snapWindowSpinBox->setValue(m_snapWindow);
	m_snapWindowLabel = new QLabel("Snap Window");
	m_commonForm->addRow(m_snapWindowLabel, m_snapWindowSpinBox);

	m_useMedianCheckBox = new QCheckBox;
	m_useMedianCheckBox->setCheckState(m_useMedian ? Qt::Checked : Qt::Unchecked);
	m_useMedianLabel = new QLabel("Use median");
	m_commonForm->addRow(m_useMedianLabel, m_useMedianCheckBox);

	m_halfLwxMedianFilterSpinBox = new QSpinBox;
	m_halfLwxMedianFilterSpinBox->setMinimum(1);
	m_halfLwxMedianFilterSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_halfLwxMedianFilterSpinBox->setValue(m_halfLwxMedianFilter);
	m_halfLwxMedianFilterLabel = new QLabel("half lwx median filter");
	m_commonForm->addRow(m_halfLwxMedianFilterLabel, m_halfLwxMedianFilterSpinBox);

	m_sizeCorrSpinBox = new QSpinBox;
	m_sizeCorrSpinBox->setMinimum(1);
	m_sizeCorrSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_sizeCorrSpinBox->setValue(m_sizeCorr);
	m_sizeCorrLabel = new QLabel("Half window correlation");
	m_commonForm->addRow(m_sizeCorrLabel, m_sizeCorrSpinBox);

	m_seuilCorrSpinBox = new QDoubleSpinBox;
	m_seuilCorrSpinBox->setMinimum(0);
	m_seuilCorrSpinBox->setMaximum(std::numeric_limits<float>::max());
	m_seuilCorrSpinBox->setDecimals(2);
	m_seuilCorrSpinBox->setValue(m_seuilCorr);
	m_seuilCorrLabel = new QLabel("Correlation Threshold");
	m_commonForm->addRow(m_seuilCorrLabel, m_seuilCorrSpinBox);

	/*m_seedReductionWindowSpinBox = new QSpinBox;
	m_seedReductionWindowSpinBox->setMinimum(1);
	m_seedReductionWindowSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_seedReductionWindowSpinBox->setValue(m_seedReductionWindow);
	m_seedReductionWindowLabel = new QLabel("Half window Seed Reduction");
	m_commonForm->addRow(m_seedReductionWindowLabel, m_seedReductionWindowSpinBox);*/

	m_seedFilterNumberSpinBox = new QSpinBox;
	m_seedFilterNumberSpinBox->setMinimum(1);
	m_seedFilterNumberSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_seedFilterNumberSpinBox->setValue(m_seedFilterNumber);
	m_seedFilterNumberLabel = new QLabel("Half window Seed Reduction");
	m_commonForm->addRow(m_seedFilterNumberLabel, m_seedFilterNumberSpinBox);

	m_propagationTypeComboBox = new QComboBox;
	m_propagationTypeComboBox->addItem("Only new seeds", QVariant(2));
	m_propagationTypeComboBox->addItem("All seeds", QVariant(1));
	m_propagationTypeComboBox->setCurrentIndex(1);
	m_propagationTypeLabel = new QLabel("Propagation Mode");
	m_commonForm->addRow(m_propagationTypeLabel, m_propagationTypeComboBox);

	m_numIterSpinBox = new QSpinBox;
	m_numIterSpinBox->setMinimum(1);
	m_numIterSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_numIterSpinBox->setValue(m_numIter);
	m_numIterLabel = new QLabel("Propagator Nb Iter");
	m_commonForm->addRow(m_numIterLabel, m_numIterSpinBox);

	// SPECTRUM
	m_spectrumGroupBox = new QGroupBox("Spectrum Parameters :");

	QFormLayout* spectrumForm = new QFormLayout;

	m_spectrumWindowSizeSpinBox = new QSpinBox;
	m_spectrumWindowSizeSpinBox->setMinimum(1);
	m_spectrumWindowSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_spectrumWindowSizeSpinBox->setValue(m_spectrumWindowSize);
	spectrumForm->addRow("Window Size", m_spectrumWindowSizeSpinBox);

	m_spectrumHatPower = new QDoubleSpinBox;
	m_spectrumHatPower->setMinimum(0);
	m_spectrumHatPower->setMaximum(std::numeric_limits<float>::max());
	m_spectrumHatPower->setDecimals(2);
	m_spectrumHatPower->setValue(m_hatPower);
	m_spectrumHatPowerLabel = new QLabel("Hat Power");
	spectrumForm->addRow(m_spectrumHatPowerLabel, m_spectrumHatPower);


	m_spectrumGroupBox->setLayout(spectrumForm);
	mainLayout->addWidget(m_spectrumGroupBox);

	// MORLET
	m_morletGroupBox = new QGroupBox("Morlet Parameters :");

	QFormLayout* morletForm = new QFormLayout;

	m_minFreqSpinBox = new QSpinBox;
	m_minFreqSpinBox->setMinimum(15);
	m_minFreqSpinBox->setMaximum(200);
	m_minFreqSpinBox->setValue(m_freqMin);
	morletForm->addRow("Frequency Min", m_minFreqSpinBox);

	m_maxFreqSpinBox = new QSpinBox;
	m_maxFreqSpinBox->setMinimum(15);
	m_maxFreqSpinBox->setMaximum(200);
	m_maxFreqSpinBox->setValue(m_freqMax);
	morletForm->addRow("Frequency Max ", m_maxFreqSpinBox);

	m_stepFreqSpinBox = new QSpinBox;
	m_stepFreqSpinBox->setMinimum(1);
	m_stepFreqSpinBox->setMaximum(20);
	m_stepFreqSpinBox->setValue(m_freqStep);
	morletForm->addRow("Frequency Step ", m_stepFreqSpinBox);

	m_morletGroupBox->setLayout(morletForm);
	mainLayout->addWidget(m_morletGroupBox);

	m_gccGroupBox = new QGroupBox("GCC Parameters :");

	QFormLayout* gccForm = new QFormLayout;

	m_gccOffsetSpinBox = new QSpinBox;
	m_gccOffsetSpinBox->setMinimum(1);
	m_gccOffsetSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_gccOffsetSpinBox->setValue(m_gccOffset);
	gccForm->addRow("Window size ", m_gccOffsetSpinBox);

	m_wSpinBox = new QSpinBox;
	m_wSpinBox->setMinimum(std::numeric_limits<int>::min());
	m_wSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_wSpinBox->setValue(m_w);
	gccForm->addRow("W ", m_wSpinBox);

	m_shiftSpinBox = new QSpinBox;
	m_shiftSpinBox->setMinimum(std::numeric_limits<int>::min());
	m_shiftSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_shiftSpinBox->setValue(m_shift);
	gccForm->addRow("Shift ", m_shiftSpinBox);

	m_gccGroupBox->setLayout(gccForm);
	mainLayout->addWidget(m_gccGroupBox);

	m_tmapGroupBox = new QGroupBox("TMap Parameters :");

	QFormLayout* tmapForm = new QFormLayout;

	m_tmapExampleSizeSpinBox = new QSpinBox;
	m_tmapExampleSizeSpinBox->setMinimum(1);
	m_tmapExampleSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_tmapExampleSizeSpinBox->setValue(m_tmapExampleSize);
	tmapForm->addRow("Example Size ", m_tmapExampleSizeSpinBox);

	m_tmapSizeSpinBox = new QSpinBox;
	m_tmapSizeSpinBox->setMinimum(1);
	m_tmapSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_tmapSizeSpinBox->setValue(m_tmapSize);
	tmapForm->addRow("TMap Size ", m_tmapSizeSpinBox);

	m_tmapExampleStepSpinBox = new QSpinBox;
	m_tmapExampleStepSpinBox->setMinimum(1);
	m_tmapExampleStepSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_tmapExampleStepSpinBox->setValue(m_tmapExampleStep);
	tmapForm->addRow("Example selection step ", m_tmapExampleStepSpinBox);

	m_tmapGroupBox->setLayout(tmapForm);
	mainLayout->addWidget(m_tmapGroupBox);

	m_meanGroupBox = new QGroupBox("Mean Parameters :");

	QFormLayout* meanForm = new QFormLayout;

	m_meanWindowSizeSpinBox = new QSpinBox;
	m_meanWindowSizeSpinBox->setMinimum(1);
	m_meanWindowSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	m_meanWindowSizeSpinBox->setValue(m_meanWindowSize);
	meanForm->addRow("Window Size ", m_meanWindowSizeSpinBox);

	//	m_meanDatasetListWidget = new QListWidget;
	//	m_meanDatasetListWidget->setSelectionMode(QAbstractItemView::MultiSelection);
	//	const QList<Seismic3DAbstractDataset*>& surveyDatasets = m_datasetS->survey()->datasets();
	//	for (std::size_t i=0; i<surveyDatasets.count(); i++) {
	//		Seismic3DDataset* cpuDataset = dynamic_cast<Seismic3DDataset*>(surveyDatasets[i]);
	//		if (cpuDataset!=nullptr) {
	//			std::size_t id = nextDatasetIndex();
	//			m_allDatasets[id] = cpuDataset;
	//
	//			QListWidgetItem* item = new QListWidgetItem;
	//			item->setText(cpuDataset->name());
	//			item->setData(Qt::UserRole, (qulonglong) id);
	//			m_meanDatasetListWidget->addItem(item);
	//		}
	//	}

	//	meanForm->addRow("Datasets ", m_meanDatasetListWidget);

	m_meanGroupBox->setLayout(meanForm);
	mainLayout->addWidget(m_meanGroupBox);

	m_anisotropyGroupBox = new QGroupBox("Anisotropy Parameters : ");

	QVBoxLayout* anisotropyForm = new QVBoxLayout;
	m_anisotropyTree = new QTreeWidget;
	m_anisotropyTree->setStyleSheet("QTreeWidget{min-height: 4.5em}");
	m_anisotropyTree->setColumnCount(2);
	m_anisotropyTree->setHeaderLabels(QStringList() << "Dataset" << "Angle");
	anisotropyForm->addWidget(m_anisotropyTree);

	m_anisotropyGroupBox->setLayout(anisotropyForm);
	mainLayout->addWidget(m_anisotropyGroupBox);

	m_historySpectrumList = new QListWidget;
	fillHistorySpectrum();
	mainLayout->addWidget(m_historySpectrumList);
	m_historySpectrumList->setStyleSheet("QListWidget{min-height: 3em}");

	m_historyMorletList = new QListWidget;
	fillHistoryMorlet();
	mainLayout->addWidget(m_historyMorletList);
	m_historyMorletList->setStyleSheet("QListWidget{min-height: 3em}");

	m_historyGccList = new QListWidget;
	fillHistoryGcc();
	mainLayout->addWidget(m_historyGccList);
	m_historyGccList->setStyleSheet("QListWidget{min-height: 3em}");

	m_historyTmapList = new QListWidget;
	fillHistoryTmap();
	mainLayout->addWidget(m_historyTmapList);
	m_historyTmapList->setStyleSheet("QListWidget{min-height: 3em}");

	m_historyMeanList = new QListWidget;
	fillHistoryMean();
	mainLayout->addWidget(m_historyMeanList);
	m_historyMeanList->setStyleSheet("QListWidget{min-height: 3em}");

	m_historyAnisotropyList = new QListWidget;
	fillHistoryAnisotropy();
	mainLayout->addWidget(m_historyAnisotropyList);
	m_historyAnisotropyList->setStyleSheet("QListWidget{min-height: 3em}");

	QWidget* holderComputeButtonOnly = new QWidget;
	QHBoxLayout* layoutComputeButtonOnly = new QHBoxLayout;
	mainLayout->addWidget(holderComputeButtonOnly);
	holderComputeButtonOnly->setLayout(layoutComputeButtonOnly);

	QLabel* labelComputeButtonOnly = new QLabel("Only compute if button pressed");
	layoutComputeButtonOnly->addWidget(labelComputeButtonOnly);

	m_computeButtonOnlyCheckBox = new QCheckBox;
	m_computeButtonOnlyCheckBox->setCheckState(m_computeButtonOnly ? Qt::Checked : Qt::Unchecked);
	layoutComputeButtonOnly->addWidget(m_computeButtonOnlyCheckBox);

//	QWidget* holderDebug = new QWidget;
//	QHBoxLayout* layoutDebug = new QHBoxLayout;
//	mainLayout->addWidget(holderDebug);
//	holderDebug->setLayout(layoutDebug);

	QLabel* labelDebug = new QLabel("Debug mode");
	layoutComputeButtonOnly->addWidget(labelDebug);

	m_debugModeCheckBox = new QCheckBox;
	m_debugModeCheckBox->setCheckState(m_debugMode ? Qt::Checked : Qt::Unchecked);
	layoutComputeButtonOnly->addWidget(m_debugModeCheckBox);

	QPushButton *pbRgtModifications = new QPushButton("Rgt adjust");
	mainLayout->addWidget(pbRgtModifications);
	connect(pbRgtModifications, SIGNAL(clicked()), this, SLOT(trt_rgtModifications()));
#ifndef __DEBUG_RGTPICKING__
	pbRgtModifications->setVisible(false);
#endif

	QScrollArea* scrollArea = new QScrollArea;
	scrollArea->setWidget(widget);
	scrollArea->setWidgetResizable(true);
	topLevelLayout->addWidget(scrollArea);

	QHBoxLayout* computeLayout = new QHBoxLayout;
	m_computeButton = new QPushButton("Compute");
	m_progressbar = new QProgressBar;
	m_progressbar->setOrientation(Qt::Horizontal);
	m_progressbar->setMinimum(0);
	m_progressbar->setMaximum(100);
	computeLayout->addWidget(m_computeButton);
	computeLayout->addWidget(m_progressbar);
	topLevelLayout->addLayout(computeLayout);
	// m_progressbar->hide();

	connect(m_methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
		m_method = index;
		if (m_method == 0) {
			m_morletGroupBox->show();
			m_historyMorletList->show();
		} else {
			m_morletGroupBox->hide();
			m_historyMorletList->hide();
		}

		if (m_method == 1) {
			m_spectrumGroupBox->show();
			m_historySpectrumList->show();
		} else {
			m_spectrumGroupBox->hide();
			m_historySpectrumList->hide();
		}

		if (m_method == 2) {
			m_gccGroupBox->show();
			m_historyGccList->show();
		} else {
			m_gccGroupBox->hide();
			m_historyGccList->hide();
		}

		if (m_method == 3) {
			m_tmapGroupBox->show();
			m_historyTmapList->show();
		} else {
			m_tmapGroupBox->hide();
			m_historyTmapList->hide();
		}

		if (m_method == 4) {
			m_meanGroupBox->show();
			m_historyMeanList->show();
		} else {
			m_meanGroupBox->hide();
			m_historyMeanList->hide();
		}

		if (m_method == 5) {
			m_anisotropyGroupBox->show();
			m_historyAnisotropyList->show();
		} else {
			m_anisotropyGroupBox->hide();
			m_historyAnisotropyList->hide();
		}

		if ( m_method > 3 )// && m_method < 25 )
		{
			m_gccGroupBox->hide();
			m_historyGccList->hide();
			m_spectrumGroupBox->hide();
			m_morletGroupBox->hide();
			m_historyMorletList->hide();
		}

		if (m_horizon!=nullptr) {
			if ( m_method == 0 || m_method == 2 || m_method == 5 ) {
				m_horizon->setHorizonMode(DEFAULT);
			} else if (m_method == 1) {
				m_horizon->setHorizonMode(DELTA_T);
				m_horizon->setDelta(m_spectrumWindowSize/2);
			} else if (m_method == 3) {
				m_horizon->setHorizonMode(DELTA_T);
				m_horizon->setDeltaTop(0);
				m_horizon->setDeltaBottom(m_tmapExampleSize);
			} else if (m_method == 4) {
				m_horizon->setHorizonMode(DELTA_T);
				m_horizon->setDelta(m_meanWindowSize/2); // false if m_meanWindowSize%2==0 but only one pixel of error
			}
		}

		this->m_isComputationValid = false;
		this->m_isWindowChanged = true;
	});

	connect(m_seedModeButton, &QPushButton::toggled, this, &LayerSpectrumDialog::toggleSeedMode);

	connect(m_seedEditButton, &QPushButton::toggled, [this](bool checked) {
		m_seedEdit = checked;
		if (m_seedEdit) {
			m_seedEditButton->setText("Stop Edit");
			m_horizon->setBehaviorMode(POINTPICKING);
		} else {
			m_seedEditButton->setText("Start Edit");
			m_horizon->setBehaviorMode(FIXED);
		}
	});

	connect(m_undoButton, &QPushButton::clicked, this, &LayerSpectrumDialog::undoHorizonModification);
	connect(m_redoButton, &QPushButton::clicked, this, &LayerSpectrumDialog::redoHorizonModification);
	connect(m_loadHorizonButton, &QPushButton::clicked, this, &LayerSpectrumDialog::loadHorizon);
	connect(m_saveButton, &QPushButton::clicked, this, &LayerSpectrumDialog::saveHorizon);
	connect(m_releaseButton, &QPushButton::clicked, this, &LayerSpectrumDialog::clearHorizon);

	connect(m_tauSpinBox, &LayerSpectrumDialogSpinBox::editingFinished, [this]() {
		int tau = this->m_tauSpinBox->value();
		setPseudoGeologicalTime(tau);
		this->updateData();
		//		trt_compute();
	});

	connect(minusButton, &QPushButton::clicked, [this]() {
		m_tauSpinBox->stepDown();
		m_tauSpinBox->editingFinished();
	});

	connect(plusButton, &QPushButton::clicked, [this]() {
		m_tauSpinBox->stepUp();
		m_tauSpinBox->editingFinished();
	});

	connect(m_tauStepSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int tauStep) {
		m_tauSpinBox->setSingleStep(tauStep);
	});

	connect(m_distancePowerSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		m_distancePower = val;
		if (m_horizon!=nullptr) {
			m_horizon->setDistancePower(m_distancePower);
			this->m_isComputationValid = false;
		}
	});

	connect(m_useSnapCheckBox, &QCheckBox::stateChanged, [this](int state) {
		if (m_useSnap!=(state==Qt::Checked)) {
			m_useSnap = state==Qt::Checked;
			if (m_horizon!=nullptr) {
				m_horizon->toggleSnap(m_useSnap);
			}
			if (m_useSnap && m_debugMode) {
				m_polarityComboBox->show();
				m_polarityLabel->show();
				m_snapWindowSpinBox->show();
				m_snapWindowLabel->show();
			} else {
				m_polarityComboBox->hide();
				m_polarityLabel->hide();
				m_snapWindowSpinBox->hide();
				m_snapWindowLabel->hide();
			}
			m_isComputationValid = false;
		}
	});

	connect(m_polarityComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
		QVariant var = m_polarityComboBox->itemData(index);
		bool ok;
		int varInt = var.toInt(&ok);
		if (ok) {
			//m_polarity = varInt;
			setPolarity(varInt);
			if (m_useSnap) {
				m_isComputationValid = false;
			}
		}
	});

	connect(m_snapWindowSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		if (val != m_snapWindow) {
			m_snapWindow = val;
			if (m_useSnap) {
				if (m_horizon!=nullptr) {
					m_horizon->setSnapWindow(m_snapWindow);
				}
				m_isComputationValid = false;
			}
		}
	});

	connect(m_useMedianCheckBox, &QCheckBox::stateChanged, [this](int state) {
		if (m_useMedian!=(state==Qt::Checked)) {
			m_useMedian = state==Qt::Checked;
			if (m_debugMode && m_useMedian) {
				m_halfLwxMedianFilterLabel->show();
				m_halfLwxMedianFilterSpinBox->show();
			} else {
				m_halfLwxMedianFilterLabel->hide();
				m_halfLwxMedianFilterSpinBox->hide();
			}
			if (m_horizon!=nullptr) {
				m_horizon->toggleMedian(m_useMedian);
			}
			m_isComputationValid = false;
		}
	});

	connect(m_halfLwxMedianFilterSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		m_halfLwxMedianFilter = val;
		if (m_horizon!=nullptr) {
			m_horizon->setLWXMedianFilter(m_halfLwxMedianFilter*2+1);
			this->m_isComputationValid = false;
		}

	});

	connect(m_sizeCorrSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		m_sizeCorr = val;
	});

	connect(m_seuilCorrSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double val) {
		m_seuilCorr = val;
	});

	connect(m_seedFilterNumberSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		m_seedFilterNumber = val;
	});

	connect(m_numIterSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		m_numIter = val;
	});

	connect(m_spectrumWindowSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int ws) {
		if (this->m_spectrumWindowSize != ws) {
			this->m_spectrumWindowSize = ws;
			m_horizon->setDelta(m_spectrumWindowSize/2);
			this->m_isComputationValid = false;
			this->m_isWindowChanged = true;
		}
	});

	connect(m_spectrumHatPower, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double val) {
		if (this->m_hatPower != val) {
			this->m_hatPower = val;
			this->m_isComputationValid = false;
		}
	});

	connect(m_minFreqSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int ws) {
		if (this->m_freqMin != ws) {
			this->m_freqMin = ws;
			this->m_isComputationValid = false;
			this->m_isWindowChanged = true;
		}
	});

	connect(m_maxFreqSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int ws) {
		if (this->m_freqMax != ws) {
			this->m_freqMax = ws;
			this->m_isComputationValid = false;
			this->m_isWindowChanged = true;
		}
	});

	connect(m_stepFreqSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int ws) {
		if (this->m_freqStep != ws) {
			this->m_freqStep = ws;
			this->m_isComputationValid = false;
			this->m_isWindowChanged = true;
		}
	});

	connect(m_gccOffsetSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int ws) {
		if (this->m_gccOffset != ws) {
			this->m_gccOffset = ws;
			this->m_isComputationValid = false;
			this->m_isWindowChanged = true;
		}
	});

	connect(m_wSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int ws) {
		if (this->m_w != ws) {
			this->m_w = ws;
			this->m_isComputationValid = false;
		}
	});

	connect(m_shiftSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int ws) {
		if (this->m_shift != ws) {
			this->m_shift = ws;
			this->m_isComputationValid = false;
		}
	});

	connect(m_tmapExampleSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		if (this->m_tmapExampleSize != val) {
			this->m_tmapExampleSize = val;
			this->m_isComputationValid = false;
		}
	});

	connect(m_tmapSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		if (this->m_tmapSize!= val) {
			this->m_tmapSize = val;
			this->m_isComputationValid = false;
		}
	});

	connect(m_tmapExampleStepSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		if (this->m_tmapExampleStep != val) {
			this->m_tmapExampleStep = val;
			this->m_isComputationValid = false;
		}
	});

	connect(m_meanWindowSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		if (this->m_meanWindowSize != val) {
			this->m_meanWindowSize = val;
			this->m_isComputationValid = false;
			if (this->m_horizon) {
				this->m_horizon->setDelta(this->m_meanWindowSize/2);
			}
		}
	});



	connect(m_computeButton, &QPushButton::clicked, [this]() {
		this->updateData(true);
		//		trt_compute();
	});

	connect(m_historySpectrumList, &QListWidget::currentItemChanged, [this](QListWidgetItem* item) {
		bool test;
		long currentId = item->data(Qt::UserRole).toLongLong(&test);
		long i=0;
		while (i<this->m_historySpectrum.size() && this->m_historySpectrum[i].id!=currentId) {
			i++;
		}
		if (i<this->m_historySpectrum.size()) {
			this->setSeeds(this->m_historySpectrum[i].seeds);
			this->setPolarity(this->m_historySpectrum[i].polarity);
			this->m_distancePowerSpinBox->setValue(this->m_historySpectrum[i].distancePower);
			this->m_halfLwxMedianFilterSpinBox->setValue(m_historySpectrum[i].halfLwx);
			this->m_spectrumWindowSizeSpinBox->setValue(this->m_historySpectrum[i].windowSize);
			this->m_spectrumHatPower->setValue(this->m_historySpectrum[i].hatPower);
			this->m_methodCombo->setCurrentIndex(1);
			// this->updateData();
			//			trt_compute();
		}
	});

	connect(m_historyMorletList, &QListWidget::currentItemChanged, [this](QListWidgetItem* item) {
		bool test;
		long currentId = item->data(Qt::UserRole).toLongLong(&test);
		long i=0;
		while (i<this->m_historyMorlet.size() && this->m_historyMorlet[i].id!=currentId) {
			i++;
		}
		if (i<this->m_historyMorlet.size()) {
			this->setSeeds(this->m_historyMorlet[i].seeds);
			this->setPolarity(this->m_historyMorlet[i].polarity);
			this->m_distancePowerSpinBox->setValue(this->m_historyMorlet[i].distancePower);
			this->m_halfLwxMedianFilterSpinBox->setValue(m_historyMorlet[i].halfLwx);
			this->m_minFreqSpinBox->setValue(this->m_historyMorlet[i].freqMin);
			this->m_maxFreqSpinBox->setValue(this->m_historyMorlet[i].freqMax);
			this->m_stepFreqSpinBox->setValue(this->m_historyMorlet[i].freqMax);
			this->m_methodCombo->setCurrentIndex(0);
			this->updateData();
			//			trt_compute();
		}
	});

	connect(m_historyGccList, &QListWidget::currentItemChanged, [this](QListWidgetItem* item) {
		bool test;
		long currentId = item->data(Qt::UserRole).toLongLong(&test);
		long i=0;
		while (i<this->m_historyGcc.size() && this->m_historyGcc[i].id!=currentId) {
			i++;
		}
		if (i<this->m_historyGcc.size()) {
			this->setSeeds(this->m_historyGcc[i].seeds);
			this->setPolarity(this->m_historyGcc[i].polarity);
			this->m_distancePowerSpinBox->setValue(this->m_historyGcc[i].distancePower);
			this->m_halfLwxMedianFilterSpinBox->setValue(m_historyGcc[i].halfLwx);
			this->m_gccOffsetSpinBox->setValue(this->m_historyGcc[i].offset);
			this->m_wSpinBox->setValue(this->m_historyGcc[i].w);
			this->m_shiftSpinBox->setValue(this->m_historyGcc[i].shift);
			this->m_methodCombo->setCurrentIndex(2);
			this->updateData();
			//			trt_compute();
		}
	});

	connect(m_historyTmapList, &QListWidget::currentItemChanged, [this](QListWidgetItem* item) {
		bool test;
		long currentId = item->data(Qt::UserRole).toLongLong(&test);
		long i=0;
		while (i<this->m_historyTmap.size() && this->m_historyTmap[i].id!=currentId) {
			i++;
		}
		if (i<this->m_historyTmap.size()) {
			this->setSeeds(this->m_historyTmap[i].seeds);
			this->setPolarity(this->m_historyTmap[i].polarity);
			this->m_distancePowerSpinBox->setValue(this->m_historyTmap[i].distancePower);
			this->m_halfLwxMedianFilterSpinBox->setValue(m_historyTmap[i].halfLwx);
			this->m_tmapExampleSizeSpinBox->setValue(m_historyTmap[i].tmapExampleSize);
			this->m_tmapSizeSpinBox->setValue(m_historyTmap[i].tmapSize);
			this->m_tmapExampleStepSpinBox->setValue(m_historyTmap[i].tmapExampleStep);
			this->m_methodCombo->setCurrentIndex(3);
			this->updateData();
			//			trt_compute();
		}
	});

	connect(m_historyMeanList, &QListWidget::currentItemChanged, [this](QListWidgetItem* item) {
		bool test;
		long currentId = item->data(Qt::UserRole).toLongLong(&test);
		long i=0;
		while (i<this->m_historyMean.size() && this->m_historyMean[i].id!=currentId) {
			i++;
		}
		if (i<this->m_historyMean.size()) {
			this->setSeeds(this->m_historyMean[i].seeds);
			this->setPolarity(this->m_historyMean[i].polarity);
			this->m_distancePowerSpinBox->setValue(this->m_historyMean[i].distancePower);
			this->m_halfLwxMedianFilterSpinBox->setValue(m_historyMean[i].halfLwx);
			this->m_meanWindowSizeSpinBox->setValue(m_historyMean[i].meanWindowSize);
			this->m_methodCombo->setCurrentIndex(4);
			this->updateData();
			//			trt_compute();
		}
	});

	connect(m_historyAnisotropyList, &QListWidget::currentItemChanged, [this](QListWidgetItem* item) {
		bool test;
		long currentId = item->data(Qt::UserRole).toLongLong(&test);
		long i=0;
		while (i<this->m_historyAnisotropy.size() && this->m_historyAnisotropy[i].id!=currentId) {
			i++;
		}
		if (i<this->m_historyAnisotropy.size()) {
			this->setSeeds(this->m_historyAnisotropy[i].seeds);
			this->setPolarity(this->m_historyAnisotropy[i].polarity);
			this->m_distancePowerSpinBox->setValue(this->m_historyAnisotropy[i].distancePower);
			this->m_halfLwxMedianFilterSpinBox->setValue(m_historyAnisotropy[i].halfLwx);
			this->m_methodCombo->setCurrentIndex(5);
			this->updateData();
			//			trt_compute();
		}
	});

	connect(m_debugModeCheckBox, &QCheckBox::stateChanged, [this](int state) {
		toggleDebugMode(state==Qt::Checked);
	});

	connect(m_computeButtonOnlyCheckBox, &QCheckBox::stateChanged, [this](int state) {
		m_computeButtonOnly = state==Qt::Checked;
	});

	connect(m_propagateButton, &QPushButton::clicked, this, &LayerSpectrumDialog::propagateHorizon);
	connect(m_cloneAndKeepButton, &QPushButton::clicked, this, &LayerSpectrumDialog::cloneAndKeep);
	connect(m_eraseDataButton, &QPushButton::clicked, this, &LayerSpectrumDialog::eraseData);
	connect(m_undoPropagationButton, &QPushButton::clicked, this, &LayerSpectrumDialog::undoPropagation);
	connect(m_undoPatchButton, &QPushButton::clicked, this, &LayerSpectrumDialog::undoPropagation);
	connect(m_erasePatchButton, &QPushButton::clicked, this, &LayerSpectrumDialog::erasePatch);
	connect(m_toggleInterpolation, &QPushButton::toggled, this, &LayerSpectrumDialog::interpolateHorizon);
	connect(m_propagationTypeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
		QVariant type = m_propagationTypeComboBox->itemData(index);
		bool ok;
		m_propagationType = type.toInt(&ok);
	});

	connect(m_referenceList, &ItemListSelectorWithColor::itemSelectionChanged, this, &LayerSpectrumDialog::updateReferenceHorizonList);
	connect(m_referenceList, &ItemListSelectorWithColor::colorChanged, this, &LayerSpectrumDialog::updateReferenceHorizonColor);

//	connect(seedFromMarkerButton, &QPushButton::clicked, [this, widget]() {
//		CubeSeismicAddon addon = m_datasetT->cubeSeismicAddon();
//		SelectSeedFromMarkers dialog(m_datasetT->workingSetManager(), m_datasetT, m_channelT, addon.getSampleUnit(), "Load Seed From Markers", widget);
//		int result = dialog.exec();
//		if (result==QDialog::Accepted) {
//			QList<RgtSeed> newSeeds = dialog.getSelectedSeeds();
//			std::vector<RgtSeed> newSeedsVec(newSeeds.begin(), newSeeds.end());
//
//			if (m_horizon) {
//				m_seeds = newSeedsVec;
//				m_horizon->setSeeds(newSeedsVec);
//				m_isComputationValid = false;
//			}
//		}
//	});

	connect(geotimeQCWithPicksButton, &QPushButton::clicked, [this, widget]() {
		//		CubeSeismicAddon addon = m_datasetT->cubeSeismicAddon();
		//		SelectSeedFromMarkers dialog(m_datasetT->workingSetManager(), m_datasetT, addon.getSampleUnit(), "Load Seed From Markers", widget);
		//		int result = dialog.exec();
		//		if (result==QDialog::Accepted) {
		//			QList<RgtSeed> newSeeds = dialog.getSelectedSeeds();
		//			std::vector<RgtSeed> newSeedsVec(newSeeds.begin(), newSeeds.end());
		//
		//			if (m_horizon) {
		//				m_seeds = newSeedsVec;
		//				m_horizon->setSeeds(newSeedsVec);
		//				m_isComputationValid = false;
		//			}
		//		}
		GeotimeMarkerQCDialog* dialog = new GeotimeMarkerQCDialog(this,
				m_datasetS->workingSetManager(), m_datasetT, m_channelT, m_originViewer->depthLengthUnit());
		dialog->resize(550*2, 950);

		connect(dialog, &GeotimeMarkerQCDialog::choosedPicks, this, &LayerSpectrumDialog::choosedPicksFromQC);
		connect(m_originViewer, &GeotimeGraphicsView::depthLengthUnitChanged, dialog, &GeotimeMarkerQCDialog::setDepthLengthUnit);
		connect(dialog, &GeotimeMarkerQCDialog::destroyed, [dialog, this]() {
			// this should not crash because dialog is a child of this so this->m_originViewer should always be valid
			disconnect(this->m_originViewer, nullptr, dialog, nullptr);
		});
		dialog->show();
	});

	m_conn.push_back(connect(m_datasetS->survey(), &SeismicSurvey::datasetAdded, this, &LayerSpectrumDialog::addDataset));

	connect(m_meanWindowSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int val) {
		if (val!=m_meanWindowSize) {
			m_meanWindowSize = val;
			this->m_isComputationValid = false;
		}
	});

	//	connect(m_meanDatasetListWidget, &QListWidget::itemSelectionChanged, [this]() {
	//		this->m_isComputationValid = false;
	//		this->m_isWindowChanged = true;
	//		m_meanDatasets.clear();
	//		QList<QListWidgetItem*> selection = m_meanDatasetListWidget->selectedItems();
	//		for (QListWidgetItem* item : selection) {
	//			bool ok;
	//			std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);
	//			if (ok) {
	//				Seismic3DDataset* dataset = m_allDatasets[id];
	//				m_meanDatasets.push_back(std::pair<Seismic3DDataset*, int>(dataset, 0));
	//			}
	//		}
	//	});

	QAbstractItemModel* model = m_anisotropyTree->model();
	connect(model, &QAbstractItemModel::dataChanged, this, &LayerSpectrumDialog::anisotropyDataChanged);

	connect(patchButton, &QPushButton::clicked, this, &LayerSpectrumDialog::patchConstrain);
	connect(patchNeighbourButton, &QPushButton::clicked, this, &LayerSpectrumDialog::patchNeightbourConstrain);


	m_methodCombo->setCurrentIndex(m_method);
	m_seedModeButton->setChecked(m_horizon->isMultiSeed());
	m_seedModeButton->toggled(m_horizon->isMultiSeed());
	toggleDebugMode(m_debugMode);
	m_horizon->setMultiSeed(!m_horizon->isMultiSeed());
	toggleSeedMode(!m_horizon->isMultiSeed());

	updateAngleTree();

	QTimer *timer = new QTimer(this);
	timer->start(1000);
	timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));

	return topLevelWidget;
}

void LayerSpectrumDialog::setLayerData(){

	m_data->setSeeds(m_seeds);
	m_data->setMethod(m_method);
	m_data->setUseSnap(m_useSnap);
	m_data->setSnapWindow(m_snapWindow);
	m_data->setUseMedian(m_useMedian);
	m_data->setLWXMedianFilter(m_halfLwxMedianFilter*2+1);
	QString isoName("isochrone");
	m_data->setConstrainLayer(m_constrainData.get(), isoName);

	if (m_method == 0) {
		m_data->setFreqMin(m_freqMin);
		m_data->setFreqMax(m_freqMax);
		m_data->setFreqStep(m_freqStep);
	}
	else if (m_method == 1) {
		m_data->setWindowSize(m_spectrumWindowSize);
		m_data->setHatPower(m_hatPower);
	}
	else if (m_method == 2) {
		m_data->setGccOffset(m_gccOffset);
		m_data->setW(m_w);
		m_data->setShift(m_shift);
		m_data->setType(0);
	}
	else if ( m_method == 3 )
	{
		m_data->setTmapExampleSize(m_tmapExampleSize);
		m_data->setTmapSize(m_tmapSize);
		m_data->setTmapExampleStep(m_tmapExampleStep);
		/*
			m_data->setWindowSize(m_gccWindowSize);
			m_data->setW(m_w);
			m_data->setShift(m_shift);
			m_data->setType(0);
		 */
	}
	else if ( m_method == 4 )
	{
		m_data->setWindowSize(m_meanWindowSize);
		QList<std::pair<Seismic3DDataset*, int>> meanDatasets;
		// do not use this->m_meanDatasets because each mean will be compute by the related layer slice
		meanDatasets.push_back(std::pair<Seismic3DDataset*, int>(m_data->getDatasetS(), m_data->getChannelS()));
		m_data->setAttributDatasets(meanDatasets);
	}
	else if ( m_method == 5 && m_anisotropyTree!=nullptr)
	{
		QList<std::tuple<Seismic3DDataset*, int, float>> anisotropyDatasetsAndAngles;
		for (int i=0; i<m_anisotropyTree->topLevelItemCount(); i++) {
			bool okId, okAngle;
			std::size_t id = m_anisotropyTree->topLevelItem(i)->data(0, Qt::UserRole).toULongLong(&okId);
			float angle = m_anisotropyTree->topLevelItem(i)->data(1, Qt::EditRole).toULongLong(&okAngle);

			if (okId && okAngle) {
				// TODO get correct channel, but there is currently no means to get the channel (same issue with Mean process)
				std::tuple<Seismic3DDataset*, int, float> tuple = std::tuple<Seismic3DDataset*, int, float>(m_allDatasets[id], 0, angle);
				anisotropyDatasetsAndAngles.append(tuple);
			}
		}
		m_data->setDatasetsAndAngles(anisotropyDatasetsAndAngles);
	}

	if (m_constrainData!=nullptr && m_horizon!=nullptr) {
		m_data->setDTauReference(m_horizon->getDTauReference());
	}

	// Above code will set Polarity, else mark as true
	if (m_polarity==0) {
		m_polarity = 1;
	}
	m_data->setPolarity(m_polarity>=0);
}

void LayerSpectrumDialog::updateData(bool fromComputeButton) {
	if ((!fromComputeButton && m_computeButtonOnly)) {
		return;
	}
	DEBUG0
	if (m_isComputationValid || m_isComputationRunning) {
		return;
	}
	if ((m_method == 4 || m_method == 5 ) && m_meanDatasets.count()==0) {
		// method 5 also used selected datasets like the one stored in m_meanDatasets
		return;
	}

	//  MZR 18082021
	if(m_data->ProcessDeletion() == true){
		QMessageBox msgBox;
		msgBox.setText(tr("Layer was deleted !"));
		msgBox.setInformativeText(tr("Do you want to continue ?"));
		msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
		msgBox.setDefaultButton(QMessageBox::Ok);
		int ret = msgBox.exec();
		if(ret == QMessageBox::Ok){
			// Spectrum calculation
			if (m_method == 1) {
				m_datasetS->workingSetManager()->addRGBLayerSlice(m_dataRGB);
			}
			m_data->setProcessDeletion(false);
			m_data->getDatasetS()->workingSetManager()->addLayerSlice(m_data);
		}
		else{
			return;
		}
	}

	m_isComputationRunning = true;
	DEBUG0
	m_horizon->applyDtauToSeeds();
	m_horizon->updateSeedsPositionWithRead(this);
	DEBUG0

	// To redraw horizon before computing
	//TODO m_originViewer->getGLView()->update();
	qApp->processEvents();
	DEBUG0

	tryAppendHistory();
	DEBUG0
	// m_data->computeProcess(this);  // JDTODO
	trt_compute();

	//DEBUG0
	//	if (m_isWindowChanged) {
	//		crunch();
	//	}
	//DEBUG0
	//	m_isComputationValid = true;
	//DEBUG0
	//	tryAppendHistory();
	//DEBUG0
}

void LayerSpectrumDialog::threadableCompute() {

	//QList<QVector<IData*>>  list = m_mapData.values();

	for(IData* pData :m_VectorData){
		m_data =  dynamic_cast<LayerSlice*>(pData);
		setLayerData();
		qDebug() << m_data->name();
		m_data->computeProcess(this);  // JDTODO
	}
	this->set_progressbar_values((double)0, (double)1);
}

void LayerSpectrumDialog::postComputeStep() {
	if (m_isWindowChanged) {
		crunch();
	}
	m_isComputationValid = true;

	m_isComputationRunning = false;
}

void LayerSpectrumDialog::setupHorizonExtension() {

	/*view2d::DatasetSyncViewer2dVisual* rgtVisual = nullptr;
	view2d::DatasetSyncViewer2dVisual* seismicVisual = nullptr;
	std::vector<view2d::SyncViewer2dVisual*> visuals = m_originViewer->getVisuals();
	int i=0;
	while(i<visuals.size()) {
		view2d::DatasetSyncViewer2dVisual* visual = dynamic_cast<view2d::DatasetSyncViewer2dVisual*>(visuals[i]);
		if (visual!=nullptr && visual->getDataset()==m_datasetT) {
			rgtVisual = visual;
		}
		if (visual!=nullptr && visual->getDataset()==m_datasetS) {
			seismicVisual = visual;
		}
		i++;
	}*/
	//if (rgtVisual!=nullptr && seismicVisual!=nullptr) {
	m_horizon = new MultiSeedHorizon("seed_horizon", m_datasetS->workingSetManager(),
			dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS, dynamic_cast<Seismic3DDataset*>(m_datasetT), m_channelT);
	//		m_horizonExtenstion->setDirection(m_originViewer->getDirection());
	//		m_horizonExtenstion->setCurrentSlice(m_originViewer->getSliceToBeAnalyse());
	//		m_horizonExtenstion->setPixelSampleRateX(rgtVisual->getDatasetGroup()->getDBStepX());
	//		if (m_originViewer->getDirection()==view2d::View2dDirection::Z) {
	//			m_horizonExtenstion->setPixelSampleRateY(rgtVisual->getDatasetGroup()->getDBStepY());
	//		} else if (m_originViewer->getDirection()==view2d::View2dDirection::Y) {
	//			m_horizonExtenstion->setPixelSampleRateY(rgtVisual->getDatasetGroup()->getDBStepZ());
	//		} else {
	//			// TODO Direction X should block this
	//		}
	m_horizon->toggleSnap(m_useSnap);
	m_horizon->toggleMedian(m_useMedian);
	m_horizon->setLWXMedianFilter(m_halfLwxMedianFilter*2+1);
	m_horizon->setSeeds(m_seeds);
	m_horizon->setPolarity(m_polarity);
	m_horizon->setSnapWindow(m_snapWindow);

	//m_horizonExtenstion->connectViewer(m_originViewer);
	//		m_horizonExtenstion->setBehaviorMode(view2d::MultiSeedHorizonExtension::FIXED);
	if (m_method==1) {
		m_horizon->setDelta(m_spectrumWindowSize/2);
		m_horizon->setHorizonMode(DELTA_T);
	} else if ( m_method == 0 || m_method == 2 ) {
		m_horizon->setHorizonMode(DEFAULT);
	} else if (m_method == 3) {
		m_horizon->setDeltaTop(0);
		m_horizon->setDeltaBottom(m_tmapExampleSize);
		m_horizon->setHorizonMode(DELTA_T);
	}
	m_horizon->setBehaviorMode(POINTPICKING);
	//m_horizonExtenstion->setCurrentTau(m_tau, m_polarity);
	setPolarity(m_horizon->getPolarity());


	connect(m_horizon, &MultiSeedHorizon::newPointCreated, this, &LayerSpectrumDialog::newPointCreatedFromHorizon);

	connect(m_horizon, &MultiSeedHorizon::pointRemoved, [this](RgtSeed seed, int id) {
		std::vector<RgtSeed>::iterator it = m_seeds.begin();
		while (it!=m_seeds.end() && !((*it)==seed)) {
			it++;
		}
		if (it!=m_seeds.end()) {
			m_seeds.erase(it);
			if (m_seeds.size()>0) {
				setPseudoGeologicalTime(m_horizon->getPseudoTau());
			}
			m_isComputationValid = false;
		}
	});

	connect(m_horizon, &MultiSeedHorizon::pointMoved, [this](RgtSeed oldSeed,
			RgtSeed newSeed, int id) {
		std::vector<RgtSeed>::iterator it = m_seeds.begin();
		while (it!=m_seeds.end() && !((*it)==oldSeed)) {
			it++;
		}
		if (it!=m_seeds.end()) {
			*it = newSeed;
			setPseudoGeologicalTime(m_horizon->getPseudoTau());
			setPolarity(m_horizon->getPolarity());
			m_isComputationValid = false;
		}
	});
	connect(m_horizon, &MultiSeedHorizon::dtauPolygonMainCacheChanged, [this]() {
		m_isComputationValid = false;
	});
	connect(m_horizon, &MultiSeedHorizon::constrainChanged, [this]() {
		if (m_horizon->constrainLayer()==nullptr && m_constrainData!=nullptr) {
			// clear stored data
			QString iso("isochrone"), amp("amplitude");
			std::vector<float> tab;
			tab.resize(m_constrainData->width()*m_constrainData->depth(), -9999.0);
			m_constrainData->writeProperty(tab.data(), iso);
			m_constrainData->writeProperty(tab.data(), amp);
			m_isDataInterpolated = false;
			m_data->setConstrainLayer(nullptr, iso);
			m_data->setDTauReference(0);

			if (m_propagator!=nullptr) {
				m_propagator->clearTabs();
				m_propagatorUndoCache.clear();
				m_propagatorNextId = 1;
				m_propagatorNewSeedsIdLimit.clear();
			}
		}
	});
	m_datasetS->workingSetManager()->addMultiSeedHorizon(m_horizon);

	// Check data on tree
	GeotimeGraphicsView* geotimeView = m_originViewer;
	if (geotimeView!=nullptr) {
		//			QInnerViewTreeWidgetItem* rootItem = geotimeView->getItemFromView(m_originViewer);

		for (AbstractInnerView* innerView : geotimeView->getInnerViews()) {
			if (innerView->viewType()!=ViewType::InlineView && innerView->viewType()!=ViewType::XLineView &&
					innerView->viewType()!=ViewType::RandomView) {
				continue;
			}
			QInnerViewTreeWidgetItem* rootItem = geotimeView->getItemFromView(innerView);

			QStack<QTreeWidgetItem*> stack;
			stack.push(rootItem);
			QGraphicsRepTreeWidgetItem* itemData = nullptr;
			while (stack.size()>0 && itemData==nullptr) {
				QTreeWidgetItem* item = stack.pop();

				std::size_t N = item->childCount();
				for (std::size_t index=0; index<N; index++) {
					stack.push(item->child(index));
				}
				QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
				if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
					const MultiSeedHorizon* data = dynamic_cast<const MultiSeedHorizon*>(_item->getRep()->data());
					if (data!=nullptr && data==m_horizon) {
						itemData = _item;
					}
				}
			}
			if (itemData!=nullptr) {
				itemData->setCheckState(0, Qt::Checked);
				//				m_horizonRep = dynamic_cast<MultiSeedSliceRep*>(itemData->getRep());
			}
		}
	}
	//		this->m_originMainViewer->getViewExtensions().addExtension(m_horizonExtenstion);
	//}
}

void LayerSpectrumDialog::newPointCreatedFromHorizon(RgtSeed seed, int id) {
	m_seeds.push_back(seed);
	setPseudoGeologicalTime(m_horizon->getPseudoTau());
	setPolarity(m_horizon->getPolarity());
	if (m_horizon->isMultiSeed()) {
		m_redoButtonList.clear();
	}
	m_isComputationValid = false;
	m_patchSeeds.push_back(seed);
	m_patchSeedId.push_back(m_propagatorNextId);
}

void LayerSpectrumDialog::toggleDebugMode(bool state) {
	m_debugMode = state;
	if (m_spectrumHatPower!=nullptr && m_spectrumHatPowerLabel!=nullptr) {
		if (m_debugMode) {
			m_spectrumHatPower->show();
			m_spectrumHatPowerLabel->show();
			m_useSnapCheckBox->show();
			m_useSnapLabel->show();
			m_useMedianCheckBox->show();
			m_useMedianLabel->show();
			if (m_useMedian) {
				m_halfLwxMedianFilterSpinBox->show();
				m_halfLwxMedianFilterLabel->show();
			}
			m_distancePowerLabel->show();
			m_distancePowerSpinBox->show();
			if(m_useSnap) {
				m_polarityLabel->show();
				m_polarityComboBox->show();
				m_snapWindowLabel->show();
				m_snapWindowSpinBox->show();
			}
			if (m_horizon->isMultiSeed()) {
				m_sizeCorrLabel->show();
				m_sizeCorrSpinBox->show();
				m_seuilCorrLabel->show();
				m_seuilCorrSpinBox->show();
				//m_seedReductionWindowLabel->show();
				//m_seedReductionWindowSpinBox->show();
				m_seedFilterNumberLabel->show();
				m_seedFilterNumberSpinBox->show();
				m_propagationTypeLabel->show();
				m_propagationTypeComboBox->show();
				m_numIterSpinBox->show();
				m_numIterLabel->show();
			}
		} else {
			m_spectrumHatPower->hide();
			m_spectrumHatPowerLabel->hide();
			m_useSnapCheckBox->hide();
			m_useSnapLabel->hide();
			m_useMedianCheckBox->hide();
			m_useMedianLabel->hide();
			m_halfLwxMedianFilterSpinBox->hide();
			m_halfLwxMedianFilterLabel->hide();
			m_distancePowerLabel->hide();
			m_distancePowerSpinBox->hide();
			m_polarityLabel->hide();
			m_polarityComboBox->hide();
			m_snapWindowLabel->hide();
			m_snapWindowSpinBox->hide();
			m_sizeCorrLabel->hide();
			m_sizeCorrSpinBox->hide();
			m_seuilCorrLabel->hide();
			m_seuilCorrSpinBox->hide();
			m_seedFilterNumberLabel->hide();
			m_seedFilterNumberSpinBox->hide();
			m_propagationTypeLabel->hide();
			m_propagationTypeComboBox->hide();
			m_numIterSpinBox->hide();
			m_numIterLabel->hide();
		}
	}
	m_commonForm->setSpacing(4);
}

void LayerSpectrumDialog::fillHistorySpectrum() {
	if (m_historySpectrumList==nullptr) {
		return;
	}
	m_historySpectrumList->clear();

	for (SpectrumValue& val : m_historySpectrum) {
		appendHistorySpectrum(val);
	}
}

void LayerSpectrumDialog::appendHistorySpectrum(void){
	bool found = false;
	long i = 0;
	SpectrumValue val;
	val.seeds = m_seeds;
	val.distancePower = m_distancePower;
	val.polarity = m_polarity>=0;
	val.halfLwx = m_halfLwxMedianFilter;
	val.useSnap = m_useSnap;
	val.useMedian = m_useMedian;
	val.windowSize = m_spectrumWindowSize;
	val.hatPower = m_hatPower;
	val.id = currentId;
	while (i<m_historySpectrum.size() && !found) {
		found = m_historySpectrum[i].seeds==val.seeds && m_historySpectrum[i].windowSize==val.windowSize &&
				m_historySpectrum[i].hatPower==val.hatPower && m_historySpectrum[i].polarity==val.polarity &&
				m_historySpectrum[i].halfLwx==val.halfLwx && m_historySpectrum[i].useSnap==val.useSnap &&
				m_historySpectrum[i].useMedian==val.useMedian && m_historySpectrum[i].distancePower==val.distancePower;
		i++;
	}

	if (!found) {
		m_historySpectrum.push_back(val);
		appendHistorySpectrum(val);
		currentId++;
	}
}

void LayerSpectrumDialog::appendHistorySpectrum(SpectrumValue val) {
	if (m_historySpectrumList==nullptr) {
		return;
	}

	QString name;
	long tau;
	if (val.seeds.size()>0) {
		tau = val.seeds[0].rgtValue;
	} else {
		// TODO
		tau = getMeanTauFromMultiSeedHorizon();
	}
	name = "tau : " + QString::number(tau);
	if ( val.useSnap ) {
		name += ((val.polarity) ? "p": "n");
	}
	if ( val.useMedian ) {
		name += ",half lwx : " + QString::number(val.halfLwx);
	}
	name += ", ws : " + QString::number(val.windowSize);
	if (m_debugMode) {
		name  = name + ", hat power : " + QString::number(val.hatPower);
	}
	QListWidgetItem* item = new QListWidgetItem(name);
	item->setData(Qt::UserRole, static_cast<qlonglong>(val.id));
	m_historySpectrumList->addItem(item);
}

void LayerSpectrumDialog::fillHistoryMorlet() {
	if (m_historyMorletList==nullptr) {
		return;
	}
	m_historyMorletList->clear();

	for (SpectrumValue& val : m_historySpectrum) {
		appendHistorySpectrum(val);
	}
}

void LayerSpectrumDialog::appendHistoryMorlet(void){
	bool found = false;
	long i = 0;

	MorletValue val;
	val.seeds = m_seeds;
	val.distancePower = m_distancePower;
	val.polarity = m_polarity>=0;
	val.halfLwx = m_halfLwxMedianFilter;
	val.useSnap = m_useSnap;
	val.useMedian = m_useMedian;
	val.freqMin = m_freqMin;
	val.freqMax = m_freqMax;
	val.freqStep = m_freqStep;
	val.id = currentId;
	while (i<m_historyMorlet.size() && !found) {
		found = m_historyMorlet[i].seeds==val.seeds && m_historyMorlet[i].freqMin==val.freqMin &&
				m_historyMorlet[i].freqMax==val.freqMax && m_historyMorlet[i].freqStep==val.freqStep &&
				m_historyMorlet[i].polarity==val.polarity && m_historyMorlet[i].halfLwx==val.halfLwx &&
				m_historyMorlet[i].useSnap==val.useSnap && m_historyMorlet[i].useMedian==val.useMedian &&
				m_historyMorlet[i].distancePower==val.distancePower;
		i++;
	}

	if (!found) {
		m_historyMorlet.push_back(val);
		appendHistoryMorlet(val);
		currentId++;
	}
}

void LayerSpectrumDialog::appendHistoryMorlet(MorletValue val) {
	if (m_historyMorletList==nullptr) {
		return;
	}

	QString name;
	long tau;
	if (val.seeds.size()>0) {
		tau = val.seeds[0].rgtValue;
	} else {
		// TODO
		tau = getMeanTauFromMultiSeedHorizon();
	}
	name = "tau : " + QString::number(tau);
	if ( val.useSnap ) {
		name += ((val.polarity) ? "p": "n");
	}
	if ( val.useMedian ) {
		name += ",half lwx : " + QString::number(val.halfLwx);
	}
	name += ", fmin : " + QString::number(val.freqMin) +
			", fmax : " + QString::number(val.freqMax) + ", fstep : " + QString::number(val.freqStep);

	QListWidgetItem* item = new QListWidgetItem(name);
	item->setData(Qt::UserRole, static_cast<qlonglong>(val.id));
	m_historyMorletList->addItem(item);
}

void LayerSpectrumDialog::fillHistoryGcc() {
	if (m_historyGccList==nullptr) {
		return;
	}
	m_historyGccList->clear();

	for (GccValue& val : m_historyGcc) {
		appendHistoryGcc(val);
	}
}

void LayerSpectrumDialog::appendHistoryGcc(void) {
	bool found = false;
	long i = 0;
	GccValue val;
	val.seeds = m_seeds;
	val.distancePower = m_distancePower;
	val.polarity = m_polarity>=0;
	val.halfLwx = m_halfLwxMedianFilter;
	val.useSnap = m_useSnap;
	val.useMedian = m_useMedian;
	val.offset = m_gccOffset;
	val.w = m_w;
	val.shift = m_shift;
	while (i<m_historyGcc.size() && !found) {
		found = m_historyGcc[i].seeds==val.seeds && m_historyGcc[i].offset==val.offset &&
				m_historyGcc[i].w==val.w &&  m_historyGcc[i].shift==val.shift && m_historyGcc[i].polarity==val.polarity &&
				m_historyGcc[i].halfLwx==val.halfLwx && m_historyGcc[i].useSnap==val.useSnap && m_historyGcc[i].useMedian==val.useMedian &&
				m_historyGcc[i].distancePower==val.distancePower;
		i++;
	}

	if (!found) {
		m_historyGcc.push_back(val);
		appendHistoryGcc(val);
		currentId++;
	}
}

void LayerSpectrumDialog::appendHistoryGcc(GccValue val) {
	if (m_historyGccList==nullptr) {
		return;
	}
	QString name;
	long tau;
	if (val.seeds.size()>0) {
		tau = val.seeds[0].rgtValue;
	} else {
		// TODO
		tau = getMeanTauFromMultiSeedHorizon();
	}
	name = "tau : " + QString::number(tau);
	if ( val.useSnap ) {
		name += ((val.polarity) ? "p": "n");
	}
	if ( val.useMedian ) {
		name += ",half lwx : " + QString::number(val.halfLwx);
	}

	name += ", offset : " + QString::number(val.offset) +
			", w : " + QString::number(val.w) + ", shift : " + QString::number(val.shift);

	QListWidgetItem* item = new QListWidgetItem(name);
	item->setData(Qt::UserRole, static_cast<qlonglong>(val.id));
	m_historyGccList->addItem(item);
}

void LayerSpectrumDialog::fillHistoryTmap() {
	if (m_historyTmapList==nullptr) {
		return;
	}
	m_historyTmapList->clear();

	for (TmapValue& val : m_historyTmap) {
		appendHistoryTmap(val);
	}
}

void LayerSpectrumDialog::appendHistoryTmap(void){
	bool found = false;
	long i = 0;
	TmapValue val;
	val.seeds = m_seeds;
	val.distancePower = m_distancePower;
	val.polarity = m_polarity>=0;
	val.halfLwx = m_halfLwxMedianFilter;
	val.useSnap = m_useSnap;
	val.useMedian = m_useMedian;
	val.tmapExampleSize = m_tmapExampleSize;
	val.tmapSize = m_tmapSize;
	val.tmapExampleStep = m_tmapExampleStep;
	while (i<m_historyTmap.size() && !found) {
		found = m_historyTmap[i].seeds==val.seeds && m_historyTmap[i].tmapExampleSize==val.tmapExampleSize && m_historyTmap[i].tmapSize==val.tmapSize &&
				m_historyTmap[i].tmapExampleStep==val.tmapExampleStep && m_historyTmap[i].polarity==val.polarity &&
				m_historyTmap[i].halfLwx==val.halfLwx && m_historyTmap[i].useSnap==val.useSnap && m_historyTmap[i].useMedian==val.useMedian &&
				m_historyTmap[i].distancePower==val.distancePower;
		i++;
	}

	if (!found) {
		m_historyTmap.push_back(val);
		appendHistoryTmap(val);
		currentId++;
	}
}

void LayerSpectrumDialog::appendHistoryTmap(TmapValue val) {
	if (m_historyTmapList==nullptr) {
		return;
	}
	QString name;
	long tau;
	if (val.seeds.size()>0) {
		tau = val.seeds[0].rgtValue;
	} else {
		// TODO
		tau = getMeanTauFromMultiSeedHorizon();
	}
	name = "tau : " + QString::number(tau);

	if ( val.useSnap ) {
		name += ((val.polarity) ? "p": "n");
	}
	if ( val.useMedian ) {
		name += ",half lwx : " + QString::number(val.halfLwx);
	}

	name += ", exSize : " + QString::number(val.tmapExampleSize) +
			", tmapSize : " + QString::number(val.tmapSize) + ", exStep : " + QString::number(val.tmapExampleStep);

	QListWidgetItem* item = new QListWidgetItem(name);
	item->setData(Qt::UserRole, static_cast<qlonglong>(val.id));
	m_historyTmapList->addItem(item);
}

void LayerSpectrumDialog::fillHistoryMean() {
	if (m_historyMeanList==nullptr) {
		return;
	}
	m_historyMeanList->clear();

	for (MeanValue& val : m_historyMean) {
		appendHistoryMean(val);
	}
}

void LayerSpectrumDialog::appendHistoryMean(void){
	bool found = false;
	long i = 0;
	MeanValue val;
	val.seeds = m_seeds;
	val.distancePower = m_distancePower;
	val.polarity = m_polarity>=0;
	val.halfLwx = m_halfLwxMedianFilter;
	val.useSnap = m_useSnap;
	val.useMedian = m_useMedian;
	val.meanWindowSize = m_meanWindowSize;
	while (i<m_historyMean.size() && !found) {
		found = m_historyMean[i].seeds==val.seeds && m_historyMean[i].meanWindowSize==val.meanWindowSize &&
				m_historyMean[i].polarity==val.polarity &&
				m_historyMean[i].halfLwx==val.halfLwx && m_historyMean[i].useSnap==val.useSnap && m_historyMean[i].useMedian==val.useMedian &&
				m_historyMean[i].distancePower==val.distancePower;
		i++;
	}

	if (!found) {
		m_historyMean.push_back(val);
		appendHistoryMean(val);
		currentId++;
	}
}

void LayerSpectrumDialog::appendHistoryMean(MeanValue val) {
	bool found = false;
	long i = 0;
	if (m_historyMeanList==nullptr) {
		return;
	}
	QString name;
	long tau;
	if (val.seeds.size()>0) {
		tau = val.seeds[0].rgtValue;
	} else {
		// TODO
		tau = getMeanTauFromMultiSeedHorizon();
	}
	name = "tau : " + QString::number(tau);

	if ( val.useSnap ) {
		name += ((val.polarity) ? "p": "n");
	}
	if ( val.useMedian ) {
		name += ",half lwx : " + QString::number(val.halfLwx);
	}

	name += ", winSize : " + QString::number(val.meanWindowSize);

	QListWidgetItem* item = new QListWidgetItem(name);
	item->setData(Qt::UserRole, static_cast<qlonglong>(val.id));
	m_historyMeanList->addItem(item);
}

void LayerSpectrumDialog::fillHistoryAnisotropy() {
	if (m_historyAnisotropyList==nullptr) {
		return;
	}
	m_historyAnisotropyList->clear();

	for (AnisotropyValue& val : m_historyAnisotropy) {
		appendHistoryAnisotropy(val);
	}
}

void LayerSpectrumDialog::appendHistoryAnisotropy(void){
	bool found = false;
	long i = 0;
	AnisotropyValue val;
	val.seeds = m_seeds;
	val.distancePower = m_distancePower;
	val.polarity = m_polarity>=0;
	val.halfLwx = m_halfLwxMedianFilter;
	val.useSnap = m_useSnap;
	val.useMedian = m_useMedian;
	while (i<m_historyAnisotropy.size() && !found) {
		found = m_historyAnisotropy[i].seeds==val.seeds && m_historyAnisotropy[i].polarity==val.polarity &&
				m_historyAnisotropy[i].halfLwx==val.halfLwx && m_historyAnisotropy[i].useSnap==val.useSnap &&
				m_historyAnisotropy[i].useMedian==val.useMedian && m_historyAnisotropy[i].distancePower==val.distancePower;
		i++;
	}

	if (!found) {
		m_historyAnisotropy.push_back(val);
		appendHistoryAnisotropy(val);
		currentId++;
	}
}

void LayerSpectrumDialog::appendHistoryAnisotropy(AnisotropyValue val) {
	bool found = false;
	long i = 0;
	if (m_historyAnisotropyList==nullptr) {
		return;
	}
	QString name;
	long tau;
	if (val.seeds.size()>0) {
		tau = val.seeds[0].rgtValue;
	} else {
		// TODO
		tau = getMeanTauFromMultiSeedHorizon();
	}
	name = "tau : " + QString::number(tau);

	if ( val.useSnap ) {
		name += ((val.polarity) ? "p": "n");
	}
	if ( val.useMedian ) {
		name += ",half lwx : " + QString::number(val.halfLwx);
	}

	QListWidgetItem* item = new QListWidgetItem(name);
	item->setData(Qt::UserRole, static_cast<qlonglong>(val.id));
	m_historyAnisotropyList->addItem(item);
}

void LayerSpectrumDialog::tryAppendHistory() {
	switch(m_method){
	case eComputeMethd_Morlet :
		appendHistoryMorlet();
		break;
	case eComputeMethd_Spectrum :
		appendHistorySpectrum();
		break;
	case eComputeMethd_Gcc:
		appendHistoryGcc();
		break;
	case eComputeMethd_TMAP:
		appendHistoryTmap();
		break;
	case eComputeMethd_Mean:
		appendHistoryMean();
		break;
	case eComputeMethd_Anisotropy:
		appendHistoryAnisotropy();
		break;
	default:
		break;
	}
}

void LayerSpectrumDialog::clearHorizon() {
	if (m_horizon!=nullptr) {
		std::vector<RgtSeed> emptyVector;
		m_horizon->setSeeds(emptyVector);
	}
	m_seeds.clear();
	if (m_propagator!=nullptr) {
		m_propagator->clearTabs();
	}
	m_propagatorUndoCache.clear();
	m_propagatorNextId = 1;
	m_propagatorNewSeedsIdLimit.clear();

	if (m_constrainData!=nullptr) {
		if (m_horizon!=nullptr) {
			m_horizon->setConstrainLayer(nullptr);
			m_horizon->setDTauReference(0);
		}
		// clear stored data
		QString iso("isochrone"), amp("amplitude");
		std::vector<float> tab;
		tab.resize(m_constrainData->width()*m_constrainData->depth(), -9999.0);
		m_constrainData->writeProperty(tab.data(), iso);
		m_constrainData->writeProperty(tab.data(), amp);
		m_isDataInterpolated = false;
		m_data->setConstrainLayer(nullptr, iso);
		m_data->setDTauReference(0);
	}
	m_redoButtonList.clear();
	m_patchSeeds.clear();
	m_patchSeedId.clear();
	m_patchTabIso.clear();
	patchConstrain();
}

void LayerSpectrumDialog::fullClearHorizon() {
	clearHorizon();

	if (m_horizon!=nullptr) {
		m_horizon->setConstrainLayer(nullptr);

		WorkingSetManager* manager = m_horizon->workingSetManager();
		manager->removeMultiSeedHorizon(m_horizon);
		m_horizon->deleteLater();
		m_horizon = nullptr;
	}

	if (m_constrainData!=nullptr) {
		FixedLayerFromDataset* layer = m_constrainData.release();
		WorkingSetManager* manager = layer->workingSetManager();
		manager->removeFixedLayerFromDataset(layer); // remove delete IData
	}
	m_propagator.reset(nullptr);
}

// clear horizon and add a single point pt
void LayerSpectrumDialog::setPoint(Abstract2DInnerView* inner2DView, QPointF pt) {
	if (inner2DView->viewType()==ViewType::InlineView || inner2DView->viewType()==ViewType::XLineView) {
		clearHorizon();
		double imageX, imageY, worldX, worldY;
		worldX = pt.x();
		worldY = pt.y();
		if (inner2DView->viewType()==ViewType::InlineView) {
			dynamic_cast<Seismic3DDataset*>(m_datasetS)->ijToInlineXlineTransfoForInline()->worldToImage(worldX, worldY, imageX, imageY);
		} else {
			dynamic_cast<Seismic3DDataset*>(m_datasetS)->ijToInlineXlineTransfoForXline()->worldToImage(worldX, worldY, imageX, imageY);
		}

		MultiSeedSliceRep* horizonRep = nullptr;
		std::size_t index=0;

		const QList<AbstractGraphicRep*> visibleReps = inner2DView->getVisibleReps();
		while(index<visibleReps.size() && horizonRep==nullptr) {
			horizonRep = dynamic_cast<MultiSeedSliceRep*>(visibleReps[index]);
			index++;
		}
		if (horizonRep!=nullptr) {
			QPoint ptImage(imageX, imageY);
			horizonRep->addPointAndSelect(ptImage);
		}
		DEBUG0
	} else {
		clearHorizon();
		double imageX, imageY;
		//		worldX = pt.x();
		//		worldY = pt.y();
		imageX = dynamic_cast<RandomLineView*>(inner2DView)->getDiscreatePolyLineIndexFromScenePos(pt);
		m_datasetS->sampleTransformation()->indirect(pt.y(), imageY);

		MultiSeedRandomRep* horizonRep = nullptr;
		std::size_t index=0;

		const QList<AbstractGraphicRep*> visibleReps = inner2DView->getVisibleReps();
		while(index<visibleReps.size() && horizonRep==nullptr) {
			horizonRep = dynamic_cast<MultiSeedRandomRep*>(visibleReps[index]);
			index++;
		}
		if (horizonRep!=nullptr) {
			QPoint ptImage(imageX, imageY);
			horizonRep->addPointAndSelect(ptImage);
		}
	}
	//TODO BIDOUILLE
	/*RgtSeed seed;
	seed.x = 100;
	seed.y = 100;
	seed.rgtValue = 10000;
	seed.seismicValue = 100;
	seed.z = 200;
	m_seeds.push_back(seed);*/
}

void LayerSpectrumDialog::undoHorizonModification() {
	if (!m_horizon->isMultiSeed()) {
		return;
	}
	if (m_horizon!=nullptr) {
		std::vector<std::size_t> ids = m_horizon->getAllIds();
		if (ids.size()>0) {
			std::size_t indexNewSelect = 0;
			bool indexNewSelectSet = false;
			std::size_t id = ids[0];
			for (long index=1; index<ids.size(); index++) {
				if (id<ids[index]) {
					indexNewSelect = id;
					id = ids[index];
					indexNewSelectSet = true;
				} else if (indexNewSelect<ids[index]) {
					indexNewSelect = ids[index];
					indexNewSelectSet = true;
				}
			}
			if (id==m_horizon->getSelectedId() && indexNewSelectSet) {
				m_horizon->selectPoint(indexNewSelect);
			}
			RgtSeed seed = (*m_horizon)[id];
			m_horizon->removePoint(id);
			m_redoButtonList.push_back(seed);
		}
	}
	if ( m_patchSeeds.size() == 0 ) return;
	m_patchSeeds.pop_back();
	m_patchSeedId.pop_back();
	m_patchTabIso.clear();
}

void LayerSpectrumDialog::redoHorizonModification() {
	if (m_redoButtonList.size()==0 || !m_horizon->isMultiSeed()) {
		return;
	}

	RgtSeed seed = m_redoButtonList.back();
	m_redoButtonList.pop_back();
	if (m_horizon!=nullptr) {
		// to avoid a clear of redoButtonList
		disconnect(m_horizon, &MultiSeedHorizon::newPointCreated, this, &LayerSpectrumDialog::newPointCreatedFromHorizon);
		m_horizon->addPointAndSelect(seed);
		connect(m_horizon, &MultiSeedHorizon::newPointCreated, this, &LayerSpectrumDialog::newPointCreatedFromHorizon);
	} else {
		m_seeds.push_back(seed);
	}
}

template<typename InputType>
struct UpdateSeedsRgtValueKernel {
	static void run(Seismic3DDataset* rgt, int channelT, std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers,
			QString isoName, QString rgtName, std::vector<RgtSeed>& seeds, float tdeb, float pasech) {
		if (seeds.size()==0) {
			return;
		}

		long dimx = rgt->height();
		long dimy = rgt->width();
		long dimz = rgt->depth();

		bool isReferenceValid = referenceLayers.size()!=0;
		std::vector<std::pair<std::vector<float>, std::vector<float>>> referenceVec;
		//std::vector<float> referenceTauVec, referenceIsoVec;
		if (isReferenceValid) {
			referenceVec.resize(referenceLayers.size(), std::pair<std::vector<float>, std::vector<float>>());
			for (std::size_t index=0; index<referenceVec.size(); index++) {
				std::pair<std::vector<float>, std::vector<float>>& pair = referenceVec[index];
				std::get<0>(pair).resize(dimy*dimz);
				std::get<1>(pair).resize(dimy*dimz);
				isReferenceValid = referenceLayers[index]->readProperty(std::get<0>(pair).data(), isoName);
				isReferenceValid = isReferenceValid && referenceLayers[index]->readProperty(std::get<1>(pair).data(), rgtName);
			}
		}

		//		if (dynamic_cast<io::CubeCwt<InputType>*>(cubeRgt)!=nullptr) {
		//			std::vector<process::RgtSeed*> seedsCopy;
		//			seedsCopy.resize(seeds.size());
		//			for (std::size_t index=0; index<seedsCopy.size(); index++) {
		//				seedsCopy[index] = &(seeds[index]);
		//			}
		//
		//			std::sort(seedsCopy.begin(), seedsCopy.end(), [](process::RgtSeed* a, process::RgtSeed* b){
		//				return a->z<b->z;
		//			});
		//			std::vector<InputType> buffer;
		//			buffer.resize(dimx*dimy);
		//			long currentZ = -1;
		//
		//			for (process::RgtSeed* seed : seedsCopy) {
		//				if (currentZ!=seed->z) {
		//					currentZ = seed->z;
		//					cubeRgt->readSubVolume(0, 0, seed->z, dimx, dimy, 1, buffer.data());
		//				}
		//				seed->rgtValue = buffer[dimx*seed->y+seed->x];
		//				if (isReferenceValid) {
		//					for (std::size_t index=0; index<referenceVec.size(); index++) {
		//						std::pair<std::vector<float>, std::vector<float>>& pair = referenceVec[index];
		//						long ix = (std::get<0>(pair)[seed->z*dimy+seed->y] - tdeb) / pasech;
		//						std::get<1>(pair)[seed->z*dimy+seed->y] = buffer[ix+dimx*seed->y];
		//					}
		//				}
		//			}
		//		} else {
		std::vector<InputType> valTab;
		valTab.resize(rgt->dimV());
		for (RgtSeed& seed : seeds) {
			rgt->readSubTraceAndSwap(valTab.data(), seed.x, seed.x+1, seed.y, seed.z);
			seed.rgtValue = valTab[channelT];
			if (isReferenceValid) {
				for (std::size_t index=0; index<referenceVec.size(); index++) {
					std::pair<std::vector<float>, std::vector<float>>& pair = referenceVec[index];
					long ix = (std::get<0>(pair)[seed.z*dimy+seed.y] - tdeb) / pasech;
					rgt->readSubTraceAndSwap(valTab.data(), ix, ix+1, seed.y, seed.z);
					std::get<1>(pair)[seed.z*dimy+seed.y] = valTab[channelT];
				}
			}
		}
		//		}
		if (isReferenceValid) {
			for (std::size_t index=0; index<referenceVec.size(); index++) {
				std::pair<std::vector<float>, std::vector<float>>& pair = referenceVec[index];
				referenceLayers[index]->writeProperty(std::get<1>(pair).data(), rgtName);
			}
		}
	}
};

template<typename InputType>
struct CreatePropagatorObjectKernel {
	static SeedsGenericPropagator* run(Seismic3DDataset* dataset, int channel) {
		return new SeedsPropagator<InputType>(dataset, channel);
	}
};

void LayerSpectrumDialog::loadHorizon() {
	//	QFileInfo dataFileInfo(m_datasetS->getDataPath());
	//	QDir surveyDir = dataFileInfo.absoluteDir();
	//	QDir horizonDir;
	QString saveName;
	bool isValid = true;
	//	bool isValid = surveyDir.cdUp() && surveyDir.cdUp();
	//
	//
	//	isValid = isValid && surveyDir.cd("ImportExport") && surveyDir.cd("IJK");
	//	if (isValid) {
	//		// TODO redo when specification will be more precices
	//		QStringList list = surveyDir.entryList(QDir::NoDotAndDotDot | QDir::Dirs | QDir::Executable | QDir::Readable);
	//		isValid = list.size()>0;
	//		if (isValid) {
	//			horizonDir = QDir(surveyDir.absoluteFilePath(list[0]));
	//			isValid = horizonDir.cd("HORIZON_GRIDS");
	//		}
	//	}
	//	if (isValid) {
	//		QFileInfoList list = horizonDir.entryInfoList(QStringList() << "*.raw", QDir::Files | QDir::Readable);
	QStringList nameList;
	std::vector<long> idVector;
	idVector.reserve(m_horizonNames.size());
	for (const std::pair<long, QString>& e : m_horizonNames) {
		nameList << e.second; //e.baseName();
		idVector.push_back(e.first);
	}
	if (m_horizonNames.size()!=m_horizonDatas.size()) {
		qDebug() << "LayerSpectrumDialog::loadHorizon mismatch sizes";
	}
	QString title("Horizon selection");
	StringSelectorDialog dialog(&nameList, title);
	isValid = dialog.exec() == QDialog::Accepted;

	long id;
	if (isValid) {
		int index = dialog.getSelectedIndex();
		isValid = index>=0;
		if (isValid) {
			id = idVector[index];
			saveName = m_horizonPaths[id]; //.baseName();
		}
	}
	//	}
	if (isValid) {
		//		saveName = horizonDir.absoluteFilePath(saveName+".raw");

		//		view2d::DatasetSyncViewer2dVisual* rgtVisual = nullptr;
		//		view2d::DatasetSyncViewer2dVisual* seismicVisual = nullptr;
		//		std::vector<view2d::SyncViewer2dVisual*> visuals = m_originViewer->getVisuals();
		//		int i=0;
		//		while(i<visuals.size()) {
		//			view2d::DatasetSyncViewer2dVisual* visual = dynamic_cast<view2d::DatasetSyncViewer2dVisual*>(visuals[i]);
		//			if (visual!=nullptr && visual->getDataset()==m_datasetT) {
		//				rgtVisual = visual;
		//			}
		//			if (visual!=nullptr && visual->getDataset()==m_datasetS) {
		//				seismicVisual = visual;
		//			}
		//			i++;
		//		}
		//		if ( rgtVisual==nullptr || seismicVisual==nullptr) {
		//			return;
		//		}
		//		DimensionHolder dims = rgtVisual->getDimensions();
		Seismic3DDataset* rgtDataset = dynamic_cast<Seismic3DDataset*>(m_datasetT);
		float tdeb = rgtDataset->sampleTransformation()->b();
		float pasech = rgtDataset->sampleTransformation()->a();


		if (m_constrainData==nullptr) {
			createConstrain();
		}

		//		std::string filenameStd = saveName.toStdString();
		qDebug() << "save name " << saveName;
		QString isoName("isochrone");
		if (m_horizonDatas[id]==nullptr) {
			isValid = m_constrainData->loadProperty(saveName, isoName);
		} else {
			std::string isoFilePath = m_horizonDatas[id]->path().toStdString() + "/" + FreeHorizonManager::isoDataName;
			FreeHorizonManager::PARAM params = FreeHorizonManager::dataSetGetParam(isoFilePath);
			std::vector<float> buffer;
			buffer.resize(params.dimx*params.dimy);
			std::string errorStr = FreeHorizonManager::read(isoFilePath, buffer.data());
			if (errorStr!="ok") {
				return;
			}
			m_constrainData->writeProperty(buffer.data(), isoName);
		}
		if (isValid) {
			std::vector<float> inputLayer;

			if (m_propagator==nullptr) {
				//				std::unique_ptr<io::Cube> seismic = process::openAsGenericCube(m_datasetS);

				SampleTypeBinder binder(m_datasetS->sampleType());
				m_propagator.reset(binder.bind<CreatePropagatorObjectKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS));
			}
			//			std::unique_ptr<io::Cube> seismic = process::openAsGenericCube(m_datasetS);
			//			std::unique_ptr<io::Cube> rgt = process::openAsGenericCube(m_datasetT);
			std::vector<int>& tabSeed = m_propagator->getTabSeedType();
			std::vector<float>& tabIso = m_propagator->getIsochroneTab();
			std::vector<float>& tabAmp = m_propagator->getAmplitudeTab();
			tabSeed.clear();
			tabSeed.resize(rgtDataset->width()*rgtDataset->depth(), 0);
			tabIso.clear();
			tabIso.resize(rgtDataset->width()*rgtDataset->depth(), -9999);
			tabAmp.clear();
			tabAmp.resize(rgtDataset->width()*rgtDataset->depth(), -9999);

			m_constrainData->readProperty(tabIso.data(), isoName);

			QString ampName("amplitude"), ampFilename(saveName+"_amp.pickingType");
			//			FixedLayerFromDataset ampLayer(ampName, m_datasetS->workingSetManager(),
			//							dynamic_cast<Seismic3DDataset*>(m_datasetS), this);
			bool ampResult = false;
			if (m_horizonDatas[id]==nullptr) {
				ampResult = m_constrainData->loadProperty(ampFilename, ampName);
				ampResult = ampResult && m_constrainData->readProperty(tabAmp.data(), ampName);
			}
			/*if (!ampResult) {
				io::SampleTypeBinder binder(seismic->getNativeType());
				binder.bind<ExtractPropertyFromIsochroneKernel>(tabIso, seismic.get(), tabAmp, tdeb, pasech);
			}*/
			QString historyName("history"), historyFilename(saveName+"_history.pickingType");
			FixedLayerFromDataset historyLayer(historyName, m_datasetS->workingSetManager(),
					dynamic_cast<Seismic3DDataset*>(m_datasetS), this);
			bool historyResult = false;
			if (m_horizonDatas[id]==nullptr) {
				historyResult = historyLayer.loadProperty(historyFilename, historyName);
			}

			m_propagatorUndoCache.resize(rgtDataset->width()*rgtDataset->depth());
			if (historyResult) {
				std::vector<float> tabHistory;
				tabHistory.resize(rgtDataset->width()*rgtDataset->depth());
				historyResult =  historyLayer.readProperty(tabHistory.data(), historyName);
				for (std::size_t index=0; index<tabSeed.size(); index++) {
					m_propagatorUndoCache[index] = tabHistory[index];
					if (tabHistory[index]!=0) {
						tabSeed[index] = 1;
					}
				}
			}
			bool isInterpolated = false;
			if (!historyResult) {
				bool idUsed = false;
				for (long z=0; z<rgtDataset->depth(); z++) {
					for (long y=0; y<rgtDataset->width(); y++) {
						if (tabIso[z*rgtDataset->width()+y]!=-9999) {
							tabSeed[z*rgtDataset->width()+y] = 1;
							m_propagatorUndoCache[z*rgtDataset->width()+y] = 1;
							idUsed = true;
						}
					}
				}
				if (idUsed) {
					m_propagatorNextId = 2;
				} else {
					m_propagatorNextId = 1;
				}
			} else {
				m_propagatorNextId = 0;
				for (std::size_t index=0; index<tabSeed.size(); index++) {
					unsigned int& val = m_propagatorUndoCache[index];
					if (val>m_propagatorNextId) {
						m_propagatorNextId = val;
					}
					isInterpolated = isInterpolated && (val==0 && tabIso[index]!=-9999.0);
				}
				m_propagatorNextId += 1;
			}


			// task already done above
			//			if(m_dataAmp!=nullptr && m_dataIso!=nullptr) {
			//				io::ArrayCube<float>* cubeAmp = dynamic_cast<io::ArrayCube<float>*>(m_dataAmp->getCubeForEdition());
			//				cubeAmp->writeSubVolume(0, 0, 0, tabAmp.data(), 1, cubeAmp->getDim().getJ(), cubeAmp->getDim().getK());
			//				m_dataAmp->releaseEdition();

			//				io::ArrayCube<float>* cubeIso = dynamic_cast<io::ArrayCube<float>*>(m_dataIso->getCubeForEdition());
			//				cubeIso->writeSubVolume(0, 0, 0, tabIso.data(), 1, cubeIso->getDim().getJ(), cubeIso->getDim().getK());
			//				m_dataIso->releaseEdition();
			m_isDataInterpolated = false;
			//			}

			updatePropagationExtension();


			std::vector<RgtSeed> newSeeds = m_propagator->extractSeedsFromTabs(m_seedFilterNumber);
			QString rgtName("rgt");
			SampleTypeBinder binder(rgtDataset->sampleType());
			//			binder.bind<UpdateSeedsRgtValueKernel>(rgt.get(), m_referenceLayers, isoName, rgtName, newSeeds, tdeb, pasech);
			binder.bind<UpdateSeedsRgtValueKernel>(rgtDataset, m_channelT, m_referenceLayers, isoName, rgtName, newSeeds, tdeb, pasech);
			setSeeds(newSeeds);
			/*setPolarity(correctPolarity); // no polarity after loading
			if (m_horizonExtenstion!=nullptr) {
				m_horizonExtenstion->setPolarity(m_polarity);
			}*/
			//propagationCacheSetup();
			updatePropagationView();
			if (m_horizon!=nullptr) {
				m_horizon->applyDtauToSeeds();
			}

			bool wasBlocked = m_toggleInterpolation->blockSignals(true);
			m_toggleInterpolation->setChecked(isInterpolated);
			m_toggleInterpolation->blockSignals(wasBlocked);

			m_toggleInterpolation->show();
			m_undoPropagationButton->show();
			m_isComputationValid = false;
		}
	}
}

template<typename RgtCubeType>
struct LayerRGTInterpolatorKernel {
	template<typename SeismicCubeType>
	struct LayerRGTInterpolatorKernelLevel2 {
		static void run(const std::vector<float>& inputLayer, long dtauReference, std::vector<float>& outputLayerIso,
				std::vector<float>& outputLayerSeismic, std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers,
				std::vector<RgtSeed> seeds, const Seismic3DDataset* rgt, int channelT, const Seismic3DDataset* seismic, int channelS, bool useSnap,
				bool useMedian, int lwx, int distancePower, int snapWindow, int polarity, float tdeb, float pasech) {
			layerRGTInterpolatorMultiSeed<RgtCubeType, SeismicCubeType>(inputLayer, dtauReference, outputLayerIso,
					outputLayerSeismic, referenceLayers,
					seeds, rgt, channelT, seismic, channelS, useSnap,
					useMedian, lwx, distancePower, snapWindow, polarity, tdeb, pasech);
		}
	};

	static void run(ImageFormats::QSampleType seismicSampleType, const std::vector<float>& inputLayer, long dtauReference, std::vector<float>& outputLayerIso,
			std::vector<float>& outputLayerSeismic, std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers,
			std::vector<RgtSeed> seeds, const Seismic3DDataset* rgt, int channelT, const Seismic3DDataset* seismic, int channelS, bool useSnap,
			bool useMedian, int lwx, int distancePower, int snapWindow, int polarity, float tdeb, float pasech) {
		SampleTypeBinder binder(seismicSampleType);
		binder.bind<LayerRGTInterpolatorKernelLevel2>(inputLayer, dtauReference, outputLayerIso, outputLayerSeismic, referenceLayers,
				seeds, rgt, channelT, seismic, channelS, useSnap, useMedian, lwx, distancePower, snapWindow, polarity, tdeb, pasech);
	}
};

void LayerSpectrumDialog::saveHorizon() {
	if (m_constrainData ==nullptr && m_seeds.size()==0) {
		return;
	}
	QFileInfo dataFileInfo(QString::fromStdString(dynamic_cast<Seismic3DDataset*>(
			m_datasetS)->path()));
	QDir surveyDir = dataFileInfo.absoluteDir();
	QDir horizonDir;
	QString saveName, saveRgbName;
	bool isValid = surveyDir.cdUp() && surveyDir.cdUp();
	bool saveInterpolate = false;
	bool saveRgb = false;
	bool isNameNew = false;

	if (!surveyDir.exists("ImportExport")) {
		isValid = surveyDir.mkdir("ImportExport");
	}
	isValid = isValid && surveyDir.cd("ImportExport");
	if (isValid) {
		if (!surveyDir.exists("IJK")) {
			isValid = surveyDir.mkdir("IJK");
		}
		isValid = isValid && surveyDir.cd("IJK");
	}
	if (isValid) {
		// TODO redo when specification will be more precices
		//QStringList list = surveyDir.entryList(QDir::NoDotAndDotDot | QDir::Dirs | QDir::Writable | QDir::Executable | QDir::Readable);

		QString seismicSismageName = DataSelectorDialog::getSismageNameFromSeismicFile(
				QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetS)->path()));
		QString rgtSismageName = DataSelectorDialog::getSismageNameFromSeismicFile(
				QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetT)->path()));
		horizonDir = surveyDir;
		if (!horizonDir.cd(seismicSismageName) && !horizonDir.cd(rgtSismageName))  {
			isValid = horizonDir.mkdir(seismicSismageName) && horizonDir.cd(seismicSismageName);
		}

		if (isValid && !horizonDir.exists("HORIZON_GRIDS")) {
			isValid = horizonDir.mkdir("HORIZON_GRIDS");
		}
		isValid = horizonDir.cd("HORIZON_GRIDS");
	}
	//	if (isValid) {
	//		QStringList nameList;
	//		nameList << "Only save propagation" << "Save propagation and interpolation";
	//		QString title("Horizon object selection");
	//		StringSelectorDialog dialog(&nameList, title);
	//		isValid = dialog.exec() == QDialog::Accepted;
	//		saveInterpolate = dialog.getSelectedIndex()==1;
	//	}
	if (isValid) {
		QFileInfoList list = horizonDir.entryInfoList(QStringList() << "*.raw", QDir::Files | QDir::Writable | QDir::Readable);
		QStringList nameList;
		//		nameList << "Create New Horizon";
		for (QFileInfo e : list) {
			nameList << e.baseName();
		}
		// NV-HORIZONS
		surveyDir = dataFileInfo.absoluteDir();
		isValid = surveyDir.cdUp() && surveyDir.cdUp();
		QString nvHorizonpath = surveyDir.absolutePath() + "/" + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_PATH) + "/";
		QDir nvHorizonDir(nvHorizonpath);
		list = nvHorizonDir.entryInfoList(QStringList() << "*", QDir::Dirs | QDir::NoDot | QDir::NoDotDot);
		for (QFileInfo e : list) {
			nameList << e.baseName();
		}
		nameList.removeDuplicates();

		QString title("Horizon selection");
		//		StringSelectorDialog dialog(&nameList, title);
		SaveHorizonDialog dialog(nameList, title);
		isValid = dialog.exec() == QDialog::Accepted;

		if (isValid) {
			saveInterpolate = dialog.doInterpolation();
			saveRgb = dialog.doRgb();
			isNameNew = dialog.isNameNew();
			saveName = dialog.getSaveName();
			isValid = !saveName.isEmpty() && !saveName.isNull();
		}

		//		if (isValid) {
		//			int index = dialog.getSelectedIndex();
		//			if (index==0) {
		//				saveName = QInputDialog::getText(nullptr, "New horizon name", "Define new horizon name");
		//				isValid = !saveName.isNull() && !saveName.isEmpty();
		//				isNameNew = true;
		//			} else {
		//				saveName = list[index-1].baseName();
		//				isNameNew = false;
		//			}
		//		}
	}
	if (isValid) {
		QString horizonBaseName = saveName;
		saveRgbName = horizonDir.absoluteFilePath(saveName+".png");
		saveName = horizonDir.absoluteFilePath(saveName+".raw");

		m_horizon->applyDtauToSeeds();

		//		view2d::DatasetSyncViewer2dVisual* rgtVisual = nullptr;
		//		view2d::DatasetSyncViewer2dVisual* seismicVisual = nullptr;
		//		std::vector<view2d::SyncViewer2dVisual*> visuals = m_originViewer->getVisuals();
		//		int i=0;
		//		while(i<visuals.size()) {
		//			view2d::DatasetSyncViewer2dVisual* visual = dynamic_cast<view2d::DatasetSyncViewer2dVisual*>(visuals[i]);
		//			if (visual!=nullptr && visual->getDataset()==m_datasetT) {
		//				rgtVisual = visual;
		//			}
		//			if (visual!=nullptr && visual->getDataset()==m_datasetS) {
		//				seismicVisual = visual;
		//			}
		//			i++;
		//		}
		//		if ( rgtVisual==nullptr || seismicVisual==nullptr) {
		//			return;
		//		}
		//		DimensionHolder dims = rgtVisual->getDimensions();

		QString isoName("isochrone"), ampName("amplitude"), historyName("history");

		FixedLayerFromDataset saveProperties("", m_datasetS->workingSetManager(),
				dynamic_cast<Seismic3DAbstractDataset*>(m_datasetS));
		//		data::StorelessLayerProperty isoProperty(dims.getDimY(), dims.getDimZ(), isoName);
		//		data::StorelessLayerProperty ampProperty(dims.getDimY(), dims.getDimZ(), ampName);
		//		data::StorelessLayerProperty historyProperty(dims.getDimY(), dims.getDimZ(), historyName);

		if (saveInterpolate && m_isDataInterpolated && m_constrainData!=nullptr) {
			//			const io::ArrayCube<float>* cubeIso = dynamic_cast<const io::ArrayCube<float>*>(m_dataIso->getCube());
			//			const io::ArrayCube<float>* cubeAmp = dynamic_cast<const io::ArrayCube<float>*>(m_dataAmp->getCube());
			std::vector<float> buffer;
			buffer.resize(saveProperties.width()*saveProperties.depth());
			//			cubeIso->readSubVolume(0, 0, 0, 1, dims.getDimY(), dims.getDimZ(), buffer.data());
			m_constrainData->readProperty(buffer.data(), isoName);
			saveProperties.writeProperty(buffer.data(), isoName);
			//			cubeAmp->readSubVolume(0, 0, 0, 1, dims.getDimY(), dims.getDimZ(), buffer.data());
			m_constrainData->readProperty(buffer.data(), ampName);
			saveProperties.writeProperty(buffer.data(), ampName);
		} else {
			std::vector<float> inputLayer;
			inputLayer.resize(saveProperties.width()*saveProperties.depth(), -9999);
			if (m_constrainData!=nullptr) {
				m_constrainData->readProperty(inputLayer.data(), isoName);
			}

			if (saveInterpolate) {
				std::vector<float> outputLayerIso, outputLayerSeismic;
				outputLayerIso.resize(saveProperties.width()*saveProperties.depth());
				outputLayerSeismic.resize(saveProperties.width()*saveProperties.depth());
				if (m_isComputationValid && m_data->isModuleComputed()) {
					const float* rgtBuf = m_data->getModuleData(0);
					const float* seismicBuf = m_data->getModuleData(1);

					float tdeb = saveProperties.getOriginSample();
					float pasech = saveProperties.getStepSample();

					for (std::size_t index=0; index<outputLayerIso.size(); index++) {
						outputLayerIso[index] = rgtBuf[index] * pasech + tdeb;
						outputLayerSeismic[index] = seismicBuf[index];
					}

				} else {
					float tdeb = saveProperties.getOriginSample();
					float pasech = saveProperties.getStepSample();
					//					io::Cube* rgt = dynamic_cast<io::Cube*>(io::openCube<float>(rgtVisual->getDataset()->getDataPath().toStdString()));
					//					io::Cube* seismic = dynamic_cast<io::Cube*>(io::openCube<float>(seismicVisual->getDataset()->getDataPath().toStdString()));

					// interpolate with rgt
					SampleTypeBinder binder(m_datasetT->sampleType());
					binder.bind<LayerRGTInterpolatorKernel>(m_datasetS->sampleType(), inputLayer, m_horizon->getDTauReference(), outputLayerIso, outputLayerSeismic, m_referenceLayers, m_seeds,
							dynamic_cast<Seismic3DDataset*>(m_datasetT), m_channelT, dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS,
							m_useSnap, m_useMedian, m_halfLwxMedianFilter*2+1, m_distancePower,  m_snapWindow,  m_polarity, tdeb, pasech);
				}

				float *tmp = outputLayerIso.data();


				saveProperties.writeProperty(outputLayerIso.data(), isoName);
				saveProperties.writeProperty(outputLayerSeismic.data(), ampName);
			} else {
				bool isPropagatorValid = m_propagator!=nullptr;
				if (isPropagatorValid) {
					std::vector<float>& amplitudeTab = m_propagator->getAmplitudeTab();
					isPropagatorValid = amplitudeTab.size()==saveProperties.width()*saveProperties.depth();
				}
				if (isPropagatorValid) {
					std::vector<float>& amplitudeTab = m_propagator->getAmplitudeTab();
					saveProperties.writeProperty(amplitudeTab.data(), ampName);
				} else {
					std::vector<float> amplitudeTab;
					amplitudeTab.resize(saveProperties.width()*saveProperties.depth(), 0);
					saveProperties.writeProperty(amplitudeTab.data(), ampName);
				}

				saveProperties.writeProperty(inputLayer.data(), isoName);
			}
		}

		bool isPropagatorValid = m_propagatorUndoCache.size()==saveProperties.width()*saveProperties.depth();
		if (isPropagatorValid) {
			std::vector<unsigned int>& historyLayer = m_propagatorUndoCache;
			std::vector<float> buffer;
			buffer.resize(historyLayer.size());
			for (std::size_t index=0; index<buffer.size(); index++) {
				buffer[index] = historyLayer[index];
			}
			saveProperties.writeProperty(buffer.data(), historyName);
		} else {
			std::vector<float> historyLayer;
			historyLayer.resize(saveProperties.width()*saveProperties.depth(), 0);
			saveProperties.writeProperty(historyLayer.data(), historyName);
		}

		QString filenameStd = saveName;
		QString filenameAmp(filenameStd+"_amp.pickingType");
		QString filenameHistory(filenameStd+"_history.pickingType");
		saveProperties.saveProperty(filenameStd, isoName);
		saveProperties.saveProperty(filenameAmp, ampName);
		saveProperties.saveProperty(filenameHistory, historyName);
		if (saveRgb) {
			saveRGB(saveRgbName);
		}
		// save horizon in IJK FREE HORIZON
		int dimy = saveProperties.width();
		int dimz = saveProperties.depth();
		float tdeb = saveProperties.getOriginSample();
		float pasech = saveProperties.getStepSample();
		std::vector<float> isoData;
		isoData.resize(dimy*dimz);
		float *pIsodata = isoData.data();
		saveProperties.readProperty(pIsodata, isoName);
		QString dataSetPath = QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetT)->path());
		QString horizonFullBaseName = horizonBaseName;// + "_(" + SeismicManager::seismicFullFilenameToTinyName(dataSetPath) + ")";
		saveFreeHorizon(dataSetPath, horizonFullBaseName, pIsodata, saveProperties.width(), dimz, tdeb, pasech);

		if (isNameNew) {
			IJKHorizon* horizon = new IJKHorizon(horizonBaseName, filenameStd, QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetS)->path()),
					m_datasetS->workingSetManager());

			m_datasetS->workingSetManager()->addIJKHorizon(horizon);

			//			// add to internal list
			//			m_horizonNames.push_back(horizon->name());//(*horizonNames)[i]);
			//			m_horizonPaths.push_back(horizon->path());//(*horizonPaths)[i]);
			//
			//			// apply to m_referenceList
			//			QListWidgetItem* item = new QListWidgetItem(horizon->name());
			//			item->setData(Qt::UserRole, horizon->path());
			//			m_referenceList->addItem(item);
		}
	}
}

void LayerSpectrumDialog::saveFreeHorizon(QString rgtPath, QString filename, float *data, int dimy, int dimz, float tdeb, float pasech)
{
	// QString path0 = QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetS)->path());
	// qDebug() << path0;
	QFileInfo dataFileInfo(rgtPath);
	QDir surveyDir = dataFileInfo.absoluteDir();
	bool isValid = surveyDir.cdUp() && surveyDir.cdUp();
	if ( !isValid ) return;
	QString path = surveyDir.absolutePath();
//	path += "/ImportExport"; mkdirPathIfNotExist(path);
//	path += "/IJK"; mkdirPathIfNotExist(path);
//	path += "/HORIZONS"; mkdirPathIfNotExist(path);
//	path += "/" + QString::fromStdString(FreeHorizonManager::BaseDirectory) + "/";
	path += "/" + QString::fromStdString(GeotimePath::NEXTVISION_NVHORIZON_PATH) + "/";

	mkdirPathIfNotExist(path);
	path = path + "/" + filename; mkdirPathIfNotExist(path);
	// QString horizonPath = path + "/";
	// path = path + "/" + QString::fromStdString(FreeHorizonManager::isoDataName);
	QString rgtName = SeismicManager::seismicFullFilenameToTinyName(rgtPath);
	FreeHorizonManager::write(rgtPath.toStdString(), rgtName.toStdString(), path.toStdString(), data);
	// FreeHorizonManager::displayParams(path.toStdString());


	// freeHorizonSaveWithoutTdebAndPasech(path.toStdString(), data, dimy, dimz, tdeb, pasech);
	/*
	FILE *pf = fopen(path.toStdString().c_str(), "w");
	if ( pf == nullptr ) { qDebug() << "problem in creating file: " + path; return; }
	fwrite(data, sizeof(float), (long)dimy*dimz, pf);
	fclose(pf);
	freeHorizonWriteDims(horizonPath, dimy, dimz);
	*/

	/*
	float *out = (float*)calloc(dimy*dimz, sizeof(float));
	for (long add=0; add<dimy*dimz; add++)
		out[add] = ( data[add] - tdeb ) / pasech;

	for (int i=0; i<20; i++)
		fprintf(stderr, "* %d - %f\n", i, out[i]);

	const murat::io::InputOutputCube<float> *cube0 = murat::io::createCube<float>(path.toStdString(), 1, dimy, dimz, murat::io::SampleType::E::FLOAT32, true);
	// cube = murat::io::openCube<float>(path.toStdString());
	const murat::io::InputOutputCube<float> *cube = murat::io::openRAWCube<float>(path.toStdString(), 1, dimy, dimz, 0, murat::io::SampleType::E::FLOAT32, true);
	cube->writeSubVolume(0, 0, 0, out, 1, dimy, dimz);
	murat::io::CubeDimension dims = cube->getDim();
	int ni = dims.getI();
	int nj = dims.getJ();
	int nk = dims.getK();

	for (long add=0; add<dimy*dimz; add++) out[add] = 1.0f;

	const murat::io::InputOutputCube<float> *cube2 = murat::io::openCube<float>(path.toStdString());
	cube2->readSubVolume(0, 0, 0, 1, dimy, dimz, out);
	for (int i=0; i<20; i++)
		fprintf(stderr, "** %d - %f\n", i, out[i]);
	qDebug() << "end";
	*/
}

template<typename InputType>
struct CopyPaletteToBufferKernel {
	static void run(CUDARGBImage* image, pixel_t *pixel_out, int dimW, int dimH) {
		CUDAImagePaletteHolder* red = image->get(0);
		CUDAImagePaletteHolder* green = image->get(1);
		CUDAImagePaletteHolder* blue = image->get(2);

		QVector2D redRange = red->range();
		QVector2D greenRange = green->range();
		QVector2D blueRange = blue->range();

		image->lockPointer();
		InputType* redBuf = static_cast<InputType*>(red->backingPointer());
		InputType* greenBuf = static_cast<InputType*>(green->backingPointer());
		InputType* blueBuf = static_cast<InputType*>(blue->backingPointer());

		for (int h = 0; h < dimH; h++) {
			for (int w = 0; w < dimW; w++) {
				int iIn = (h * dimW + w);
				int iOut = (w * dimH + h);
				pixel_out[iOut].red  = Cultural::getComponent(redBuf[iIn], redRange);
				pixel_out[iOut].green = Cultural::getComponent(greenBuf[iIn], greenRange);
				pixel_out[iOut].blue   = Cultural::getComponent(blueBuf[iIn], blueRange);
			}
		}
		image->unlockPointer();
	}
};

bool LayerSpectrumDialog::saveRGB(QString saveRGBName) {
	bool ok = m_isComputationValid && m_data->isModuleComputed();

	// read rgb data
	if (ok) {
		int dimW = m_dataRGB->layerSlice()->width();
		int dimH = m_dataRGB->layerSlice()->depth();
		QRect sourceRegion(0, 0, dimW, dimH);

		//	RawImage1 * rawImage  = imageRgb->ReadData1( sourceRegion );
		//	QByteArray& buf = rawImage->buffer;
		//	unsigned char* charTab = static_cast<unsigned char*> (static_cast<void*>(buf.data()));


		bitmap_t bitmap;
		bitmap.width = dimH; // reversed
		bitmap.height = dimW;
		bitmap.pixels = new pixel_t[dimW * dimH];

		//uint8_t *pixel_in = charTab;
		pixel_t *pixel_out = bitmap.pixels;

		CUDARGBImage* image = m_dataRGB->image();

		SampleTypeBinder binder(image->get(0)->sampleType());
		binder.bind<CopyPaletteToBufferKernel>(image, pixel_out, dimW, dimH);


		int retval = Cultural::save_png_to_file (&bitmap, saveRGBName.toStdString().c_str());

		if (retval != 0) {
			printf ("Error saving PNG.\n");
			ok = false;
			//goto free_val;
		}

		//????free(buffer);
		delete[] (bitmap.pixels);
	}
	return ok;
}

void LayerSpectrumDialog::toggleSeedMode(bool isMultiSeed) {
	if (isMultiSeed!=m_horizon->isMultiSeed()) {
		m_horizon->setMultiSeed(isMultiSeed);
		clearHorizon();
		if (m_horizon->isMultiSeed()) {
			m_seedModeButton->setText("Multi Seed Mode");
			//m_seedEditButton->show();
			m_undoButton->show();
			m_redoButton->show();
			//m_saveButton->show();
			m_releaseButton->show();
			m_tauSpinBox->hide();
			m_seedEditButton->setChecked(true);
		} else {
			m_seedModeButton->setText("Mono Seed Mode");
			//m_seedEditButton->hide();
			m_undoButton->hide();
			m_redoButton->hide();
			//m_saveButton->hide();
			m_releaseButton->hide();
			m_seedEditButton->setChecked(false);
			m_redoButtonList.clear();

			if (m_seeds.size()>1) {
				std::vector<RgtSeed> new_seeds;
				new_seeds.push_back(m_seeds[0]);
				setSeeds(new_seeds);
				if (m_horizon!=nullptr) {
					std::size_t id = m_horizon->getAllIds().back();
					m_horizon->selectPoint(id);
				}
			}
			m_tauSpinBox->show();

		}
	}
}

template<typename InputType>
struct PropagatorKernel {
	static std::vector<RgtSeed> run(SeedsGenericPropagator* _propagator,
			const std::vector<RgtSeed>& seeds, int type_snap, int isx,
			int propagation_type, int sizeCorr, float seuilCorr, int numIter, int seedReductionWindow) {
		SeedsPropagator<InputType>* propagator = dynamic_cast<SeedsPropagator<InputType>*>(_propagator);
		std::vector<RgtSeed> newSeeds = propagator->propagate(seeds, propagation_type, type_snap, sizeCorr,
				isx, seuilCorr, numIter, seedReductionWindow);

		return (newSeeds);
	}
};

template<typename ImageType>
struct CloneAndKeepKernel {
	static bool run(IImagePaletteHolder* image, const QByteArray& mask, float* copyBuffer) {
		long dimX = image->width();
		long dimZ = image->height();
		long N = dimX * dimZ;

		bool atLeastOneModification = false;

		image->lockPointer();

		ImageType* imageTab = static_cast<ImageType*>(image->backingPointer());
		for (long i=0; i<N; i++) {
			if (mask[static_cast<unsigned int>(i)]!=0 && copyBuffer[i]==-9999.0) {
				copyBuffer[i] = imageTab[i];
				atLeastOneModification = true;
			}
		}

		image->unlockPointer();

		return atLeastOneModification;
	}
};

template<typename InputType>
struct UpdateSeismicBufferKernel {
	static void run(Seismic3DDataset* datasetS, int channelS,
					double tdeb, double pasech, FixedLayerFromDataset* layer, QString ampName, QString isoName) {
		std::vector<float> ampVect, isoVect;
		float nullValue = -9999.0f;

		std::size_t dimI = layer->getNbProfiles();
		std::size_t dimJ = layer->getNbTraces();

		ampVect.resize(dimI*dimJ, nullValue);
		isoVect.resize(dimI*dimJ, nullValue);

		layer->readProperty(isoVect.data(), isoName);
		layer->readProperty(ampVect.data(), ampName);

		std::vector<InputType> valTab;
		valTab.resize(datasetS->dimV());

		for (std::size_t i=0; i<dimI; i++) {
			for (std::size_t j=0; j<dimJ; j++) {
				std::size_t idx = j + i * dimJ;
				if (isoVect[idx]!=nullValue && ampVect[idx]==nullValue) {
					int x = (isoVect[idx] - tdeb) / pasech;
					if (x>=0 && x<datasetS->height()) {
						datasetS->readSubTraceAndSwap(valTab.data(), x, x+1, j, i);
						ampVect[idx] = valTab[channelS];
					}
				}
			}
		}

		layer->writeProperty(ampVect.data(), ampName);
	}
};

void LayerSpectrumDialog::cloneAndKeep()
{
	st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();
	if (st_GraphicSettings.pActiveScene != nullptr)
	{
		// this isoSurface is time/depth
		BaseMapSurface *baseMapSurface = st_GraphicSettings.pActiveScene->cloneAndKeepFromLayerSpectrum();
		IImagePaletteHolder* isoSurface = nullptr;
		CUDAImageMask* item = nullptr;
		bool valid = baseMapSurface!=nullptr && baseMapSurface->isoType()==m_datasetS->cubeSeismicAddon().getSampleUnit();
		if (valid)  {
			isoSurface = baseMapSurface->isoSurface();
			item = dynamic_cast<CUDAImageMask*>(baseMapSurface->basemapItem());

			valid = item!=nullptr;
		}

		if (valid) {
			QByteArray mask = item->getArray();

			long width = m_datasetS->width();
			long depth = m_datasetS->depth();
			// check sizes before buffer manipulations
			if (isoSurface->width()==width && isoSurface->height()==depth) {
				std::vector<float> iso;
				iso.resize(width * depth, -9999.0f);
				if (m_constrainData!=nullptr) {
					// output bool ignored because iso buffer is already initialized
					m_constrainData->readProperty(iso.data(), FixedLayerFromDataset::ISOCHRONE);
				}

				SampleTypeBinder binder(isoSurface->sampleType());
				bool updated = binder.bind<CloneAndKeepKernel>(isoSurface, mask, iso.data());

				if (updated) {
					bool created = m_constrainData==nullptr;

					if (created) {
						createConstrain();
					}
					m_constrainData->writeProperty(iso.data(), FixedLayerFromDataset::ISOCHRONE);

					if (m_propagator==nullptr) {
						SampleTypeBinder binder(m_datasetS->sampleType());
						m_propagator.reset(binder.bind<CreatePropagatorObjectKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS));
					}
					std::vector<int>& tabSeed = m_propagator->getTabSeedType();
					std::vector<float>& tabIso = m_propagator->getIsochroneTab();
					std::vector<float>& tabAmp = m_propagator->getAmplitudeTab();
					tabSeed.clear();
					tabSeed.resize(m_datasetS->width()*m_datasetS->depth(), 0);
					tabIso.clear();
					tabIso.resize(m_datasetS->width()*m_datasetS->depth(), -9999);
					tabAmp.clear();
					tabAmp.resize(m_datasetS->width()*m_datasetS->depth(), -9999);

					m_constrainData->readProperty(tabIso.data(), FixedLayerFromDataset::ISOCHRONE);

					// TODO update amplitude
					double pasech = m_datasetS->sampleTransformation()->a();
					double tdeb = m_datasetS->sampleTransformation()->b();
					SampleTypeBinder binderS(m_datasetS->sampleType());
					binderS.bind<UpdateSeismicBufferKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS,
							tdeb, pasech, m_constrainData.get(), "amplitude", FixedLayerFromDataset::ISOCHRONE);


					for (std::size_t i=0; i<tabSeed.size(); i++) {
						if (tabSeed[i]==0 && tabIso[i]!=-9999.0f) {
							tabSeed[i] = 1;
						}
					}

					if (created) {
						m_datasetS->workingSetManager()->addFixedLayerFromDataset(m_constrainData.get());

						if (m_horizon!=nullptr) {
							m_horizon->setConstrainLayer(m_constrainData.get());
						}
					}

					updatePropagationExtension();

	//				std::vector<RgtSeed> newSeeds = m_propagator->extractSeedsFromTabs(m_seedFilterNumber);
	//				QString rgtName("rgt");
	//				SampleTypeBinder binder(m_datasetT->sampleType());
	//				binder.bind<UpdateSeedsRgtValueKernel>(m_datasetT, m_channelT, m_referenceLayers, "isochrone", "rgt", newSeeds, tdeb, pasech);
	//				setSeeds(newSeeds);
	//				if (m_horizon!=nullptr) {
	//					m_horizon->setPolarity(polarity);
	//				}
					updatePropagationView();
					m_undoPropagationButton->show();
					propagationCacheSetup();
				}
			}
		}
		if (baseMapSurface!=nullptr) {
			baseMapSurface->deleteLater();
		}
	}
}

void LayerSpectrumDialog::eraseData()
{
	st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();
	bool valid = st_GraphicSettings.pActiveScene != nullptr;
	Abstract2DInnerView* innerView = nullptr;
	if (valid)
	{
		innerView = st_GraphicSettings.pActiveScene->innerView();
		valid = innerView!=nullptr && (innerView->viewType()==BasemapView || innerView->viewType()==StackBasemapView);
	}

	QList<QGraphicsItem*> selectedItems;
	std::vector<float> tabFloat;
	if (valid) {
		// get items
		selectedItems = st_GraphicSettings.pActiveScene->selectedItems();

		bool notFound = true;
		int i = 0;
		float pasech = m_datasetS->sampleTransformation()->a();
		float tdeb = m_datasetS->sampleTransformation()->b();
		while (notFound && i < m_VectorData.size()) {
			if (dynamic_cast<LayerSlice*>(m_VectorData[i])!=nullptr &&
					dynamic_cast<LayerSlice*>(m_VectorData[i])->getMethod()==eComputeMethd_Spectrum) {
				const float * tabShort = dynamic_cast<LayerSlice*>(m_VectorData[i])->getModuleData(0);
				notFound = tabShort==nullptr;
				if (!notFound) {
					tabFloat.resize(m_datasetS->width()*m_datasetS->depth());
					for (std::size_t j = 0; j<tabFloat.size(); j++) {
						tabFloat[j] = tabShort[j] * pasech + tdeb;
					}
				}
			}
			i++;
		}
		valid = !notFound;
	}

	if (valid) {
		if (m_constrainData==nullptr) {
			if (m_propagator==nullptr) {
				SampleTypeBinder binder(m_datasetS->sampleType());
				m_propagator.reset(binder.bind<CreatePropagatorObjectKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS));
			}

			std::vector<int>& tabSeed = m_propagator->getTabSeedType();
			std::vector<float>& tabIso = m_propagator->getIsochroneTab();
			std::vector<float>& tabAmp = m_propagator->getAmplitudeTab();
			tabSeed.clear();
			tabSeed.resize(m_datasetS->width()*m_datasetS->depth(), 0);
			tabIso.clear();
			tabIso.resize(m_datasetS->width()*m_datasetS->depth(), -9999);
			tabAmp.clear();
			tabAmp.resize(m_datasetS->width()*m_datasetS->depth(), -9999);

			updatePropagationExtension();
		} else {
			if (m_propagator==nullptr) {
				SampleTypeBinder binder(m_datasetS->sampleType());
				m_propagator.reset(binder.bind<CreatePropagatorObjectKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS));
			}
			std::vector<int>& tabSeed = m_propagator->getTabSeedType();
			std::vector<float>& tabIso = m_propagator->getIsochroneTab();
			std::vector<float>& tabAmp = m_propagator->getAmplitudeTab();
			// tabSeed.clear();
			tabSeed.resize(m_datasetS->width()*m_datasetS->depth(), 0);
			// tabIso.clear();
			tabIso.resize(m_datasetS->width()*m_datasetS->depth(), -9999);
			// tabAmp.clear();
			tabAmp.resize(m_datasetS->width()*m_datasetS->depth(), -9999);
		}
		m_constrainData->writeProperty(tabFloat.data(), FixedLayerFromDataset::ISOCHRONE);
		tabFloat.clear();
	}

	if (valid) {
		// create of datas to erase
		QList<iGraphicToolDataControl*> datasToModify;
		datasToModify.append(m_constrainData.get()); // constrain
		// keep below commented code in case specifications change
//			for (int i=0; i<m_VectorData.size(); i++) {
//				datasToModify.append(m_VectorData[i]); // gray datas
//
//				if (m_VectorData[i]->getMethod()==eComputeMethd_Spectrum &&
//						m_mapData.contains(m_VectorData[i]->seismic()) &&
//						m_mapData[m_VectorData[i]->seismic()].size()>eComputeMethd_RGB) { // rgb only for spectrum
//					IData* rgbData = m_mapData[m_VectorData[i]->seismic()][eComputeMethd_RGB];
//					if (rgbData!=nullptr) {
//						datasToModify.append(rgbData); // rgb datas
//					}
//				}
//			}

		// apply to all datas the suppresion
		for (int i=0; i<datasToModify.size(); i++) {
			for (int j = 0; j<selectedItems.size(); j++) {
				if (datasToModify[i]!=nullptr) {
					datasToModify[i]->deleteGraphicItemDataContent(selectedItems[j]);
				}
			}
		}

		if (m_constrainData!=nullptr) {
			std::vector<float> tabIso, tabAmp;
			tabIso.resize(m_datasetS->width()*m_datasetS->depth());
			tabAmp.resize(m_datasetS->width()*m_datasetS->depth());

			bool ok = m_constrainData->readProperty(tabIso.data(), FixedLayerFromDataset::ISOCHRONE) &&
					m_constrainData->readProperty(tabAmp.data(), "amplitude");

			if (ok) {
				CubeSeismicAddon addon = m_datasetS->cubeSeismicAddon();
				std::vector<RgtSeed> noSeedsVector;

				int correctPolarity = m_polarity;

				setSeeds(noSeedsVector);
				m_patchSeeds.clear();
				m_patchSeedId.clear();

				setPolarity(correctPolarity);
				if (m_horizon!=nullptr) {
					m_horizon->setPolarity(m_polarity);
				}

				std::vector<float>& tabIso = m_propagator->getIsochroneTab();
				std::vector<float>& tabAmp = m_propagator->getAmplitudeTab();
				m_constrainData->readProperty(tabIso.data(), FixedLayerFromDataset::ISOCHRONE);
				QString ampName("amplitude");
				m_constrainData->readProperty(tabAmp.data(), ampName);

				std::vector<int>& tabType = m_propagator->getTabSeedType();
				for (std::size_t i=0; i<tabType.size(); i++) {
					if (tabIso[i]!=-9999.0f) {
						tabType[i] = 1;
					} else {
						tabType[i] = 0;
					}
				}

				// clear undo cache
				m_propagatorUndoCache.clear();
				m_propagatorNextId = 1;

				updatePropagationView();
				updatePropagationExtension();
				m_undoPropagationButton->show();
				propagationCacheSetup();
			}
		}
	}
}

void LayerSpectrumDialog::propagateHorizon() {
	if (m_horizon->isMultiSeed()) {
		m_horizon->applyDtauToSeeds();
		m_horizon->updateSeedsPositionWithRead(this);

		if (m_propagator==nullptr) {
			SampleTypeBinder binder(m_datasetS->sampleType());
			m_propagator.reset(binder.bind<CreatePropagatorObjectKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS));
		}

		long highestSeedId = 0;

		// io::SampleTypeBinder binder(m_propagator->getSampleType());
		// std::vector<RgtSeed> newSeeds = /*binder.bind<*/PropagatorKernel<short>/*>*/::run(m_propagator.get(),
		// 				m_seeds, m_polarity, m_snapWindow, m_propagationType, m_sizeCorr, m_seuilCorr,
		// 				m_seedFilterNumber);


		for (int i=0; i<m_seeds.size(); i++)
			fprintf(stderr, "m_seeds: %d --> %d %d %d\n", i, m_seeds[i].x, m_seeds[i].y, m_seeds[i].z);

		SampleTypeBinder binderS(m_datasetS->sampleType());
		std::vector<RgtSeed> newSeeds = binderS.bind<PropagatorKernel>(m_propagator.get(),
				m_seeds, m_polarity, m_snapWindow, m_propagationType, m_sizeCorr, m_seuilCorr,
				m_numIter, m_seedFilterNumber);

		// newSeeds.resize(0);


		updatePropagationExtension();
		int correctPolarity = m_polarity;
		//std::unique_ptr<io::Cube> rgt = process::openAsGenericCube(m_datasetT);
		QString isoName("isochrone");
		QString rgtName("rgt");
		//model::DatasetGroup* dsg = m_datasetS->getDatasetGroup();
		float tdeb = m_constrainData->getOriginSample();
		float pasech = m_constrainData->getStepSample();

		SampleTypeBinder binderT(m_datasetT->sampleType());
		binderT.bind<UpdateSeedsRgtValueKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetT), m_channelT,
				m_referenceLayers, isoName, rgtName, newSeeds, tdeb, pasech);

		setSeeds(newSeeds);

		setPolarity(correctPolarity);
		if (m_horizon!=nullptr) {
			m_horizon->setPolarity(m_polarity);
		}

		updatePropagationView();

		m_toggleInterpolation->show();
		m_undoPropagationButton->show();


		/*
		char *filename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/tmp/HORIZON_GRIDS/base.raw";
		float *data = (float*)calloc(1500*700, sizeof(float));
		FILE *pfile = fopen(filename, "r");
		fread(data, sizeof(float), 1500*700, pfile);
		fclose(pfile);


		std::vector<std::vector<RgtSeed>> vseed;
		vseed.resize(1);
		vseed[0].resize(1500*700);
		int idx = 0;
		for (int zz=0; zz<700; zz++)
			for (int xx=0; xx<1500; xx++)
			{
				vseed[0][idx].x = (int)(data[zz*1500+xx]/2.0);
				vseed[0][idx].y = xx;
				vseed[0][idx++].z = zz;
			}

		m_horizon->setSeeds2(vseed);
		 */


	}

	propagationCacheSetup();
}

void LayerSpectrumDialog::updatePropagationExtension() {
	if (m_propagator==nullptr) {
		return;
	}
	if (m_constrainData==nullptr) {
		createConstrain();
	}
	QString isoName("isochrone"), ampName("amplitude");
	m_constrainData->writeProperty(m_propagator->getIsochroneTab().data(), isoName);
	m_constrainData->writeProperty(m_propagator->getAmplitudeTab().data(), ampName);
	m_data->setConstrainLayer(m_constrainData.get(), isoName);

	if (m_horizon!=nullptr) {
		m_horizon->setConstrainLayer(m_constrainData.get());
	}

	//	PropertiesRegister& propertiesregister = m_originViewer->getPropertiesRegister();
	//	view2d::DatasetSyncViewer2dVisual* rgtVisual = nullptr;
	//	std::vector<view2d::SyncViewer2dVisual*> visuals = m_originViewer->getVisuals();
	//	int i=0;
	//	while(i<visuals.size()) {
	//		view2d::DatasetSyncViewer2dVisual* visual = dynamic_cast<view2d::DatasetSyncViewer2dVisual*>(visuals[i]);
	//		if (visual!=nullptr && visual->getDataset()==m_datasetT) {
	//			rgtVisual = visual;
	//		}
	//		i++;
	//	}
	//	if (m_constrainLayerExtension!=nullptr) {
	//		this->m_originMainViewer->getViewExtensions().removeExtension(m_constrainLayerExtension.get());
	//	}
	//	m_constrainLayerExtension.reset(new view2d::ViewLayerExtension(propertiesregister, rgtVisual, m_constrainLayer.get(), isoName));
	//	this->m_originMainViewer->getViewExtensions().addExtension(m_constrainLayerExtension.get());
	//	m_constrainLayerExtension->setDirection(m_originMainViewer->getDirection());
	//	m_constrainLayerExtension->setCurrentSlice(m_originMainViewer->getCurrentSlice());
	//	m_constrainLayerExtension->setPixelSampleRateX(rgtVisual->getDatasetGroup()->getDBStepX());
	//	if (m_originViewer->getDirection()==view2d::View2dDirection::Z) {
	//		m_constrainLayerExtension->setPixelSampleRateY(rgtVisual->getDatasetGroup()->getDBStepY());
	//	} else if (m_originViewer->getDirection()==view2d::View2dDirection::Y) {
	//		m_constrainLayerExtension->setPixelSampleRateY(rgtVisual->getDatasetGroup()->getDBStepZ());
	//	} else {
	//		// TODO Direction X should block this
	//	}
	//	m_constrainLayerExtension->updateMainHorizon();
}


void LayerSpectrumDialog::propagationCacheSetup() {
	if (m_propagator==nullptr) {
		return;
	}
	const std::vector<int>& propagatorSeedCache = m_propagator->getTabSeedType();
	m_propagatorUndoCache.resize(propagatorSeedCache.size(), 0);
	for (long index=0; index<propagatorSeedCache.size(); index++) {
		if (m_propagatorUndoCache[index]==0 && propagatorSeedCache[index]!=0) {
			m_propagatorUndoCache[index] = m_propagatorNextId;
		}
	}

	m_propagatorNextId ++;
}

// todo JD
void LayerSpectrumDialog::undoPropagation() {
	if (m_constrainData==nullptr || m_propagator==nullptr ||
			m_propagatorNextId<=1 || m_propagatorUndoCache.size()==0) {
		return;
	}
	unsigned int toRemoveId = m_propagatorNextId - 1;
	std::vector<int>& propagatorSeedCache = m_propagator->getTabSeedType();
	std::vector<float>& propagatorAmpCache = m_propagator->getAmplitudeTab();
	std::vector<float>& propagatorIsoCache = m_propagator->getIsochroneTab();
	for (long index=0; index<m_propagatorUndoCache.size(); index++) {
		if (m_propagatorUndoCache[index]==toRemoveId) {
			m_propagatorUndoCache[index] = 0;
			propagatorSeedCache[index] = 0;
			propagatorAmpCache[index] = -9999;
			propagatorIsoCache[index] = -9999;
		}
	}
	m_propagatorNextId--;

	updatePropagationExtension();

	std::vector<RgtSeed> newSeeds = m_propagator->extractSeedsFromTabs(m_seedFilterNumber);
	int correctPolarity = m_polarity;
	//std::unique_ptr<io::Cube> rgt = process::openAsGenericCube(m_datasetT);
	QString isoName("isochrome");
	QString rgtName("rgt");
	float tdeb = m_constrainData->getOriginSample();
	float pasech = m_constrainData->getStepSample();
	SampleTypeBinder binder(m_datasetT->sampleType());
	binder.bind<UpdateSeedsRgtValueKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetT),
			m_channelT, m_referenceLayers, isoName, rgtName, newSeeds, tdeb, pasech);
	setSeeds(newSeeds);
	setPolarity(correctPolarity);
	if (m_horizon!=nullptr) {
		m_horizon->setPolarity(m_polarity);
	}

	if ( m_patchSeeds.size() > 0 )
	{
		m_patchSeeds.pop_back();
		m_patchSeedId.pop_back();
	}

/*
	int iBegin = -1;
	int iEnd = -1;
	for (int  n=0; n<m_patchSeedId.size(); n++)
	{
		if ( iBegin < 0 && m_patchSeedId[n] == m_propagatorNextId ) iBegin = n;
		if ( m_patchSeedId[n] == m_propagatorNextId ) iEnd = n;
	}
	if ( iBegin >= 0 && iEnd >=0 )
	{
		m_patchSeedId.erase(m_patchSeedId.begin()+iBegin, m_patchSeedId.begin()+iEnd);
		m_patchSeeds.erase(m_patchSeeds.begin()+iBegin, m_patchSeeds.begin()+iEnd);
	}
*/

	updatePropagationView();
}

void LayerSpectrumDialog::updatePropagationView() {
	if (m_propagator==nullptr) {
		return;
	}

	if (!m_isConstrainInitialized) {
		m_datasetS->workingSetManager()->addFixedLayerFromDataset(m_constrainData.get());
		m_isConstrainInitialized = true;

	}

	// search second BaseMapView
	StackBaseMapView* basemap = nullptr;
	GeotimeGraphicsView* geotimeView = m_originViewer;
	if (geotimeView!=nullptr) {
		QVector<AbstractInnerView*> innerViews = geotimeView->getInnerViews();
		std::size_t index = 0;
		while(index<innerViews.size() && (dynamic_cast<StackBaseMapView*>(innerViews[index])==nullptr ||
				dynamic_cast<StackBaseMapView*>(innerViews[index])->windowTitle().compare("Propagation"))) {
			index++;
		}
		if (index<innerViews.size()) {
			basemap = dynamic_cast<StackBaseMapView*>(innerViews[index]);
		}
	}

	if (basemap!=nullptr){
		QInnerViewTreeWidgetItem* rootItem = geotimeView->getItemFromView(basemap);

		QStack<QTreeWidgetItem*> stack;
		stack.push(rootItem);
		QGraphicsRepTreeWidgetItem* itemData = nullptr;
		while (stack.size()>0 && itemData==nullptr) {
			QTreeWidgetItem* item = stack.pop();

			std::size_t N = item->childCount();
			for (std::size_t index=0; index<N; index++) {
				stack.push(item->child(index));
			}
			QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
			if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
				const FixedLayerFromDataset* data = dynamic_cast<const FixedLayerFromDataset*>(_item->getRep()->data());
				if (data!=nullptr && data==m_constrainData.get()) {
					itemData = _item;
				}
			}
		}
		if (itemData!=nullptr) {
			itemData->setCheckState(0, Qt::Checked);
			dynamic_cast<FixedLayerFromDatasetRep*>(itemData->getRep())->chooseAttribute("amplitude");
		}
	}
	{
		// activate on origin viewer
		for (AbstractInnerView* innerView : m_originViewer->getInnerViews()) {
			if (innerView->viewType()!=ViewType::InlineView && innerView->viewType()!=ViewType::XLineView &&
					innerView->viewType()!=ViewType::RandomView ) {
				continue;
			}
			QInnerViewTreeWidgetItem* rootItem = geotimeView->getItemFromView(innerView);

			QStack<QTreeWidgetItem*> stack;
			stack.push(rootItem);
			QGraphicsRepTreeWidgetItem* itemData = nullptr;
			while (stack.size()>0 && itemData==nullptr) {
				QTreeWidgetItem* item = stack.pop();

				std::size_t N = item->childCount();
				for (std::size_t index=0; index<N; index++) {
					stack.push(item->child(index));
				}
				QGraphicsRepTreeWidgetItem* _item = dynamic_cast<QGraphicsRepTreeWidgetItem*>(item);
				if (_item!=nullptr && (item->flags() & Qt::ItemIsUserCheckable)) {
					const FixedLayerFromDataset* data = dynamic_cast<const FixedLayerFromDataset*>(_item->getRep()->data());
					if (data!=nullptr && data==m_constrainData.get()) {
						itemData = _item;
					}
				}
			}
			if (itemData!=nullptr) {
				itemData->setCheckState(0, Qt::Checked);
			}
		}
	}

}

void LayerSpectrumDialog::interpolateHorizon(bool toggled) {
	if ( m_constrainData==nullptr) {
		return;
	}
	m_isDataInterpolated = toggled;
	if (toggled) {
		m_horizon->applyDtauToSeeds();
		//		view2d::DatasetSyncViewer2dVisual* rgtVisual = nullptr;
		//		view2d::DatasetSyncViewer2dVisual* seismicVisual = nullptr;
		//		std::vector<view2d::SyncViewer2dVisual*> visuals = m_originViewer->getVisuals();
		//		int i=0;
		//		while(i<visuals.size()) {
		//			view2d::DatasetSyncViewer2dVisual* visual = dynamic_cast<view2d::DatasetSyncViewer2dVisual*>(visuals[i]);
		//			if (visual!=nullptr && visual->getDataset()==m_datasetT) {
		//				rgtVisual = visual;
		//			}
		//			if (visual!=nullptr && visual->getDataset()==m_datasetS) {
		//				seismicVisual = visual;
		//			}
		//			i++;
		//		}
		//		if ( rgtVisual==nullptr || seismicVisual==nullptr) {
		//			return;
		//		}
		//		DimensionHolder dims = rgtVisual->getDimensions();
		float tdeb = m_constrainData->getOriginSample();
		float pasech = m_constrainData->getStepSample();
		//		io::Cube* rgt = dynamic_cast<io::Cube*>(io::openCube<float>(rgtVisual->getDataset()->getDataPath().toStdString()));
		//		io::Cube* seismic = dynamic_cast<io::Cube*>(io::openCube<float>(seismicVisual->getDataset()->getDataPath().toStdString()));
		// interpolate with rgt
		std::vector<float> inputLayer, outputLayerIso, outputLayerSeismic;
		inputLayer.resize(m_constrainData->width()*m_constrainData->depth());
		QString isoName("isochrone"), ampName("amplitude");
		m_constrainData->readProperty(inputLayer.data(), isoName);
		SampleTypeBinder binder(m_datasetT->sampleType());
		binder.bind<LayerRGTInterpolatorKernel>(m_datasetS->sampleType(), inputLayer, m_horizon->getDTauReference(), outputLayerIso, outputLayerSeismic, m_referenceLayers, m_seeds,
				dynamic_cast<Seismic3DDataset*>(m_datasetT), m_channelT, dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS,
				m_useSnap, m_useMedian, m_halfLwxMedianFilter*2+1, m_distancePower,  m_snapWindow,  m_polarity, tdeb, pasech);

		//		io::ArrayCube<float>* cubeAmp = dynamic_cast<io::ArrayCube<float>*>(m_dataAmp->getCubeForEdition());
		//		cubeAmp->writeSubVolume(0, 0, 0, outputLayerSeismic.data(), 1, cubeAmp->getDim().getJ(), cubeAmp->getDim().getK());
		//		m_dataAmp->releaseEdition();

		m_constrainData->writeProperty(outputLayerSeismic.data(), ampName);
		m_constrainData->writeProperty(outputLayerIso.data(), isoName);

		//		io::ArrayCube<float>* cubeIso = dynamic_cast<io::ArrayCube<float>*>(m_dataIso->getCubeForEdition());
		//		cubeIso->writeSubVolume(0, 0, 0, outputLayerIso.data(), 1, cubeIso->getDim().getJ(), cubeIso->getDim().getK());
		//		m_dataIso->releaseEdition();
	} else {
		//		io::ArrayCube<float>* cubeAmp = dynamic_cast<io::ArrayCube<float>*>(m_dataAmp->getCubeForEdition());
		//		cubeAmp->writeSubVolume(0, 0, 0, m_propagator->getAmplitudeTab().data(), 1, cubeAmp->getDim().getJ(), cubeAmp->getDim().getK());
		//		m_dataAmp->releaseEdition();
		QString isoName("isochrone"), ampName("amplitude");
		m_constrainData->readProperty(m_propagator->getAmplitudeTab().data(), ampName);
		m_constrainData->readProperty( m_propagator->getIsochroneTab().data(), isoName);

		//		io::ArrayCube<float>* cubeIso = dynamic_cast<io::ArrayCube<float>*>(m_dataIso->getCubeForEdition());
		//		cubeIso->writeSubVolume(0, 0, 0, m_propagator->getIsochroneTab().data(), 1, cubeIso->getDim().getJ(), cubeIso->getDim().getK());
		//		m_dataIso->releaseEdition();
	}
}

void LayerSpectrumDialog::initReferenceComboBox() {
	m_referenceList->clear();
	if ( m_horizonNames.size() == 0 ) return;
	for (auto it=m_horizonNames.cbegin(); it!=m_horizonNames.cend(); it++)
	{
		QColor color;
		if (m_horizonDatas[it->first]==nullptr) {
			QFileInfo baseFileInfo(m_horizonPaths[it->first]);
			QString colorFilePath = baseFileInfo.dir().absoluteFilePath(baseFileInfo.baseName() + "_color.txt");
			bool colorValid;
			color = FixedLayerFromDataset::loadColorFromFile(colorFilePath, &colorValid);
			if (!colorValid) {
				color = Qt::blue;
			}
		} else {
			color = m_horizonDatas[it->first]->color();
		}
		m_referenceList->addItem(it->second, color, (qlonglong)it->first);
	}
}

int getIndexFromVectorString(std::vector<QString> list, QString txt)
{
	for (int i=0; i<list.size(); i++)
	{
		if ( list[i].compare(txt) == 0 )
			return i;
	}
	return -1;
}

void LayerSpectrumDialog::updateReferenceHorizonList() {

	//	if (m_constrainData==nullptr) {
	//		createConstrain();
	//	}

	//	Seismic3DDataset* rgt = dynamic_cast<Seismic3DDataset*>(m_datasetT);
	//	long dimx = rgt->height();
	//	long dimy = rgt->width();
	//	long dimz = rgt->depth();
	//	float tdeb = rgt->sampleTransformation()->b();
	//	float pasech = rgt->sampleTransformation()->a();
	//
	//	QList<QListWidgetItem*> items = m_referenceList->selectedItems();
	//	std::vector<std::vector<RgtSeed>> vseed;
	//	vseed.resize(items.size());
	//	float *data = (float*)calloc(dimy*dimy, sizeof(float));


	/*
	for (int i=0; i<items.size(); i++)
	{
		QListWidgetItem* item = items[i];
		int idx = getIndexFromVectorString(*m_horizonNames, item->text());
		if ( idx >=0 && idx < m_horizonPaths->size() )
		{
			QString name = (*m_horizonPaths)[idx];
			fprintf(stderr, "reference: %s\n", name.toStdString().c_str());

			FILE *pfile = fopen(name.toStdString().c_str(), "r");
			fread(data, sizeof(float), dimy*dimz, pfile);
			fclose(pfile);

			vseed[i].resize(dimy*dimz);
			int idx = 0;
			for (int zz=0; zz<dimz; zz++)
				for (int xx=0; xx<dimy; xx++)
				{
					if ( data[zz*dimy+xx] >= 0.0f )
					{
						vseed[i][idx].x = (int)(data[zz*dimy+xx]/pasech);
						vseed[i][idx].y = xx;
						vseed[i][idx++].z = zz;

						if ( idx < 100 ) fprintf(stderr, "[%d %d] %f %d\n", zz, xx, data[zz*dimy+xx], vseed[i][idx-1].x);
					}
				}
		}
	}
	free(data);
	m_horizon->setSeeds2(vseed);
	 */

	//	for (int i=0; i<m_referenceData.size(); i++)
	//	{
	//		free(m_referenceData[i]);
	//	}
	//
	//	int N = items.size();
	//	m_referenceData.resize(N);
	//	for (int i=0; i<items.size(); i++)
	//	{
	//		QListWidgetItem* item = items[i];
	//		int idx = getIndexFromVectorString(m_horizonNames, item->text());
	//		if ( idx >=0 && idx < m_horizonPaths.size() )
	//		{
	//			QString name = m_horizonPaths[idx];
	//			fprintf(stderr, "reference: %s\n", name.toStdString().c_str());
	//			m_referenceData[i] = (float*)calloc(dimy*dimz, sizeof(float));
	//			FILE *pfile = fopen(name.toStdString().c_str(), "r");
	//			fread(m_referenceData[i], sizeof(float), dimy*dimz, pfile);
	//			fclose(pfile);
	//		}
	//	}
	//
	//	m_horizon->setReferences(m_referenceData);

	QList<QTreeWidgetItem*> items = m_referenceList->selectedItems();
	QList<QTreeWidgetItem*> newSelection;
	std::vector<bool> isItemInSelection;
	isItemInSelection.resize(m_referenceLayers.size(), false);

	for (QTreeWidgetItem* item : items) {
		std::size_t index_ref = 0;
		while(index_ref<m_referenceLayers.size() && item->text(0).compare(m_referenceLayers[index_ref]->name())) {
			index_ref++;
		}
		if (index_ref==m_referenceLayers.size()) {
			newSelection.push_back(item);
		} else {
			isItemInSelection[index_ref] = true;
		}
	}

	for(long index_ref=m_referenceLayers.size()-1; index_ref>=0; index_ref--) {
		if (!isItemInSelection[index_ref]) {
			unloadReferenceHorizon(index_ref);
		}
	}

	for (QTreeWidgetItem* item : newSelection) {
		loadReferenceHorizon(item);
	}
}

void LayerSpectrumDialog::updateReferenceHorizonColor(QTreeWidgetItem* item, QColor color) {
	std::size_t index_ref = 0;
	while(index_ref<m_referenceLayers.size() && item->text(0).compare(m_referenceLayers[index_ref]->name())) {
		index_ref++;
	}

	bool ok;
	long id = item->data(0, Qt::UserRole).toLongLong(&ok);
	if (!ok) {
		return;
	}
	auto dataIt = m_horizonDatas.find(id);
	if (dataIt==m_horizonDatas.end()) {
		return;
	}
	if (dataIt->second==nullptr) {
		QFileInfo baseFileInfo(m_horizonPaths[id]);
		QString colorFilePath = baseFileInfo.dir().absoluteFilePath(baseFileInfo.baseName() + "_color.txt");
		if (index_ref!=m_referenceLayers.size()) {
			m_referenceLayers[index_ref]->setColor(color);

			m_referenceLayers[index_ref]->saveColor(colorFilePath);
		} else {
			FixedLayerFromDataset::saveColorToFile(colorFilePath, color);
		}
	} else {
		if (index_ref!=m_referenceLayers.size()) {
			m_referenceLayers[index_ref]->setColor(color);
		}
		dataIt->second->setColor(color);
	}
}

void LayerSpectrumDialog::loadReferenceHorizon(QTreeWidgetItem* item) {

	//	Seismic3DDataset* rgt = dynamic_cast<Seismic3DDataset*>(m_datasetT);
	//	long dimx = rgt->height();
	//	long dimy = rgt->width();
	//	long dimz = rgt->depth();
	//	float tdeb = rgt->sampleTransformation()->b();
	//	float pasech = rgt->sampleTransformation()->a();
	//
	//	fprintf(stderr, "%d %d %d %f %f\n", dimx, dimy, dimz, tdeb, pasech);
	//	return;




	//	char *filename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/tmp/HORIZON_GRIDS/base.raw";
	//			float *data = (float*)calloc(1500*700, sizeof(float));
	//			FILE *pfile = fopen(filename, "r");
	//			fread(data, sizeof(float), 1500*700, pfile);
	//			fclose(pfile);
	//
	//
	//			std::vector<std::vector<RgtSeed>> vseed;
	//			vseed.resize(1);
	//			vseed[0].resize(1500*700);
	//			int idx = 0;
	//			for (int zz=0; zz<700; zz++)
	//				for (int xx=0; xx<1500; xx++)
	//				{
	//					vseed[0][idx].x = (int)(data[zz*1500+xx]/2.0);
	//					vseed[0][idx].y = xx;
	//					vseed[0][idx++].z = zz;
	//				}
	//
	//			m_horizon->setSeeds2(vseed);

	bool idValid;
	long id = item->data(0, Qt::UserRole).toLongLong(&idValid);
	if (!idValid) {
		return;
	}

	Seismic3DDataset* seismic = dynamic_cast<Seismic3DDataset*>(m_datasetS);
	float tdeb = seismic->sampleTransformation()->b();
	float pasech = seismic->sampleTransformation()->a();

	std::shared_ptr<FixedLayerFromDataset> layer(new FixedLayerFromDataset(item->text(0), seismic->workingSetManager(), seismic));
	bool colorValid;
	QColor layerColor = m_referenceList->getColor(item, &colorValid);
	if (colorValid) {
		layer->setColor(layerColor);
	}

	bool isValid = true;
	QString isoName("isochrone");
	if (m_horizonDatas[id]==nullptr) {
		QString saveName(m_horizonPaths[id]);
		QFileInfo saveInfo(saveName);

		qDebug() << "save name " << saveName;
		isValid = layer->loadProperty(saveName, isoName);
	} else {
		std::string isoFilePath = m_horizonDatas[id]->path().toStdString() + "/" + FreeHorizonManager::isoDataName;
		FreeHorizonManager::PARAM params = FreeHorizonManager::dataSetGetParam(isoFilePath);
		std::vector<float> buffer;
		buffer.resize(params.dimx*params.dimy);
		std::string errorStr = FreeHorizonManager::read(isoFilePath, buffer.data());
		if (errorStr!="ok") {
			return;
		}
		layer->writeProperty(buffer.data(), isoName);
	}

	QString rgtName("rgt");
	std::vector<float> rgtVec, isoVec;
	isoVec.resize(layer->width()* layer->depth());
	rgtVec.resize(layer->width()* layer->depth(), -9999.0);
	layer->readProperty(isoVec.data(), isoName);

	// scan to check if there is holes
	//	for(std::size_t index=0; index<isoVec.size() && isValid; index++) {
	//		isValid = isoVec[index] != -9999.0;
	//	}

	//io::SampleTypeBinder binder(m_datasetT->getSampleType());
	//binder.bind<LoadReferenceRgtKernel>(isoVec, m_datasetT, rgtVec, dims.getOriginX(), dims.getStepX());
	isValid = isValid && layer->writeProperty(rgtVec.data(), rgtName);
	if (isValid) {
		// read all layers to allow point sorting
		std::vector<std::vector<float>> layers;
		std::vector<float> init;
		layers.resize(m_referenceLayers.size(), init);
		for (std::size_t index_ref=0; index_ref<layers.size(); index_ref++) {
			layers[index_ref].resize(m_referenceLayers[0]->getNbTraces()*m_referenceLayers[0]->getNbProfiles());
			m_referenceLayers[index_ref]->readProperty(layers[index_ref].data(), isoName);
		}

		// find insertion index
		long insert_index = -1;
		int rgtSearchVal, rgtCurrentVal;
		long ymid = layer->width()/2;
		long zmid = layer->depth()/2;
		if (m_referenceLayers.size()!=0) {
			do {
				insert_index++;
				rgtSearchVal = isoVec[layer->width() * zmid + ymid];
				std::vector<float>& currentMap = layers[insert_index];
				//currentMap.resize(isoVec.size());
				//m_referenceLayers[insert_index]->readProperty(currentMap.data(), isoName);
				rgtCurrentVal = currentMap[layer->width() * zmid + ymid];
			}
			while (insert_index+1<m_referenceLayers.size() && rgtCurrentVal<rgtSearchVal);
			if (rgtCurrentVal<rgtSearchVal) {
				insert_index=m_referenceLayers.size();
			}
		} else {
			insert_index = 0;
		}

		std::size_t Ndim = layer->width()* layer->depth();
//#pragma omp parallel for schedule(dynamic)
//		for (std::size_t index_carte=0; index_carte<Ndim; index_carte++) {
//			std::vector<float> vect;
//			vect.resize(m_referenceLayers.size()+1);
//			// get values
//			for (std::size_t index_ref=0; index_ref<m_referenceLayers.size(); index_ref++) {
//				vect[index_ref] = layers[index_ref][index_carte];
//			}
//			vect[m_referenceLayers.size()] = isoVec[index_carte];
//
//			std::sort(vect.begin(), vect.end());
//
//			// apply switch
//			for(std::size_t index_ref=0; index_ref<layers.size()+1; index_ref++) {
//				if (index_ref < insert_index) {
//					layers[index_ref][index_carte] = vect[index_ref];
//				} else if (index_ref==insert_index) {
//					isoVec[index_carte] = vect[index_ref];
//				} else {
//					layers[index_ref-1][index_carte] = vect[index_ref];
//				}
//			}
//		}
//
//		// apply changes
//		for (std::size_t index_ref=0; index_ref<layers.size(); index_ref++) {
//			m_referenceLayers[index_ref]->writeProperty(layers[index_ref].data(), isoName);
//		}

		//		PropertiesRegister propertiesregister;
		//		std::shared_ptr<view2d::ViewLayerExtension> layerExtension(new view2d::ViewLayerExtension(propertiesregister, rgtVisual, layer.get(), isoName));
		//
		//		this->m_originMainViewer->getViewExtensions().addExtension(layerExtension.get());
		//		layerExtension->setDirection(m_originMainViewer->getDirection());
		//		layerExtension->setCurrentSlice(m_originMainViewer->getCurrentSlice());
		//		layerExtension->setPixelSampleRateX(rgtVisual->getDatasetGroup()->getDBStepX());
		//		if (m_originViewer->getDirection()==view2d::View2dDirection::Z) {
		//			layerExtension->setPixelSampleRateY(rgtVisual->getDatasetGroup()->getDBStepY());
		//		} else if (m_originViewer->getDirection()==view2d::View2dDirection::Y) {
		//			layerExtension->setPixelSampleRateY(rgtVisual->getDatasetGroup()->getDBStepZ());
		//		} else {
		//			// TODO Direction X should block this
		//		}
		//		layerExtension->updateMainHorizon();
		//
		//		m_referenceLayerExtensions.insert(m_referenceLayerExtensions.begin()+insert_index, layerExtension);
		m_referenceLayers.insert(m_referenceLayers.begin()+insert_index, layer);
		m_data->setReferenceLayers(m_referenceLayers, isoName, rgtName);
		m_horizon->setReferences(m_referenceLayers);
		//		clearHorizon();
	}
}

void LayerSpectrumDialog::unloadReferenceHorizon(std::size_t index_ref) {
	m_referenceLayers.erase(m_referenceLayers.begin()+index_ref);

	if(m_horizon!=nullptr) {
		m_horizon->setReferences(m_referenceLayers);
	}
	// TODO
	m_data->setReferenceLayers(m_referenceLayers, QString("isochrone"), QString("rgt"));
}

bool LayerSpectrumDialog::filterHorizon(IJKHorizon* horizon) {
	if (m_datasetS==nullptr && m_datasetT==nullptr) {
		return false;
	}

	Seismic3DAbstractDataset* refDataset = nullptr;

	// check dataset caracteristics
	QString seismicPath;
	if (m_datasetS!=nullptr) {
		seismicPath = QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetS)->path());
		refDataset = m_datasetS;
	}
	QString rgtPath;
	if (m_datasetT!=nullptr) {
		rgtPath = QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetT)->path());
		refDataset = m_datasetT;
	}

	// get seismic used for extraction of horizon
	QString horizonExtractionDatasetPath = horizon->seismicOriginPath(); //(*horizonExtractionDataPaths)[i];

	bool isCubeCompatible = seismicPath.compare(horizonExtractionDatasetPath)==0 || rgtPath.compare(horizonExtractionDatasetPath)==0;

	// check compatibility on carte (inline & xline)
	if (!isCubeCompatible) {
		Seismic3DDataset* dataset = dynamic_cast<Seismic3DDataset*>(refDataset);
		CubeSeismicAddon addon = dataset->cubeSeismicAddon();

		std::size_t nbTraces, nbProfiles;
		float oriTraces, pasTraces, oriProfiles, pasProfiles;
		SampleUnit sampleUnit;

		inri::Xt xt(horizonExtractionDatasetPath.toStdString().c_str());
		if (!xt.is_valid()) {
			isCubeCompatible = false;
		} else {
			nbTraces = xt.nRecords();
			nbProfiles = xt.nSlices();
			oriTraces = xt.startRecord();
			pasTraces = xt.stepRecords();
			oriProfiles = xt.startSlice();
			pasProfiles = xt.stepSlices();
			if (xt.axis()==inri::Xt::Time) {
				sampleUnit = SampleUnit::TIME;
			} else if (xt.axis()==inri::Xt::Depth) {
				sampleUnit = SampleUnit::DEPTH;
			} else {
				sampleUnit = SampleUnit::NONE;
			}

			isCubeCompatible = nbTraces==dataset->width() && nbProfiles==dataset->depth() && oriProfiles==addon.getFirstInline() &&
					pasProfiles==addon.getInlineStep() && oriTraces==addon.getFirstXline() && pasTraces==addon.getXlineStep() &&
					sampleUnit==addon.getSampleUnit();
		}
	}
	return isCubeCompatible;
}

bool LayerSpectrumDialog::filterHorizon(FreeHorizon* horizon) {
	if (m_datasetS==nullptr && m_datasetT==nullptr) {
		return false;
	}

	Seismic3DAbstractDataset* refDataset = nullptr;

	// check dataset caracteristics
	QString seismicPath;
	//std::string surveyPath;
	if (m_datasetS!=nullptr) {
		seismicPath = QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetS)->path());
		refDataset = m_datasetS;

		//surveyPath = ;
	}
	QString rgtPath;
	if (m_datasetT!=nullptr) {
		rgtPath = QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetT)->path());
		refDataset = m_datasetT;
	}

	// get seismic used for extraction of horizon
	//std::string currentDatasetName = FreeHorizonManager::dataSetNameGet(horizon->path().toStdString());
	//QString horizonExtractionDatasetPath = QString::fromStdString(SismageDBManager::datasetPathFromDatasetFileNameAndSurveyPath(currentDatasetName, surveyPath));

	bool isCubeCompatible = false;//seismicPath.compare(horizonExtractionDatasetPath)==0 || rgtPath.compare(horizonExtractionDatasetPath)==0;

	// check compatibility on carte (inline & xline)
	if (!isCubeCompatible) {
		Seismic3DDataset* dataset = dynamic_cast<Seismic3DDataset*>(refDataset);
		CubeSeismicAddon addon = dataset->cubeSeismicAddon();

		std::size_t nbTraces, nbProfiles;
		float oriTraces, pasTraces, oriProfiles, pasProfiles;
		SampleUnit sampleUnit;

		FreeHorizonManager::PARAM params = FreeHorizonManager::dataSetGetParam(horizon->path().toStdString()+"/"+FreeHorizonManager::isoDataName);

		// size match the iso as a single inline
		nbTraces = params.dimx;
		nbProfiles = params.dimy;
		// start and step match ori dataset
		oriTraces = params.startRecord;
		pasTraces = params.stepRecords;
		oriProfiles = params.startSlice;
		pasProfiles = params.stepSlices;
		if (params.axis==inri::Xt::Time) {
			sampleUnit = SampleUnit::TIME;
		} else if (params.axis==inri::Xt::Depth) {
			sampleUnit = SampleUnit::DEPTH;
		} else {
			sampleUnit = SampleUnit::NONE;
		}

		isCubeCompatible = nbTraces==dataset->width() && nbProfiles==dataset->depth() && oriProfiles==addon.getFirstInline() &&
				pasProfiles==addon.getInlineStep() && oriTraces==addon.getFirstXline() && pasTraces==addon.getXlineStep() &&
				sampleUnit==addon.getSampleUnit();
	}
	return isCubeCompatible;
}


//
void LayerSpectrumDialog::set_progressbar_values(double val, double valmax)
{
	this->progressbar_val = val;
	this->progressbar_valmax = valmax;
	// this->pthread->set_progressbar_values(val, valmax);
}



void LayerSpectrumDialog::trt_compute()
{
	// this->updateData(true);
	if ( pthread == nullptr )
	{
		pthread = new MyThread_LayerSpectrumDialogCompute();
		connect(pthread, &MyThread_LayerSpectrumDialogCompute::finished, this, &LayerSpectrumDialog::postComputeStep);
	}

	pthread->pspectrum_dialog = this;
	pthread->start();
}


void LayerSpectrumDialog::showTime()
{
	if ( pthread == NULL ) return;
	int type = 0;
	long idx = 0;
	long vmax = 1;
	char txt[1000];
	int ret = ihm_get_global_msg(&type, &idx, &vmax, txt);
	int val = 0;
	if ( ret == 1 )
	{
		val = (int)(100.0*idx/(double)vmax);
	}
	else
	{
		val = (int)(100.0*progressbar_val/progressbar_valmax);
	}
	m_progressbar->setValue(val);
}

long LayerSpectrumDialog::getMeanTauFromMultiSeedHorizon() const {
	long tau = 0;
	long N = 0;

	if (m_horizon!=nullptr && m_originViewer!=nullptr) {
		// get all sections with MultiSeedSliceRep and defined SliceReps
		QVector<AbstractInnerView *> views = m_originViewer->getInnerViews();
		QList<AbstractSectionView*> sections;

		for (AbstractInnerView* view : views) {
			if (AbstractSectionView* section = dynamic_cast<AbstractSectionView*>(view)) {
				const QList<AbstractGraphicRep*>& reps = view->getVisibleReps();
				bool isValid = false;
				long index = 0;
				while (!isValid && index<reps.count()) {
					MultiSeedSliceRep* horizonRep = dynamic_cast<MultiSeedSliceRep*>(reps[index]);
					isValid = reps[index]->data()==m_horizon && horizonRep!=nullptr;
					if (isValid) {
						tau += horizonRep->getMeanTauOnCurve(&isValid);
						N++;
					}
					index++;
				}
			}
		}
	}
	if (N>0) {
		tau = tau / N;
	}

	return tau;
}

std::size_t LayerSpectrumDialog::nextDatasetIndex() const {
	return m_nextDatasetIndex++;
}


void LayerSpectrumDialog::trt_rgtModifications()
{
	/*
	fprintf(stderr, "start rgt modifications\n");
	if ( m_rgtVolumicDialog == nullptr )
	{
		m_rgtVolumicDialog = new RgtVolumicDialog(nullptr, 0, nullptr, 0, nullptr);
		m_rgtVolumicDialog->show();
		m_rgtVolumicDialog->setLayerSpectrumDialog(this);
		m_horizon->setRgtPickingMode(RGTPICKINGOK);
		m_horizon->setRgtVolumicDialog(m_rgtVolumicDialog);
	}
	*/
}

void LayerSpectrumDialog::newLayerAdded(IData *data) {
	IJKHorizon* ijkHorizon = dynamic_cast<IJKHorizon*>(data);
	if (ijkHorizon==nullptr) {
		newFreeHorizonAdded(data);
		return;
	}

	bool isCubeCompatible = filterHorizon(ijkHorizon);

	if (isCubeCompatible) {
		long id = m_horizonNextId++;
		m_horizonNames[id] = ijkHorizon->name();
		m_horizonPaths[id] = ijkHorizon->path();
		m_horizonDatas[id] = nullptr;

		QFileInfo baseFileInfo(ijkHorizon->path());
		QString colorFilePath = baseFileInfo.dir().absoluteFilePath(baseFileInfo.baseName() + "_color.txt");
		bool colorValid;
		QColor color = FixedLayerFromDataset::loadColorFromFile(colorFilePath, &colorValid);
		if (!colorValid) {
			color = Qt::blue;
		}
		m_referenceList->addItem(ijkHorizon->name(), color, (qlonglong)id);
	}
}

void LayerSpectrumDialog::newFreeHorizonAdded(IData *data) {
	FreeHorizon* freeHorizon = dynamic_cast<FreeHorizon*>(data);
	if (freeHorizon==nullptr) {
		return;
	}

	bool isCubeCompatible = filterHorizon(freeHorizon);

	if (isCubeCompatible) {
		long id = m_horizonNextId++;
		m_horizonNames[id] = freeHorizon->name();
		m_horizonPaths[id] = "";
		m_horizonDatas[id] = freeHorizon;

		QColor color = freeHorizon->color();
		m_referenceList->addItem(freeHorizon->name(), color, (qlonglong)id);
	}
}

Seismic3DAbstractDataset *LayerSpectrumDialog::getDataSetS()
{
	return m_datasetS;
}

LayerSlice* LayerSpectrumDialog::getMdata()
{
	return m_data;
}

std::unique_ptr<FixedLayerFromDataset> LayerSpectrumDialog::getMConstrainData()
{
	return nullptr; // (std::unique_ptr<FixedLayerFromDataset>) m_constrainData;
}

std::vector<RgtSeed> LayerSpectrumDialog::getSeeds()
{
	return m_seeds;
}

// ==================================================================================
float *LayerSpectrumDialog::getHorizonFromSeed()
{
	float *ret = nullptr;
	if ( m_seeds.size()==0) {
		return ret;
	}
	QFileInfo dataFileInfo(QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetS)->path()));
	QDir surveyDir = dataFileInfo.absoluteDir();
	QDir horizonDir;
	QString saveName, saveRgbName;
	bool isValid = surveyDir.cdUp() && surveyDir.cdUp();
	bool saveInterpolate = false;
	bool saveRgb = false;
	bool isNameNew = false;

	if (!surveyDir.exists("ImportExport")) { isValid = surveyDir.mkdir("ImportExport"); }
	isValid = isValid && surveyDir.cd("ImportExport");
	if (isValid) {
		if (!surveyDir.exists("IJK")) {
			isValid = surveyDir.mkdir("IJK");
		}
		isValid = isValid && surveyDir.cd("IJK");
	}
	if (isValid) {
		QString seismicSismageName = DataSelectorDialog::getSismageNameFromSeismicFile(QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetS)->path()));
		QString rgtSismageName = DataSelectorDialog::getSismageNameFromSeismicFile(QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetT)->path()));
		horizonDir = surveyDir;
		if (!horizonDir.cd(seismicSismageName) && !horizonDir.cd(rgtSismageName))  {
			isValid = horizonDir.mkdir(seismicSismageName) && horizonDir.cd(seismicSismageName);
		}

		if (isValid && !horizonDir.exists("HORIZON_GRIDS")) {
			isValid = horizonDir.mkdir("HORIZON_GRIDS");
		}
		isValid = horizonDir.cd("HORIZON_GRIDS");
	}

	if (isValid) {
		saveInterpolate = true;
		saveRgb = false;
		isNameNew = true;
		saveName = "tmp";
		isValid = !saveName.isEmpty() && !saveName.isNull();
	}
	if (isValid) {
		QString horizonBaseName = saveName;
		saveRgbName = horizonDir.absoluteFilePath(saveName+".png");
		saveName = horizonDir.absoluteFilePath(saveName+".raw");
		m_horizon->applyDtauToSeeds();

		QString isoName("isochrone"), ampName("amplitude"), historyName("history");

		FixedLayerFromDataset saveProperties("", m_datasetS->workingSetManager(), dynamic_cast<Seismic3DAbstractDataset*>(m_datasetS));

		if (saveInterpolate && m_isDataInterpolated && m_constrainData!=nullptr) {
			std::vector<float> buffer;
			buffer.resize(saveProperties.width()*saveProperties.depth());

			m_constrainData->readProperty(buffer.data(), isoName);
			saveProperties.writeProperty(buffer.data(), isoName);

			m_constrainData->readProperty(buffer.data(), ampName);
			saveProperties.writeProperty(buffer.data(), ampName);
		} else {
			std::vector<float> inputLayer;
			inputLayer.resize(saveProperties.width()*saveProperties.depth(), -9999);
			if (m_constrainData!=nullptr) {
				m_constrainData->readProperty(inputLayer.data(), isoName);
			}

			if (saveInterpolate) {
				std::vector<float> outputLayerIso, outputLayerSeismic;
				outputLayerIso.resize(saveProperties.width()*saveProperties.depth());
				outputLayerSeismic.resize(saveProperties.width()*saveProperties.depth());
				if (m_isComputationValid && m_data->isModuleComputed()) {
					const float* rgtBuf = m_data->getModuleData(0);
					const float* seismicBuf = m_data->getModuleData(1);

					float tdeb = saveProperties.getOriginSample();
					float pasech = saveProperties.getStepSample();

					for (std::size_t index=0; index<outputLayerIso.size(); index++) {
						outputLayerIso[index] = rgtBuf[index] * pasech + tdeb;
						outputLayerSeismic[index] = seismicBuf[index];
					}

				} else {
					float tdeb = saveProperties.getOriginSample();
					float pasech = saveProperties.getStepSample();
					//					io::Cube* rgt = dynamic_cast<io::Cube*>(io::openCube<float>(rgtVisual->getDataset()->getDataPath().toStdString()));
					//					io::Cube* seismic = dynamic_cast<io::Cube*>(io::openCube<float>(seismicVisual->getDataset()->getDataPath().toStdString()));

					// interpolate with rgt
					SampleTypeBinder binder(m_datasetT->sampleType());
					binder.bind<LayerRGTInterpolatorKernel>(m_datasetS->sampleType(), inputLayer, m_horizon->getDTauReference(), outputLayerIso, outputLayerSeismic, m_referenceLayers, m_seeds,
							dynamic_cast<Seismic3DDataset*>(m_datasetT), m_channelT, dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS,
							m_useSnap, m_useMedian, m_halfLwxMedianFilter*2+1, m_distancePower,  m_snapWindow,  m_polarity, tdeb, pasech);
				}

				float tdeb = saveProperties.getOriginSample();
				float pasech = saveProperties.getStepSample();
				long size0 = saveProperties.width()*saveProperties.depth();
				float *tmp = outputLayerIso.data();
				ret = new float[size0];
				for (long add=0; add<size0; add++)
					ret[add] = ( tmp[add] - tdeb ) / pasech;
				//				std::vector<float>& iso = m_propagator->getIsochroneTab();
				//				tmp = iso.data();
				//				for (long add=0; add<10; add++)
				//					fprintf(stderr, ">>> %f\n", tmp[add]);
				return ret;


				// saveProperties.writeProperty(outputLayerIso.data(), isoName);
				// saveProperties.writeProperty(outputLayerSeismic.data(), ampName);
			} else {
				bool isPropagatorValid = m_propagator!=nullptr;
				if (isPropagatorValid) {
					std::vector<float>& amplitudeTab = m_propagator->getAmplitudeTab();
					// isPropagatorValid = amplitudeTab.size()==saveProperties.width()*saveProperties.depth();
				}
				if (isPropagatorValid) {
					std::vector<float>& amplitudeTab = m_propagator->getAmplitudeTab();
					// saveProperties.writeProperty(amplitudeTab.data(), ampName);
				} else {
					std::vector<float> amplitudeTab;
					amplitudeTab.resize(saveProperties.width()*saveProperties.depth(), 0);
					// saveProperties.writeProperty(amplitudeTab.data(), ampName);
				}

				// saveProperties.writeProperty(inputLayer.data(), isoName);
			}
		}


		bool isPropagatorValid = m_propagatorUndoCache.size()==saveProperties.width()*saveProperties.depth();
		if (isPropagatorValid) {
			std::vector<unsigned int>& historyLayer = m_propagatorUndoCache;
			std::vector<float> buffer;
			buffer.resize(historyLayer.size());
			for (std::size_t index=0; index<buffer.size(); index++) {
				buffer[index] = historyLayer[index];
			}
			saveProperties.writeProperty(buffer.data(), historyName);
		} else {
			std::vector<float> historyLayer;
			historyLayer.resize(saveProperties.width()*saveProperties.depth(), 0);
			saveProperties.writeProperty(historyLayer.data(), historyName);
		}

		QString filenameStd = saveName;
		QString filenameAmp(filenameStd+"_amp.pickingType");
		QString filenameHistory(filenameStd+"_history.pickingType");
		qDebug() << "save name " << saveName;
		saveProperties.saveProperty(filenameStd, isoName);
		saveProperties.saveProperty(filenameAmp, ampName);
		saveProperties.saveProperty(filenameHistory, historyName);
		if (saveRgb) {
			saveRGB(saveRgbName);
		}

		if (isNameNew) {
			IJKHorizon* horizon = new IJKHorizon(horizonBaseName, filenameStd, QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetS)->path()),
					m_datasetS->workingSetManager());

			m_datasetS->workingSetManager()->addIJKHorizon(horizon);

			//			// add to internal list
			//			m_horizonNames.push_back(horizon->name());//(*horizonNames)[i]);
			//			m_horizonPaths.push_back(horizon->path());//(*horizonPaths)[i]);
			//
			//			// apply to m_referenceList
			//			QListWidgetItem* item = new QListWidgetItem(horizon->name());
			//			item->setData(Qt::UserRole, horizon->path());
			//			m_referenceList->addItem(item);
		}
	}

}

void LayerSpectrumDialog::choosedPicksFromQC(QList<WellBore*> choosenWells, QList<int> geotimes, QList<int> mds) {
	if (!m_horizon || choosenWells.count()==0 || choosenWells.count()!=mds.count() || geotimes.count()!=mds.count()) {
		return;
	}

	std::vector<RgtSeed> newSeeds;

	CubeSeismicAddon addon = m_datasetT->cubeSeismicAddon();
	for (std::size_t i=0; i<choosenWells.count(); i++) {
		std::pair<RgtSeed, bool> pair = WellPick::getProjectionOnDataset(
				m_datasetS, addon.getSampleUnit(), choosenWells[i], mds[i], WellUnit::MD);
		RgtSeed seed = pair.first;
		bool ok = pair.second;

		if (ok) {
			seed.rgtValue = geotimes[i];
			seed.seismicValue = 0;

			// add min/max to avoid having a seed outside of the volume bounds
			if (seed.x<0) {
				seed.x = 0;
			} else if (seed.x>=m_datasetS->height()) {
				seed.x = m_datasetS->height()-1;
			}
			if (seed.y<0) {
				seed.y = 0;
			} else if (seed.y>=m_datasetS->width()) {
				seed.y = m_datasetS->width()-1;
			}
			if (seed.z<0) {
				seed.z = 0;
			} else if (seed.z>=m_datasetS->depth()) {
				seed.z = m_datasetS->depth()-1;
			}

			newSeeds.push_back(seed);
		}
	}

	if (newSeeds.size()>0) {
		m_seeds = newSeeds;
		m_horizon->setSeeds(newSeeds);
		m_isComputationValid = false;
	} else {
		qDebug() << "LayerSpectrumDialog::choosedPicksFromQC empty seed list, ignore new list";
	}
}

std::pair<bool, LayerSlice*> LayerSpectrumDialog::searchFirstValidDataFromMethod(int method) {
	LayerSlice* firstFoundData = nullptr;

	bool foundDataIsInWorkingSet = false;
	QList<QListWidgetItem*> selection = m_sismiqueCombo->selectedItems();
	int i=0;
	while (!foundDataIsInWorkingSet && i<selection.count()) {
		QListWidgetItem* item = selection[i];
		bool ok;
		std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);

		std::map<std::size_t, Seismic3DDataset*>::iterator it;
		if (ok) {
			it = m_allDatasets.find(id);
			ok = it!=m_allDatasets.end();
		}

		Seismic3DDataset* dataset = nullptr;
		if (ok) {
			dataset = it->second;
			ok = m_mapData.contains(dataset);
		}

		QVector<IData*> vectorData;
		if (ok) {
			vectorData =  m_mapData[dataset];
			ok = method<vectorData.size();
		}

		LayerSlice* data = nullptr;
		if (ok) {
			data = dynamic_cast<LayerSlice*>(vectorData[method]);
			ok = data!=nullptr;
		}

		if (ok) {
			foundDataIsInWorkingSet = data->getDatasetS()->workingSetManager()->containsLayerSlice(data);
			if (foundDataIsInWorkingSet || firstFoundData==nullptr) {
				firstFoundData = data;
			}
		}
		i++;
	}

	return std::pair<bool, LayerSlice*>(foundDataIsInWorkingSet, firstFoundData);
}

void LayerSpectrumDialog::methodChanged(int newIndex) {
	m_method = newIndex;
	m_VectorData.clear();

	if (m_method == eComputeMethd_Anisotropy) {
		std::pair<bool, LayerSlice*> foundData = searchFirstValidDataFromMethod(m_method);
		LayerSlice* firstFoundData = foundData.second;

		bool foundDataIsInWorkingSet = foundData.first;

		if (firstFoundData!=nullptr) {
			m_data = firstFoundData;
			m_VectorData.push_back(firstFoundData);
			if (!foundDataIsInWorkingSet) {
				firstFoundData->getDatasetS()->workingSetManager()->addLayerSlice(firstFoundData);
			}

			if(firstFoundData->ProcessDeletion() == true){
				firstFoundData->setProcessDeletion(false);
			}
		}
	} else {
		QList<QListWidgetItem*> selection = m_sismiqueCombo->selectedItems();
		for (QListWidgetItem* item : selection) {
			bool ok;
			std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);
			if (ok) {
				Seismic3DDataset* dataset = m_allDatasets[id];
				QVector<IData*>  vectorData =  m_mapData[dataset];
				m_data = dynamic_cast<LayerSlice*>(vectorData[m_method]);
				m_VectorData.push_back(m_data);

				m_data->getDatasetS()->workingSetManager()->addLayerSlice(m_data);

				if(m_method == eComputeMethd_Spectrum){
					m_dataRGB = dynamic_cast<RGBLayerSlice*>(vectorData[eComputeMethd_RGB]);
					m_data->getDatasetS()->workingSetManager()->addRGBLayerSlice(m_dataRGB);
				}
				if(m_data->ProcessDeletion() == true){
					m_data->setProcessDeletion(false);
				}
			}
		}
	}
}

template<typename InputType>
struct UpdateSeedsSeismicValueKernel {
	static void run(Seismic3DDataset* seismic, int channelS,
			std::vector<RgtSeed>& seeds) {
		if (seeds.size()==0) {
			return;
		}

		long dimx = seismic->height();
		long dimy = seismic->width();
		long dimz = seismic->depth();

		std::vector<InputType> valTab;
		valTab.resize(seismic->dimV());
		for (RgtSeed& seed : seeds) {
			seismic->readSubTraceAndSwap(valTab.data(), seed.x, seed.x+1, seed.y, seed.z);
			seed.seismicValue = valTab[channelS];
		}
	}
};

/*
template<typename InputType>
struct UpdateSeismicBufferKernel {
	static void run(Seismic3DDataset* datasetS, int channelS,
					double tdeb, double pasech, FixedLayerFromDataset* layer, QString ampName, QString isoName) {
		std::vector<float> ampVect, isoVect;
		float nullValue = -9999.0f;

		std::size_t dimI = layer->getNbProfiles();
		std::size_t dimJ = layer->getNbTraces();

		ampVect.resize(dimI*dimJ);
		isoVect.resize(dimI*dimJ);

		layer->readProperty(isoVect.data(), isoName);
		layer->readProperty(ampVect.data(), ampName);

		std::vector<InputType> valTab;
		valTab.resize(datasetS->dimV());

		for (std::size_t i=0; i<dimI; i++) {
			for (std::size_t j=0; j<dimJ; j++) {
				std::size_t idx = j + i * dimJ;
				if (isoVect[idx]!=nullValue && ampVect[idx]==nullValue) {
					int x = (isoVect[idx] - tdeb) / pasech;
					if (x>=0 && x<datasetS->height()) {
						datasetS->readSubTraceAndSwap(valTab.data(), x, x+1, j, i);
						ampVect[idx] = valTab[channelS];
					}
				}
			}
		}

		layer->writeProperty(ampVect.data(), ampName);
	}
};
*/

void LayerSpectrumDialog::patchConstraintType(int type, std::vector<int> &vy, std::vector<int> &vz)
{
	if (m_constrainData==nullptr) {
		if (m_propagator==nullptr) {
			SampleTypeBinder binder(m_datasetS->sampleType());
			m_propagator.reset(binder.bind<CreatePropagatorObjectKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS));
		}

		std::vector<int>& tabSeed = m_propagator->getTabSeedType();
		std::vector<float>& tabIso = m_propagator->getIsochroneTab();
		std::vector<float>& tabAmp = m_propagator->getAmplitudeTab();
		tabSeed.clear();
		tabSeed.resize(m_datasetS->width()*m_datasetS->depth(), 0);
		tabIso.clear();
		tabIso.resize(m_datasetS->width()*m_datasetS->depth(), -9999);
		tabAmp.clear();
		tabAmp.resize(m_datasetS->width()*m_datasetS->depth(), -9999);

		updatePropagationExtension();
	}
	else
	{
		std::vector<int>& tabSeed = m_propagator->getTabSeedType();
		std::vector<float>& tabIso = m_propagator->getIsochroneTab();
		std::vector<float>& tabAmp = m_propagator->getAmplitudeTab();
		// tabSeed.clear();
		tabSeed.resize(m_propagator->mapWidth()*m_propagator->mapHeight(), 0);
		// tabIso.clear();
		tabIso.resize(m_propagator->mapWidth()*m_propagator->mapHeight(), -9999);
		// tabAmp.clear();
		tabAmp.resize(m_propagator->mapWidth()*m_propagator->mapHeight(), -9999);
	}
	PatchCompositionProcess patchProcess;
	patchProcess.setSeeds(m_patchSeeds);
	patchProcess.setTabIso(m_patchTabIso);
	patchProcess.setOutputLayer(m_constrainData.get());

	Seismic3DDataset* selectedDataset = m_cachePatchDataset;
	if (selectedDataset==nullptr) {
		std::vector<Seismic3DDataset*> potentialVolumes;
		QList<IData*> datas = m_datasetT->workingSetManager()->folders().seismics->data();
		for (IData* potentialSurvey : datas) {
			SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(potentialSurvey);
			if (survey!=nullptr) {
				for (Seismic3DAbstractDataset* dataset : survey->datasets()) {
					Seismic3DDataset* cpuDataset = dynamic_cast<Seismic3DDataset*>(dataset);
					if (cpuDataset!=nullptr && dataset->type()==Seismic3DAbstractDataset::CUBE_TYPE::Patch) {
						//if ( cpuDataset->name().contains("nextvisionpatch") )
							potentialVolumes.push_back(cpuDataset);
					}
				}
			}
		}

		if (potentialVolumes.size()==1) {
			selectedDataset = potentialVolumes[0];
			if ( rgtGraphLabelRead == nullptr )
				rgtGraphLabelRead = new RgtGraphLabelRead(potentialVolumes[0]->path());
		} else if (potentialVolumes.size()>1) {
			QStringList nameList;
			for (Seismic3DDataset* dataset : potentialVolumes) {
				nameList << dataset->name();
				std::cout << dataset->path() << std::endl;
			}
			StringSelectorDialog dialog(&nameList, "Select Patch volume");
			int code = dialog.exec();
			if (code==QDialog::Accepted) {
				int selectedIdx = dialog.getSelectedIndex();
				if (selectedIdx>=0 && selectedIdx<potentialVolumes.size()) {
					selectedDataset = potentialVolumes[selectedIdx];
					if ( rgtGraphLabelRead == nullptr )
						rgtGraphLabelRead = new RgtGraphLabelRead(potentialVolumes[selectedIdx]->path());
				}
			}
		}
		m_cachePatchDataset = selectedDataset;
	}

	if (selectedDataset) {
		short seismicThreshold = (short)(m_patchThreshold->text().toInt());
		bool ok = patchProcess.setPatchVolume(selectedDataset, 0);
		patchProcess.setLabelReader(rgtGraphLabelRead);
		ok = ok && patchProcess.compute(type, seismicThreshold, vy, vz);
		if (!ok) {
			// QMessageBox::warning(this, "Patch constrain", "Failed to use the volume to patch the constrain (" + selectedDataset->name() + ")");
		} else {
			std::vector<float>& tabIso = m_propagator->getIsochroneTab();
			std::vector<float>& tabAmp = m_propagator->getAmplitudeTab();
			m_constrainData->readProperty(tabIso.data(), FixedLayerFromDataset::ISOCHRONE);
			QString ampName("amplitude");
			m_constrainData->readProperty(tabAmp.data(), ampName);

			updatePropagationExtension();

			std::vector<RgtSeed> newSeeds = m_propagator->extractSeedsFromTabs(m_seedFilterNumber);
			fprintf(stderr, "newSeeds size: %d\n", newSeeds.size());

			int correctPolarity = m_polarity;

			QString isoName("isochrone");
			QString rgtName("rgt");

			float tdeb = m_constrainData->getOriginSample();
			float pasech = m_constrainData->getStepSample();

			SampleTypeBinder binderT(m_datasetT->sampleType());
			binderT.bind<UpdateSeedsRgtValueKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetT), m_channelT,
					m_referenceLayers, isoName, rgtName, newSeeds, tdeb, pasech);

			SampleTypeBinder binderS(m_datasetS->sampleType());
			binderS.bind<UpdateSeedsSeismicValueKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS,
					newSeeds);

			binderS.bind<UpdateSeismicBufferKernel>(dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS,
					tdeb, pasech, m_constrainData.get(), ampName, FixedLayerFromDataset::ISOCHRONE);

			std::vector<int>& tabType = m_propagator->getTabSeedType();
			for (std::size_t i=0; i<tabType.size(); i++) {
				if (tabType[i]==0 && tabIso[i]!=-9999.0f) {
					tabType[i] = 1;
				}
			}

			setSeeds(newSeeds);

			setPolarity(correctPolarity);
			if (m_horizon!=nullptr) {
				m_horizon->setPolarity(m_polarity);
			}

			updatePropagationView();
			m_undoPropagationButton->show();
			propagationCacheSetup();
		}
	} else if (!m_constrainMissingWarned) {
		// Comment warning message box and replace by a qDebug because only few users need this message, most do not use patch files
		//QMessageBox::warning(this, "Patch constrain", "Failed to find a volume to patch the constrain");
		qDebug() << "Failed to find a volume to patch the constrain";
		m_constrainMissingWarned = true;
	}
}

void LayerSpectrumDialog::patchConstrain() {
	std::vector<int> vy;
	std::vector<int> vz;
	patchConstraintType(0, vy, vz);
}

void LayerSpectrumDialog::patchNeightbourConstrain()
{
	std::vector<int> vy;
	std::vector<int> vz;
	patchConstraintType(1, vy, vz);
}


void LayerSpectrumDialog::erasePatch() {
	st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();
	bool valid = st_GraphicSettings.pActiveScene != nullptr && m_constrainData!=nullptr;
	Abstract2DInnerView* innerView = nullptr;
	if (valid)
	{
		innerView = st_GraphicSettings.pActiveScene->innerView();
		valid = innerView!=nullptr && (innerView->viewType()==BasemapView || innerView->viewType()==StackBasemapView);
	}

	QList<QGraphicsItem*> selectedItems;
	std::vector<float> tabFloat;
	if (valid) {
		// get items
		selectedItems = st_GraphicSettings.pActiveScene->selectedItems();

		valid = selectedItems.size()>0;
	}

	std::vector<QPoint> points;
	if (valid) {
		// get points in ij
		const Affine2DTransformation* transfo = m_constrainData->dataset()->ijToXYTransfo();
		for (QGraphicsItem* item : selectedItems) {
			std::vector<QPoint> fromItemPoints = iGraphicToolDataControl::getDeletePointsOnGrid(transfo, item);
			points.insert(points.end(), fromItemPoints.begin(), fromItemPoints.end());
		}

		valid = points.size()>0;
	}

	if ( valid )
	{
		std::vector<int> Y(points.size());
		std::vector<int> Z(points.size());
		for (int n=0; n<points.size(); n++) {
			Y[n] = points[n].x();
			Z[n] = points[n].y();
		}
		patchConstraintType(2, Y, Z);
	}
}

void LayerSpectrumDialog::eraseParentPatch(Abstract2DInnerView* inner2DView, QPointF pt) {
	double imageX, imageY, worldX, worldY;
	worldX = pt.x();
	worldY = pt.y();
	if (inner2DView->viewType()==ViewType::InlineView) {
		dynamic_cast<Seismic3DDataset*>(m_datasetS)->ijToInlineXlineTransfoForInline()->worldToImage(worldX, worldY, imageX, imageY);
	} else {
		dynamic_cast<Seismic3DDataset*>(m_datasetS)->ijToInlineXlineTransfoForXline()->worldToImage(worldX, worldY, imageX, imageY);
	}

	fprintf(stderr, "worldx: %f\nworldy: %f\nimagex: %f\nimagey: %f", worldX, worldY, imageX, imageY);
}

void LayerSpectrumDialog::reloadHorizonList() {
	m_horizonNames.clear();
	m_horizonPaths.clear();
	m_horizonDatas.clear();
	if (m_datasetT==nullptr && m_datasetS==nullptr) {
		return;
	}

	WorkingSetManager *wm = nullptr;
	if (m_datasetS!=nullptr) {
		wm = m_datasetS->workingSetManager();
	} else {
		wm = m_datasetT->workingSetManager();
	}

	QList<IData*> datas = wm->folders().horizonsFree->data();
	for (int i=0; i<datas.size(); i++) {
		IJKHorizon* horizon = dynamic_cast<IJKHorizon*>(datas[i]);
		FreeHorizon* freeHorizon = dynamic_cast<FreeHorizon*>(datas[i]);
		if (horizon!=nullptr) {
			bool isCubeCompatible = filterHorizon(horizon);

			if (isCubeCompatible) {
				long id = m_horizonNextId++;
				m_horizonNames[id] = horizon->name();//(*horizonNames)[i]);
				m_horizonPaths[id] = horizon->path();//(*horizonPaths)[i]);
				m_horizonDatas[id] = nullptr;
			}
		} else if (freeHorizon!=nullptr) {
			bool isCubeCompatible = filterHorizon(freeHorizon);

			if (isCubeCompatible) {
				long id = m_horizonNextId++;
				m_horizonNames[id] = freeHorizon->name();//(*horizonNames)[i]);
				m_horizonPaths[id] = "";//(*horizonPaths)[i]);
				m_horizonDatas[id] = freeHorizon;
			}
		}
	}

	initReferenceComboBox();
}

void LayerSpectrumDialog::addDataset(Seismic3DAbstractDataset* dataset) {
	if (!m_datasetT->isCompatible(dataset)) {
		return;
	}

	Seismic3DDataset* cpuDataset = dynamic_cast<Seismic3DDataset*>(dataset);
	if (  (cpuDataset!=nullptr) && (cpuDataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::Seismic)
			&& (cpuDataset->dimV()==1)){

		std::size_t id = nextDatasetIndex();
		m_allDatasets[id] = cpuDataset;

		QListWidgetItem* item = new QListWidgetItem;
		item->setText(cpuDataset->name());
		item->setData(Qt::UserRole, (qulonglong) id);
		m_sismiqueCombo->addItem(item);

		if (m_datasetS==nullptr && m_horizon==nullptr) {
			m_datasetS = cpuDataset;
			item->setSelected(true);
			setupHorizonExtension();
		} else if (m_datasetS==dataset) {
			item->setSelected(true);
		}
	}

	if ((cpuDataset!=nullptr) && (cpuDataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::Patch)) {
		std::size_t id = nextDatasetIndex();
		m_allDatasets[id] = cpuDataset;
		QListWidgetItem* patchItem = new QListWidgetItem;
		patchItem->setText(cpuDataset->name());
		patchItem->setData(Qt::UserRole, (qulonglong) id);
		m_patchList->addItem(patchItem);
		m_constrainMissingWarned = false;
	}
}

std::vector<Seismic3DAbstractDataset*> LayerSpectrumDialog::getSelectedDatasetsInView(GeotimeGraphicsView* view) {
	std::vector<Seismic3DAbstractDataset*> output;

	QVector<AbstractInnerView*> innerViews = view->getInnerViews();
	std::vector<ViewType> allowedTypes = {ViewType::InlineView, ViewType::XLineView, ViewType::RandomView};

	for (AbstractInnerView* innerView : innerViews) {
		bool typeValid = false;
		int iType = 0;
		while (!typeValid) {
			typeValid = allowedTypes[iType]==innerView->viewType();
			iType++;
		}

		if (!typeValid) {
			continue;
		}

		const QList<AbstractGraphicRep*>& visibleReps = innerView->getVisibleReps();
		for (AbstractGraphicRep* rep : visibleReps) {
			Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(rep->data());
			if (dataset==nullptr) {
				continue;
			}

			std::vector<Seismic3DAbstractDataset*>::iterator it = std::find(output.begin(), output.end(), dataset);
			if (it==output.end()) {
				output.push_back(dataset);
			}
		}
	}

	return output;
}


void LayerSpectrumDialog::updatePatchDataSet() {
	m_cachePatchDataset = nullptr;
	QList<QListWidgetItem*> selection = m_patchList->selectedItems();
	for (QListWidgetItem* item : selection) {
		bool ok;
		std::size_t id = item->data(Qt::UserRole).toULongLong(&ok);
		if (ok) {
			Seismic3DDataset* dataset = m_allDatasets[id];
			m_cachePatchDataset = dataset;
		}
	}
}

void LayerSpectrumDialog::createConstrain() {
	QString name("constrain");
	m_constrainData.reset(new FixedLayerFromDataset(name, m_datasetS->workingSetManager(),
			dynamic_cast<Seismic3DDataset*>(m_datasetS)));

	m_constrainData->setColor(Qt::blue);
	m_constrainData->toggleTemporaryData(true);
	connect(m_constrainData.get(), &QObject::destroyed, this, &LayerSpectrumDialog::releaseConstrainFromDestroyed);
}

void LayerSpectrumDialog::releaseConstrainFromDestroyed() {
	// this should only be called by the QObject::destroyed signal of the constrain
	m_constrainData.release();
}

void MyThread_LayerSpectrumDialogCompute::run()
{
	pspectrum_dialog->threadableCompute();
}

