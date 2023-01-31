#ifndef WorkingSetManager_H
#define WorkingSetManager_H

#include <QObject>
class IData;
class SeismicSurvey;
class StratiSlice;
class LayerSlice;
class RGBLayerSlice;
class MultiSeedHorizon;
class RGBStratiSliceAttribute;
class FixedLayerFromDataset;
class FixedRGBLayersFromDataset;
class FixedRGBLayersFromDatasetAndCube;
class FixedLayersFromDatasetAndCube;
class IJKHorizon;
class WellHead;
class Marker;
class FolderData;
class HorizonFolderData;
class VideoLayer;
class RgbDataset;
class RgbLayerFromDataset;
class RgbComputationOnDataset;
class ScalarComputationOnDataset;
class GeotimeProjectManagerWidget;
class ProjectManagerWidget;
class GraphicTool_GraphicLayer;
class NurbsDataset;
class RandomDataset;
class ComputationOperatorDataset;
class FreeHorizon;
class IsoHorizon;

// class MultiSeedRgt;

class WorkingSetManager: public QObject {
Q_OBJECT
public:
	typedef struct FolderList {
		FolderData* seismics;
		FolderData* wells;
		FolderData* graphicLayers;
		FolderData* markers;
		FolderData* nurbs;
		FolderData* randoms;
		FolderData* horizonsIso;
		FolderData* horizonsFree;
		FolderData* horizonsAnim;
		FolderData* others;
	} FolderList;

	WorkingSetManager(QObject *parent = 0);
	virtual ~WorkingSetManager();
	void addSeismicSurvey(SeismicSurvey *survey);
	void addStratiSlice(StratiSlice *stratiSlice);
	void addLayerSlice(LayerSlice *stratiSlice);
	void addRGBLayerSlice(RGBLayerSlice *stratiSlice);
	void addMultiSeedHorizon(MultiSeedHorizon *horizon);
	void addRGBStratiSlice(RGBStratiSliceAttribute *stratiSlice);
	void addFixedLayerFromDataset(FixedLayerFromDataset* layer);
	void addIJKHorizon(IJKHorizon* horizon);
	void addWellHead(WellHead* wellHead);
	void addMarker(Marker* marker);
	void addFixedRGBLayersFromDataset(FixedRGBLayersFromDataset* layer);
	void addFixedRGBLayersFromDatasetAndCube(FixedRGBLayersFromDatasetAndCube* layer);
	void addFixedLayersFromDatasetAndCube(FixedLayersFromDatasetAndCube* layer);
	// void addHorizonsFromDirectories(FixedRGBLayersFromDatasetAndCube* layer);
	void addHorizonsIsoFromDirectories(FixedRGBLayersFromDatasetAndCube* layer);
	void addHorizonsFreeFromDirectories(FixedRGBLayersFromDatasetAndCube* layer);

	void addFreeHorizons(FreeHorizon* layer);
	void addIsoHorizons(IsoHorizon* layer);
	void removeFreeHorizons(FreeHorizon* layer);
	void removeIsoHorizons(IsoHorizon* layer);

	void addFolderData(HorizonFolderData * horizonfolderdata);
	void removeFolderData(HorizonFolderData * horizonfolderdata);

	void addHorizonAnimData(HorizonFolderData * horizonfolderdata);
	void removeHorizonAnimData(HorizonFolderData * horizonfolderdata);

	bool containsFreeHorizon(QString path);
	FreeHorizon *getFreeHorizon(QString path);
	bool containsHorizonAnim(const QString& name);

	void addVideoLayer(VideoLayer* layer);
	void addRgbDataset(RgbDataset* rgbDataset);
	void addRgbLayerFromDataset(RgbLayerFromDataset* layer);
	void addRgbComputationOnDataset(RgbComputationOnDataset* dataset);
	void addScalarComputationOnDataset(ScalarComputationOnDataset* dataset);
	void addComputationOperatorDataset(ComputationOperatorDataset* dataset);
	void addGraphicLayer(GraphicTool_GraphicLayer* dataset);
	void addNurbs(NurbsDataset* dataset);
	void addRandom(RandomDataset* dataset);
	// void addMultiSeedRgt(MultiSeedRgt *rgt);
	void addNurbs(QString path, QString name);

	void removeSeismicSurvey(SeismicSurvey *survey);
	void removeStratiSlice(StratiSlice *stratiSlice);
	void removeLayerSlice(LayerSlice *stratiSlice);
	void deleteLayerSlice(LayerSlice *layerSlice);// MZR 17082021
	void removeRGBLayerSlice(RGBLayerSlice *stratiSlice);
	void deleteRGBLayerSlice(RGBLayerSlice *stratiSlice);// MZR 17082021
	void removeMultiSeedHorizon(MultiSeedHorizon *horizon);
	void removeRGBStratiSlice(RGBStratiSliceAttribute *stratiSlice);
	void removeFixedLayerFromDataset(FixedLayerFromDataset* layer);
	void removeIJKHorizon(IJKHorizon* horizon);
	void removeWellHead(WellHead* wellHead);
	void deleteWellHead(WellHead* wellHead);// MZR 17082021
	void removeMarker(Marker* marker);
	void deleteMarker(Marker* marker);// MZR 17082021
	void removeFixedRGBLayersFromDataset(FixedRGBLayersFromDataset* layer);
	void removeFixedRGBLayersFromDatasetAndCube(FixedRGBLayersFromDatasetAndCube* layer);
	void removeFixedLayersFromDatasetAndCube(FixedLayersFromDatasetAndCube* layer);
	void removeVideoLayer(VideoLayer* layer);
	void removeRgbDataset(RgbDataset* rgbDataset);
	void removeRgbLayerFromDataset(RgbLayerFromDataset* layer);
	void removeRgbComputationOnDataset(RgbComputationOnDataset* dataset);
	void removeComputationOperatorDataset(ComputationOperatorDataset* dataset);
	void removeGraphicLayer(GraphicTool_GraphicLayer* dataset);
	void removeNurbs(NurbsDataset* dataset);
	void removeRandom(RandomDataset* dataset);
	// void removeMultiSeedRgt(MultiSeedRgt *rgt);

	bool containsLayerSlice(LayerSlice *stratiSlice);

	QList<IJKHorizon*> listIJKHorizons();
	QList<WellHead*> listWellHead();

	QList<IData*> data(){return m_data;}
	FolderList folders() {return m_folders;}
	void setManagerWidget(GeotimeProjectManagerWidget* pDataManager);
	GeotimeProjectManagerWidget* getManagerWidget(){return m_DiaglogSelector;}

	void setManagerWidgetV2(ProjectManagerWidget* pDataManager) { m_DiaglogSelectorV2 = pDataManager; }
	ProjectManagerWidget* getManagerWidgetV2(){return m_DiaglogSelectorV2;}


	QList<QString> getDataset(RandomDataset* dataset);
private:
	void addData(IData *data);
	void removeData(IData *data);

signals:
	void dataAdded(IData *d);
	void dataRemoved(IData *d);
protected:
	QList<IData*> m_data;
	FolderList m_folders;
	GeotimeProjectManagerWidget* m_DiaglogSelector = nullptr;
	ProjectManagerWidget* m_DiaglogSelectorV2 = nullptr;
};

#endif
