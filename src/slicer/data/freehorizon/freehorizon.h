#ifndef __FREEHORIZON__
#define __FREEHORIZON__

#include <QObject>
#include <QPointer>
#include <QVector2D>
#include <QList>
#include <QColor>
#include <vector>

#include <fixedrgblayersfromdatasetandcube.h>
#include <fixedlayerfromdataset.h>
#include "itreewidgetitemdecoratorprovider.h"

class SeismicSurvey;
#include "idata.h"
#include "imageformats.h"
#include "viewutils.h"
// #include "RgtLayerProcessUtil.h"

class FreeHorizonGraphicRepFactory;
class FreeHorizonAttribut;
class AffineTransformation;
class Affine2DTransformation;
class FixedLayerImplFreeHorizonFromDataset;
class IconTreeWidgetItemDecorator;
class RGBLayerImplFreeHorizonOnSlice;
class SurfaceMeshCache;
class FixedLayerImplFreeHorizonFromDatasetAndCube;

class FreeHorizon: public IData, public ITreeWidgetItemDecoratorProvider {
Q_OBJECT

public:
class Attribut
{

protected:
	FixedRGBLayersFromDatasetAndCube *pFixedRGBLayersFromDatasetAndCube = nullptr;

	FixedLayerImplFreeHorizonFromDataset *pFixedLayerFromDataset = nullptr;
	RGBLayerImplFreeHorizonOnSlice *pRGBLayerImplFreeHorizonOnSlice = nullptr;
	FixedLayerImplFreeHorizonFromDatasetAndCube *pFixedLayersImplFreeHorizonFromDatasetAndCube = nullptr;

public:
	QString name() const;
	IData* getData() const;
	const AffineTransformation* sampleTransformation() const;
	CubeSeismicAddon cubeSeismicAddon() const;
	const Affine2DTransformation* ijToXYTransfo() const;
	bool isIsoInT() const;
	bool isIndexCache(int index) const;
	int currentImageIndex() const;
	SurfaceMeshCache* getMeshCache(int index) const;
	int getSimplifyMeshSteps() const;
	int getCompressionMesh()const;
	int width() const;
	int depth() const;
	int heightFor3D() const;
	ImageFormats::QSampleType imageType() const;
	ImageFormats::QSampleType isoType() const;
	void copyImageData(CUDARGBInterleavedImage* data);
	void copyIsoData(CPUImagePaletteHolder* data);


	const Affine2DTransformation* ijToInlineXlineTransfoForInline() const;
	const Affine2DTransformation* ijToInlineXlineTransfoForXline()const;

	CPUImagePaletteHolder* getIsoSurface();

	void setFixedRGBLayersFromDatasetAndCube(FixedRGBLayersFromDatasetAndCube * d){pFixedRGBLayersFromDatasetAndCube =d;}
	FixedRGBLayersFromDatasetAndCube *getFixedRGBLayersFromDatasetAndCube() {return pFixedRGBLayersFromDatasetAndCube;}

	void setFixedLayerFromDataset(FixedLayerImplFreeHorizonFromDataset * d){pFixedLayerFromDataset =d;}
	FixedLayerImplFreeHorizonFromDataset *getFixedLayerFromDataset() {return pFixedLayerFromDataset;}

	void setRGBLayerImplFreeHorizonOnSlice(RGBLayerImplFreeHorizonOnSlice * d){pRGBLayerImplFreeHorizonOnSlice =d;}
	RGBLayerImplFreeHorizonOnSlice *getRGBLayerImplFreeHorizonOnSlice() {return pRGBLayerImplFreeHorizonOnSlice;}

	void setFixedLayersImplFreeHorizonFromDatasetAndCube(FixedLayerImplFreeHorizonFromDatasetAndCube * d){pFixedLayersImplFreeHorizonFromDatasetAndCube =d;}
	FixedLayerImplFreeHorizonFromDatasetAndCube *getFixedLayersImplFreeHorizonFromDatasetAndCube() {return pFixedLayersImplFreeHorizonFromDatasetAndCube;}
};


public:
	FreeHorizon(WorkingSetManager * workingSet, SeismicSurvey *survey, const QString &path, const QString &name, QObject *parent = 0);
	virtual ~FreeHorizon();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}
	QString path() { return m_path; }

	// QList<WellPick*> getWellPickFromWell(WellBore* bore);
	// const QList<WellPick*>& wellPicks() const;
	QColor color() const;
	void setColor(const QColor& color);
	// QList<RgtSeed> getProjectedPicksOnDataset(Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit);

	const AffineTransformation* sampleTransformation() const;
	CubeSeismicAddon cubeSeismicAddon() const;

	const Affine2DTransformation* ijToXYTransfo(QString name) const;

	Attribut getLayer(QString name);

	// ITreeWidgetItemDecoratorProvider
	virtual ITreeWidgetItemDecorator* getTreeWidgetItemDecorator() override;

	bool isAttributExists(QString name);
	void freeHorizonAttributCreate(QString path);
	void freeHorizonAttributRemove(QString path);

	void openSismageExporter(QWidget* parent=0);

	IData* getIsochronData();

	Attribut attribut(int index) const;
	long numAttributs() const;

	SeismicSurvey* survey() const;
	/*
public slots:
	void addWellPick(WellPick* pick);
	void removeWellPick(WellPick* pick);

signals:
	void wellPickAdded(WellPick* pick);
	void wellPickRemoved(WellPick* pick);
	*/
	// std::vector<FreeHorizonAttribut*> m_attribut;
	// std::vector<FixedRGBLayersFromDatasetAndCube*> m_attribut;
	std::vector<Attribut> m_attribut;

signals:
	void colorChanged(QColor color);

	// only used by item decorator
	void iconChanged(QIcon icon);
	// void freeHorizonAdded();
	// void freeHorizonAttributAdded(Attribut *att);
	// void freeHorizonAttributAdded(FixedRGBLayersFromDatasetAndCube *data);
	void attributAdded(FreeHorizon::Attribut *data);
	void attributRemoved(FreeHorizon::Attribut *data);




private slots:
	void updateIcon(QColor color);

private:
	QString m_path = "";
	QString m_name;
	QUuid m_uuid;
	QObject *m_parent = nullptr;
	WorkingSetManager * m_workingSet = nullptr;
	SeismicSurvey *m_survey = nullptr;
	QPointer<FixedLayerImplFreeHorizonFromDatasetAndCube> m_isoData;

	FreeHorizonGraphicRepFactory* m_repFactory;
	// QList<WellPick*> m_wellPicks;
	QColor m_color;
	IconTreeWidgetItemDecorator* m_decorator;
	SampleUnit m_sampleUnit;
	void freeHorizonAttributCreate();
	int attributListRemove(QString name);


	QString findCompatibleDataSetForHorizon(QString horizonPath, QString dataSetPath);

};

#endif
