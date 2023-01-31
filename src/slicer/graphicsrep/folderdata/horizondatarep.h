#ifndef HorizonDataRep_H
#define HorizonDataRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isampledependantrep.h"
#include "iGraphicToolDataControl.h"
#include "iCUDAImageClone.h"
#include "horizonfolderdata.h"

#include "cudargbinterleavedimage.h"
#include "cpuimagepaletteholder.h"
class CUDAImagePaletteHolder;
class CUDARGBImage;
class QGraphicsObject;

class HorizonFolderData;
class HorizonPropPanel;
class HorizonFolder3DLayer;

class FixedRGBLayersFromDatasetAndCubePropPanel;
class FixedRGBLayersFromDatasetAndCube3DLayer;
class FixedRGBLayersFromDatasetAndCubeLayerOnMap;
class FixedRGBLayersFromDatasetAndCube;
class HorizonFolderLayerOnMap;
class HorizonFolderLayerOnSlice;

class HorizonAnimInformation;
class QGLLineItem;

//Representation d'une slice RGB
class HorizonDataRep: public AbstractGraphicRep,
		public IMouseImageDataProvider, public ISampleDependantRep,
		public iGraphicToolDataControl, public iCUDAImageClone {
Q_OBJECT
public:


	struct HorizonAnimParams
	{
		QString attribut;
		int nbHorizons;
		bool lockPalette;
		bool paletteRGB;
		QVector2D rangeR;
		QVector2D rangeG;
		QVector2D rangeB;

		QStringList horizons;
		std::vector<int> orderIndex;
	};

	HorizonDataRep(HorizonFolderData *layer, AbstractInnerView *parent = 0);
	virtual ~HorizonDataRep();

	HorizonFolderData* horizonFolderData() const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;
	Graphic3DLayer* layer3D(QWindow *parent, Qt3DCore::QEntity *root,
			Qt3DRender::QCamera *camera) override;

	IData* data() const override;

	virtual void buildContextMenu(QMenu *menu) override;

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y, MouseInfo &info) override;
	virtual TypeRep getTypeGraphicRep() override;

	// iGraphicToolDataControl
	void deleteGraphicItemDataContent(QGraphicsItem *item) override;

	QGraphicsObject* cloneCUDAImageWithMask(QGraphicsItem *parent);


	void setBuffer();


	CUDARGBInterleavedImage* image()
	{
		return m_image;
	}

	CPUImagePaletteHolder* isoSurfaceHolder()
	{
		return m_isoSurfaceHolder;
	}

	void setNameAttribut(QString nom)
	{
		m_nameAttribut= nom;
	}

	QString getNameAttribut()
		{
			return m_nameAttribut;
		}

	FreeHorizon::Attribut currentLayerWithAttribut()
	{
		return m_data->currentLayerWithAttribut(m_nameAttribut);
	}



	//void showCache(int index);

	// the read is strictly controlling that the file is compatible
	static HorizonAnimParams readAnimationHorizon(QString ,bool*);
	static bool writeAnimationHorizon(QString path, HorizonAnimParams params);
	static void addAnimationHorizon(QString,QString,WorkingSetManager*);
	static void removeAnimationHorizon(QString,QString,WorkingSetManager*);
	static HorizonAnimInformation* newAnimationHorizon(WorkingSetManager* manager);
//	static QStringList getAttributesAvailable();


private slots:
	void addData();
	void addSismageHorizon();
	void computeAttributHorizon();
	void dataChanged();
	void computeCache();
	void clearCache();
	void showCache(int index);


protected:
	HorizonPropPanel *m_propPanel;
	HorizonFolder3DLayer *m_layer3D;
//	FixedRGBLayersFromDatasetAndCubeLayerOnMap *m_layer;
	HorizonFolderLayerOnMap* m_layer;

	HorizonFolderLayerOnSlice* m_layerSlice;

	CUDARGBInterleavedImage* m_image = nullptr;
	CPUImagePaletteHolder* m_isoSurfaceHolder= nullptr;

	QString m_nameAttribut;



	HorizonFolderData *m_data;

	//static QPointer<HorizonFolderData> m_dataAnim;
	SampleUnit m_sampleUnit;
};

#endif
