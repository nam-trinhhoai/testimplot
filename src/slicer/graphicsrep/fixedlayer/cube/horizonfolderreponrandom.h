#ifndef HorizonFolderRepOnRandom_H
#define HorizonFolderRepOnRandom_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "sliceutils.h"
#include "horizonfolderdata.h"

#include <QMenu>
#include <QAction>
// #include "GraphEditor_PolyLineShape.h"
#include "GraphEditor_MultiPolyLineShape.h"

class HorizonFolderPropPanelOnRandom;
class HorizonFolderLayerOnRandom;
class HorizonFolderData;
class FixedRGBLayersFromDatasetAndCube;
class IGeorefImage;

class HorizonFolderRepOnRandom : public AbstractGraphicRep,  public ISliceableRep, public ISampleDependantRep {
Q_OBJECT
public:
HorizonFolderRepOnRandom(HorizonFolderData *fixedLayer, AbstractInnerView *parent = 0);
	virtual ~HorizonFolderRepOnRandom();

	//FixedRGBLayersFromDatasetAndCube* fixedRGBLayersFromDataset() const;
	HorizonFolderData* horizonFolderData() const;

	void setSliceIJPosition(int imageVal) override;
	int currentIJSliceRep()const{return m_currentSlice;}
	virtual bool setSampleUnit(SampleUnit unit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	virtual void deleteLayer() override;
	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;

	IData* data() const override;
	//SliceDirection direction() const {return m_dir;}
	virtual TypeRep getTypeGraphicRep() override;

	GraphEditor_MultiPolyLineShape *getHorizonShape() { return m_polylineShape; }

	void setBuffer(CPUImagePaletteHolder* isoSurface );

	CPUImagePaletteHolder* isoSurfaceHolder()
		{
			return m_fixedLayer->isoSurfaceHolder();
		//	return m_isoSurfaceHolder;
		}

	virtual void buildContextMenu(QMenu *menu) override;

	public slots:
	void refresh();
	void addData();
	void addSismageHorizon();
	void computeAttributHorizon();

private:

	GraphEditor_MultiPolyLineShape *m_polylineShape = nullptr;



protected:
	HorizonFolderPropPanelOnRandom *m_propPanel;
	HorizonFolderLayerOnRandom *m_layer;
//	FixedRGBLayersFromDatasetAndCube *m_fixedLayer;
	CPUImagePaletteHolder* m_isoSurfaceHolder= nullptr;

	HorizonFolderData *m_fixedLayer;

//	SliceDirection m_dir;
	int m_currentSlice;


};

#endif
