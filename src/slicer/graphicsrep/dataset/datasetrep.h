#ifndef DatasetRep_H
#define DatasetRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "isampledependantrep.h"
#include "idatacontrolerprovider.h"

class SeismicDataset3DLayer;
class Seismic3DAbstractDataset;


class DatasetRep: public AbstractGraphicRep, public ISampleDependantRep ,public IDataControlerProvider{
Q_OBJECT
public:
	DatasetRep(Seismic3DAbstractDataset *data, AbstractInnerView *parent = 0);
	virtual ~DatasetRep();

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer* layer(QGraphicsScene *scene, int defaultZDepth,
			QGraphicsItem *parent) override;

	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;

	virtual void buildContextMenu(QMenu *menu) override;
	IData* data() const override;

	virtual bool setSampleUnit(SampleUnit sampleUnit) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;

	virtual void setDataControler(DataControler *controler) override;
	virtual DataControler* dataControler() const override;
	virtual TypeRep getTypeGraphicRep() override;
private slots:
	void createInline();
	void createXline();
	void createAnimatedSurfaces();
	void createAnimatedSurfacesFromCube();
	void createVideoLayer();
	void deleteDatasetRep();
signals:
	void deletedRep(AbstractGraphicRep *rep);// MZR 15072021
	void delete3DRep();
private:
	SeismicDataset3DLayer *m_layer;
	Seismic3DAbstractDataset *m_data;

	DataControler *m_controler;
};

#endif
