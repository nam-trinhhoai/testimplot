#ifndef WellBoreRepOnMap_H
#define WellBoreRepOnMap_H

#include <QObject>
#include <QMap>

#include "abstractgraphicrep.h"
#include "isampledependantrep.h"
#include "iRepGraphicItem.h"
#include "wellborelayeronmap.h"

class WellBore;
#include "wellborelayeronmap.h"

class WellBoreLayer3D;

class WellBoreRepOnMap: public AbstractGraphicRep, public ISampleDependantRep, public iRepGraphicItem {
Q_OBJECT
public:
	WellBoreRepOnMap(WellBore *wellBore, AbstractInnerView *parent = 0);
	virtual ~WellBoreRepOnMap();

	WellBore* wellBore() const;
	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override {
		return true;
	}
	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;
	WellBoreLayerOnMap* layer();

	QGraphicsItem* graphicsItem() const override
	{
		if (m_layer)
		{
			return m_layer->graphicsItem();
		}
		return nullptr;
	}

	void autoDeleteRep() override
	{
		deleteWellBoreRepOnMap();
	}

	SampleUnit sampleUnit() const;
	virtual bool setSampleUnit(SampleUnit type) override;
	virtual QList<SampleUnit> getAvailableSampleUnits() const override;
	virtual QString getSampleUnitErrorMessage(SampleUnit sampleUnit) const override;
	virtual void buildContextMenu(QMenu *menu) override;
	virtual TypeRep getTypeGraphicRep() override;
private slots:
	void viewWellsLogRepOnMap();
	void deleteWellBoreRepOnMap();
signals:
    void deletedRep(AbstractGraphicRep *rep);// MZR 15072021

private:
	WellBoreLayerOnMap *m_layer;
	WellBore* m_data;

	SampleUnit m_sampleUnit = SampleUnit::NONE;
};

#endif
