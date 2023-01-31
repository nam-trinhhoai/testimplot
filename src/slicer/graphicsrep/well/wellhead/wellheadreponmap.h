#ifndef WellHeadRepOnMap_H
#define WellHeadRepOnMap_H

#include <QObject>
#include <QMap>

#include "abstractgraphicrep.h"
#include "iRepGraphicItem.h"
#include "wellheadlayeronmap.h"

class WellHead;
#include "wellheadlayeronmap.h"

class WellHeadRepOnMap: public AbstractGraphicRep, public iRepGraphicItem {
Q_OBJECT
public:
	WellHeadRepOnMap(WellHead *wellHead, AbstractInnerView *parent = 0);
	virtual ~WellHeadRepOnMap();

	WellHead* wellHead() const;
	virtual IData* data() const override;
	virtual QString name() const override;

	virtual bool canBeDisplayed() const override {
		return true;
	}
	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	WellHeadLayerOnMap* layer();

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

	}

	virtual TypeRep getTypeGraphicRep() override;
private:
	WellHeadLayerOnMap *m_layer;
	WellHead* m_data;
};

#endif
