/*
 * GraphicTool_GraphicLayer.h
 *
 *  Created on: Nov 16, 2021
 *      Author: l1046262
 */

#ifndef SRC_GRAPHICEDITOR_GraphicTool_GraphicLayer_H_
#define SRC_GRAPHICEDITOR_GraphicTool_GraphicLayer_H_

#include <QList>
#include "idata.h"
#include "marker.h"

#include "GraphicEditorFormsRepFactory.h"
#include "abstract2Dinnerview.h"

typedef enum {
	eSliceLayer,
	eLoadedLayer
} eLayerType ;

class GraphicTool_GraphicLayer : public IData
{
public:
	GraphicTool_GraphicLayer(WorkingSetManager *workingSet, QString name, QList<QGraphicsItem *> items,
			eLayerType type, Abstract2DInnerView* view): IData(workingSet), m_name(name), m_Type(type),
			m_ItemsList(items), m_View(view)
	{
		m_uuid = QUuid::createUuid();
		m_repFactory = new GraphicEditorFormsRepFactory(this, m_View);
	}

	virtual IGraphicRepFactory *graphicRepFactory() override {
		return m_repFactory;
	}

	void setType(eLayerType eType)
	{
		m_Type = eType;
	}

	eLayerType type()
	{
		return m_Type;
	}

	QList<QGraphicsItem *> itemsList()
	{
		return m_ItemsList;
	}

	QUuid dataID() const override{return m_uuid;}
	QString name() const override{return m_name;}

private:
	QString m_name;
	QUuid m_uuid;
	eLayerType m_Type;
	QList<QGraphicsItem *>m_ItemsList;
	Abstract2DInnerView* m_View;
	GraphicEditorFormsRepFactory* m_repFactory;
};

#endif /* SRC_GRAPHICEDITOR_GraphicTool_GraphicLayer_H_ */
