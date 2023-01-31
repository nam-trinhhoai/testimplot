/*
 * GraphicEditorFormsLayer.h
 *
 *  Created on: Nov 18, 2021
 *      Author: l1046262
 */

#ifndef SRC_GRAPHICEDITOR_GraphicEditorFormsLayer_H_
#define SRC_GRAPHICEDITOR_GraphicEditorFormsLayer_H_

#include "graphiclayer.h"
#include <QObject>
#include <QRectF>

class QGraphicsView;
class QGraphicsItem;
class QEvent;
class GraphicEditorFormsRep;

class GraphicEditorFormsLayer : public GraphicLayer {
	Q_OBJECT
public:
	GraphicEditorFormsLayer(GraphicEditorFormsRep *rep,QGraphicsScene *scene, int defaultZDepth,QGraphicsItem * parent);
	virtual ~GraphicEditorFormsLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRectF boundingRect() const override;

	protected slots:
	virtual void refresh() override;

	private:
	GraphicEditorFormsRep* m_rep;
};



#endif /* SRC_GRAPHICEDITOR_GraphicEditorFormsLayer_H_ */
