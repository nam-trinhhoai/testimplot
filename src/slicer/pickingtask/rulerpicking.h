/*
 * PolygonMaskPainter.h
 *
 *  Created on: 29 mars 2018
 *      Author: l0380577
 */

#ifndef MURATAPP_SRC_VIEW_CANVAS2D_EXTENSIONS_RulerPicking_H_
#define MURATAPP_SRC_VIEW_CANVAS2D_EXTENSIONS_RulerPicking_H_

#include "pickingtask.h"

#include <QObject>
#include <QGraphicsLineItem>
#include <QGraphicsPathItem>
#include <QGraphicsSimpleTextItem>

//#include "data/MtLengthUnit.h"
class QGraphicsScene;

class RulerPicking : public PickingTask{
	Q_OBJECT
public:
	RulerPicking(QObject *parent, int thickness);
	virtual ~RulerPicking();

	//void setWorldUnit(  data::MtLengthUnit* worldUnit ) { this->worldUnit = worldUnit; };
	/**
	 * Extension methods
	 */
	virtual void mouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;
	virtual void mousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) override;
	virtual void initCanvas(QGraphicsScene* canvas);
	virtual void releaseCanvas(QGraphicsScene* canvas);
//	virtual void setEnabled(bool enabled) override;
//	virtual void keyReleaseEvent(QKeyEvent *keyEvent) override;
	
signals:
	void linePointed(const RulerPicking& origin, const QLineF& line);

private:
	QGraphicsLineItem line;
	QGraphicsSimpleTextItem* text;
	QLineF current;
	bool isElasticMove = false;
	int thickness = 5;
//	const data::MtLengthUnit* worldUnit = nullptr;
};

#endif /* MURATAPP_SRC_VIEW_CANVAS2D_EXTENSIONS_RulerPicking_H_ */
