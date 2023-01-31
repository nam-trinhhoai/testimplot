#ifndef Random3dTexRep_H
#define Random3dTexRep_H

#include <QObject>
#include "abstractgraphicrep.h"
class RandomTexDataset;
class Random3dTexLayer;

class Random3dTexRep: public AbstractGraphicRep {
Q_OBJECT
public:
Random3dTexRep(RandomTexDataset *nurbs, AbstractInnerView *parent = 0);
	virtual ~Random3dTexRep();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return true;
	}

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;
	virtual IData* data() const override;
	virtual AbstractGraphicRep::TypeRep getTypeGraphicRep() override;

private:
	RandomTexDataset *m_data;

	Random3dTexLayer *m_layer3D;
};

#endif
