#ifndef Random3dRep_H
#define Random3dRep_H

#include <QObject>
#include "abstractgraphicrep.h"
class RandomDataset;
class Random3dLayer;

class Random3dRep: public AbstractGraphicRep {
Q_OBJECT
public:
Random3dRep(RandomDataset *nurbs, AbstractInnerView *parent = 0);
	virtual ~Random3dRep();

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
	RandomDataset *m_data;

	Random3dLayer *m_layer3D;
};

#endif
