#ifndef NurbsRep_H
#define NurbsRep_H

#include <QObject>
#include "abstractgraphicrep.h"
class NurbsDataset;
class NurbsLayer;
class QMenu;

class NurbsRep: public AbstractGraphicRep {
Q_OBJECT
public:
NurbsRep(NurbsDataset *nurbs, AbstractInnerView *parent = 0);
	virtual ~NurbsRep();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return true;
	}

	virtual void buildContextMenu(QMenu *menu) override;

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;
	virtual IData* data() const override;
	virtual AbstractGraphicRep::TypeRep getTypeGraphicRep() override;


	public slots:
//	void removeNurbs();
	void editerNurbs();

private:
	NurbsDataset *m_data;

	NurbsLayer *m_layer3D;
};

#endif
